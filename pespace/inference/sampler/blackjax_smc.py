"""Adaptive-tempered SMC samplers with automatically tuned inner kernels.

Two samplers are provided, both running an adaptive-tempered SMC loop built
from blackjax primitives (ESS-based tempering, systematic resampling) with an
inner mutation kernel whose parameters are re-tuned from the particle cloud
at every tempering level:

- :class:`BlackjaxSMCwithNUTS` mutates the particles with NUTS on the
  unconstrained space prepared by :class:`NumpyroPriorSamplerBase` (the prior
  bijections and the inverse mapping back to physical parameters are handled
  by the base class).  Following SHARPy, a fixed shared ``step_size`` is
  used, and the per-particle dense ``inverse_mass_matrix`` is re-computed at
  every level as the inverse of the SoftAbs-regularized negative Hessian of
  the tempered log-posterior at each particle's position (position-dependent
  Riemannian metric, frozen along each NUTS trajectory).
- :class:`BlackjaxSMCwithIRMH` mutates the particles with an independent
  Metropolis-Hastings kernel directly in the physical parameter space; the
  Gaussian proposal ``mean`` / ``covariance`` are re-fitted to the particle
  cloud and the output is already physical.

For both samplers the number of mutation rounds per tempering level is a
fixed number of rounds given by ``num_mcmc_steps``.

With ``show_progress=True`` each level's mutation is wrapped in
:func:`blackjax.progress_bar` (requires the optional ``jaxtap`` package)
and per-level diagnostics (temperature, tempering increment and kernel
statistics) are printed.  With ``save_diagnosis=True`` each level's
mutation and tuning info and the diagnostic figures are stored under
``outdir/_diagnosis_info/level_{i}_mutate/``.
"""

from functools import partial

import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import corner
import numpy as np
import h5py

from .base import NumpyroPriorSamplerBase
from ..utils import map_fn

try:
    import blackjax
    from blackjax.smc import resampling
    from blackjax.smc.base import update_and_take_last
    from blackjax.smc.ess import log_ess
    from blackjax.smc.from_mcmc import unshared_parameters_and_step_fn
    from blackjax.smc.solver import dichotomy
    from blackjax.smc.tuning.from_particles import (
        particles_covariance_matrix,
        particles_means,
    )
except ImportError as err:
    raise ImportError(
        "blackjax (with `blackjax.irmh` and `blackjax.smc.tuning`) is not "
        "installed by default, please install blackjax manually."
    ) from err

__all__ = ["BlackjaxSMCwithNUTS", "BlackjaxSMCwithIRMH"]


def _stack_particles(particles) -> jnp.ndarray:
    """Ravel each particle into a row of an ``(n_particles, dim)`` matrix."""
    return jax.vmap(lambda particle: ravel_pytree(particle)[0])(particles)


def _square_mahal_distance(diff, inverse_mass_matrix):
    """Squared distance per row under the mass metric, ``dq^T M dq``.

    ``M = A^{-1}`` with ``A`` the per-particle blackjax
    ``inverse_mass_matrix`` of shape ``(n, d, d)`` (the quadratic form
    entering the kinetic energy); for a covariance ``A`` this is the
    classical Mahalanobis distance.  ``diff`` has shape ``(n, d)``.  The
    batched linear solve applies the metric without inverting every
    particle's matrix.
    """
    solved = jnp.linalg.solve(inverse_mass_matrix, diff[..., None])[..., 0]
    return jnp.einsum("ni,ni->n", diff, solved)


def _std_ratio(previous, particles):
    """Per-dimension ratio of the post/pre-mutation population std.

    Follows the plain-ratio semantics: a zero/zero dimension yields
    ``nan`` (degenerate before and after) and a positive/zero dimension
    yields ``inf`` (spread created by the mutation).
    """
    start = np.asarray(_stack_particles(previous))
    end = np.asarray(_stack_particles(particles))
    std_before = start.std(axis=0)
    std_after = end.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return std_after / std_before


def _hessian_matrix(fn, position) -> jnp.ndarray:
    """Dense ``(d, d)`` Hessian of a scalar pytree function.

    The position is flattened once with ``ravel_pytree`` and the Hessian is
    taken in the flattened coordinates, so the result is directly an
    ``(d, d)`` array (no pytree-of-pytrees assembly) in the same ordering
    ``ravel_pytree`` uses for the momentum -- the kernel's parameter
    ordering is matched by construction.
    """
    flat_position, unflatten = ravel_pytree(position)
    return jax.hessian(lambda flat: fn(unflatten(flat)))(flat_position)


def _softabs_metric(hessian, alpha):
    """SoftAbs-regularized metric of the Hessian (Betancourt 2013).

    ``G = V diag(lambda_i / tanh(alpha * lambda_i)) V^T`` with
    ``hessian = V diag(lambda_i) V^T``.  The eigenvalues are smoothed to
    ``|lambda_i|`` for ``|lambda_i| >> 1 / alpha`` and to the finite floor
    ``1 / alpha`` near zero, so the metric is positive definite and
    invertible everywhere, including saddle and negative-curvature regions
    of the log-posterior.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    soft_eigenvalues = eigenvalues / jnp.tanh(alpha * eigenvalues)
    return eigenvectors @ jnp.diag(soft_eigenvalues) @ eigenvectors.T


class _TunedAdaptiveTemperedSMC(NumpyroPriorSamplerBase):
    """Shared machinery for adaptive-tempered SMC with self-tuned kernels.

    Concrete samplers provide the inner mutation kernel and its tuning rule;
    this base class implements the tempering loop and the fixed-step mutation
    schedule.

    Hooks expected from subclasses (set in ``__init__``):

    The generic interfaces (``logprior_fn`` / ``loglikelihood_fn`` /
    ``logdensity_fn``) are bound on the instance to the space-specific
    implementations (unbounded or physical) before sampling.
    ``_mcmc_step_fn`` / ``_mcmc_init_fn``
        The inner MCMC kernel and its state initialiser.
    ``_get_particles_physical``
        Maps one particle (in the sampler's working space) to the physical
        sampled-parameter space.
    """

    # Type annotations, avoid initialize these hooks with None in __init__ of base class,
    # as that would cause complains of type checker.
    _mcmc_step_fn: Callable
    _mcmc_init_fn: Callable
    _get_particles_physical: Callable

    def __init__(
        self,
        likelihood,
        prior_dict,
        label=None,
        outdir=None,
        rkey_seed: int = 0,
        num_particles: int = 4096,
        target_ess: float = 0.9,
        num_mcmc_steps: int = 10,
        batch_size: int = 0,
        show_progress: bool = False,
        save_diagnosis: bool = False,
    ) -> None:
        super().__init__(likelihood, prior_dict, label, outdir, rkey_seed)
        self.num_particles = int(num_particles)
        self.target_ess = float(target_ess)
        self.num_mcmc_steps = int(num_mcmc_steps)
        self.batch_size = int(batch_size)
        self.show_progress = bool(show_progress)
        self.save_diagnosis = bool(save_diagnosis)
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative.")
        if self.num_mcmc_steps < 1:
            raise ValueError("num_mcmc_steps must be positive.")

        # Ravel/unravel utilities shared by both samplers; the template fixes
        # the pytree structure and the flattened dimension.
        self.rng_key, test_key = jax.random.split(self.rng_key)
        test_sample = self._sample_priors_unbounded(test_key, ())
        sample_data, self._unflatten_fn = ravel_pytree(test_sample)
        self._dim = int(sample_data.shape[0])

    # ------------------------------------------------------------------
    # mutation rounds
    # ------------------------------------------------------------------
    def _mutate(
        self,
        rng_key,
        particles,
        parameters,
        tempering_param,
        num_mcmc_steps: int = 1,
    ):
        """One mutation round: ``num_mcmc_steps`` kernel steps per particle.

        Runs ``num_mcmc_steps`` inner-kernel steps for every particle and
        keeps only the final states (blackjax's ``update_and_take_last``).
        The tempering loop passes the level's full ``self.num_mcmc_steps``.
        """

        def tempered_logposterior(position):
            return self.logprior_fn(position) + tempering_param * self.loglikelihood_fn(
                position
            )

        unshared, shared_step = unshared_parameters_and_step_fn(
            parameters, self._mcmc_step_fn
        )
        update_fn, _ = update_and_take_last(
            self._mcmc_init_fn,
            tempered_logposterior,
            shared_step,
            num_mcmc_steps,
            self.num_particles,
            batch_size=self.batch_size,
        )
        keys = jax.random.split(rng_key, self.num_particles)
        return update_fn(keys, particles, unshared)

    # ------------------------------------------------------------------
    # hooks implemented by the concrete samplers
    # ------------------------------------------------------------------
    def _initial_particles(self, key, num_particles):
        """Draw the initial particle population."""
        raise NotImplementedError

    def _tune_mcmc_parameters(self, key, particles, tempering_param):
        """Build the inner kernel parameters for the mutation rounds.

        Called after the resampling, so the parameters are re-fitted from
        the resampled particle cloud at every level.  Returns the tuned
        ``parameters`` and the tuning diagnostics ``tune_info`` (the empty
        tuple for both current kernels: NUTS recomputes its per-particle
        mass matrices, IRMH refits its proposal); tuning never modifies
        the particles.
        """
        raise NotImplementedError

    def _mutate_eval_counts(self, mutate_info):
        """Number of log-density and gradient evaluations of one mutation
        call.

        Implemented by the concrete samplers; ``mutate_info`` is the info
        pytree returned by the level's mutation call.
        """
        raise NotImplementedError

    def _tune_eval_counts(self, tune_info):
        """Number of log-density and gradient evaluations of the tuning
        probe, read from ``tune_info``.

        Implemented by the concrete samplers; kernels without a tuning
        probe report zero counts.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # inference loop
    # ------------------------------------------------------------------
    def _temper(
        self,
        key,
        particles,
        tempering_param,
        log_evidence,
    ):
        """One tempering level up to (but excluding) the mutation.

        Computes the ESS-optimal tempering increment from the current
        particle likelihoods (evaluated by the instance-level
        ``_batched_loglikelihood``, bound by :meth:`run`), accumulates the
        normalising-constant increment into the log-evidence,
        systematic-resamples the particles and re-tunes the inner kernel
        from the resampled cloud (both current kernels re-fit all their
        kernel parameters there, so no per-particle parameter resampling
        is needed).  The input RNG key is
        consumed by the split into the resampling and tuning subkeys;
        nothing is returned for the caller.  Returns the resampled
        particles, the tuned parameters, the new tempering parameter, the
        updated log-evidence, the tempering increment and the tuning
        diagnostics.
        """
        resample_key, tune_key = jax.random.split(key, 2)

        loglikelihoods = self._batched_loglikelihood(particles)
        # Solve for the ESS-optimal tempering increment, mirroring
        # ``blackjax.smc.ess.ess_solver`` but reusing the likelihood
        # population evaluated above (instead of re-evaluating it).
        target_val = jnp.log(self.num_particles * self.target_ess)

        def fun_to_solve(delta):
            log_weights = jnp.nan_to_num(delta * loglikelihoods)
            return log_ess(log_weights) - target_val

        delta = dichotomy(fun_to_solve, 0.0, 1.0 - tempering_param)
        tempering_param = tempering_param + delta
        log_weights = delta * loglikelihoods
        log_evidence = (
            log_evidence + logsumexp(log_weights) - jnp.log(self.num_particles)
        )
        indices = resampling.systematic(
            resample_key,
            jnp.exp(log_weights - logsumexp(log_weights)),
            self.num_particles,
        )
        particles = jax.tree.map(lambda p: p[indices], particles)
        mcmc_params, tune_info = self._tune_mcmc_parameters(
            tune_key, particles, tempering_param
        )
        return (
            particles,
            mcmc_params,
            tune_info,
            tempering_param,
            log_evidence,
            delta,
        )

    def run(self) -> dict:
        """Run the adaptive-tempered SMC sampler.

        Returns
        -------
        dict
            Run result containing ``label``, ``outdir``, ``rkey_seed``,
            ``prior_dict``, ``likelihood_class``, ``runing_time``,
            ``sampler_class``, ``sampler_config``, ``posteriors`` (physical
            sampled/derived/fixed parameters), ``log_likelihoods``, the
            log-evidence ``log_evidence`` and normalized importance
            ``weights``.  The result is also saved to
            ``outdir/{label}_results.json``.
        """
        self.rng_key, init_key, loop_key = jax.random.split(self.rng_key, 3)
        particles = self._initial_particles(init_key, self.num_particles)
        self._batched_loglikelihood = map_fn(self.loglikelihood_fn, self.batch_size)
        start = time.perf_counter()
        particles, log_evidence = self._inference_loop(loop_key, particles)
        output = self._finalize(
            particles, log_evidence, runing_time=time.perf_counter() - start
        )
        self._save_output(output)
        return output

    def _inference_loop(self, key, particles):
        """Reweight-resample-mutate loop, terminating at tempering parameter 1.

        A plain Python ``while`` drives the levels; each level runs the
        jitted :meth:`_temper` (tempering increment, resampling and
        inner-kernel tuning) followed by the jitted :meth:`_mutate` (the
        fixed number of mutation rounds).  With ``show_progress`` the
        mutation call is wrapped in :func:`blackjax.progress_bar` and each
        level's diagnostics are printed.

        Notes
        -----
        The progress bar relies on ``jaxtap`` instrumenting the scan at
        trace time.  The jitted closures are recreated on every
        ``_inference_loop`` call, so the mutation kernel is (re)traced
        inside the active context whenever ``show_progress=True`` -- no
        ``jax.clear_caches()`` dance is needed when flipping the flag
        between runs.
        """
        temper_jit = jax.jit(self._temper)
        # ``num_mcmc_steps`` is bound once by ``partial``, so the jitted
        # wrapper needs no ``static_argnums``.
        mutate_jit = jax.jit(partial(self._mutate, num_mcmc_steps=self.num_mcmc_steps))

        tempering_param = 0.0
        log_evidence = 0.0
        level = 0
        cum_ll = 0
        cum_grad = 0
        n_batch = (
            (self.num_particles + self.batch_size - 1) // self.batch_size
            if self.batch_size > 0
            else 1
        )
        while tempering_param < 1.0:
            level += 1
            start = time.perf_counter()
            # Derive this level's subkeys in the Python scope: the incoming
            # ``key`` is consumed by the split and replaced by the parent
            # of the next level; ``temper_key`` and ``move_key`` are each
            # used exactly once below and never again.
            key, temper_key, move_key = jax.random.split(key, 3)
            (
                particles,
                mcmc_params,
                tune_info,
                tempering_param,
                log_evidence,
                delta,
            ) = temper_jit(
                temper_key,
                particles,
                tempering_param,
                log_evidence,
            )
            previous = particles
            if self.show_progress:
                print(
                    f"level {level}: lambda={float(tempering_param):.6g} "
                    f"(delta={float(delta):.4g})  "
                    f"logZ {float(log_evidence):.4f}"
                )
                with blackjax.progress_bar(label=f"mutate (n_batch={n_batch})"):
                    particles, mutate_info = mutate_jit(
                        move_key,
                        particles,
                        mcmc_params,
                        tempering_param,
                    )
                # temper: one batched likelihood evaluation of all
                # particles, plus the tuning probe's evaluations.
                tune_ll, tune_grad = self._tune_eval_counts(tune_info)
                cum_ll += self.num_particles + tune_ll
                cum_grad += tune_grad
                mutate_ll, mutate_grad = self._mutate_eval_counts(mutate_info)
                cum_ll += mutate_ll
                cum_grad += mutate_grad
                self._print_diagnose_info(
                    mcmc_params,
                    mutate_info,
                    previous,
                    particles,
                    cum_ll,
                    cum_grad,
                    time.perf_counter() - start,
                )
            else:
                particles, mutate_info = mutate_jit(
                    move_key,
                    particles,
                    mcmc_params,
                    tempering_param,
                )
            if self.save_diagnosis:
                self._save_diagnosis(
                    level,
                    previous,
                    particles,
                    mcmc_params,
                    mutate_info,
                    tune_info,
                )
        return particles, log_evidence

    def _print_diagnose_info(
        self,
        mcmc_params,
        mutate_info,
        previous,
        particles,
        cum_ll,
        cum_grad,
        elapsed,
    ):
        """Print kernel-specific mutation diagnostics for one level.

        Implemented by the concrete samplers; ``mutate_info`` is the info
        pytree returned by the level's mutation call (leaf shape
        ``(num_particles, num_mcmc_steps)``), ``previous`` / ``particles``
        are the pre/post-mutation populations and ``cum_ll`` /
        ``cum_grad`` the cumulative evaluation counts.
        """
        raise NotImplementedError

    def _save_diagnosis(
        self, level, previous, particles, mcmc_params, mutate_info, tune_info
    ):
        """Save the level's mutation and tuning info and figures to disk.

        Creates ``outdir/_diagnosis_info/level_{level}_mutate/`` and stores
        the mutation info pytree (always a namedtuple, possibly with nested
        namedtuples) in ``mutate_info.h5`` and, when ``tune_info`` is not
        the empty tuple, the tuning diagnostics in ``tune_info.h5``
        (HDF5).  Each pytree is expanded once with
        ``jax.tree_util.tree_flatten_with_path``; every key-path entry is
        then a ``GetAttrKey`` field name, which becomes the nested group
        layout, and only the array leaves become datasets (shapes kept).
        The kernel-specific figures produced by :meth:`_diagnosis_plots`
        receive the pre/post-mutation populations, the mutation
        parameters, the original ``mutate_info`` and ``tune_info``.
        """
        level_dir = os.path.join(
            self.outdir, "_diagnosis_info", f"level_{level}_mutate"
        )
        os.makedirs(level_dir, exist_ok=True)
        leaves, _ = jax.tree_util.tree_flatten_with_path(mutate_info)
        with h5py.File(os.path.join(level_dir, "mutate_info.h5"), "w") as f:
            for path, leaf in leaves:
                names = [entry.name for entry in path]
                group = f
                for name in names[:-1]:
                    group = group.require_group(name)
                group.create_dataset(names[-1], data=np.asarray(leaf))
        if tune_info:
            tune_leaves, _ = jax.tree_util.tree_flatten_with_path(tune_info)
            with h5py.File(os.path.join(level_dir, "tune_info.h5"), "w") as f:
                for path, leaf in tune_leaves:
                    names = [entry.name for entry in path]
                    group = f
                    for name in names[:-1]:
                        group = group.require_group(name)
                    group.create_dataset(names[-1], data=np.asarray(leaf))
        self._diagnosis_plots(
            previous, particles, mcmc_params, mutate_info, tune_info, level_dir
        )

    def _diagnosis_plots(
        self, previous, particles, mcmc_params, mutate_info, tune_info, level_dir
    ):
        """Draw kernel-specific diagnostic figures into ``level_dir``.

        ``previous`` / ``particles`` are the pre/post-mutation
        populations, ``mcmc_params`` the mutation parameters,
        ``mutate_info`` the level's original mutation info namedtuple and
        ``tune_info`` the tuning diagnostics (or the empty tuple).
        """
        raise NotImplementedError

    def _finalize(self, particles, log_evidence, runing_time) -> dict:
        """Assemble the final population into the run-result dictionary.

        The particles are restored to the physical space and reported as
        ``sampled_params``; the ``fixed_params`` are broadcast to the
        particle count, and the ``derived_params`` are taken from the
        likelihood-input assembly, which also provides the parameters used
        to evaluate the likelihood directly.  The particles are equally
        weighted, so ``weights`` is left as None (stored as an empty
        array).
        """
        particles_phy = self._get_particles_physical(particles)
        ll_params = map_fn(self._likelihood_params_from_physical, self.batch_size)(
            particles_phy
        )
        log_likelihoods = map_fn(self.likelihood.log_likelihood, self.batch_size)(
            ll_params
        )

        sampled = {
            name: np.asarray(value) for name, value in particles_phy._asdict().items()
        }
        fixed = {
            name: np.full(self.num_particles, value)
            for name, value in self.fixed_params.items()
        }
        derived_names = [
            name
            for outputs in self.derived_params
            for name in (outputs if isinstance(outputs, tuple) else (outputs,))
        ]
        derived = {name: np.asarray(getattr(ll_params, name)) for name in derived_names}

        posterior_samples = {
            "sampled_params": sampled,
            "derived_params": derived,
            "fixed_params": fixed,
        }

        return self.format_output(
            posterior_samples, log_likelihoods, log_evidence, runing_time, weights=None
        )


class BlackjaxSMCwithNUTS(_TunedAdaptiveTemperedSMC):
    """Adaptive-tempered SMC with a NUTS mutation kernel.

    NUTS requires an unbounded domain, so sampling happens on the
    unconstrained space prepared by :class:`NumpyroPriorSamplerBase` (the
    prior bijections and the inverse transform back to physical parameters
    are handled by the base class).

    At every tempering level the per-particle dense ``inverse_mass_matrix``
    is re-computed as the inverse of the SoftAbs-regularized negative
    Hessian of the tempered log-posterior evaluated at each particle's
    position (a position-dependent Riemannian metric, frozen along each
    NUTS trajectory, as in SHARPy).  A single fixed ``step_size`` is used
    for all particles and levels; no step-size tuning is performed.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing ``log_likelihood`` and the
        ``likelihood_param_type`` namedtuple type of the likelihood parameters.
    prior : dict
        Prior declaration, see
        :class:`pespace.inference.sampler.base.NumpyroPriorSamplerBase`.
    label : str, optional
        Label of the run, converted to ``str``; also the JSON filename
        (``outdir/{label}_results.json``).  Default is None (-> ``"None"``).
    outdir : str, optional
        Output directory for the saved result.  It must not exist yet: the
        base class creates it and refuses to reuse an existing path, so that
        the results of an earlier run are never overwritten.  Default is None,
        which resolves to an ``outdir_{label}`` directory under the current
        working directory.
    rkey_seed : int, optional
        Seed for the JAX random number generator key.  Default is 0.
    num_particles : int, optional
        Number of SMC particles.  Default is 4096.
    target_ess : float, optional
        Target ESS ratio of the adaptive tempering schedule.  Default is 0.9.
    num_mcmc_steps : int, optional
        Fixed number of mutation rounds per tempering level.  Default is 1.
    batch_size : int, optional
        Number of particles processed per sequential batch when positive.
        ``0`` (default) keeps the plain ``jax.vmap`` behaviour.  A positive
        value bounds the peak memory of the likelihood intermediates and the
        mutation updates, at the cost of cross-batch parallelism.  The
        per-particle Hessian tuning is chunked separately via
        ``hessian_batch_size``.
    show_progress : bool, optional
        When True, wrap each level's mutation in
        :func:`blackjax.progress_bar` (requires the optional ``jaxtap``
        package) and print per-level diagnostics (temperature, tempering
        increment and the step-size / leapfrog / acceptance statistics
        below).  Default is False.
    save_diagnosis : bool, optional
        When True, save each level's full mutation info and the diagnostic
        figures under ``outdir/_diagnosis_info/level_{i}_mutate/``
        (mutate_info.h5, hist_num_integration_steps.png, particles_Mahal.png
        and particles_corner.png; no per-particle tuning info is written).
        Default is False.
    step_size : float, optional
        Fixed NUTS step size shared by all particles and levels (no
        step-size tuning is performed).  Default is 0.2.
    softabs_alpha : float, optional
        SoftAbs regularization parameter of the position-dependent metric
        (Betancourt 2013): the metric eigenvalues are
        ``lambda / tanh(alpha * lambda)``, so near-flat directions get a
        finite mass ``1 / alpha`` and the metric stays positive definite
        everywhere.  Default is 0.3.
    hessian_batch_size : int, optional
        Number of particles per sequential chunk when the per-particle
        Hessians of the mass-matrix tuning are evaluated.  The Hessian
        evaluation is far more memory-hungry than a plain likelihood
        evaluation (roughly ``dim`` forward tangents are materialised per
        particle), so it is chunked separately from ``batch_size``: ``0``
        keeps the plain ``jax.vmap`` behaviour and a positive value caps
        the chunk (and hence the peak memory) of the tuning evaluation.
        Lower it when the GPU runs out of memory.  Default is 32.
    max_num_doublings : int, optional
        Maximum number of trajectory doublings of the NUTS kernel (at most
        ``2**max_num_doublings`` leapfrog steps per trajectory).  Default
        is 7.
    divergence_threshold : int, optional
        Absolute energy error above which a transition is flagged as
        divergent.  Default is 100.
    """

    def __init__(
        self,
        likelihood,
        prior_dict,
        label=None,
        outdir=None,
        rkey_seed: int = 0,
        num_particles: int = 4096,
        target_ess: float = 0.9,
        num_mcmc_steps: int = 1,
        batch_size: int = 0,
        show_progress: bool = False,
        save_diagnosis: bool = False,
        step_size: float = 0.2,
        softabs_alpha: float = 0.3,
        hessian_batch_size: int = 32,
        max_num_doublings: int = 7,
        divergence_threshold: int = 100,
    ) -> None:
        super().__init__(
            likelihood,
            prior_dict,
            label,
            outdir,
            rkey_seed,
            num_particles,
            target_ess,
            num_mcmc_steps,
            batch_size,
            show_progress,
            save_diagnosis,
        )
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if hessian_batch_size < 0:
            raise ValueError("hessian_batch_size must be non-negative.")
        self.step_size = float(step_size)
        self.softabs_alpha = float(softabs_alpha)
        self.hessian_batch_size = int(hessian_batch_size)
        self.max_num_doublings = int(max_num_doublings)
        self.divergence_threshold = int(divergence_threshold)

        # NUTS is gradient-based: bind the generic density interface to the
        # unbounded-space functions (setattr keeps strict type checkers quiet
        # about the differing parameter names).
        setattr(self, "logprior_fn", self.logprior_unbounded_fn)
        setattr(self, "loglikelihood_fn", self.loglikelihood_unbounded_fn)
        setattr(self, "logdensity_fn", self.logdensity_unbounded_fn)
        kernel = blackjax.nuts.build_kernel(
            divergence_threshold=self.divergence_threshold
        )
        self._mcmc_step_fn = partial(kernel, max_num_doublings=self.max_num_doublings)
        self._mcmc_init_fn = blackjax.nuts.init
        self._get_particles_physical = self._to_physical_params

    def _initial_particles(self, key, num_particles):
        return self._sample_priors_unbounded(key, (num_particles,))

    def _tune_mcmc_parameters(self, key, particles, tempering_param):
        """Build the per-particle Hessian-based mass matrices.

        The per-particle dense ``inverse_mass_matrix`` is
        ``SoftAbs(-hessian)^{-1}`` of the tempered log-posterior at each
        (resampled) particle's position (blackjax convention: the
        quadratic form entering the kinetic energy).  SoftAbs keeps the
        metric positive definite everywhere and gives near-flat directions
        a finite mass ``1 / alpha``; the matrix is frozen along each NUTS
        trajectory, so the standard velocity-Verlet integrator and the
        detailed-balance argument remain valid.  The evaluation is chunked
        with ``hessian_batch_size`` to bound the peak memory (one Hessian
        costs roughly ``dim`` gradient evaluations per particle).  The
        fixed step size is rebuilt here and no probe move is performed, so
        the tuning returns the empty tuple as diagnostics.
        """
        del key

        def logposterior(position):
            return self.logprior_fn(position) + tempering_param * self.loglikelihood_fn(
                position
            )

        def per_particle(position):
            hessian = _hessian_matrix(logposterior, position)
            return jnp.linalg.inv(_softabs_metric(-hessian, self.softabs_alpha))

        inverse_mass_matrix = map_fn(per_particle, self.hessian_batch_size)(particles)
        return {
            # Shared parameter (shape[0] == 1): bound once by the SMC
            # machinery instead of being mapped per particle.
            "step_size": jnp.full((1,), self.step_size),
            "inverse_mass_matrix": inverse_mass_matrix,
        }, ()

    def _mutate_eval_counts(self, mutate_info):
        """NUTS: one value-and-gradient evaluation per particle for the
        init, plus one per leapfrog step of every trajectory."""
        leaps = int(np.asarray(mutate_info.num_integration_steps).sum())
        return self.num_particles + leaps, self.num_particles + leaps

    def _tune_eval_counts(self, tune_info):
        """NUTS: one value evaluation per particle for the level's mass
        matrices, plus roughly ``dim`` gradient evaluations per particle
        for the Hessians (``jax.hessian`` is forward-over-reverse)."""
        del tune_info
        return self.num_particles, self.num_particles * self._dim

    def _print_diagnose_info(
        self,
        mcmc_params,
        mutate_info,
        previous,
        particles,
        cum_ll,
        cum_grad,
        elapsed,
    ):
        """NUTS diagnostics for one level: the fixed step size, leapfrog
        length, acceptance rate, divergent-transition fraction, Mahalanobis
        distances, the fraction of stationary particles (unchanged
        positions), per-dimension std ratios and cumulative evaluation
        counts."""
        # The step size is fixed and shared by all particles and levels.
        step_size = np.asarray(mcmc_params["step_size"]).item()
        steps = np.asarray(mutate_info.num_integration_steps)
        acceptance = np.asarray(mutate_info.acceptance_rate)
        divergences = np.asarray(mutate_info.is_divergent)
        start = np.asarray(_stack_particles(previous))
        end = np.asarray(_stack_particles(particles))
        inverse_mass_matrix = np.asarray(mcmc_params["inverse_mass_matrix"])
        mahal = np.sqrt(
            np.asarray(_square_mahal_distance(start - end, inverse_mass_matrix))
        )
        unmoved = int(np.count_nonzero(mahal == 0.0))
        std_ratio = _std_ratio(previous, particles)

        print(f"  step_size   {step_size:.4g}")
        print(
            f"  leapfrog    mean {steps.mean():.0f}  "
            f"med {np.median(steps):.0f}  "
            f"[{steps.min()}, {steps.max()}]"
        )
        print(
            f"  acceptance  mean {acceptance.mean():.3f}  "
            f"med {np.median(acceptance):.3f}"
        )
        print(f"  divergence  {100.0 * divergences.mean():.1f}%")
        print(
            f"  Mahal       mean {mahal.mean():.4g}  "
            f"med {np.median(mahal):.4g}  "
            f"[{mahal.min():.4g}, {mahal.max():.4g}]"
        )
        print(
            f"  unmoved     {unmoved}/{mahal.size}  "
            f"({100.0 * unmoved / mahal.size:.1f}%)"
        )
        print(f"  std ratio   " f"{np.array2string(std_ratio, precision=4)}")
        print(f"  cum ll      {cum_ll:,}")
        print(f"  cum grad    {cum_grad:,}")
        print(f"  ({elapsed:.2f}s)")

    def _diagnosis_plots(
        self, previous, particles, mcmc_params, mutate_info, tune_info, level_dir
    ):
        """NUTS diagnostic figures: the integration-step histogram, the
        Mahalanobis distance histogram and the pre/post-mutation corner
        plot."""

        steps = np.asarray(mutate_info.num_integration_steps).ravel()

        fig, ax = plt.subplots()
        n_bins = int(steps.max() - steps.min() + 1) if steps.size else 1
        ax.hist(steps, bins=min(40, n_bins))
        ax.set_xlabel("num_integration_steps")
        ax.set_ylabel("counts")
        ax.set_title("hist num_integration_steps")
        fig.savefig(os.path.join(level_dir, "hist_num_integration_steps.png"))
        plt.close(fig)

        start = np.asarray(_stack_particles(previous))
        end = np.asarray(_stack_particles(particles))
        inverse_mass_matrix = np.asarray(mcmc_params["inverse_mass_matrix"])
        mahal = np.sqrt(
            np.asarray(_square_mahal_distance(start - end, inverse_mass_matrix))
        )
        fig, ax = plt.subplots()
        ax.hist(mahal, bins=40)
        ax.set_xlabel("Mahalanobis distance")
        ax.set_ylabel("counts")
        ax.set_title("Mahalanobis distance after mutation")
        fig.savefig(os.path.join(level_dir, "particles_Mahal.png"))
        plt.close(fig)

        before = np.asarray(_stack_particles(previous))
        after = np.asarray(_stack_particles(particles))
        labels = [f"x{i}" for i in range(before.shape[1])]
        fig = corner.corner(
            before,
            labels=labels,
            color="tab:blue",
            show_titles=True,
        )
        corner.corner(
            after,
            labels=labels,
            color="tab:orange",
            fig=fig,
            show_titles=True,
        )
        fig.savefig(os.path.join(level_dir, "particles_corner.png"))
        plt.close(fig)


class BlackjaxSMCwithIRMH(_TunedAdaptiveTemperedSMC):
    """Adaptive-tempered SMC with an independent MH mutation kernel.

    Sampling is performed directly in the physical parameter space, so the
    output is already physical and needs no post-processing.  The Gaussian
    proposal ``mean`` / ``covariance`` are re-fitted to the particle cloud at
    every tempering level.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing ``log_likelihood`` and the
        ``likelihood_param_type`` namedtuple type of the likelihood parameters.
    prior : dict
        Prior declaration, see
        :class:`pespace.inference.sampler.base.NumpyroPriorSamplerBase`.
    label : str, optional
        Label of the run, converted to ``str``; also the JSON filename
        (``outdir/{label}_results.json``).  Default is None (-> ``"None"``).
    outdir : str, optional
        Output directory for the saved result.  It must not exist yet: the
        base class creates it and refuses to reuse an existing path, so that
        the results of an earlier run are never overwritten.  Default is None,
        which resolves to an ``outdir_{label}`` directory under the current
        working directory.
    rkey_seed : int, optional
        Seed for the JAX random number generator key.  Default is 0.
    num_particles : int, optional
        Number of SMC particles.  Default is 4096.
    target_ess : float, optional
        Target ESS ratio of the adaptive tempering schedule.  Default is 0.9.
    num_mcmc_steps : int, optional
        Fixed number of mutation rounds per tempering level.  Default is 100.
    batch_size : int, optional
        Number of particles processed per sequential batch when positive.
        ``0`` (default) keeps the plain ``jax.vmap`` behaviour.  A positive
        value bounds the peak memory of the likelihood intermediates and the
        mutation updates, at the cost of cross-batch parallelism.
    show_progress : bool, optional
        When True, wrap each level's mutation in
        :func:`blackjax.progress_bar` (requires the optional ``jaxtap``
        package) and print per-level diagnostics (temperature, tempering
        increment and the acceptance / proposal-scale statistics below).
        Default is False.
    save_diagnosis : bool, optional
        When True, save each level's full mutation info under
        ``outdir/_diagnosis_info/level_{i}_mutate/`` (mutate_info.h5;
        IRMH has no per-particle tuning info, so no tune_info.h5 is
        written).  No diagnostic figures are drawn yet for IRMH.  Default
        is False.
    covariance_regularization : float, optional
        Diagonal regularization added to the fitted proposal covariance.
        Default is 1e-6.
    """

    def __init__(
        self,
        likelihood,
        prior_dict,
        label=None,
        outdir=None,
        rkey_seed: int = 0,
        num_particles: int = 4096,
        target_ess: float = 0.9,
        num_mcmc_steps: int = 100,
        batch_size: int = 0,
        show_progress: bool = False,
        save_diagnosis: bool = False,
        covariance_regularization: float = 1e-6,
    ) -> None:
        super().__init__(
            likelihood,
            prior_dict,
            label,
            outdir,
            rkey_seed,
            num_particles,
            target_ess,
            num_mcmc_steps,
            batch_size,
            show_progress,
            save_diagnosis,
        )
        self.covariance_regularization = float(covariance_regularization)

        # IRMH is not gradient-based and samples directly in the physical
        # space: bind the generic density interface to the physical-space
        # functions.
        setattr(self, "logprior_fn", self.logprior_physical_fn)
        setattr(self, "loglikelihood_fn", self.loglikelihood_physical_fn)
        setattr(self, "logdensity_fn", self.logdensity_physical_fn)
        self._irmh_kernel = blackjax.irmh.build_kernel()
        self._mcmc_step_fn = self._irmh_step_fn
        self._mcmc_init_fn = blackjax.irmh.init
        self._get_particles_physical = lambda particles: particles

    def _irmh_step_fn(self, rng_key, state, logdensity_fn, mean, covariance):
        """IRMH kernel with a shared Gaussian proposal ``N(mean, covariance)``."""

        def sample_proposal(key):
            return self._unflatten_fn(
                jax.random.multivariate_normal(key, mean, covariance)
            )

        def proposal_logdensity(from_state, to_state):
            # blackjax calls this as ``(new_state, prev_state)``; the density
            # of the ``from_state -> to_state`` transition is the proposal
            # density at the destination ``to_state`` for an independent
            # proposal, so ``from_state`` is ignored and the standard IMH
            # acceptance ratio ``p(x') q(x) / p(x) q(x')`` is recovered.
            del from_state
            return jax.scipy.stats.multivariate_normal.logpdf(
                ravel_pytree(to_state.position)[0], mean, covariance
            )

        return self._irmh_kernel(
            rng_key, state, logdensity_fn, sample_proposal, proposal_logdensity
        )

    def _initial_particles(self, key, num_particles):
        return self._sample_priors_physical(key, (num_particles,))

    def _tune_mcmc_parameters(self, key, particles, tempering_param):
        # the proposal is fitted from the particles before the level's
        # mutation; no measurement move is needed.
        del key, tempering_param
        return self._fitted_proposal(particles), ()

    def _tune_eval_counts(self, tune_info):
        """IRMH: the proposal is fitted without a measurement move, so the
        tuning probe contributes no evaluations."""
        del tune_info
        return 0, 0

    def _mutate_eval_counts(self, mutate_info):
        """IRMH: one log-density evaluation per particle for the init plus
        one per mutation round; no gradients."""
        del mutate_info
        return self.num_particles * (self.num_mcmc_steps + 1), 0

    def _print_diagnose_info(
        self,
        mcmc_params,
        mutate_info,
        previous,
        particles,
        cum_ll,
        cum_grad,
        elapsed,
    ):
        """IRMH diagnostics for one level: acceptance rate and the proposal
        scale range."""
        del previous, particles, cum_ll, cum_grad
        acceptance = np.asarray(mutate_info.acceptance_rate)
        std = np.sqrt(np.diag(np.asarray(mcmc_params["covariance"][0])))
        print(
            f"  acceptance  mean {acceptance.mean():.3f}  "
            f"med {np.median(acceptance):.3f}"
        )
        print(
            f"  proposal sigma[{std.min():.4g}, {std.max():.4g}]  "
            f"log10-span {np.log10(std.max() / std.min()):.2f}  "
            f"({elapsed:.2f}s)"
        )

    def _diagnosis_plots(
        self, previous, particles, mcmc_params, mutate_info, tune_info, level_dir
    ):
        """Placeholder: no IRMH diagnostic figures implemented yet."""
        del previous, particles, mcmc_params, mutate_info, tune_info, level_dir

    def _fitted_proposal(self, particles):
        """Gaussian proposal parameters fitted to the current particle cloud."""
        covariance = particles_covariance_matrix(particles)
        covariance = covariance + self.covariance_regularization * jnp.eye(self._dim)
        return {
            "mean": particles_means(particles)[None, ...],
            "covariance": covariance[None, ...],
        }
