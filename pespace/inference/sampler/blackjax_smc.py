"""Adaptive-tempered SMC samplers with automatically tuned inner kernels.

Two samplers are provided, both running an adaptive-tempered SMC loop built
from blackjax primitives (ESS-based tempering, systematic resampling) with an
inner mutation kernel whose parameters are re-tuned from the particle cloud
at every tempering level:

- :class:`BlackjaxSMCwithNUTS` mutates the particles with NUTS on the
  unconstrained space prepared by :class:`NumpyroPriorSamplerBase` (the prior
  bijections and the inverse mapping back to physical parameters are handled
  by the base class).  The per-particle ``step_size`` is tuned with a
  feedback scheme based on the realised squared-jump distance, and the shared
  ``inverse_mass_matrix`` is re-estimated from the particle variance.
- :class:`BlackjaxSMCwithIRMH` mutates the particles with an independent
  Metropolis-Hastings kernel directly in the physical parameter space; the
  Gaussian proposal ``mean`` / ``covariance`` are re-fitted to the particle
  cloud and the output is already physical.

For both samplers the number of mutation rounds per tempering level is
determined automatically from the inter-generation lag-1 autocorrelation of
the particles (evaluated on the ``x + x**2`` transform), capped by
``max_num_moves``.
"""

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp

from .base import NumpyroPriorSamplerBase

try:
    import blackjax
    from blackjax.smc import resampling
    from blackjax.smc.base import update_and_take_last
    from blackjax.smc.ess import ess_solver
    from blackjax.smc.from_mcmc import unshared_parameters_and_step_fn
    from blackjax.smc.solver import dichotomy
    from blackjax.smc.tuning.from_particles import (
        inverse_mass_matrix_from_particles,
        particles_covariance_matrix,
        particles_means,
    )
except ImportError:
    logging.error(
        "blackjax (with `blackjax.irmh` and `blackjax.smc.tuning`) is not "
        "installed by default, please install blackjax manually."
    )
    raise

__all__ = ["BlackjaxSMCwithNUTS", "BlackjaxSMCwithIRMH"]


def _stack_particles(particles) -> jnp.ndarray:
    """Ravel each particle into a row of an ``(n_particles, dim)`` matrix."""
    return jax.vmap(lambda particle: ravel_pytree(particle)[0])(particles)


def _lag1_corr(previous: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
    """Component-wise lag-1 correlation on the ``x + x**2`` transform.

    The transform makes the criterion insensitive to the sign of the
    components (following the HSMC move-length criterion).
    """
    previous = previous + previous**2
    current = current + current**2
    previous = previous - jnp.mean(previous, axis=0)
    current = current - jnp.mean(current, axis=0)
    denominator = jnp.sqrt(jnp.sum(previous**2, axis=0) * jnp.sum(current**2, axis=0))
    return jnp.sum(previous * current, axis=0) / (denominator + 1e-16)


class _TunedAdaptiveTemperedSMC(NumpyroPriorSamplerBase):
    """Shared machinery for adaptive-tempered SMC with self-tuned kernels.

    Concrete samplers provide the inner mutation kernel and its tuning rule;
    this base class implements the tempering loop and the autocorrelation-
    based mutation stopping rule.

    Hooks expected from subclasses (set in ``__init__``):

    ``_logprior_fn`` / ``_loglikelihood_fn``
        Space-specific log-density functions used by the tempered kernel and
        the mutation rounds.
    ``_mcmc_step_fn`` / ``_mcmc_init_fn``
        The inner MCMC kernel and its state initialiser.
    ``_params_to_likelihood``
        Maps one particle to the likelihood-namespace parameter dict.
    """

    # Define the default value as the class-level attribute.
    # Subclasses can override this default without redefining __init__.
    _DEFAULT_MAX_MOVES = 50

    # Type annotations, avoid initialize these hooks with None in __init__ of base class,
    # as that would cause complains of type checker.
    _logprior_fn: Callable
    _loglikelihood_fn: Callable
    _mcmc_step_fn: Callable
    _mcmc_init_fn: Callable
    _params_to_likelihood: Callable

    def __init__(
        self,
        likelihood,
        prior,
        rkey_seed: int = 0,
        num_particles: int = 200,
        target_ess: float = 0.8,
        max_num_moves: int | None = None,
        correlation_threshold: float = 0.1,
        min_correlated_fraction: float = 0.3,
    ) -> None:
        super().__init__(likelihood, prior, rkey_seed)
        self.num_particles = int(num_particles)
        self.target_ess = float(target_ess)
        self.max_num_moves = (
            self._DEFAULT_MAX_MOVES if max_num_moves is None else int(max_num_moves)
        )
        self.correlation_threshold = float(correlation_threshold)
        self.min_correlated_fraction = float(min_correlated_fraction)

        # Ravel/unravel utilities shared by both samplers; the template fixes
        # the pytree structure and the flattened dimension.  The template key
        # is derived from the sampler's key stream (never create a fresh key),
        # so that no key is ever consumed twice.
        self.rng_key, template_key = jax.random.split(self.rng_key)
        template = self._sample_initial_position(template_key, ())
        self._dim, self._unflatten_fn = ravel_pytree(template)
        self._dim = int(self._dim.shape[0])

    # ------------------------------------------------------------------
    # mutation rounds
    # ------------------------------------------------------------------
    def _mutate_until_decorrelated(
        self, rng_key, particles, parameters, tempering_param
    ):
        """Mutate the particles at the current temperature until decorrelation.

        All mutation rounds share the kernel parameters tuned beforehand.
        Implements the ``move_steps`` rule: start from a running correlation
        product of ones and multiply it by the lag-1 correlation between
        successive populations after every mutation round, until fewer than
        ``min_correlated_fraction`` of the components remain correlated.
        """
        init = (rng_key, particles, _stack_particles(particles), jnp.ones(self._dim), 0)

        def cond(carry):
            _, _, _, correlations, num_moves = carry
            still_correlated = jnp.mean(correlations > self.correlation_threshold)
            return (num_moves < self.max_num_moves) & (
                still_correlated >= self.min_correlated_fraction
            )

        def body(carry):
            key, particles, previous, correlations, num_moves = carry
            key, subkey = jax.random.split(key)
            particles, _ = self._mutate(subkey, particles, parameters, tempering_param)
            current = _stack_particles(particles)
            correlations = correlations * _lag1_corr(previous, current)
            return key, particles, current, correlations, num_moves + 1

        _, particles, _, _, _ = jax.lax.while_loop(cond, body, init)
        return particles

    def _mutate(self, rng_key, particles, parameters, tempering_param):
        """One mutation round: a single kernel step for every particle."""

        def tempered_logposterior(position):
            return self._logprior_fn(
                position
            ) + tempering_param * self._loglikelihood_fn(position)

        unshared, shared_step = unshared_parameters_and_step_fn(
            parameters, self._mcmc_step_fn
        )
        update_fn, _ = update_and_take_last(
            self._mcmc_init_fn,
            tempered_logposterior,
            shared_step,
            1,
            self.num_particles,
        )
        keys = jax.random.split(rng_key, self.num_particles)
        return update_fn(keys, particles, unshared)

    # ------------------------------------------------------------------
    # hooks implemented by the concrete samplers
    # ------------------------------------------------------------------
    def _initial_particles(self, key, num_particles):
        """Draw the initial particle population."""
        raise NotImplementedError

    def _initial_mcmc_parameters(self, key, particles):
        """Initial inner-kernel parameters."""
        raise NotImplementedError

    def _resample_mcmc_parameters(self, parameters, indices):
        """Resample the per-particle kernel parameters with their particles."""
        raise NotImplementedError

    def _tune_inner_mcmc_parameters(self, key, particles, parameters, tempering_param):
        """Tune the inner kernel parameters before the mutation rounds.

        Returns ``(particles, parameters)``.  Kernels that need a
        performance measurement may advance the particles with one MCMC
        move; kernels whose parameters only depend on the particle cloud
        leave them unchanged.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # inference loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run the adaptive-tempered SMC sampler.

        Returns
        -------
        dict
            Physical-space samples keyed by parameter name, plus
            ``'log_likelihood'``, normalized importance ``'weights'`` and the
            log-evidence ``'log_evidence'``.
        """
        self.rng_key, init_key_1, init_key_2, loop_key = jax.random.split(
            self.rng_key, 4
        )
        particles = self._initial_particles(init_key_1, self.num_particles)
        mcmc_params = self._initial_mcmc_parameters(init_key_2, particles)
        batched_loglikelihood = jax.vmap(self._loglikelihood_fn)
        particles, log_evidence = self._inference_loop(
            loop_key, particles, mcmc_params, batched_loglikelihood
        )
        return self._finalize(particles, log_evidence)

    def _inference_loop(self, key, particles, mcmc_params, batched_loglikelihood):
        """Reweight-resample-mutate loop, terminating at tempering parameter 1.

        Each iteration computes the ESS-optimal tempering increment,
        reweights and resamples the particles, tunes the inner kernel
        parameters from the particle cloud, and mutates the particles at the
        new temperature until they decorrelate.  The particles carry uniform
        weights after resampling; the normalising-constant increments are
        accumulated into the log-evidence.
        """

        def cond(carry):
            return carry[2] < 1.0

        def body(carry):
            key, particles, tempering_param, mcmc_params, log_evidence = carry
            key, resample_key, tune_key, move_key = jax.random.split(key, 4)
            delta = ess_solver(
                batched_loglikelihood,
                particles,
                self.target_ess,
                1.0 - tempering_param,
                dichotomy,
            )
            tempering_param = tempering_param + delta
            log_weights = delta * batched_loglikelihood(particles)
            log_evidence = (
                log_evidence + logsumexp(log_weights) - jnp.log(self.num_particles)
            )
            indices = resampling.systematic(
                resample_key,
                jnp.exp(log_weights - logsumexp(log_weights)),
                self.num_particles,
            )
            particles = jax.tree.map(lambda p: p[indices], particles)
            mcmc_params = self._resample_mcmc_parameters(mcmc_params, indices)
            particles, mcmc_params = self._tune_inner_mcmc_parameters(
                tune_key, particles, mcmc_params, tempering_param
            )
            particles = self._mutate_until_decorrelated(
                move_key, particles, mcmc_params, tempering_param
            )
            return key, particles, tempering_param, mcmc_params, log_evidence

        _, particles, _, _, log_evidence = jax.lax.while_loop(
            cond, body, (key, particles, 0.0, mcmc_params, 0.0)
        )
        return particles, log_evidence

    def _finalize(self, particles, log_evidence) -> dict:
        """Convert the final population into a physical-parameter dict."""
        physical = jax.vmap(self._params_to_likelihood)(particles)
        log_likelihoods = jax.vmap(self._loglikelihood_fn)(particles)
        weights = jnp.ones(self.num_particles) / self.num_particles
        output = self._format_output(physical, log_likelihoods, weights=weights)
        output["log_evidence"] = float(np.asarray(log_evidence))
        return output


class BlackjaxSMCwithNUTS(_TunedAdaptiveTemperedSMC):
    """Adaptive-tempered SMC with a NUTS mutation kernel.

    NUTS requires an unbounded domain, so sampling happens on the
    unconstrained space prepared by :class:`NumpyroPriorSamplerBase` (the
    prior bijections and the inverse transform back to physical parameters
    are handled by the base class).

    At every tempering level the shared ``inverse_mass_matrix`` is
    re-estimated from the particle variance, and the per-particle
    ``step_size`` is updated by a feedback scheme from one measured MCMC
    move at the level's temperature (squared Mahalanobis distance divided
    by the trajectory length and scaled by the acceptance rate): the step
    sizes are resampled by that measure and then jittered multiplicatively
    in log space.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing ``log_likelihood``, ``parameter_names``
        and ``params_type``.
    prior : dict
        Prior declaration, see
        :class:`pespace.inference.sampler.base.NumpyroPriorSamplerBase`.
    seed : int, optional
        Seed for the JAX random number generator.  Default is 0.
    num_particles : int, optional
        Number of SMC particles.  Default is 200.
    target_ess : float, optional
        Target ESS ratio of the adaptive tempering schedule.  Default is 0.8.
    max_num_moves : int, optional
        Upper bound on the number of mutation rounds per tempering level.
        Default is 50.
    correlation_threshold : float, optional
        Lag-1 autocorrelation threshold of the mutation stopping rule.
        Default is 0.1.
    min_correlated_fraction : float, optional
        Fraction of components allowed to stay correlated before the
        mutation rounds stop.  Default is 0.3.
    log_step_size_range : tuple, optional
        ``(low, high)`` bounds of the log-uniform distribution the initial
        per-particle step sizes are drawn from (in log space).  Default is
        ``(log 1e-4, log 1.0)``.
    tuning_alpha_frac : float, optional
        Floor of the reweighting measure in the step-size feedback tuning,
        expressed as a fraction of the mean realised measure (keeps the
        floor a non-dominant term).  Default is 0.1.
    tuning_sigma_log : float, optional
        Log-space perturbation scale applied to the resampled step sizes.
        Default is 0.2.
    """

    _DEFAULT_MAX_MOVES = 50

    def __init__(
        self,
        likelihood,
        prior,
        seed: int = 0,
        num_particles: int = 200,
        target_ess: float = 0.8,
        max_num_moves: int | None = None,
        correlation_threshold: float = 0.1,
        min_correlated_fraction: float = 0.3,
        log_step_size_range: tuple = (jnp.log(1e-4), jnp.log(1.0)),
        tuning_alpha_frac: float = 0.1,
        tuning_sigma_log: float = 0.2,
    ) -> None:
        super().__init__(
            likelihood,
            prior,
            seed,
            num_particles,
            target_ess,
            max_num_moves,
            correlation_threshold,
            min_correlated_fraction,
        )
        low, high = log_step_size_range
        if not low < high:
            raise ValueError(
                "log_step_size_range must be a (low, high) tuple with low < high."
            )
        self.log_step_size_range = (float(low), float(high))
        self.tuning_alpha_frac = float(tuning_alpha_frac)
        self.tuning_sigma_log = float(tuning_sigma_log)

        self._logprior_fn = self.logprior_fn
        self._loglikelihood_fn = self.loglikelihood_fn
        self._mcmc_step_fn = blackjax.nuts.build_kernel()
        self._mcmc_init_fn = blackjax.nuts.init
        self._params_to_likelihood = self._build_likelihood_params

    def _initial_particles(self, key, num_particles):
        return self._sample_initial_position(key, (num_particles,))

    def _initial_mcmc_parameters(self, key, particles):
        low, high = self.log_step_size_range
        log_step_size = jax.random.uniform(
            key,
            (self.num_particles,),
            minval=low,
            maxval=high,
        )
        inverse_mass_matrix = inverse_mass_matrix_from_particles(particles)[None, ...]
        return {
            "step_size": jnp.exp(log_step_size),
            "inverse_mass_matrix": inverse_mass_matrix,
        }

    def _resample_mcmc_parameters(self, parameters, indices):
        """Resample the per-particle step sizes with their particles."""
        return {
            "step_size": parameters["step_size"][indices],
            "inverse_mass_matrix": parameters["inverse_mass_matrix"],
        }

    def _tune_inner_mcmc_parameters(self, key, particles, parameters, tempering_param):
        """Tune the inner kernel from one measured MCMC move.

        The move advances the particles and doubles as the measurement
        step: its realised squared-jump measure reweights the particle step
        sizes, which are resampled first and then jittered multiplicatively
        in log space (so duplicates receive independent perturbations); the
        shared inverse mass matrix is re-estimated from the pre-move
        particle cloud.
        """
        move_key, resample_key, perturb_key = jax.random.split(key, 3)
        previous = particles
        particles, info = self._mutate(move_key, particles, parameters, tempering_param)
        measure = self._perf_metric(parameters, previous, particles, info)

        weights = self.tuning_alpha_frac * jnp.mean(measure) + measure
        selection = resampling.systematic(
            resample_key,
            weights / jnp.sum(weights),
            parameters["step_size"].shape[0],
        )
        step_size = parameters["step_size"][selection]
        step_size = step_size * jnp.exp(
            self.tuning_sigma_log * jax.random.normal(perturb_key, step_size.shape)
        )

        inverse_mass_matrix = inverse_mass_matrix_from_particles(previous)[None, ...]
        return particles, {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
        }

    def _perf_metric(self, parameters, previous, probe, info):
        """Realised squared-jump measure of a mutation round.

        The squared Mahalanobis distance between the pre- and post-mutation
        positions is divided by the NUTS trajectory length and scaled by the
        acceptance rate.
        """
        start = _stack_particles(previous)
        end = _stack_particles(probe)
        inverse_mass_diag = jnp.diag(parameters["inverse_mass_matrix"][0])
        trajectory_length = jnp.ravel(info.num_integration_steps)
        acceptance = jnp.ravel(info.acceptance_rate)
        measure = jnp.sum((start - end) ** 2 / inverse_mass_diag, axis=-1)
        return measure / trajectory_length * acceptance


class BlackjaxSMCwithIRMH(_TunedAdaptiveTemperedSMC):
    """Adaptive-tempered SMC with an independent MH mutation kernel.

    Sampling is performed directly in the physical parameter space, so the
    output is already physical and needs no post-processing.  The Gaussian
    proposal ``mean`` / ``covariance`` are re-fitted to the particle cloud at
    every tempering level.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing ``log_likelihood``, ``parameter_names``
        and ``params_type``.
    prior : dict
        Prior declaration, see
        :class:`pespace.inference.sampler.base.NumpyroPriorSamplerBase`.
    seed : int, optional
        Seed for the JAX random number generator.  Default is 0.
    num_particles : int, optional
        Number of SMC particles.  Default is 200.
    target_ess : float, optional
        Target ESS ratio of the adaptive tempering schedule.  Default is 0.8.
    max_num_moves : int, optional
        Upper bound on the number of mutation rounds per tempering level.
        Default is 100.
    correlation_threshold : float, optional
        Lag-1 autocorrelation threshold of the mutation stopping rule.
        Default is 0.1.
    min_correlated_fraction : float, optional
        Fraction of components allowed to stay correlated before the
        mutation rounds stop.  Default is 0.3.
    covariance_regularization : float, optional
        Diagonal regularization added to the fitted proposal covariance.
        Default is 1e-6.
    """

    _DEFAULT_MAX_MOVES = 100

    def __init__(
        self,
        likelihood,
        prior,
        seed: int = 0,
        num_particles: int = 200,
        target_ess: float = 0.8,
        max_num_moves: int | None = None,
        correlation_threshold: float = 0.1,
        min_correlated_fraction: float = 0.3,
        covariance_regularization: float = 1e-6,
    ) -> None:
        super().__init__(
            likelihood,
            prior,
            seed,
            num_particles,
            target_ess,
            max_num_moves,
            correlation_threshold,
            min_correlated_fraction,
        )
        self.covariance_regularization = float(covariance_regularization)

        self._logprior_fn = self.logprior_physical_fn
        self._loglikelihood_fn = self.loglikelihood_physical_fn
        self._irmh_kernel = blackjax.irmh.build_kernel()
        self._mcmc_step_fn = self._irmh_step_fn
        self._mcmc_init_fn = blackjax.irmh.init
        self._params_to_likelihood = self._likelihood_params_from_physical

    def _irmh_step_fn(self, rng_key, state, logdensity_fn, mean, covariance):
        """IRMH kernel with a shared Gaussian proposal ``N(mean, covariance)``."""

        def proposal_distribution(key):
            return self._unflatten_fn(
                jax.random.multivariate_normal(key, mean, covariance)
            )

        def proposal_logdensity(_proposal_state, current_state):
            # blackjax subtracts ``proposal_logdensity_fn(new_state,
            # prev_state)`` from the energy of the new state; evaluating the
            # independent proposal density at the *current* position recovers
            # the standard IMH acceptance ratio ``p(x') q(x) / p(x) q(x')``.
            return jax.scipy.stats.multivariate_normal.logpdf(
                ravel_pytree(current_state.position)[0], mean, covariance
            )

        return self._irmh_kernel(
            rng_key, state, logdensity_fn, proposal_distribution, proposal_logdensity
        )

    def _initial_particles(self, key, num_particles):
        return self._sample_initial_position_physical(key, (num_particles,))

    def _initial_mcmc_parameters(self, key, particles):
        del key
        return self._fitted_proposal(particles)

    def _resample_mcmc_parameters(self, parameters, indices):
        del indices
        return parameters

    def _tune_inner_mcmc_parameters(self, key, particles, parameters, tempering_param):
        # the proposal is fitted from the particles before the level's
        # mutation; no measurement move is needed.
        del key, parameters, tempering_param
        return particles, self._fitted_proposal(particles)

    def _fitted_proposal(self, particles):
        """Gaussian proposal parameters fitted to the current particle cloud."""
        covariance = particles_covariance_matrix(particles)
        covariance = covariance + self.covariance_regularization * jnp.eye(self._dim)
        return {
            "mean": particles_means(particles)[None, ...],
            "covariance": covariance[None, ...],
        }
