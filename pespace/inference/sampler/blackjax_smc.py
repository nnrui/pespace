"""SMC sampler interface based on blackjax.

:class:`BlackjaxSMC` runs :func:`blackjax.adaptive_tempered_smc` on the
unconstrained space provided by :class:`NumpyroPriorSamplerBase`, supporting
several mutation kernels (RMH / HMC / NUTS) and automatic tuning of the inner
kernel parameters (pretuning / inner-kernel tuning).  The inference loop is a
pure-JAX ``jax.lax.while_loop`` (terminating when the tempering parameter
reaches 1), which is the prerequisite for the optional ``jax.pmap``
multi-chain backend (Phase 4).
"""

import logging

import jax
import jax.numpy as jnp

from .base import NumpyroPriorSamplerBase

try:
    import blackjax
    import blackjax.smc.resampling as resampling
    from blackjax.mcmc import random_walk
    from blackjax.smc.pretuning import build_pretune
except ImportError:
    logging.error(
        "blackjax is not installed by default, if the blackjax sampler "
        "interface is needed, please install blackjax manually."
    )
    raise


class BlackjaxSMC(NumpyroPriorSamplerBase):
    """Adaptive-tempered SMC sampler using blackjax.

    Parameters
    ----------
    likelihood : WhittleLikelihoodJax
        Likelihood instance providing ``log_likelihood``, ``parameter_names``
        and ``params_type``.
    prior : dict
        Prior declaration, see
        :class:`pespace.inference.sampler.base.NumpyroPriorSamplerBase`.
    seed : int, optional
        Seed for the JAX random number generator.  Default is 0.
    mutation_kernel : {'rmh', 'hmc', 'nuts'}, optional
        Mutation kernel used to move the particles inside the tempering loop.
        ``'rmh'`` is the robust gradient-free baseline (additive-step random
        walk); ``'hmc'`` / ``'nuts'`` operate on the unconstrained space and
        require a differentiable log-density.  Default is ``'rmh'``.
    mcmc_parameters : dict, optional
        Initial parameters passed to the mutation kernel.  Each value must be
        an array whose leading dimension is 1 (shared by all particles) or
        ``num_particles`` (per-particle).  Sensible defaults are derived per
        kernel when not given.
    tuning_method : {'none', 'pretuning', 'inner_kernel_tuning',
        'pretuning+inner_kernel_tuning'}, optional
        Automatic tuning of the inner-kernel parameters.  ``'none'`` keeps the
        (possibly adapted) parameters fixed; ``'pretuning'`` (default) adapts
        particle-wise step sizes; ``'inner_kernel_tuning'`` re-estimates the
        kernel parameters (e.g. the mass matrix) at every tempering step;
        ``'pretuning+inner_kernel_tuning'`` combines both.
    target_ess : float, optional
        Target effective sample size ratio for the adaptive tempering
        schedule.  Default is 0.5.
    num_mcmc_steps : int, optional
        Number of MCMC steps applied to each particle per tempering
        iteration.  Default is 10.
    resampling_fn : callable, optional
        Blackjax resampling function.  Default is ``resampling.systematic``.
    pretune_config : dict, optional
        Extra keyword arguments for
        :func:`blackjax.smc.pretuning.build_pretune` (e.g. ``alpha``,
        ``sigma_parameters``, ``natural_parameters``,
        ``positive_parameters``).
    """

    _KERNELS = ("rmh", "hmc", "nuts")
    _TUNING = (
        "none",
        "pretuning",
        "inner_kernel_tuning",
        "pretuning+inner_kernel_tuning",
    )

    def __init__(
        self,
        likelihood,
        prior,
        seed: int = 0,
        mutation_kernel: str = "rmh",
        mcmc_parameters: dict | None = None,
        tuning_method: str = "pretuning",
        target_ess: float = 0.5,
        num_mcmc_steps: int = 10,
        resampling_fn=None,
        pretune_config: dict | None = None,
        nuts_max_num_doublings: int = 10,
    ) -> None:
        super().__init__(likelihood, prior, seed)

        if mutation_kernel not in self._KERNELS:
            raise ValueError(
                f"mutation_kernel must be one of {self._KERNELS}, "
                f"got '{mutation_kernel}'."
            )
        if tuning_method not in self._TUNING:
            raise ValueError(
                f"tuning_method must be one of {self._TUNING}, "
                f"got '{tuning_method}'."
            )
        self.mutation_kernel = mutation_kernel
        self.tuning_method = tuning_method
        self.target_ess = target_ess
        self.num_mcmc_steps = num_mcmc_steps
        self.resampling_fn = resampling_fn or resampling.systematic
        self.pretune_config = dict(pretune_config or {})
        self.nuts_max_num_doublings = int(nuts_max_num_doublings)

        self._mcmc_step_fn, self._mcmc_init_fn, default_parameters = (
            self._build_mcmc_kernel(mutation_kernel)
        )
        self.mcmc_parameters = (
            default_parameters if mcmc_parameters is None else dict(mcmc_parameters)
        )

    # ------------------------------------------------------------------
    # kernel factory
    # ------------------------------------------------------------------
    def _build_mcmc_kernel(self, name: str):
        """Return ``(mcmc_step_fn, mcmc_init_fn, default_parameters)``.

        The step function must have the signature
        ``(rng_key, state, logdensity_fn, **step_parameters)`` expected by
        blackjax's SMC machinery (parameters are unpacked from
        ``mcmc_parameters``).
        """
        dim = len(self.parameter_names)
        if name == "rmh":
            additive_step = random_walk.build_additive_step()

            def kernel(rng_key, state, logdensity_fn, sigma=None, **_):
                if sigma is None:  # pragma: no cover - SMC always passes sigma
                    sigma = jnp.array([0.5])
                random_step = random_walk.normal(sigma)
                return additive_step(rng_key, state, logdensity_fn, random_step)

            default = {"sigma": jnp.array([0.5])}
            return kernel, blackjax.rmh.init, default

        if name == "hmc":
            default = {
                "step_size": jnp.array([1e-2]),
                "inverse_mass_matrix": jnp.eye(dim)[None],
                "num_integration_steps": jnp.array([10]),
            }
            return blackjax.hmc.build_kernel(), blackjax.hmc.init, default

        if name == "nuts":
            # max_num_doublings must be a static Python int (it sizes an array),
            # so it is bound here instead of being passed through the SMC
            # mcmc_parameters machinery (which would make it a tracer).
            base_kernel = blackjax.nuts.build_kernel()
            max_doublings = self.nuts_max_num_doublings

            def kernel(rng_key, state, logdensity_fn, step_size=None, inverse_mass_matrix=None, **_):
                return base_kernel(
                    rng_key,
                    state,
                    logdensity_fn,
                    step_size,
                    inverse_mass_matrix,
                    max_num_doublings=max_doublings,
                )

            default = {
                "step_size": jnp.array([1e-2]),
                "inverse_mass_matrix": jnp.eye(dim)[None],
            }
            return kernel, blackjax.nuts.init, default

        raise ValueError(f"Unknown mutation kernel '{name}'.")

    # ------------------------------------------------------------------
    # tuning
    # ------------------------------------------------------------------
    def _build_pretune(self, num_particles: int):
        """Build a ``pretune_fn`` for the configured mutation kernel.

        Blackjax's default chain-mixing measure is the ESJD normalised by the
        inverse mass matrix, so it only applies to ``hmc``/``nuts``.  For the
        gradient-free ``rmh`` kernel a unit-mass-matrix ESJD is used instead.
        """
        cfg = self.pretune_config
        measure_factory = None
        if self.mutation_kernel == "rmh":
            sigma_parameters = cfg.get(
                "sigma_parameters", {"sigma": jnp.array(0.5)}
            )
            natural = cfg.get("natural_parameters", [])
            positive = cfg.get("positive_parameters", ["sigma"])
            from blackjax.smc.pretuning import esjd

            dim = len(self.parameter_names)
            measure_factory = lambda state: esjd(jnp.eye(dim)[None])
        elif self.mutation_kernel == "hmc":
            sigma_parameters = cfg.get(
                "sigma_parameters",
                {
                    "step_size": jnp.array(1e-2),
                    "num_integration_steps": jnp.array(2),
                },
            )
            natural = cfg.get("natural_parameters", ["num_integration_steps"])
            positive = cfg.get("positive_parameters", ["step_size"])
        else:  # nuts
            sigma_parameters = cfg.get(
                "sigma_parameters", {"step_size": jnp.array(1e-2)}
            )
            natural = cfg.get("natural_parameters", [])
            positive = cfg.get("positive_parameters", ["step_size"])

        kwargs = dict(
            mcmc_init_fn=self._mcmc_init_fn,
            mcmc_step_fn=self._mcmc_step_fn,
            alpha=cfg.get("alpha", 1),
            n_particles=num_particles,
            sigma_parameters=sigma_parameters,
            natural_parameters=natural,
            positive_parameters=positive,
        )
        if measure_factory is not None:
            kwargs["performance_of_chain_measure_factory"] = measure_factory
        return build_pretune(**kwargs)

    def _build_inner_update_fn(self, combined: bool):
        """Return an ``mcmc_parameter_update_fn`` for inner-kernel tuning.

        For ``hmc``/``nuts`` the inverse mass matrix is re-estimated from the
        particle covariance; for ``rmh`` the (scalar) proposal scale is kept
        fixed.
        """
        from blackjax.smc.tuning.from_particles import (
            inverse_mass_matrix_from_particles,
        )

        kernel = self.mutation_kernel
        if combined:
            # state is StateWithParameterOverride
            def update_fn(key, state, info):
                particles = state.sampler_state.particles
                if kernel in ("hmc", "nuts"):
                    return {"inverse_mass_matrix": inverse_mass_matrix_from_particles(particles)}
                return {}
        else:
            def update_fn(key, state, info):
                particles = state.particles
                if kernel in ("hmc", "nuts"):
                    return {"inverse_mass_matrix": inverse_mass_matrix_from_particles(particles)}
                return {}
        return update_fn

    def _pretune_initial_parameters(self, num_particles: int) -> dict:
        """Per-particle initial parameters for pretuning.

        Pretuning assigns a value to each inner chain, so the tuned parameters
        must already carry a per-particle leading dimension at initialisation;
        otherwise the ``jax.lax.while_loop`` carry type would change shape
        after the first pretuning step.
        """
        n = num_particles
        if self.mutation_kernel == "rmh":
            sigma = float(jnp.ravel(self.mcmc_parameters["sigma"])[0])
            return {"sigma": jnp.full((n,), sigma)}
        if self.mutation_kernel == "hmc":
            step = float(jnp.ravel(self.mcmc_parameters["step_size"])[0])
            integ = int(jnp.ravel(self.mcmc_parameters["num_integration_steps"])[0])
            return {
                "inverse_mass_matrix": self.mcmc_parameters["inverse_mass_matrix"],
                "step_size": jnp.full((n,), step),
                "num_integration_steps": jnp.full((n,), integ, dtype=jnp.int32),
            }
        # nuts
        step = float(jnp.ravel(self.mcmc_parameters["step_size"])[0])
        return {
            "inverse_mass_matrix": self.mcmc_parameters["inverse_mass_matrix"],
            "step_size": jnp.full((n,), step),
        }

    def _build_smc(self, num_particles: int, num_mcmc_steps: int):
        """Construct the (possibly tuned) sampling algorithm.

        Returns ``(smc, is_tuned)`` where ``is_tuned`` indicates that the
        state carries a ``sampler_state`` attribute.
        """
        base_kwargs = dict(
            logprior_fn=self.logprior_fn,
            loglikelihood_fn=self.loglikelihood_fn,
            mcmc_step_fn=self._mcmc_step_fn,
            mcmc_init_fn=self._mcmc_init_fn,
            resampling_fn=self.resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            target_ess=self.target_ess,
        )

        if self.tuning_method == "none":
            smc = blackjax.adaptive_tempered_smc(
                mcmc_parameters=self.mcmc_parameters, **base_kwargs
            )
            return smc, False

        if self.tuning_method == "pretuning":
            pretune = self._build_pretune(num_particles)
            smc = blackjax.pretuning(
                blackjax.adaptive_tempered_smc,
                initial_parameter_value=self._pretune_initial_parameters(num_particles),
                pretune_fn=pretune,
                **base_kwargs,
            )
            return smc, True

        if self.tuning_method == "inner_kernel_tuning":
            smc = blackjax.inner_kernel_tuning(
                blackjax.adaptive_tempered_smc,
                mcmc_parameter_update_fn=self._build_inner_update_fn(combined=False),
                initial_parameter_value=self.mcmc_parameters,
                **base_kwargs,
            )
            return smc, True

        # pretuning + inner_kernel_tuning
        pretune = self._build_pretune(num_particles)

        def pretuning_factory(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            mcmc_parameters,
            resampling_fn,
            num_mcmc_steps,
            initial_parameter_value,
            target_ess,
        ):
            return blackjax.pretuning(
                blackjax.adaptive_tempered_smc,
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                mcmc_step_fn=mcmc_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                resampling_fn=resampling_fn,
                num_mcmc_steps=num_mcmc_steps,
                initial_parameter_value=initial_parameter_value,
                pretune_fn=pretune,
                target_ess=target_ess,
            )

        smc = blackjax.inner_kernel_tuning(
            pretuning_factory,
            mcmc_parameter_update_fn=self._build_inner_update_fn(combined=True),
            initial_parameter_value=self._pretune_initial_parameters(num_particles),
            smc_returns_state_with_parameter_override=True,
            **base_kwargs,
        )
        return smc, True

    # ------------------------------------------------------------------
    # inference loop
    # ------------------------------------------------------------------
    @staticmethod
    def _run_loop(smc, key, state, tuned: bool):
        """Pure-JAX adaptive-tempering loop terminating at ``tempering_param >= 1``."""

        if tuned:

            def cond(carry):
                _, s = carry
                return s.sampler_state.tempering_param < 1.0

        else:

            def cond(carry):
                _, s = carry
                return s.tempering_param < 1.0

        def body(carry):
            k, s = carry
            k, subkey = jax.random.split(k)
            new_s, _info = smc.step(subkey, s)
            return k, new_s

        return jax.lax.while_loop(cond, body, (key, state))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(
        self,
        num_particles: int = 200,
        num_mcmc_steps: int | None = None,
        num_chains: int = 1,
        parallel_backend: str | None = None,
    ) -> dict:
        """Run the adaptive-tempered SMC sampling.

        Parameters
        ----------
        num_particles : int, optional
            Number of SMC particles.  Default is 200.
        num_mcmc_steps : int, optional
            Number of MCMC mutation steps per tempering iteration.  Defaults
            to the value given at construction.
        num_chains : int, optional
            Number of independent SMC runs.  ``1`` runs a single chain.
            ``> 1`` requires ``parallel_backend``.
        parallel_backend : {'cpu-pmap', 'gpu-pmap', None}, optional
            Multi-chain parallel backend (Phase 4).  Default is None.

        Returns
        -------
        dict
            Dictionary of numpy arrays keyed by physical parameter name, plus
            ``'log_likelihood'`` and importance ``'weights'`` (normalized).
        """
        steps = self.num_mcmc_steps if num_mcmc_steps is None else num_mcmc_steps
        if num_chains > 1:
            if parallel_backend is None:
                raise ValueError(
                    "num_chains > 1 requires a parallel_backend ('cpu-pmap' or "
                    "'gpu-pmap')."
                )
            return self._run_pmap(num_particles, steps, num_chains, parallel_backend)
        return self._run_single(num_particles, steps)

    def _run_single(self, num_particles: int, num_mcmc_steps: int) -> dict:
        smc, tuned = self._build_smc(num_particles, num_mcmc_steps)
        self.rng_key, key = jax.random.split(self.rng_key)
        state = smc.init(self.initial_position((num_particles,)), rng_key=None)
        _, final_state = self._run_loop(smc, key, state, tuned)
        return self._finalize(final_state, tuned)

    def _run_pmap(self, num_particles, num_mcmc_steps, num_chains, parallel_backend):
        """Run ``num_chains`` independent SMC runs with ``jax.pmap``.

        Each device executes one full adaptive-tempered SMC run (different
        rng key / initial particles).  Particle-level parallelism is provided
        inside the SMC kernel (``vmap`` / ``lax.map``) and is orthogonal to
        this chain-level ``pmap``, so the two do not conflict: on CPU each
        chain runs on its own core (likelihood vectorisation stays on that
        core), while on GPU the likelihood vectorisation runs on the GPU.
        The per-device final states are merged into a single physical
        parameter dict.
        """
        from jax import devices

        if parallel_backend not in ("cpu-pmap", "gpu-pmap"):
            raise ValueError(
                f"parallel_backend must be 'cpu-pmap' or 'gpu-pmap', got "
                f"'{parallel_backend}'."
            )
        n_devices = len(devices())
        if n_devices < num_chains:
            raise ValueError(
                f"parallel_backend='{parallel_backend}' requires at least "
                f"{num_chains} devices but {n_devices} are available. For CPU "
                "pmap, call numpyro.set_host_device_count(N) before the first "
                "JAX operation (e.g. before constructing the sampler)."
            )

        smc, tuned = self._build_smc(num_particles, num_mcmc_steps)

        def run_chain(key):
            particles = self._sample_initial_position(key, (num_particles,))
            state = smc.init(particles, rng_key=None)
            _, final = self._run_loop(smc, key, state, tuned)
            return final

        self.rng_key, key = jax.random.split(self.rng_key)
        keys = jax.random.split(key, num_chains)
        finals = jax.pmap(run_chain)(keys)
        if tuned:
            finals = finals.sampler_state
        return self._finalize_pmap(finals)

    def _finalize_pmap(self, finals) -> dict:
        """Merge per-device final SMC states into a physical-parameter dict."""
        particles = finals.particles  # dict name -> (num_chains, num_particles, ...)
        weights = finals.weights  # (num_chains, num_particles)
        physical = jax.vmap(jax.vmap(self._build_likelihood_params))(particles)
        log_likelihoods = jax.vmap(jax.vmap(self.loglikelihood_fn))(particles)
        physical = {k: v.reshape((-1,) + v.shape[2:]) for k, v in physical.items()}
        return self._format_output(
            physical, log_likelihoods.reshape(-1), weights=weights.reshape(-1)
        )

    def _finalize(self, final_state, tuned: bool) -> dict:
        """Convert the final SMC state into a physical-parameter dict."""
        state = final_state.sampler_state if tuned else final_state
        physical = self._to_physical(state.particles)
        log_likelihoods = jax.vmap(self.loglikelihood_fn)(state.particles)
        return self._format_output(physical, log_likelihoods, weights=state.weights)
