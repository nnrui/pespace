"""FlowMC sampler interface for Bayesian parameter estimation.

flowMC is a JAX-based normalizing-flow-enhanced MCMC sampler (Wong et al. 2023,
Gabrié et al. 2022).  A local MCMC kernel (Gaussian random walk / MALA / HMC)
walks the chains while a rational-quadratic-spline normalizing flow is trained
online on the accumulated chain history and used as an independent-Metropolis
*global* proposal, which dramatically accelerates the mixing of multi-modal or
high-dimensional posteriors.

This module provides :class:`FlowMCSampler`, which follows the same design as
the other samplers in this package: priors are declared through numpyro
distributions (:class:`~pespace.inference.sampler.base.NumpyroPriorSamplerBase`),
sampling is performed in the *unbounded* (unconstrained) space prepared by the
prior bijections, and the output is transformed back to physical parameters.

Six built-in flowMC resource-strategy bundles are available under their flowMC
names (:data:`_buildin_bundles`), with ``"RQSpline_MALA"`` as the default.
Bundle hyperparameters are supplied as a plain ``bundle_config`` dictionary and
merged on top of the built-in default configuration (:data:`_default_config`).
``n_chains`` and ``checkpoint_interval`` are sampler-level settings managed
directly by the constructor.  Fully custom setups are supported by passing a
pre-built ``ResourceStrategyBundle`` instance (``bundle_config`` is then
ignored).  flowMC is an optional dependency and is imported lazily.

Notes
-----
- The likelihood only needs to provide ``log_likelihood(params)`` and the
  ``likelihood_param_type`` namedtuple type — exactly the interface
  :class:`~pespace.inference.likelihood_jax.WhittleLikelihoodJax` exposes.
- MCMC provides no evidence estimate; ``log_evidence`` is reported as ``NaN``.
- For the parallel-tempering bundles the ``logpdf`` passed to flowMC is the
  log-likelihood, and the log-prior (including the Jacobian of the prior
  bijections, which must not be rescaled by the temperature) is passed as the
  untempered ``logprior``.
- flowMC has not reached v1.0 yet and its API is unstable; this interface is
  developed and tested against flowMC 0.6.x.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .base import NumpyroPriorSamplerBase
from ..utils import map_fn

try:
    import flowMC  # noqa: F401
    from flowMC.Sampler import Sampler as _FlowMCSampler
    from flowMC.resource.buffers import Buffer
    from flowMC.resource.states import State
    from flowMC.resource_strategy_bundle.RQSpline_GRW import RQSpline_GRW_Bundle
    from flowMC.resource_strategy_bundle.RQSpline_GRW_PT import (
        RQSpline_GRW_PT_Bundle,
    )
    from flowMC.resource_strategy_bundle.RQSpline_HMC import RQSpline_HMC_Bundle
    from flowMC.resource_strategy_bundle.RQSpline_HMC_PT import (
        RQSpline_HMC_PT_Bundle,
    )
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
    from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import (
        RQSpline_MALA_PT_Bundle,
    )
    from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle
except ImportError as err:
    raise ImportError(
        "flowMC is not installed by default; it is an optional dependency of "
        "pespace. If the flowMC sampler interface is needed, please install "
        "flowMC manually."
    ) from err

__all__ = ["FlowMCSampler"]

# Built-in resource-strategy bundles: bundle name -> bundle class.
_buildin_bundles = {
    "RQSpline_GRW": RQSpline_GRW_Bundle,
    "RQSpline_MALA": RQSpline_MALA_Bundle,
    "RQSpline_HMC": RQSpline_HMC_Bundle,
    "RQSpline_GRW_PT": RQSpline_GRW_PT_Bundle,
    "RQSpline_MALA_PT": RQSpline_MALA_PT_Bundle,
    "RQSpline_HMC_PT": RQSpline_HMC_PT_Bundle,
}

# Default configuration shared by all bundles.
_common = dict(
    n_local_steps=100,
    n_global_steps=100,
    n_training_loops=20,
    n_production_loops=20,
    n_epochs=100,
    learning_rate=1e-3,
    batch_size=10000,
    n_max_examples=10000,
    history_window=100,
    chain_batch_size=0,
    local_thinning=1,
    global_thinning=1,
    verbose=False,
    early_stopping=False,
    early_stopping_tolerance=0.05,
    early_stopping_patience=3,
    early_stopping_min_acceptance=0.1,
)

# Default configuration of the normalizing-flow model, shared by all bundles.
_nf = dict(
    rq_spline_hidden_units=(32, 32),
    rq_spline_n_bins=8,
    rq_spline_n_layers=4,
    n_NFproposal_batch_size=10000,
)

# Default configuration of the local samplers, keyed by the local kernel.
_local_kernel = {
    "grw": dict(
        grw_step_size=0.1,
        adapt_step_size=True,
        adapt_step_size_per_dim=True,
    ),
    "mala": dict(
        mala_step_size=0.1,
        adapt_step_size=True,
        adapt_step_size_per_dim=True,
    ),
    "hmc": dict(
        hmc_step_size=0.1,
        hmc_n_leapfrog=10,
        condition_matrix=1,
        adapt_step_size=True,
        adapt_step_size_per_dim=True,
    ),
}

# Default configuration of the parallel-tempering extension (PT bundles only).
_pt = dict(
    n_temperatures=5,
    max_temperature=5.0,
    n_tempered_steps=-1,
)

# Default configuration of each built-in bundle, assembled from the groups
# defined above.
_default_config = {
    "RQSpline_GRW": {**_common, **_nf, **_local_kernel["grw"]},
    "RQSpline_MALA": {**_common, **_nf, **_local_kernel["mala"]},
    "RQSpline_HMC": {**_common, **_nf, **_local_kernel["hmc"]},
    "RQSpline_GRW_PT": {**_common, **_nf, **_local_kernel["grw"], **_pt},
    "RQSpline_MALA_PT": {**_common, **_nf, **_local_kernel["mala"], **_pt},
    "RQSpline_HMC_PT": {**_common, **_nf, **_local_kernel["hmc"], **_pt},
}


class FlowMCSampler(NumpyroPriorSamplerBase):
    """Normalizing-flow-enhanced MCMC (flowMC) sampler.

    The chains evolve in the unbounded space prepared by
    :class:`~pespace.inference.sampler.base.NumpyroPriorSamplerBase` (the prior
    bijections and the transform back to physical parameters are handled by the
    base class).  Each training loop runs ``n_local_steps`` steps of the local
    kernel (GRW / MALA / HMC), trains the RQ-spline normalizing flow for
    ``n_epochs`` epochs on the accumulated chain history, and then proposes
    ``n_global_steps`` independent-Metropolis moves from the flow.  After
    ``n_training_loops`` such loops (the warmup, during which the step size is
    adapted and the flow is fitted), the flow is frozen and ``n_production_loops``
    loops of local + global steps are collected as the posterior sample.  Flow
    proposals are corrected by their proposal density, so the frozen-flow
    production chain satisfies detailed balance.

    The flowMC sampler (bundle and backend) is constructed in ``__init__``;
    :meth:`run` only executes — and times — the ``sample`` call.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing a pure ``log_likelihood(params) -> scalar``
        method (JAX-transformable) and the ``likelihood_param_type`` namedtuple
        type of the likelihood parameters.
    prior_dict : dict
        Prior declaration, see
        :class:`~pespace.inference.sampler.base.NumpyroPriorSamplerBase`.
    label : str, optional
        Label of the run, converted to ``str``; also the JSON filename
        (``outdir/{label}.json``).  Default is None (-> ``"None"``).
    outdir : str, optional
        Output directory for the saved result and the flowMC checkpoint.
        Default is None, which resolves to an ``outdir_{label}`` directory
        under the current working directory (created if missing).
    rkey_seed : int, optional
        Seed for the JAX random number generator key.  Default is 0.
    resource_strategy_bundles : str or ResourceStrategyBundle, optional
        Either the name of a built-in bundle (:data:`_buildin_bundles`) or a
        pre-constructed ``ResourceStrategyBundle`` instance.  Default is
        ``"RQSpline_MALA"``.  For pre-built instances ``bundle_config`` is
        ignored and the number of chains is inferred from the bundle's
        buffers.
    bundle_config : dict, optional
        Bundle hyperparameters, merged on top of the built-in default
        configuration :data:`_default_config`.  Defaults to
        ``_default_config["RQSpline_MALA"]``.  Unknown keys are discarded
        with a printed warning; settings that are not explicitly given fall
        back to their defaults (also with a printed warning).  Ignored when a
        pre-built bundle instance is provided.
    n_chains : int, optional
        Number of parallel chains.  Default is 64.  With a pre-built bundle
        instance, passing ``None`` infers the value from the bundle's buffers;
        a concrete value is checked against the bundle's buffers and a
        mismatch raises an error.
    likelihood_batch_size : int, optional
        Evaluate the final log-likelihoods in sequential batches of this size
        (``jax.lax.map``); 0 keeps a single ``jax.vmap``.  A positive value
        bounds the peak memory of the likelihood intermediates, at the cost of
        cross-batch parallelism.  Default is 0.
    checkpoint_interval : float, optional
        Minimum wall-clock seconds between flowMC checkpoint writes; set 0 to
        disable checkpointing.  Default is 600.0.

    Raises
    ------
    ImportError
        If flowMC is not installed.
    ValueError
        - If ``resource_strategy_bundles`` names an unknown bundle or is not a
          string / bundle instance;
        - If the number of chains cannot be inferred from a pre-built bundle
          instance.
    """

    def __init__(
        self,
        likelihood,
        prior_dict,
        label=None,
        outdir=None,
        rkey_seed: int = 0,
        resource_strategy_bundles: str | ResourceStrategyBundle = "RQSpline_MALA",
        bundle_config: dict | None = _default_config["RQSpline_MALA"],
        n_chains: int | None = 64,
        likelihood_batch_size: int = 0,
        checkpoint_interval: float = 600.0,
    ) -> None:
        super().__init__(likelihood, prior_dict, label, outdir, rkey_seed)

        self.likelihood_batch_size = int(likelihood_batch_size)
        self.n_chains = int(n_chains) if n_chains is not None else None
        self.checkpoint_interval = float(checkpoint_interval)

        # Ravel/unravel utilities between the 1-D flowMC chain positions and
        # the sampled-parameter namedtuples of the base class.
        self.rng_key, test_key = jax.random.split(self.rng_key)
        test_sample = self._sample_priors_unbounded(test_key, ())
        self._dim = len(self.sampled_params)
        self._unflatten_fn = ravel_pytree(test_sample)[1]

        # Resolve the resource-strategy bundle from the name / instance and
        # the bundle configuration.
        self._bundle = self._resolve_bundle(resource_strategy_bundles, bundle_config)

        # The flowMC sampler is constructed here (outside ``run``) so that
        # ``run`` only executes and times the sampling.
        self._sampler = self._build_sampler()

    # ------------------------------------------------------------------
    # space conversions
    # ------------------------------------------------------------------
    def _z_to_array(self, z) -> jnp.ndarray:
        """Stack a sampled-parameter namedtuple into an ``(..., n_dim)`` array.

        Handles both scalar leaves (shape ``()``) and batched leaves (shape
        ``(n_chains,)`` or ``(n_samples,)``).
        """
        return jnp.stack(
            [jnp.atleast_1d(getattr(z, name)) for name in z._fields], axis=-1
        )

    def _build_logpdfs(self, is_pt: bool):
        """Return the flowMC ``logpdf`` (and ``logprior`` for PT bundles).

        The flowMC callables take ``f(x, data)`` with ``x`` a 1-D array; they
        rewrap ``x`` into the sampled-parameter namedtuple and evaluate the
        unbounded-space functions of the base class.  PT bundles temper the
        likelihood, so ``logpdf`` is the log-likelihood there and the
        log-prior (including the bijection Jacobian, which must not be
        rescaled by the temperature) is returned as ``logprior``.
        """

        def _wrap(fn):
            def wrapped(x, data):  # noqa: ARG001  data unused (closure captures all)
                return fn(self._unflatten_fn(x))

            return wrapped

        if is_pt:
            return (
                _wrap(self.loglikelihood_unbounded_fn),
                _wrap(self.logprior_unbounded_fn),
            )
        return _wrap(self.logdensity_unbounded_fn), None

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _resolve_bundle(self, bundle_arg, bundle_config) -> ResourceStrategyBundle:
        """Resolve the ``resource_strategy_bundles`` / ``bundle_config`` inputs
        into a ready-to-use :class:`ResourceStrategyBundle`.

        Sets ``self.resource_strategy_bundles`` (the bundle class name),
        ``self.bundle_config`` (the merged configuration, or ``None`` for a
        pre-built instance) and ``self.n_chains``.

        Unknown ``bundle_config`` keys are discarded with a printed warning;
        settings that are not explicitly given fall back to their defaults
        (also with a printed warning).
        """
        if isinstance(bundle_arg, ResourceStrategyBundle):
            # Pre-built instance: used as-is, bundle_config is ignored.  Only
            # warn when the user explicitly passed a config (the default
            # ``bundle_config`` value is expected for pre-built instances).
            if bundle_config is not None:
                print(
                    "Warning: bundle_config is ignored when a pre-built "
                    "ResourceStrategyBundle instance is provided."
                )
            self.resource_strategy_bundles = type(bundle_arg).__name__
            self.bundle_config = None
            # flowMC bundles do not store ``n_chains`` explicitly; infer it
            # from the leading axis of the first multi-dimensional buffer
            # (chain buffers are shaped ``(n_chains, ...)``).
            inferred = None
            for resource in bundle_arg.resources.values():
                if isinstance(resource, Buffer) and resource.data.ndim >= 2:
                    inferred = int(resource.data.shape[0])
                    break
            if inferred is None:
                raise ValueError(
                    "Could not infer n_chains from the pre-built bundle "
                    "resources; provide a bundle whose buffers are shaped "
                    "(n_chains, ...)."
                )
            if self.n_chains is None:
                self.n_chains = inferred
            elif self.n_chains != inferred:
                raise ValueError(
                    f"n_chains={self.n_chains} does not match the number of "
                    f"chains ({inferred}) of the provided "
                    "ResourceStrategyBundle."
                )
            return bundle_arg

        if isinstance(bundle_arg, str):
            if bundle_arg not in _buildin_bundles:
                raise ValueError(
                    f"Unknown resource_strategy_bundles '{bundle_arg}'. "
                    f"Available built-in bundles are: {sorted(_buildin_bundles)}."
                )

            default = _default_config[bundle_arg]
            merged = dict(default)
            user = {} if bundle_config is None else dict(bundle_config)
            for user_key, value in user.items():
                if user_key in default:
                    merged[user_key] = value
                else:
                    print(
                        f"Warning: Unknown bundle_config key '{user_key}' for "
                        f"bundle '{bundle_arg}'; discarded. Valid keys are: "
                        f"{sorted(default)}"
                    )
            missing = [user_key for user_key in default if user_key not in user]
            if missing:
                print(
                    f"Warning: Bundle '{bundle_arg}': the following settings "
                    f"were not explicitly given and their default values are "
                    f"used: {missing}"
                )

            self.resource_strategy_bundles = _buildin_bundles[bundle_arg].__name__
            self.bundle_config = merged

            logpdf, logprior = self._build_logpdfs(bundle_arg.endswith("_PT"))
            self.rng_key, key_bundle = jax.random.split(self.rng_key)
            kwargs = dict(merged)
            kwargs.update(
                rng_key=key_bundle,
                n_dims=self._dim,
                n_chains=self.n_chains,
                logpdf=logpdf,
            )
            if logprior is not None:
                kwargs["logprior"] = logprior
            return _buildin_bundles[bundle_arg](**kwargs)

        raise ValueError(
            "resource_strategy_bundles must be a built-in bundle name (str) "
            "or a ResourceStrategyBundle instance. Got "
            f"{type(bundle_arg).__name__}."
        )

    def _build_sampler(self) -> _FlowMCSampler:
        """Construct the flowMC ``Sampler`` from the resolved bundle.

        Called from ``__init__`` so that ``run`` only executes and times
        ``sample``.
        """
        self.rng_key, key_sampler = jax.random.split(self.rng_key)
        return _FlowMCSampler(
            n_dim=self._dim,
            n_chains=self.n_chains,
            rng_key=key_sampler,
            resource_strategy_bundles=self._bundle,
            outdir=self.outdir,
            checkpoint_interval=self.checkpoint_interval,
        )

    def _get_production_samples(self) -> jnp.ndarray:
        """Collect the production samples from the bundle resources.

        Built-in bundles store them in the ``"positions_production"`` buffer;
        custom bundles are expected to contain a ``State`` resource whose
        ``"target_positions"`` entry points to the output buffer (the same
        convention the built-in bundles use).
        """
        resources = self._bundle.resources
        if "positions_production" in resources:
            return jnp.asarray(resources["positions_production"].data).reshape(
                -1, self._dim
            )

        state = next((r for r in resources.values() if isinstance(r, State)), None)
        if state is None or not isinstance(state.data.get("target_positions"), str):
            raise ValueError(
                "Could not locate the production samples: custom bundles "
                "must contain a State resource with a 'target_positions' "
                "entry naming the output buffer (as the built-in bundles do)."
            )
        buffer = resources[state.data["target_positions"]]
        return jnp.asarray(buffer.data).reshape(-1, self._dim)

    def _finalize(self, flat_samples, runing_time) -> dict:
        """Assemble the chain output into the run-result dictionary.

        The chain positions are restored to the physical space and reported as
        ``sampled_params``; the ``fixed_params`` are broadcast to the sample
        count, and the ``derived_params`` are taken from the likelihood-input
        assembly.  MCMC samples are equally weighted, so ``weights`` is left as
        None and ``log_evidence`` is NaN.
        """
        n_samples = flat_samples.shape[0]
        z_batch = jax.vmap(self._unflatten_fn)(flat_samples)
        x_batch = self._to_physical_params(z_batch)
        ll_params = jax.vmap(self._likelihood_params_from_unbounded)(z_batch)
        log_likelihoods = map_fn(
            self.likelihood.log_likelihood, self.likelihood_batch_size
        )(ll_params)

        sampled = {name: np.asarray(value) for name, value in x_batch._asdict().items()}
        fixed = {
            name: np.full(n_samples, value) for name, value in self.fixed_params.items()
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

        # flowMC is a plain MCMC sampler: no evidence estimate is available.
        return self.format_output(
            posterior_samples, log_likelihoods, np.nan, runing_time, weights=None
        )

    def run(self) -> dict:
        """Run the flowMC sampler.

        Only the ``sample`` call is executed (and timed) here; the flowMC
        sampler and its bundle were constructed in ``__init__``.

        Returns
        -------
        dict
            Run result containing ``label``, ``outdir``, ``rkey_seed``,
            ``prior_dict``, ``likelihood_class``, ``runing_time``,
            ``sampler_class``, ``sampler_config``, ``posteriors`` (physical
            sampled/derived/fixed parameters) and ``log_likelihoods``; the
            log-evidence ``log_evidence`` is NaN and ``weights`` is an empty
            array (equally weighted MCMC samples).  The result is also saved to
            ``outdir/{label}.json``.
        """
        # Initial chain positions drawn from the priors in the unbounded space.
        self.rng_key, key_init = jax.random.split(self.rng_key)
        init_z = self._sample_priors_unbounded(key_init, (self.n_chains,))
        init_array = self._z_to_array(init_z)

        start = time.perf_counter()
        self._sampler.sample(init_array, data={})
        runing_time = time.perf_counter() - start

        output = self._finalize(self._get_production_samples(), runing_time)
        self._save_output(output)
        return output
