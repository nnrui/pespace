"""Base classes for sampler interfaces.

- :class:`SamplerBase`: framework-agnostic abstract base class.
- :class:`NumpyroPriorSamplerBase`: base class for samplers where the priors are
  declared through numpyro distributions. It parses the prior dictionary
  into numpyro distribution instances, builds per-parameter bijections to an
  unconstrained space (handling bounded priors natively for gradient-based
  kernels), and constructs the pure log-prior / log-likelihood functions used
  by the sampler kernels.
"""

import inspect
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

try:
    import numpyro.distributions as dist
    from numpyro.distributions import transforms

    _NUMPYRO_AVAILABLE = True
except ImportError:  # pragma: no cover
    dist = None
    transforms = None
    _NUMPYRO_AVAILABLE = False


class SamplerBase(ABC):
    """Framework-agnostic abstract base class for samplers."""

    def __init__(self, likelihood, prior_dict: dict, rkey_seed: int = 0) -> None:
        """
        Parameters
        ----------
        likelihood : object
            Likelihood instance providing a pure ``log_likelihood(params) -> scalar`` method.
        prior_dict : dict
            Input prior specification.
        seed : int, optional
            Seed for the random number generator. Default is 0.
        """
        self.likelihood = likelihood
        self.prior_dict = prior_dict
        self.rng_key = jax.random.key(rkey_seed)

    @abstractmethod
    def logprior_fn(self, position) -> jnp.ndarray:
        """Joint log-prior density (pure JAX, scalar)."""

    @abstractmethod
    def loglikelihood_fn(self, position) -> jnp.ndarray:
        """Log-likelihood (pure JAX, scalar)."""

    def logdensity_fn(self, position) -> jnp.ndarray:
        """Log-posterior density up to a constant (pure JAX, scalar)."""
        return self.logprior_fn(position) + self.loglikelihood_fn(position)

    @abstractmethod
    def initial_position(self, sample_shape: tuple = ()):
        """Draw initial position(s)."""

    @staticmethod
    def _format_output(particles: dict, log_likelihoods, weights=None) -> dict:
        """Format sampling results as a dictionary of numpy arrays.

        Parameters
        ----------
        particles : dict
            Sampled parameter arrays in the physical parameter space, keyed
            by parameter name.
        log_likelihoods : array-like
            Log-likelihood of each sample.
        weights : array-like, optional
            Importance weight of each sample (for SMC samplers).

        Returns
        -------
        dict
            Dictionary of numpy arrays keyed by parameter name, plus
            ``'log_likelihood'`` and, if given, ``'weights'``.
        """
        output = {name: np.asarray(values) for name, values in particles.items()}
        output["log_likelihood"] = np.asarray(log_likelihoods)
        if weights is not None:
            output["weights"] = np.asarray(weights)
        return output

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        """Start the sampling and return the results as a dictionary."""
        ...


class NumpyroPriorSamplerBase(SamplerBase):
    """
    Base class for samplers using priors specified with build-in distributions in numpyro.

    The input prior dictionary is parsed into numpyro distributions which provide
    facilities to map parameters in limited ranges to the unbounded real domain for
    better supporting gradient-based (HMC/NUTS) kernels in samplers.

    Notes
    -----
    The ``prior`` dictionary maps parameter names to specifications.  Three
    kinds of entries are supported:

        (1) independent sampled priors (enter the sampling dimension)
        "param_name": {"distribution": "dist_name", **dist_params},
        e.g.,
        "chirp_mass": {"distribution": "Uniform", "low": 5e5, "high": 2e6},
        "chi_1": {"distribution": "Normal", "loc": 0.0, "scale": 0.5},

        (2) derived parameters (computed from sampled/derived ones, not sampled;
        optional "jacobian" injects the reparam log|J|)
        "param_name": {"transform": <callable>, "from": ("param_1", ...), "jacobian": <callable>},
        ("param_name_1", "param_name_2"): {"transform": <callable>, "from": ("param_1", ...), "jacobian": <callable>},
        e.g.
        "symmetric_mass_ratio": {"transform": eta_from_mass1_mass2, "from": ("mass_1", "mass_2")},
        ("mass_1", "mass_2"): {"transform": component_mass_from_Mtotal_q, "from": ("total_mass", "mass_ratio")},


        (3) fixed parameters
        "param_name": {"distribution": "Delta", "v": 0.0},
        "param_name": {"fixed": 1.0},
        e.g.,
        "coalescence_time": {"distribution": "Delta", "v": 0.0},
        "coalescence_time": {"fixed": 0.0},

    Dictionary insertion order defines the parameter order. The parameters obtained by
    transform must be specified after the original parameters from which they are derived.

    Parameters specified in prior dictionary need to satisfy:
        - All parameters required by likelihood computation are covered.
        - Unused parameters that required by neither the likelihood nor transform will
          be discarded, and a warning will be thrown.
    """

    def __init__(self, likelihood, prior: dict, seed: int = 0) -> None:
        super().__init__(likelihood, prior, seed)
        self._require_numpyro()

        self.sampled_priors = {}  # name -> numpyro Distribution (sampled)
        self.parameter_names = []  # ordered sampled-parameter names
        self.derived_transforms = {}  # name -> callable (derived parameter)
        self.derived_inputs = {}  # name -> tuple(arg names) or None
        self.derived_jacobians = {}  # name -> callable (optional reparam log|J|)
        self.fixed_values = {}  # name -> array (fixed parameter)
        self._parse_prior_spec()

        self.bijects = {
            name: transforms.biject_to(prior.support)
            for name, prior in self.sampled_priors.items()
        }

        self.likelihood_parameter_names = tuple(self.likelihood.parameter_names)
        self._validate_parameters()

        # Fail fast: evaluate the log-density once on a prior draw.
        # This also triggers JIT compilation outside the sampling loop.
        test_position = self.initial_position()
        test_value = self.logdensity_fn(test_position)
        if not jnp.isfinite(test_value):
            raise ValueError(
                f"Log-density evaluated to {test_value} at a prior draw. "
                "Check the likelihood setup and prior specification."
            )

    @staticmethod
    def _require_numpyro() -> None:
        """Raise if numpyro is unavailable (used lazily by subclasses)."""
        if not _NUMPYRO_AVAILABLE:
            raise ImportError(
                "numpyro is not installed; it is required by "
                "NumpyroPriorSamplerBase. numpyro is an optional dependency of "
                "pespace, please install it manually."
            )

    def _parse_prior_spec(self) -> None:
        """Split the prior dictionary into sampled / derived / fixed entries."""
        for name, spec in self.prior_dict.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Prior entry for '{name}' must be a dict, got "
                    f"{type(spec).__name__}."
                )
            if "distribution" in spec:
                distribution = self._build_distribution(name, spec)
                if isinstance(distribution, dist.Delta):
                    # point-mass: treat as a fixed parameter, not sampled
                    self.fixed_values[name] = np.asarray(distribution.v)
                else:
                    self.sampled_priors[name] = distribution
                    self.parameter_names.append(name)
            elif "transform" in spec:
                fn = spec["transform"]
                self.derived_transforms[name] = fn
                self.derived_inputs[name] = self._infer_transform_args(fn)
                if "jacobian" in spec:
                    self.derived_jacobians[name] = spec["jacobian"]
            elif "fixed" in spec:
                self.fixed_values[name] = np.asarray(spec["fixed"])
            else:
                raise ValueError(
                    f"Prior entry for '{name}' must contain a 'distribution', "
                    "'transform' or 'fixed' key."
                )

    @staticmethod
    def _build_distribution(name: str, spec: dict) -> dist.Distribution:
        """Instantiate a numpyro distribution from a dict specification."""
        spec = dict(spec)
        distribution = spec.pop("distribution")
        if isinstance(distribution, str):
            if not hasattr(dist, distribution):
                raise ValueError(
                    f"Unknown numpyro distribution '{distribution}' for "
                    f"parameter '{name}'. See numpyro.distributions for "
                    "available distributions."
                )
            distribution = getattr(dist, distribution)
        try:
            return distribution(**spec)
        except TypeError as err:
            raise ValueError(
                f"Invalid arguments for the distribution of parameter "
                f"'{name}': {err}"
            )

    @staticmethod
    def _infer_transform_args(fn) -> tuple | None:
        """Argument names a transform function expects.

        Returns ``None`` when the signature cannot be inspected or accepts
        ``**kwargs``, in which case the full parameter dictionary is passed.
        """
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return None
        names = []
        for param in signature.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return None
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                names.append(param.name)
        return tuple(names)

    def _validate_parameters(self) -> None:
        """Check that the prior provides every parameter the likelihood needs."""
        provided = (
            set(self.sampled_priors)
            | set(self.derived_transforms)
            | set(self.fixed_values)
        )
        required = set(self.likelihood_parameter_names)
        missing = required - provided
        if missing:
            raise ValueError(
                "The prior does not provide all parameters required by the "
                "likelihood. Missing: "
                f"{sorted(missing)}. Provided by the prior: {sorted(provided)}."
            )

    def logprior_fn(self, z: dict) -> jnp.ndarray:
        """Joint log-prior in the unconstrained space (pure JAX).

        The change-of-variables log-abs-det-Jacobian is included in the prior
        term so the tempered target stays self-consistent: the Jacobian must
        NOT be rescaled with the tempering temperature.
        """
        logp = jnp.zeros(())
        for name, prior in self.sampled_priors.items():
            biject = self.bijects[name]
            # biject maps the unconstrained value to the constrained support;
            # log_abs_det_jacobian(z, x) is the change-of-variables correction.
            x = biject(z[name])
            log_prob = jnp.where(prior.support(x), prior.log_prob(x), -jnp.inf)
            logp = logp + log_prob + biject.log_abs_det_jacobian(z[name], x)
        return logp

    def _constrained_params(self, z: dict) -> dict:
        """Restore constrained sampled parameters from unconstrained ``z``."""
        return {name: self.bijects[name](z[name]) for name in self.sampled_priors}

    def _build_likelihood_params(self, z: dict) -> dict:
        """Assemble the likelihood-namespace parameter dict from unconstrained ``z``."""
        return self._likelihood_params_from_physical(self._constrained_params(z))

    def _likelihood_params_from_physical(self, x: dict) -> dict:
        """Assemble the likelihood-namespace parameter dict from physical ``x``.

        ``x`` must provide the sampled parameters in their constrained
        (physical) domain; derived and fixed parameters are added here.
        """
        params = dict(x)
        for name, fn in self.derived_transforms.items():
            inputs = self.derived_inputs[name]
            if inputs is None:
                params[name] = fn(params)
            else:
                params[name] = fn(**{k: params[k] for k in inputs})
        for name, value in self.fixed_values.items():
            params[name] = value
        return {k: params[k] for k in self.likelihood_parameter_names}

    def loglikelihood_fn(self, z: dict) -> jnp.ndarray:
        """Log-likelihood (pure JAX), delegating to the likelihood object."""
        params = self._build_likelihood_params(z)
        return self.likelihood.log_likelihood(self.likelihood.params_type(**params))

    def logprior_physical_fn(self, x: dict) -> jnp.ndarray:
        """Joint log-prior in the physical space (no Jacobian correction)."""
        logp = jnp.zeros(())
        for name, prior in self.sampled_priors.items():
            log_prob = jnp.where(
                prior.support(x[name]), prior.log_prob(x[name]), -jnp.inf
            )
            logp = logp + log_prob
        return logp

    def loglikelihood_physical_fn(self, x: dict) -> jnp.ndarray:
        """Log-likelihood for parameters given in the physical space."""
        params = self._likelihood_params_from_physical(x)
        return self.likelihood.log_likelihood(self.likelihood.params_type(**params))

    def _sample_initial_position(self, key, sample_shape: tuple = ()) -> dict:
        """Draw initial positions in the unconstrained space from a given key.

        Pure-JAX helper (no mutable state), usable inside ``jax.pmap``.
        """
        keys = jax.random.split(key, len(self.sampled_priors))
        x = {
            name: prior.sample(keys[i], sample_shape)
            for i, (name, prior) in enumerate(self.sampled_priors.items())
        }
        return {name: self.bijects[name].inv(x[name]) for name in x}

    def initial_position(self, sample_shape: tuple = ()) -> dict:
        """Draw initial position(s) in the unconstrained space."""
        self.rng_key, key = jax.random.split(self.rng_key)
        return self._sample_initial_position(key, sample_shape)

    def _sample_initial_position_physical(self, key, sample_shape: tuple = ()) -> dict:
        """Draw initial positions in the physical space from a given key.

        Pure-JAX helper (no mutable state), usable inside ``jax.pmap``.
        """
        keys = jax.random.split(key, len(self.sampled_priors))
        return {
            name: prior.sample(keys[i], sample_shape)
            for i, (name, prior) in enumerate(self.sampled_priors.items())
        }

    def initial_position_physical(self, sample_shape: tuple = ()) -> dict:
        """Draw initial position(s) in the physical space."""
        self.rng_key, key = jax.random.split(self.rng_key)
        return self._sample_initial_position_physical(key, sample_shape)

    def _to_physical(self, z_particles: dict) -> dict:
        """Map batched unconstrained particles to physical (likelihood-namespace)
        parameter arrays by vectorising the assembly function."""
        return jax.vmap(self._build_likelihood_params)(z_particles)
