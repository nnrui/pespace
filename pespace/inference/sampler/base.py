"""Base classes for sampler interfaces.

This module provides the shared machinery for all sampler backends in
:mod:`pespace.inference.sampler`:

- :class:`SamplerBase`: framework-agnostic abstract base class.  It stores the
  likelihood and the user-declared prior dictionary *as-is* (without parsing
  it) and defines the pure density / initial-position interfaces handed to
  the sampler kernels.
- :class:`NumpyroPriorSamplerBase`: base class for samplers whose priors are
  declared through numpyro distributions.  It parses the prior dictionary
  into numpyro distribution instances, builds per-parameter bijections to an
  unconstrained space (handling bounded priors natively for gradient-based
  kernels), and constructs the pure log-prior / log-likelihood functions used
  by the sampler kernels.
"""

import inspect
import logging
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

try:
    import numpyro.distributions as dist
    from numpyro.distributions import transforms

    _NUMPYRO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numpyro is absent
    dist = None
    transforms = None
    _NUMPYRO_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)


class SamplerBase(ABC):
    """Framework-agnostic abstract base class for samplers.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing a pure
        ``log_likelihood(params) -> scalar`` method.  Concrete samplers may
        additionally require ``parameter_names`` and ``params_type``
        attributes on the likelihood (see :class:`NumpyroPriorSamplerBase`).
    prior : dict
        User-declared prior specification.  Stored *as-is*; concrete
        subclasses (e.g. :class:`NumpyroPriorSamplerBase`) are responsible for
        parsing it into the probability objects they understand.
    seed : int, optional
        Seed for the JAX random number generator.  Default is 0.
    """

    def __init__(self, likelihood, prior: dict, seed: int = 0) -> None:
        self.likelihood = likelihood
        self.prior = prior
        self.seed = seed
        self.rng_key = jax.random.key(seed)

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
    """Base class for samplers using numpyro-declared priors.

    Parses the user prior dictionary into numpyro distributions and works in
    an *unconstrained* parameter space: every sampled parameter is mapped to
    the real line through ``numpyro.distributions.transforms.biject_to(
    prior.support)``, so bounded priors are handled natively for both
    gradient-free (RMH) and gradient-based (HMC/NUTS) mutation kernels.

    Prior declaration
    -----------------
    The ``prior`` dictionary maps parameter names to specifications.  Three
    kinds of entries are supported::

        # (1) independent sampled priors (enter the sampling dimension)
        'chirp_mass': {'distribution': 'Uniform', 'low': 5e5, 'high': 2e6},
        'chi_1':      {'distribution': 'Normal', 'loc': 0.0, 'scale': 0.5},

        # (2) derived parameters (computed from sampled/derived ones,
        #     not sampled; optional 'jacobian' injects the reparam log|J|)
        'm1':         {'transform': lambda mc, q: mc * (1 + q)**0.2 / q**0.6},
        'm2':         {'transform': <callable>, 'jacobian': <callable>},

        # (3) fixed parameters
        'coalescence_time': {'distribution': 'Delta', 'v': 0.0},
        'x0':               {'fixed': 1.0},

    Dictionary insertion order defines the sampled-parameter order.  The
    likelihood-namespace parameter names (provided by ``likelihood.
    parameter_names``) need not coincide with the prior-namespace keys: they
    are assembled from sampled + derived + fixed parameters and validated on
    construction (a ``ValueError`` is raised if the prior does not provide
    every parameter the likelihood requires).
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
        for name, spec in self.prior.items():
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
        params = dict(self._constrained_params(z))
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

    def _to_physical(self, z_particles: dict) -> dict:
        """Map batched unconstrained particles to physical (likelihood-namespace)
        parameter arrays by vectorising the assembly function."""
        return jax.vmap(self._build_likelihood_params)(z_particles)
