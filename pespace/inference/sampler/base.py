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
import json
import os
from abc import ABC, abstractmethod
from collections import namedtuple

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


def _build_distribution(spec: dict) -> dist.Distribution:
    """Instantiate a numpyro distribution from a dict specification."""
    spec = dict(spec)
    distribution = spec.pop("distribution")
    if isinstance(distribution, str):
        if not hasattr(dist, distribution):
            raise ValueError(
                f"Unknown numpyro distribution '{distribution}'. See "
                "numpyro.distributions for available distributions."
            )
        distribution = getattr(dist, distribution)
    try:
        return distribution(**spec)
    except TypeError as err:
        raise ValueError(
            f"Invalid arguments for the numpyro distribution "
            f"'{distribution}': {err}"
        )


class SamplerBase(ABC):
    """Framework-agnostic abstract base class for samplers."""

    def __init__(
        self, likelihood, prior_dict: dict, rkey_seed: int = 0, label=None, outdir=None
    ) -> None:
        """
        Parameters
        ----------
        likelihood : object
            Likelihood instance providing a pure ``log_likelihood(params) -> scalar`` method.
        prior_dict : dict
            Input prior specification.
        rkey_seed : int, optional
            Seed for the random number generator key. Default is 0.
        label : str, optional
            Label attached to the sampling run, converted to ``str``. The
            default None becomes the string ``"None"``.
        outdir : str, optional
            Output directory. Default is None, which resolves to an
            ``outdir_{label}`` directory under the current working directory
            where the calling script runs (``os.getcwd()``); the directory is
            created if it does not exist.
        """
        self.likelihood = likelihood
        self.prior_dict = prior_dict
        self.rkey_seed = rkey_seed
        self.rng_key = jax.random.key(self.rkey_seed)
        self.label = str(label)
        if outdir is None:
            outdir = os.path.join(os.getcwd(), f"outdir_{self.label}")
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

    def logprior_fn(self, position) -> jnp.ndarray:
        """Joint log-prior density (pure JAX, return a scalar).

        Generic interface: concrete samplers should bind this name to the
        space-specific implementation (unbounded or physical) on the
        instance before sampling.
        """
        raise NotImplementedError(
            "The generic log-prior interface is not bound. Bind it to "
            "logprior_unbounded_fn or logprior_physical_fn on the instance."
        )

    def loglikelihood_fn(self, position) -> jnp.ndarray:
        """Log-likelihood (pure JAX, return a scalar).

        Generic interface: concrete samplers should bind this name to the
        space-specific implementation (unbounded or physical) on the
        instance before sampling.
        """
        raise NotImplementedError(
            "The generic log-likelihood interface is not bound. Bind it to "
            "loglikelihood_unbounded_fn or loglikelihood_physical_fn on the "
            "instance."
        )

    def logdensity_fn(self, position) -> jnp.ndarray:
        """Log-posterior density up to a normalization constant (pure JAX, return a scalar)."""
        return self.logprior_fn(position) + self.loglikelihood_fn(position)

    def format_output(
        self,
        posterior_samples,
        log_likelihoods,
        log_evidence,
        runing_time,
        weights=None,
    ) -> dict:
        """Assemble the run result into a dictionary.

        Parameters
        ----------
        posterior_samples : dict
            Categorized posterior samples with keys ``sampled_params``,
            ``derived_params`` and ``fixed_params``, each mapping parameter
            names to their physical parameter arrays.
        log_likelihoods : array-like
            Log-likelihood of each sample.
        log_evidence : float
            Log-evidence of the run (for SMC samplers).
        runing_time : float
            Wall-clock time of the sampling run, in seconds.
        weights : array-like, optional
            Importance weight of each sample (for SMC samplers). When None,
            an empty array is stored.

        Returns
        -------
        dict
            Dictionary with keys ``label``, ``outdir``, ``rkey_seed``,
            ``prior_dict``, ``likelihood_class``, ``runing_time``,
            ``sampler_class``, ``sampler_config``, ``posteriors``,
            ``log_likelihoods``, ``log_evidence`` and ``weights``.
        """
        # JSON-serializable copy of the prior dictionary: transform/distribution
        # callables are replaced by their names, tuple keys are stringified.
        prior_dict = {}
        for name, spec in self.prior_dict.items():
            key = str(name)
            if isinstance(spec, float):
                prior_dict[key] = spec
            elif isinstance(spec, dict):
                entry = {}
                for field, value in spec.items():
                    if field in ("distribution", "transform") and not isinstance(
                        value, str
                    ):
                        entry[field] = getattr(value, "__name__", str(value))
                    else:
                        entry[field] = value
                prior_dict[key] = entry
            else:
                raise ValueError(
                    f"Prior entry for '{name}' must be a float or a dict, "
                    f"got {type(spec).__name__}."
                )

        # Sampler configuration: constructor arguments stored as attributes,
        # collected over the MRO; the meta arguments are excluded.
        meta = {"likelihood", "prior_dict", "rkey_seed", "label", "outdir"}
        sampler_config = {}
        for cls in reversed(type(self).__mro__):
            init = cls.__dict__.get("__init__", None)
            if init is not None:
                try:
                    parameters = inspect.signature(init).parameters
                except (TypeError, ValueError):  # pragma: no cover
                    continue
                for name, parameter in parameters.items():
                    if (
                        (name != "self")
                        and (
                            parameter.kind
                            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
                        )
                        and (name not in meta)
                        and (hasattr(self, name))
                    ):
                        sampler_config[name] = getattr(self, name)

        output = {
            "label": self.label,
            "outdir": self.outdir,
            "likelihood_class": type(self.likelihood).__name__,
            "prior_dict": prior_dict,
            "rkey_seed": self.rkey_seed,
            "sampler_class": type(self).__name__,
            "sampler_config": sampler_config,
            "runing_time": runing_time,
            "posteriors": posterior_samples,
            "weights": np.asarray([]) if weights is None else np.asarray(weights),
            "log_likelihoods": np.asarray(log_likelihoods),
            "log_evidence": float(np.asarray(log_evidence)),
        }
        return output

    def _save_output(self, output: dict, filename=None) -> str:
        """Save the run result to ``outdir/{label}.json`` and return its path.

        numpy/JAX arrays and numpy scalars are converted to JSON-native types
        recursively before dumping.
        """

        def to_jsonable(value):
            if isinstance(value, dict):
                return {k: to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_jsonable(v) for v in value]
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                return to_jsonable(np.asarray(value).tolist())
            if isinstance(value, np.generic):
                return value.item()
            return value

        os.makedirs(self.outdir, exist_ok=True)
        filename = f"{self.label}.json" if filename is None else filename
        path = os.path.join(self.outdir, filename)
        with open(path, "w") as file:
            json.dump(to_jsonable(output), file, indent=2)
        return path

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        """Performing the sampling run."""
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
        maps the sampled parameters to those required by the likelihood)
        "param_name": {"transform": <callable>, "from": ("param_1", ...)},
        ("param_name_1", "param_name_2"): {"transform": <callable>, "from": ("param_1", ...)},
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

    def __init__(
        self,
        likelihood,
        prior_dict: dict,
        rkey_seed: int = 0,
        label=None,
        outdir=None,
    ) -> None:
        super().__init__(likelihood, prior_dict, rkey_seed, label, outdir)
        self._require_numpyro()

        self.sampled_params = {}  # name -> numpyro Distribution (sampled)
        self.fixed_params = {}  # name -> array (fixed parameter)
        self.derived_params = {}  # output name(s) -> (callable, *input names)
        self._parse_prior_dict()
        self._validate_parameters()
        self.bijects = {
            name: transforms.biject_to(prior.support) # mapping unbounded parameters to physical parameters # fmt: skip
            for name, prior in self.sampled_params.items()
        }

        # Particle positions are namedtuples whose fields are the sampled
        # parameters; the likelihood input namedtuple is the likelihood's
        # ``likelihood_param_type``, whose fields define the required parameters.
        self.sampled_param_type = namedtuple("SampledParams", list(self.sampled_params))

    @staticmethod
    def _require_numpyro() -> None:
        """Raise if numpyro is unavailable."""
        if not _NUMPYRO_AVAILABLE:
            raise ImportError(
                "numpyro is not installed; it is required by "
                "NumpyroPriorSamplerBase. numpyro is an optional dependency of "
                "pespace, please install it manually."
            )

    def _parse_prior_dict(self) -> None:
        """Split the prior dictionary into sampled / derived / fixed entries.

        Derived entries must provide a ``'from'`` key naming their input
        parameters; a tuple key declares a transform returning several
        parameters at once.
        """
        for name, spec in self.prior_dict.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Prior entry for '{name}' must be a dict, got "
                    f"{type(spec).__name__}."
                )
            if "distribution" in spec:
                distribution = _build_distribution(spec)
                if isinstance(distribution, dist.Delta):
                    # point-mass: treat as a fixed parameter, not sampled
                    self.fixed_params[name] = np.asarray(distribution.v)
                else:
                    self.sampled_params[name] = distribution
            elif "transform" in spec:
                fn = spec["transform"]
                if "from" not in spec:
                    raise ValueError(
                        f"Derived parameter entry for '{name}' must provide a "
                        "'from' key listing its input parameter names."
                    )
                inputs = tuple(spec["from"])
                if not inputs or not all(isinstance(k, str) for k in inputs):
                    raise ValueError(
                        f"The 'from' key of derived parameter '{name}' must "
                        "be a non-empty sequence of parameter names."
                    )
                if not isinstance(name, (str, tuple)):
                    raise ValueError(
                        f"Derived parameter key '{name}' must be a str or a "
                        f"tuple of str, got {type(name).__name__}."
                    )
                if isinstance(name, tuple) and (
                    not name or not all(isinstance(k, str) for k in name)
                ):
                    raise ValueError(
                        f"Derived parameter key '{name}' must be a non-empty "
                        "tuple of parameter-name strings."
                    )
                self.derived_params[name] = (fn, *inputs)
            elif "fixed" in spec:
                self.fixed_params[name] = np.asarray(spec["fixed"])
            else:
                raise ValueError(
                    f"Prior entry for '{name}' must contain a 'distribution', "
                    "'transform' or 'fixed' key."
                )

    def _validate_parameters(self) -> None:
        """Validate the prior declarations against the likelihood.

        Iterates the prior dictionary in insertion order and checks that

        (1) every ``'from'`` input of a derived parameter is declared
            earlier (as a sampled, derived or fixed parameter);
        (2) no name is declared twice (a fixed parameter must not conflict
            with an earlier sampled/derived declaration, and vice versa);
        (3) every parameter required by the likelihood is declared.
        """
        declared = set()
        for name, spec in self.prior_dict.items():
            if "transform" in spec:
                outputs = (name,) if isinstance(name, str) else tuple(name)
                inputs = tuple(spec["from"])
                missing_inputs = [k for k in inputs if k not in declared]
                if missing_inputs:
                    raise ValueError(
                        f"Derived parameter(s) {sorted(outputs)} depend on "
                        f"parameter(s) {missing_inputs} that are not declared "
                        "before them."
                    )
                overlap = sorted(set(outputs) & declared)
                if overlap:
                    raise ValueError(
                        f"Derived parameter(s) {overlap} are declared more "
                        "than once."
                    )
                declared.update(outputs)
            else:
                if name in declared:
                    raise ValueError(
                        f"Parameter '{name}' is declared more than once "
                        "(fixed and sampled/derived declarations must not "
                        "conflict)."
                    )
                declared.add(name)

        required = set(self.likelihood.likelihood_param_type._fields)
        missing = required - declared
        if missing:
            raise ValueError(
                "The prior does not provide all parameters required by the "
                "likelihood. Missing: "
                f"{sorted(missing)}. Provided by the prior: {sorted(declared)}."
            )

    def logprior_unbounded_fn(self, z) -> jnp.ndarray:
        """Joint log-prior in the unbounded (unconstrained) space (pure JAX).

        The change-of-variables log-abs-det-Jacobian is included in the prior
        term so the tempered target stays self-consistent: the Jacobian must
        NOT be rescaled with the tempering temperature.
        """
        logp = jnp.zeros(())
        for name, prior in self.sampled_params.items():
            biject = self.bijects[name]
            # biject maps the unconstrained value to the constrained support;
            # log_abs_det_jacobian(z, x) is the change-of-variables correction.
            value = getattr(z, name)
            x = biject(value)
            log_prob = jnp.where(prior.support(x), prior.log_prob(x), -jnp.inf)
            logp = logp + log_prob + biject.log_abs_det_jacobian(value, x)
        return logp

    def _to_physical_params(self, z):
        """Restore constrained sampled parameters from unconstrained ``z``."""
        return self.sampled_param_type(
            **{name: self.bijects[name](getattr(z, name)) for name in z._fields}
        )

    def _likelihood_params_from_unbounded(self, z):
        """Assemble the likelihood-input namedtuple from unbounded ``z``."""
        return self._likelihood_params_from_physical(self._to_physical_params(z))

    def _likelihood_params_from_physical(self, x):
        """Assemble the likelihood-input namedtuple from physical ``x``.

        ``x`` must provide the sampled parameters in their constrained
        (physical) domain; derived and fixed parameters are added here.

        A transform declared under a tuple key returns several parameters at
        once (a tuple/list matching the key order); a single-name transform
        returns the parameter value directly.
        """
        params = {name: getattr(x, name) for name in x._fields}
        for name, value in self.fixed_params.items():
            params[name] = value
        for outputs, (fn, *inputs) in self.derived_params.items():
            result = fn(**{k: params[k] for k in inputs})
            if isinstance(outputs, str):
                params[outputs] = result
            elif isinstance(outputs, tuple):
                for out, value in zip(outputs, result):
                    params[out] = value
            else:
                raise TypeError(
                    f"Derived parameter key must be a str or a tuple of "
                    f"str, got {type(outputs).__name__}."
                )
        return self.likelihood.likelihood_param_type(
            **{k: params[k] for k in self.likelihood.likelihood_param_type._fields}
        )

    def loglikelihood_unbounded_fn(self, z) -> jnp.ndarray:
        """Log-likelihood in the unbounded space (pure JAX)."""
        return self.likelihood.log_likelihood(self._likelihood_params_from_unbounded(z))

    def logdensity_unbounded_fn(self, z) -> jnp.ndarray:
        """Log-posterior density up to a constant in the unbounded space."""
        return self.logprior_unbounded_fn(z) + self.loglikelihood_unbounded_fn(z)

    def logprior_physical_fn(self, x) -> jnp.ndarray:
        """Joint log-prior in the physical space (no Jacobian correction)."""
        logp = jnp.zeros(())
        for name, prior in self.sampled_params.items():
            value = getattr(x, name)
            log_prob = jnp.where(prior.support(value), prior.log_prob(value), -jnp.inf)
            logp = logp + log_prob
        return logp

    def loglikelihood_physical_fn(self, x) -> jnp.ndarray:
        """Log-likelihood for parameters given in the physical space."""
        return self.likelihood.log_likelihood(self._likelihood_params_from_physical(x))

    def logdensity_physical_fn(self, x) -> jnp.ndarray:
        """Log-posterior density up to a constant in the physical space."""
        return self.logprior_physical_fn(x) + self.loglikelihood_physical_fn(x)

    def _sample_priors_unbounded(self, key, shape: tuple = ()):
        """Draw initial positions in the unconstrained space from a given key.

        Pure-JAX helper (no mutable state), usable inside ``jax.pmap``.
        """
        keys = jax.random.split(key, len(self.sampled_params))
        x = {
            name: prior.sample(keys[i], shape)
            for i, (name, prior) in enumerate(self.sampled_params.items())
        }
        return self.sampled_param_type(
            **{name: self.bijects[name].inv(x[name]) for name in x}
        )

    def initial_position_unbounded(self, shape: tuple = ()):
        """Draw initial position(s) in the unbounded space."""
        self.rng_key, key = jax.random.split(self.rng_key)
        return self._sample_priors_unbounded(key, shape)

    def _sample_priors_physical(self, key, shape: tuple = ()):
        """Draw initial positions in the physical space from a given key.

        Pure-JAX helper (no mutable state), usable inside ``jax.pmap``.
        """
        keys = jax.random.split(key, len(self.sampled_params))
        return self.sampled_param_type(
            **{
                name: prior.sample(keys[i], shape)
                for i, (name, prior) in enumerate(self.sampled_params.items())
            }
        )

    def initial_position_physical(self, shape: tuple = ()):
        """Draw initial position(s) in the physical space."""
        self.rng_key, key = jax.random.split(self.rng_key)
        return self._sample_priors_physical(key, shape)
