"""jaxns sampler interface for Bayesian parameter estimation.

jaxns is a pure-JAX nested sampling implementation (Albert et al.).  Nested
sampling evolves a set of *live points* through nested likelihood contours,
discarding the lowest-likelihood point and replacing it by a fresh sample drawn
from the prior restricted to the region above that likelihood.  Two things come
out of a run:

- the *evidence* ``Z = \\int L(theta) pi(theta) d theta`` with an uncertainty,
  which is exactly what MCMC-based samplers (flowMC, blackjax HMC) cannot give,
- a set of *reweighted* posterior samples with the importance weights
  ``dZ_i / Z``, appropriate for degenerate, multimodal and curving-degenerate
  posteriors.

This module provides :class:`JaxnsSampler`, which follows the same design as the
other samplers in this package: priors are declared through numpyro
distributions (:class:`~pespace.inference.sampler.base.NumpyroPriorSamplerBase`),
derived/fixed entries reuse the shared prior-dictionary machinery, and the
result dictionary is produced by
:meth:`~pespace.inference.sampler.base.SamplerBase.format_output`.

Notes
-----
- Nested sampling does not use the prior bijections of
  :class:`~pespace.inference.sampler.base.NumpyroPriorSamplerBase`: jaxns works
  directly on the unit-cube ``U`` space and applies the *quantile* transform of
  each prior, ``X = F^{-1}(U)``.  The sampled distributions are therefore
  **tensorflow_probability** ones, looked up by name in ``tfp.distributions``;
  jaxns itself rejects distributions without a proper quantile, raising
  ``InvalidDistribution`` while the model is being built.  This class inherits
  :class:`~pespace.inference.sampler.base.SamplerBase` directly -- no
  bijections, and the prior/likelihood are wrapped in the physical space only.
- jaxns evaluates the likelihood on a sharded device mesh.
- ``log_evidence`` is the jaxns estimate ``log_Z_mean``, and the posterior
  samples are *equally* weighted: nested sampling reports importance-weighted
  samples, so the interface resamples them (with replacement, down to the
  run's effective sample size) before reporting -- exactly what jaxns' own
  ``summary`` and marginalisation helpers do.  ``weights`` is therefore left
  as None, which the shared formatter stores as an empty array.
- The complete ``NestedSamplerResults`` of the run, including the importance
  weights ``log_dp_mean`` the resampling consumed, is reported under the
  ``jaxns_results`` key of the output dictionary.  ``run`` additionally prints
  the jaxns summary and saves the jaxns corner/diagnostic plots and the raw
  ``NestedSamplerResults`` into ``outdir``.
- :func:`tfpSine` / :func:`tfpCosine` give the inclination (``sin(x)/2``) and
  the ecliptic latitude (``cos(x)/2``) priors as TFP distributions, the TFP
  counterparts of :class:`~pespace.inference.sampler.base.Sine` /
  :class:`~pespace.inference.sampler.base.Cosine` (which are numpyro ones and
  cannot be consumed by jaxns).  Use them as prior factories, e.g.
  ``"inclination": {"distribution": tfpSine}``.
- jaxns is an optional dependency, imported lazily.
"""

from __future__ import annotations

import os
import time
from typing import Optional, cast

import jax
import jax.numpy as jnp
import numpy as np

from .base import SamplerBase, _validate_prior_dict

try:
    import tensorflow_probability.substrates.jax as tfp

    tfb = tfp.bijectors
    tfpd = tfp.distributions

    from jaxns import Model, NestedSampler, Prior, ShardedStaticNestedSampler
    from jaxns import load_results as _jaxns_load_results
    from jaxns import plot_cornerplot as _jaxns_plot_cornerplot
    from jaxns import plot_diagnostics as _jaxns_plot_diagnostics
    from jaxns import save_results as _jaxns_save_results
    from jaxns import summary as _jaxns_summary
    from jaxns.framework.special_priors import (
        Bernoulli as _JaxnsBernoulli,
        Categorical as _JaxnsCategorical,
    )
    from jaxns.internals.random import resample_indicies as _jaxns_resample_indicies
except ImportError as err:
    raise ImportError(
        "jaxns (with tensorflow_probability) is not installed by default; it "
        "is an optional dependency of pespace. If the jaxns sampler interface "
        "is needed, please install jaxns and tensorflow_probability manually."
    ) from err

__all__ = ["JaxnsSampler", "tfpCosine", "tfpSine"]


def _as_jnp_float64(value):
    """Convert a numeric prior argument into a float64 JAX array.

    TFP infers ``float32`` from plain Python scalars even with x64 enabled, so
    the distribution would store its parameters in single precision and the
    U-to-X quantile transform would silently lose the remaining digits.
    Handing TFP a JAX array instead makes it keep the float64 dtype.  It is the
    same fix jaxns applies to its own boundaries with ``mp_policy``; here it is
    applied one step earlier, where the arguments enter TFP.

    Only array-like arguments are touched; everything else (``None``, strings,
    other objects) is returned unchanged.  Integers stay integer, and a bool --
    a subclass of ``int`` -- comes out as a 0-d bool array, which TFP treats
    exactly like the Python ``bool`` it expects for options such as
    ``validate_args``.
    """
    if isinstance(value, (int, float, list, tuple, np.ndarray, jnp.ndarray)):
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.floating):
            return array.astype(jnp.float64)
        return array
    return value


# ---------------------------------------------------------------------------
# TFP angle priors: counterparts of base.Sine / base.Cosine
# ---------------------------------------------------------------------------
# ``pespace.inference.sampler.base.Sine`` and ``.Cosine`` are numpyro
# distributions, which jaxns cannot consume: it maps every prior onto the unit
# cube through its quantile (``X = F^{-1}(U)``) and requires the wrapped
# distribution to carry a ``_quantile``.  These factories build the same two
# densities as TFP transformed distributions of the unit-cube variable itself,
# ``u ~ Uniform(0, 1)``: the bijector's forward is verbatim the inverse CDF
# those classes sample with, and the ``Uniform`` base carries TFP's own
# ``_quantile``, which is what jaxns requires.


class _Arcsin(tfb.Bijector):
    """``u -> arcsin(2u - 1)``: ``Uniform(0, 1)`` to ``cos(x)/2`` on ``[-pi/2, pi/2]``.

    TFP names the pre-image ``x`` and the image ``y``: here ``_forward(x)``
    takes the unit-cube variable ``u`` and returns the angle.
    """

    def __init__(self, name: str = "arcsin"):
        super().__init__(forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return jnp.arcsin(2.0 * x - 1.0)

    def _inverse(self, y):
        return 0.5 * (1.0 + jnp.sin(y))

    def _forward_log_det_jacobian(self, x):
        # log|d arcsin(2x - 1)/dx| = -0.5 log(4 x (1 - x)) = -0.5 (log x +
        # log(1 - x)) -- the log 4 cancels exactly.  TFP derives the inverse
        # log-det by evaluating this at x = (1 + sin(angle)) / 2, where 1 - x
        # loses digits at the support edges, hence the log1p.
        return -0.5 * (jnp.log(x) + jnp.log1p(-x))

    def _inverse_log_det_jacobian(self, y):
        # log|du/dangle| = log(cos(angle) / 2).
        return jnp.log(0.5 * jnp.cos(y))

    @classmethod
    def _is_increasing(cls, **kwargs):
        return True


class _Arccos(tfb.Bijector):
    """``u -> arccos(1 - 2u)``: ``Uniform(0, 1)`` to ``sin(x)/2`` on ``[0, pi]``.

    Same TFP naming as :class:`_Arcsin`: ``_forward(x)`` maps the unit-cube
    variable ``u`` to the angle.
    """

    def __init__(self, name: str = "arccos"):
        super().__init__(forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return jnp.arccos(1.0 - 2.0 * x)

    def _inverse(self, y):
        return 0.5 * (1.0 - jnp.cos(y))

    def _forward_log_det_jacobian(self, x):
        # d arccos(1 - 2x)/dx = 1/sqrt(x (1 - x)), i.e. the same log-det as
        # _Arcsin: the log 4 cancels exactly, and 1 - x loses digits at the
        # support edges, hence the log1p.
        return -0.5 * (jnp.log(x) + jnp.log1p(-x))

    def _inverse_log_det_jacobian(self, y):
        # log|du/dangle| = log(sin(angle) / 2).
        return jnp.log(0.5 * jnp.sin(y))

    @classmethod
    def _is_increasing(cls, **kwargs):
        return True


def _tfp_uniform_base(validate_args: bool):
    """``Uniform(0, 1)`` with float64 boundaries, the base of both factories.

    The base *is* the unit-cube variable jaxns samples, so the bijector
    composed on top of it is exactly the quantile transform.  The boundaries
    are handed to TFP as **arrays**: TFP infers float32 from plain Python
    scalars even with x64 enabled, and the quantile transform would then lose
    the remaining digits.  (jaxns enables x64 on import, so ``jnp.asarray``
    yields float64 here.)
    """
    return tfpd.Uniform(
        low=jnp.asarray(0.0),  # type: ignore  # float64 array on purpose
        high=jnp.asarray(1.0),  # type: ignore  # float64 array on purpose
        validate_args=validate_args,
    )


def tfpCosine(name: str = "Cosine", validate_args: bool = False):
    """TFP counterpart of :class:`~pespace.inference.sampler.base.Cosine`.

    Builds the same density, ``p(x) = cos(x) / 2`` on ``[-pi/2, pi/2]``, as
    ``TransformedDistribution(Uniform(0, 1), u -> arcsin(2u - 1))`` -- the very
    inverse CDF :meth:`~pespace.inference.sampler.base.Cosine.sample` writes
    (``jnp.arcsin(2.0 * u - 1.0)``) -- whose ``Uniform`` base carries the
    ``_quantile`` jaxns requires.  Pass it as the factory of a jaxns prior
    entry::

        "ecliptic_latitude": {"distribution": tfpCosine},

    Parameters
    ----------
    name : str, optional
        Name of the TFP distribution.  Default is "Cosine".
    validate_args : bool, optional
        Passed to the TFP distribution.  Default is False.

    Returns
    -------
    tfpd.TransformedDistribution
        The TFP distribution with the ``cos(x) / 2`` density.
    """
    return tfpd.TransformedDistribution(
        distribution=_tfp_uniform_base(validate_args),
        bijector=_Arcsin(),
        name=name,
    )


def tfpSine(name: str = "Sine", validate_args: bool = False):
    """TFP counterpart of :class:`~pespace.inference.sampler.base.Sine`.

    Same construction as :func:`tfpCosine`, but the unit-cube variable is
    mapped through ``x = arccos(1 - 2u)``, giving ``p(x) = sin(x) / 2`` on
    ``[0, pi]`` -- again the inverse CDF
    :meth:`~pespace.inference.sampler.base.Sine.sample` writes
    (``jnp.arccos(1.0 - 2.0 * u)``)::

        "inclination": {"distribution": tfpSine},

    Parameters
    ----------
    name : str, optional
        Name of the TFP distribution.  Default is "Sine".
    validate_args : bool, optional
        Passed to the TFP distribution.  Default is False.

    Returns
    -------
    tfpd.TransformedDistribution
        The TFP distribution with the ``sin(x) / 2`` density.
    """
    return tfpd.TransformedDistribution(
        distribution=_tfp_uniform_base(validate_args),
        bijector=_Arccos(),
        name=name,
    )


def _make_jaxns_prior(name, dist_name, kwargs):
    """Build the prior the jaxns prior model generator yields for a parameter.

    ``dist_name`` is normally a name looked up in ``tfp.distributions``, but a
    distribution class (or any factory) may be passed directly.  Numeric
    arguments go through :func:`_as_jnp_float64`, because TFP otherwise infers
    ``float32`` from plain Python scalars.

    A continuous parameter yields ``Prior(tfp_distribution)``.  ``Categorical``
    and ``Bernoulli`` are served by the jaxns special priors instead: those
    live directly on the unit cube and, being ``BaseAbstractPrior`` rather than
    ``tfpd.Distribution``, must be yielded as they are -- never wrapped in
    ``Prior(...)``.  They build their own TFP distribution internally (available
    as ``prior.dist``), so none is built here.
    """
    args = {key: _as_jnp_float64(value) for key, value in kwargs.items()}
    if dist_name in ("Categorical", "Bernoulli"):
        try:
            if dist_name == "Categorical":
                # "cdf" suits discrete parameters whose neighbouring
                # categories are correlated.
                return _JaxnsCategorical(
                    parametrisation="cdf", name=name, **args
                )
            return _JaxnsBernoulli(name=name, **args)
        except TypeError as err:
            raise ValueError(
                f"Invalid arguments for the jaxns special prior "
                f"'{dist_name}': {err}"
            )
    if isinstance(dist_name, str):
        factory = getattr(tfpd, dist_name, None)
        if factory is None:
            raise ValueError(
                f"prior_dict['{name}']: '{dist_name}' is not a "
                "tensorflow_probability distribution.  jaxns works with "
                "TFP distributions, whose argument names differ from "
                "numpyro for a few cases (e.g. InverseGamma takes 'scale')."
            )
    else:
        factory = dist_name
    try:
        tfp_distribution = factory(**args)
    except TypeError as err:
        raise ValueError(
            f"Invalid arguments for the tensorflow_probability "
            f"distribution '{dist_name}': {err}"
        )
    return Prior(tfp_distribution, name=name)


class JaxnsSampler(SamplerBase):
    """Nested-sampling parameter estimation with jaxns.

    The prior dictionary follows the shared convention of this package:

        (1) sampled parameters
            "param_name": {"distribution": "dist_name", **dist_params},
        (2) derived parameters (computed from sampled/derived ones)
            "param_name": {"transform": <callable>, "from": ("param_1", ...)},
        (3) fixed parameters
            "param_name": {"distribution": "Delta", "v": 0.0},
            "param_name": {"fixed": 1.0},

    Unlike the other samplers of this package, jaxns does not work in the
    unbounded space: it maps every prior onto the unit cube through its quantile,
    ``X = F^{-1}(U)`` with ``U ~ Uniform([0, 1]^d)``.  The distribution names are
    therefore looked up in ``tensorflow_probability.distributions`` rather than
    numpyro.  jaxns checks itself that a distribution provides a proper quantile
    and raises ``InvalidDistribution`` while the model is being built; commonly
    usable distributions are ``Normal``, ``Uniform``, ``HalfNormal``,
    ``TruncatedNormal``, ``Beta``, ``Gamma``, ``InverseGamma``, ``StudentT``,
    ``Exponential``, ``Laplace``, ``Logistic``, ``Chi2``, ``Cauchy`` and
    ``LogNormal``.  ``Categorical`` and ``Bernoulli`` are routed to the jaxns
    special priors.

    .. note::
       TFP argument names differ from numpyro for a few distributions.  In
       particular TFP's ``InverseGamma`` takes ``scale`` where numpyro took
       ``rate``; that distribution is rarely used here, so no translation layer
       is provided.

    Parameters
    ----------
    likelihood : object
        Likelihood instance providing a pure ``log_likelihood(params)``.
    prior_dict : dict
        Input prior specification.
    label : str, optional
        Label attached to the sampling run. The default None becomes "None".
    outdir : str, optional
        Output directory.  It must not exist yet: the base class creates it
        and refuses to reuse an existing path, so that the results of an
        earlier run are never overwritten.  Default is None, which resolves to
        an ``outdir_{label}`` directory under ``os.getcwd()``.
    rkey_seed : int, optional
        Seed for the random number generator key. Default is 0.
    max_samples : int, optional
        Upper bound on the number of collected samples, **including phantom
        samples**.  It sizes the pre-allocated sample buffer, so it is the main
        memory knob of a run.  Default is None, i.e. jaxns derives
        ``c * (k + 1) * 100`` (one hundred shrinkages of the live-point set).
    num_live_points : int, optional
        *Entry value* of the live-point budget.  Its jaxns semantics is the
        sample-budget equivalent ``c * (k + 1)``, **not** the live point count
        -- see the class notes.  Default is None, i.e. jaxns picks the live
        point count itself: ``c = 30 * U_ndims`` (``100 * U_ndims`` with
        ``difficult_model=True``).
    num_slices : int, optional
        Number of slice proposals per accepted sample (the Markov-chain
        length). Default is None, i.e. jaxns uses ``s * U_ndims``.
    s : int, optional
        Slice-density multiplier, ``num_slices = s * U_ndims``. Default is None,
        i.e. 5 (10 with ``difficult_model=True``).
    k : int, optional
        Number of phantom samples saved per chain to sharpen the evidence
        uncertainty estimate. Default is None, i.e. 0, or ``U_ndims`` with
        ``parameter_estimation=True``.
    c : int, optional
        Number of parallel Markov chains.  This *is* the live point count, and
        is the knob to use when an exact number of live points is wanted.
        Default is None, i.e. jaxns derives it from ``num_live_points``.
    devices : sequence, optional
        Devices to run the sharded nested sampling over. Default is None,
        i.e. all available devices.
    difficult_model : bool, optional
        Use the more robust (slower) jaxns defaults, for degenerate or
        multimodal posteriors. Default is False.
    parameter_estimation : bool, optional
        Use the jaxns defaults tuned for parameter estimation: phantom samples
        are enabled and the run terminates once the enclosed
        ``X * L`` has dropped by a fraction of its peak. Default is False,
        which terminates on the remaining-evidence threshold ``dlogZ`` (better
        for the evidence itself).
    shell_fraction : float, optional
        Fraction of the live points discarded per iteration. Default is 0.5,
        jaxns' own default.
    gradient_guided : bool, optional
        Guide the slice directions with likelihood gradients (experimental).
        Default is False.
    init_efficiency_threshold : float, optional
        Passed through to jaxns: if positive, an initial uniform-sampling stage
        runs down to this acceptance efficiency before slice sampling starts.
        Default is 0.1, jaxns' own default.
    verbose : bool, optional
        Print the jaxns progress diagnostics during the run. Default is False.

    Notes
    -----
    **Parameter summary.**  ``likelihood`` and ``prior_dict`` are the required
    inputs; together they build the jaxns ``Model`` that ``NestedSampler``
    receives as its ``model`` argument, kept as ``sampler.model``.  The
    remaining arguments are jaxns' own configuration, grouped by whether jaxns
    may change the value:

    *Optional; the effective value may differ from the entry value* (the
    effective values are what ``sampler_config`` records):

    - ``num_live_points`` -- the live-point *budget* ``c * (k + 1)``, **not**
      the number of live points.  It is only used to derive the chain count
      ``c = max(1, num_live_points // (k + 1))``, which then overrides ``c``.
      Default ``None``: jaxns picks the count by itself.  The number of live
      points actually used is that ``c`` rounded up until the discarded shell
      divides evenly over the devices, so it is generally **not** the value
      that was passed in::

          sampler.num_live_points_input  # what was requested (budget)
          sampler.num_live_points        # live points actually used (rounded-up c)
          sampler.c                      # jaxns' c, before that round-up

      For example ``num_live_points=4000`` with ``k=2`` gives ``c = 1333``.
    - ``max_samples`` -- number of samples the pre-allocated buffer holds,
      phantom samples included; the main memory knob.  Default ``None``:
      ``c * (k + 1) * 100`` (one hundred shrinkages).  The effective value is
      rounded up so that a whole number of shells (with their phantom samples)
      fits the buffer; jaxns does not back-fill it, so it is read back from the
      running sampler.
    - ``num_slices`` -- slice proposals per accepted sample (the chain length).
      Default ``None``: ``U_ndims * s``.
    - ``s`` -- slice-density multiplier behind ``num_slices``; used only when
      ``num_slices`` was not given, and then stored as a float.  Default
      ``None``: 5, or 10 with ``difficult_model=True``.  ``sampler.s`` is
      ``None`` whenever ``num_slices`` was given explicitly.
    - ``k`` -- phantom samples saved per chain, sharpening the evidence
      uncertainty estimate.  Default ``None``: 0, or
      ``min(U_ndims, num_slices - 1)`` with ``parameter_estimation=True``;
      jaxns requires ``0 <= k < num_slices``.
    - ``c`` -- number of parallel Markov chains, which *is* the live point
      count before the device round-up.  Default ``None``: ``30 * U_ndims``,
      or ``100 * U_ndims`` with ``difficult_model=True``; ignored whenever
      ``num_live_points`` is given.  Use it to pin the count down.
    - ``shell_fraction`` -- fraction of the live points discarded per
      iteration.  Default ``0.5`` (``None`` means 0.5 too).  The effective
      value is ``max(shell_fraction, 1 / c)``, so a very small entry value is
      raised to keep at least one point per shell; it drives the round-ups of
      both the live points and the buffer.
    - ``init_efficiency_threshold`` -- if positive, jaxns first runs uniform
      sampling down to this acceptance efficiency, then switches to slice
      sampling; 0 turns that stage off.  Default ``0.1``; the value itself is
      passed through unchanged (only its sign matters).

    *Fixed configuration, passed through as given*:

    - ``devices`` -- devices to shard the run over.  Default ``None``: all
      available devices, resolved by jaxns while the sampler is built, so
      configure JAX (``XLA_FLAGS``, ``CUDA_VISIBLE_DEVICES``, ...) first.  The
      device count enters both round-ups above.
    - ``difficult_model`` -- jaxns' more robust (slower) defaults for
      degenerate or multimodal posteriors: ``s`` 5 -> 10, ``c`` ``30`` ->
      ``100`` times ``U_ndims``, no midpoint slice shrinking.  It only changes
      defaults, never values given explicitly.  Default ``False``.
    - ``parameter_estimation`` -- defaults tuned for parameter estimation:
      phantom samples are enabled (``k``) and the run terminates once the
      enclosed ``X * L`` has dropped by a fraction of its peak, instead of at
      the remaining-evidence threshold ``dlogZ``.  Default ``False``, which is
      better for the evidence itself.
    - ``gradient_guided`` -- guide the slice directions with likelihood
      gradients (experimental).  Default ``False``.
    - ``verbose`` -- print jaxns' progress diagnostics during the run.
      Default ``False``.

    Entry values are kept under ``*_input`` attributes, effective values on the
    plain attributes; the latter are also what ``format_output`` records under
    ``sampler_config``.

    Construction happens in ``__init__`` (through :meth:`_build_sampler`), which
    resolves the device list and therefore initialises the JAX backend.
    Configure JAX (``XLA_FLAGS``, ``CUDA_VISIBLE_DEVICES``,
    ``jax_platform_name``, ...) *before* instantiating the sampler.

    Calling :meth:`run` twice advances the internal RNG key and repeats the run
    with a different but statistically equivalent realisation; the configuration
    stored on the instance is not modified.

    The generic ``logprior_fn`` / ``logdensity_fn`` interfaces of
    :class:`~pespace.inference.sampler.base.SamplerBase` are not provided: the
    nested sampler works on the unit cube, where the prior is uniform by
    construction, so it never needs a physical-space prior density.  Prior
    samples and prior densities come from the jaxns model instead:
    ``sampler.model.sample_U(key)`` (unit cube), ``sample_W(key)`` (per-prior
    values), ``transform(U)`` (physical parameters) and ``log_prob_prior(U)``.
    The jaxns prior of each sampled parameter is kept under
    ``sampler.sampled_params``.

    ``loglikelihood_fn`` *is* provided, overriding the generic interface with
    the physical-space implementation: it takes the sampled parameters
    positionally (``*x``, in the declaration order of the prior dictionary),
    completes them with the fixed and derived ones, and then evaluates the
    likelihood.

    The posterior samples of a run start out *importance weighted* (the weight
    of a sample is the share of the evidence it carries, ``log_dp_mean``); they
    are resampled into an equally-weighted set of about the run's effective
    sample size before being reported, so that ``weights`` is left as None
    (stored as an empty array) and every reported sample counts once.  The
    importance weights themselves stay available under
    ``jaxns_results['log_dp_mean']``.
    """

    def __init__(
        self,
        likelihood,
        prior_dict,
        label=None,
        outdir=None,
        rkey_seed: int = 0,
        # arguments for jaxns
        max_samples: Optional[int] = None,
        num_live_points: Optional[int] = None,
        num_slices: Optional[int] = None,
        s: Optional[int] = None,
        k: Optional[int] = None,
        c: Optional[int] = None,
        devices=None,
        difficult_model: bool = False,
        parameter_estimation: bool = False,
        shell_fraction: float = 0.5,
        gradient_guided: bool = False,
        init_efficiency_threshold: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(likelihood, prior_dict, label, outdir, rkey_seed)

        # Entry configuration: what the caller asked for.  These attributes are
        # never overwritten, so the values jaxns derives (written onto the plain
        # names below by _build_sampler) remain distinguishable.  jaxns casts
        # the integer parameters itself while building the sampler, so they are
        # stored just as they came in.
        self.num_live_points_input = num_live_points
        self.max_samples_input = max_samples
        self.num_slices_input = num_slices
        self.s_input = s
        self.k_input = k
        self.c_input = c
        self.shell_fraction_input = shell_fraction
        self.init_efficiency_threshold_input = init_efficiency_threshold
        self.devices = devices
        self.difficult_model = bool(difficult_model)
        self.parameter_estimation = bool(parameter_estimation)
        self.gradient_guided = bool(gradient_guided)
        self.verbose = bool(verbose)

        # Values jaxns actually used, filled in by _build_sampler().  They are
        # also what format_output records under "sampler_config".  Every
        # placeholder is None: it only marks the attribute as "not resolved
        # yet", and _build_sampler() overwrites all of them before __init__
        # returns -- including the two whose jaxns defaults are concrete
        # floats (``shell_fraction`` and ``init_efficiency_threshold``), which
        # would otherwise advertise a value that is not the effective one.
        self.num_live_points = None
        self.max_samples = None
        self.num_slices = None
        self.s = None
        self.k = None
        self.c = None
        self.shell_fraction = None
        self.init_efficiency_threshold = None

        # Model attributes, filled in by _build_model() (which delegates to
        # _parse_prior_dict).  They are declared here as placeholders so that
        # the complete attribute set of the class can be read off __init__.
        self.sampled_params = {}        # name -> the jaxns prior object
        self.fixed_params = {}          # name -> np.ndarray; deliberately kept
                                        #   out of the jaxns model and injected
                                        #   when the likelihood runs
        self.derived_params = {}        # output name(s) -> (transform, *inputs)

        self.model = self._build_model()      # the jaxns Model
        self.sampler = self._build_sampler()  # the jaxns NestedSampler

        self._results = None  # jaxns NestedSamplerResults of the last run

    # ------------------------------------------------------------------
    # Prior dictionary parsing and model construction
    # ------------------------------------------------------------------
    def _build_model(self) -> Model:
        """Translate ``prior_dict`` + ``likelihood`` into a jaxns ``Model``.

        Three steps:

        1. split the prior dictionary into sampled / derived / fixed entries;
        2. build the jaxns prior of every sampled parameter, in declaration
           order.  Fixed parameters deliberately do **not** enter the model;
        3. wrap the likelihood so that it receives only the sampled parameters
           in the physical space, completes them with the fixed and derived
           ones, and then calls the original likelihood.
        """
        sampled, derived, fixed = self._parse_prior_dict()
        self.derived_params = derived
        self.fixed_params = fixed
        # The jaxns prior of every sampled parameter, in declaration order;
        # these are also the objects the prior model generator yields.
        self.sampled_params = {
            name: _make_jaxns_prior(name, dist_name, kwargs)
            for name, (dist_name, kwargs) in sampled.items()
        }

        def prior_model():
            """Yield the priors in declaration order; return the sampled values."""
            values = []
            for prior in self.sampled_params.values():
                values.append((yield prior))
            return tuple(values)

        model = Model(prior_model=prior_model, log_likelihood=self.loglikelihood_fn)
        return model

    def _parse_prior_dict(self):
        """Split the prior dictionary into sampled / derived / fixed entries.

        Mirrors the semantics of ``NumpyroPriorSamplerBase`` so that the same
        ``prior_dict`` can be reused across the samplers of this package.
        Sampled entries are returned as their ``(dist_name, kwargs)``
        declaration; :meth:`_build_model` turns them into objects.
        """
        _validate_prior_dict(
            self.prior_dict, self.likelihood.likelihood_param_type._fields
        )

        sampled, derived, fixed = {}, {}, {}

        for name, spec in self.prior_dict.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Prior entry for '{name}' must be a dict, got "
                    f"{type(spec).__name__}."
                )
            if "distribution" in spec:
                # Pop from a copy: the caller's prior_dict is kept as it came
                # in (it is stored on the instance and serialized into the
                # output), so it must not be modified here.
                kwargs = dict(spec)
                dist_name = kwargs.pop("distribution")
                if dist_name == "Delta":
                    # numpyro's point mass; equivalent to a fixed parameter.
                    fixed[name] = np.asarray(kwargs["v"])
                    continue
                sampled[name] = (dist_name, kwargs)
            elif "transform" in spec:
                if "from" not in spec:
                    raise ValueError(
                        f"Derived parameter entry for '{name}' must provide a "
                        "'from' key listing its input parameter names."
                    )
                derived[name] = (spec["transform"], *spec["from"])
            elif "fixed" in spec:
                fixed[name] = np.asarray(spec["fixed"])
            else:
                raise ValueError(
                    f"Prior entry for '{name}' must contain a 'distribution', "
                    "'transform' or 'fixed' key."
                )
        return sampled, derived, fixed

    # ------------------------------------------------------------------
    # Likelihood wrapping
    # ------------------------------------------------------------------
    def loglikelihood_fn(self, *x) -> jnp.ndarray:  # type: ignore[override]
        """Log-likelihood of one position, in the physical space.

        Overrides the generic
        :class:`~pespace.inference.sampler.base.SamplerBase` interface, which is
        what jaxns calls: the arguments are the position -- the sampled
        parameters only.  jaxns hands over exactly the physical values produced
        by the prior model generator, positionally, in the declaration order of
        the prior dictionary.  They are completed into the likelihood's
        ``likelihood_param_type`` by :meth:`_build_ll_params` and evaluated by
        the likelihood.
        """
        return self.likelihood.log_likelihood(self._build_ll_params(x))

    def _build_ll_params(self, sampled_values):
        """Assemble the likelihood input from the sampled parameters.

        ``sampled_values`` is a sequence holding the sampled parameters in the
        physical space, positionally, in the declaration order of the prior
        dictionary -- the order the prior model generator yields them in.
        Fixed parameters are added as they are, then the derived transforms are
        applied in declaration order; a transform declared under a tuple key
        returns several parameters at once.  Only the fields the likelihood asks
        for are kept, so a declaration it does not need is dropped.

        The likelihood's own parameter namedtuple is returned.  It is what
        jaxns calls (through ``loglikelihood_fn``) for a single position, and
        what :meth:`_finalize` calls once on the whole batch of resampled rows
        to report the derived parameters; the transforms are therefore written
        as elementwise array arithmetic, which both callers rely on.
        """
        params = dict(zip(self.sampled_params.keys(), sampled_values))
        params.update(self.fixed_params)
        for outputs, (fn, *inputs) in self.derived_params.items():
            result = fn(**{key: params[key] for key in inputs})
            if isinstance(outputs, str):
                params[outputs] = result
            else:
                params.update(zip(outputs, result))
        return self.likelihood.likelihood_param_type(
            **{key: params[key] for key in self.likelihood.likelihood_param_type._fields}
        )

    # ------------------------------------------------------------------
    # Sampler construction
    # ------------------------------------------------------------------
    def _build_sampler(self) -> NestedSampler:
        """Construct the jaxns ``NestedSampler`` and record what it resolved to.

        Called from ``__init__`` so that :meth:`run` only performs the inference
        loop.  This resolves the device list, hence the JAX backend must be
        configured before the sampler is instantiated.

        ``num_live_points_input`` is the *entry* value whose jaxns meaning is the
        sample-budget equivalent ``c * (k + 1)``.  jaxns derives ``c`` from it,
        rounds it up so that the discarded shell divides evenly over the
        devices, and uses that as the live point count.  The resolved values are
        written onto the plain attributes (``num_live_points``, ``c``, ...) so
        that entry and actual values stay distinguishable.
        """
        sampler = NestedSampler(
            model=self.model,
            max_samples=self.max_samples_input,
            num_live_points=self.num_live_points_input,
            num_slices=self.num_slices_input,
            s=self.s_input,
            k=self.k_input,
            c=self.c_input,
            devices=self.devices,
            difficult_model=self.difficult_model,
            parameter_estimation=self.parameter_estimation,
            shell_fraction=self.shell_fraction_input,
            gradient_guided=self.gradient_guided,
            init_efficiency_threshold=self.init_efficiency_threshold_input,
            verbose=self.verbose,
        )

        # Values jaxns actually used, taken exactly as jaxns reports them.
        # ``max_samples`` and ``shell_fraction`` are the exceptions: the sharded
        # sampler rounds the sample buffer up to a whole number of shells and
        # clamps ``shell_fraction`` to at least ``1 / c`` (a ``None`` becomes
        # 0.5 there too), and ``NestedSampler`` back-fills neither of those, so
        # the running sampler is the only place holding the effective values.
        inner = cast(ShardedStaticNestedSampler, sampler.nested_sampler)
        self.num_live_points = sampler.num_live_points
        self.num_slices = sampler.num_slices
        self.s = sampler.s
        self.k = sampler.k
        self.c = sampler.c
        self.max_samples = inner.max_samples
        self.shell_fraction = inner.shell_fraction
        self.init_efficiency_threshold = sampler.init_efficiency_threshold
        return sampler

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _finalize(self, jaxns_results, runing_time) -> dict:
        """Assemble the jaxns run into the result dictionary.

        Nested sampling produces *importance-weighted* samples, the weight of a
        sample being the share of the evidence it carries (``log_dp_mean``).
        They are therefore resampled into an equally-weighted set before being
        reported -- with replacement, down to the run's effective sample size,
        which is what jaxns' own ``summary`` and marginalisation helpers do as
        well -- so that the reported posterior needs no weights downstream and
        ``weights`` is left as None, stored as an empty array by the shared
        formatter.  A run whose reported effective sample size is unusable
        (NaN, infinite or below one) is refused with ``RuntimeError`` rather
        than reported as a posterior made of copies of a single sample.
        The samples are restored to the physical space
        and reported as ``sampled_params``; ``fixed_params`` are broadcast to
        the sample count and ``derived_params`` are taken from the
        likelihood-input assembly of the resampled rows.

        The log-likelihoods are the ones jaxns evaluated while sampling
        (``log_L_samples``): re-evaluating them here would give the same
        numbers and cost another pass over the likelihood.

        Nothing of the run is dropped: the complete ``NestedSamplerResults``
        are reported, field by field, under the ``jaxns_results`` key of the
        output -- including the importance weights consumed above.
        """
        num_samples = int(np.asarray(jaxns_results.total_num_samples))
        log_weights = np.asarray(jaxns_results.log_dp_mean)[:num_samples]
        log_likelihoods = np.asarray(jaxns_results.log_L_samples)[:num_samples]

        # Physical sampled parameters of the kept samples, as the tuple the
        # likelihood assembly expects.  Built field by field -- and not through
        # one stacked ``(n, n_params)`` array, which would unify the dtypes and
        # turn an integer parameter (Categorical / Bernoulli) into a float one.
        x_batch = tuple(
            np.asarray(jaxns_results.samples[name])[:num_samples]
            for name in self.sampled_params
        )

        # Importance weighted -> equally weighted.  The number drawn is the
        # run's effective sample size as jaxns reports it -- the target jaxns'
        # own summary resamples to as well -- because the ESS accounted for the
        # shrinkage randomness (its dZ^2 accumulator holds E[dZ^2], not
        # (E[dZ])^2) and is corrected for the phantom samples; recomputing a
        # Kish ESS from ``log_dp_mean`` alone would treat the weights as fixed
        # and come out larger.  Replacement is what preserves the
        # weighting -- a sample is copied in proportion to its evidence
        # contribution -- and reusing the indices for the samples and for
        # their log-likelihoods keeps the two aligned.
        #
        # An ESS that is NaN, infinite or below one cannot come from a healthy
        # run: it is exp(2 log Z - log dZ^2), so NaN means degenerate evidence
        # accumulators (no weight accumulated at all, or NaN from the
        # likelihood), +inf means underflow in log space, and a value below one
        # means the phantom correction while at most a single sample carries
        # weight.  Resampling such a run would silently produce a posterior of
        # copies of one sample, so it is refused instead.
        ess = float(np.asarray(jaxns_results.ESS))
        if not (np.isfinite(ess) and ess >= 1.0):
            raise RuntimeError(
                f"jaxns reported an unusable effective sample size "
                f"(ESS={ess!r}, collected samples {num_samples}, termination "
                f"reason "
                f"{int(np.asarray(jaxns_results.termination_reason))}).  The "
                "posterior samples are resampled from the nested-sampling "
                "weights down to the ESS, so there is no posterior to report "
                "here.  Check the likelihood (NaN or constant values?) and "
                "the run diagnostics (log_L_samples, log_dp_mean, "
                "termination_reason) before retrying."
            )
        num_ess = int(ess)
        # The master key is never consumed directly: a sub-key is derived for
        # the resampling (advancing the master key), so that no two purposes
        # can ever draw from the same key.
        self.rng_key, resample_key = jax.random.split(self.rng_key)
        idx = np.asarray(
            _jaxns_resample_indicies(resample_key, log_weights, S=num_ess)
        )
        x_batch = tuple(item[idx] for item in x_batch)
        log_likelihoods = log_likelihoods[idx]
        num_samples = num_ess

        # The fixed and derived parameters follow from the sampled ones; the
        # assembly is plain array arithmetic (the transforms are evaluated
        # elementwise over the resampled rows), so it is applied to the whole
        # batch at once -- no mapping, and no likelihood evaluation either.
        ll_params = self._build_ll_params(x_batch)

        sampled = {
            name: np.asarray(value)
            for name, value in zip(self.sampled_params, x_batch)
        }
        fixed = {
            name: np.full(num_samples, value)
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

        output = self.format_output(
            posterior_samples,
            log_likelihoods,
            jaxns_results.log_Z_mean,
            runing_time,
            weights=None,
        )
        # The complete NestedSamplerResults of the run, field by field.  The
        # posterior samples reported above are a resampled view of ``samples``;
        # here nothing is dropped, so the importance weights, the unit-cube
        # samples and every per-sample array stay available.
        jaxns_results_output = {}
        for name in jaxns_results._fields:
            value = getattr(jaxns_results, name)
            if isinstance(value, dict):  # samples, parametrised_samples
                jaxns_results_output[name] = {
                    key: np.asarray(item) for key, item in value.items()
                }
            else:
                array = np.asarray(value)
                # Counters, evidence and the termination reason are kept as
                # Python numbers rather than as 0-d arrays.
                jaxns_results_output[name] = (
                    array.item() if array.ndim == 0 else array
                )
        output["jaxns_results"] = jaxns_results_output
        return output

    def run(
        self,
        print_summary: bool = True,
        make_plots: bool = True,
        save_jaxns_results: bool = True,
    ) -> dict:
        """Run the nested sampler and post-process the result.

        Parameters
        ----------
        print_summary : bool, optional
            Print the jaxns nested-sampling summary (see :meth:`print_summary`).
        make_plots : bool, optional
            Save the jaxns corner and diagnostic plots into ``outdir`` (see
            :meth:`plot_cornerplot` and :meth:`plot_diagnostics`).
        save_jaxns_results : bool, optional
            Additionally save the full jaxns results next to the JSON output
            (see :meth:`save_jaxns_results`).

        Returns
        -------
        dict
            Run result containing ``label``, ``outdir``, ``rkey_seed``,
            ``prior_dict``, ``likelihood_class``, ``runing_time``,
            ``sampler_class``, ``sampler_config`` (the values jaxns actually
            used), ``posteriors`` (physical sampled/derived/fixed parameters,
            equally weighted by resampling), ``log_likelihoods`` (taken from
            the run, not re-evaluated) and the log-evidence ``log_evidence``
            (the jaxns ``log_Z_mean`` estimate); ``weights`` is left as None,
            stored as an empty array, and the reported samples each count once.
            A ``jaxns_results`` entry
            carries the complete ``NestedSamplerResults`` of the run (evidence
            uncertainty, ESS, termination reason, evaluation counters, the
            importance weights and the per-sample arrays).  The result is also
            saved to ``outdir/{label}_results.json``.

        Notes
        -----
        Calling ``run`` again advances the internal RNG key, so the sampler is
        run once more with a different but statistically equivalent
        realisation.  The configuration stored on the instance is not modified.
        """
        self.rng_key, run_key = jax.random.split(self.rng_key)

        start = time.perf_counter()
        termination_reason, state = self.sampler(run_key)
        results = self.sampler.to_results(termination_reason, state, trim=True)
        runing_time = time.perf_counter() - start

        # Kept on the instance so that the post-processing helpers below can be
        # called again later, and so that the shared output can be assembled
        # from the same object.
        self._results = results
        output = self._finalize(results, runing_time)

        if print_summary:
            self.print_summary()
        if make_plots:
            self.plot_cornerplot()
            self.plot_diagnostics()
        if save_jaxns_results:
            self.save_jaxns_results()

        self._save_output(output)
        return output

    # ------------------------------------------------------------------
    # Post-processing: thin wrappers around the jaxns utilities
    # ------------------------------------------------------------------
    def _require_results(self):
        """Return the jaxns results of the last run, or raise a clear error."""
        if self._results is None:
            raise RuntimeError(
                "No nested-sampling results available yet; call run() first, or "
                "load them with load_jaxns_results()."
            )
        return self._results

    def print_summary(self, f_obj = None):
        """Print the jaxns nested-sampling summary of the last run.

        Reports the log-evidence, the per-parameter quantiles, the termination
        reason and the information.  Note that jaxns 2.6.9 reports the
        information with the opposite sign (``H_mean`` equals minus the
        information gain).

        ``with_parametrised`` is deliberately not exposed: the interface has no
        parametrised variables (fixed and derived entries never enter the jaxns
        model), so only the sampled parameters can ever appear here.

        Parameters
        ----------
        f_obj : str or file-like object, optional
            Where to write the summary: a file path (opened and overwritten) or
            an open text stream (``io.TextIOBase``).  The default None prints to
            stdout.
        """
        results = self._require_results()
        return _jaxns_summary(results, f_obj=f_obj)

    def plot_cornerplot(
        self, variables=None, kde_overlay: bool = False, filename: Optional[str] = None
    ):
        """Save a corner plot of the posterior samples of the last run.

        Written to ``outdir/{label}_cornerplot.png`` unless ``filename`` is
        given; ``variables`` selects the parameters to plot (default: all).

        An already existing target file is refused with ``FileExistsError``
        rather than overwritten.
        """
        results = self._require_results()
        if filename is None:
            filename = os.path.join(self.outdir, f"{self.label}_cornerplot.png")
        # Never overwrite an existing file.
        if os.path.exists(filename):
            raise FileExistsError(
                f"Output file {filename!r} already exists. It is refused "
                "rather than overwritten; remove it or pass another 'filename'."
            )
        # with_parametrised is deliberately left False: jaxns updates the
        # results' sample mapping in place when it is True.
        _jaxns_plot_cornerplot(
            results,
            variables=variables,
            with_parametrised=False,
            save_name=filename,
            kde_overlay=kde_overlay,
        )

    def plot_diagnostics(self, filename: Optional[str] = None):
        """Save the jaxns run diagnostics of the last run (log L, X, efficiency).

        Written to ``outdir/{label}_diagnostics.png`` unless ``filename`` is
        given.  An already existing target file is refused with
        ``FileExistsError`` rather than overwritten.
        """
        results = self._require_results()
        if filename is None:
            filename = os.path.join(self.outdir, f"{self.label}_diagnostics.png")
        # Never overwrite an existing file.
        if os.path.exists(filename):
            raise FileExistsError(
                f"Output file {filename!r} already exists. It is refused "
                "rather than overwritten; remove it or pass another 'filename'."
            )
        _jaxns_plot_diagnostics(results, filename)

    def save_jaxns_results(self, filename=None):
        """Save the full jaxns results of the last run as JSON.

        Written to ``outdir/{label}_jaxns_results.json`` unless ``filename`` is
        given, and read back with :meth:`load_jaxns_results`.  An already
        existing target file is refused with ``FileExistsError`` rather than
        overwritten.  This is separate from the shared
        ``outdir/{label}_results.json`` written by :meth:`run`: here the raw
        ``NestedSamplerResults`` are stored in the jaxns format, which is what
        ``load_jaxns_results``, the jaxns summary and the jaxns plotting
        helpers consume.
        """
        results = self._require_results()
        if filename is None:
            filename = os.path.join(self.outdir, f"{self.label}_jaxns_results.json")
        # Never overwrite an existing file.
        if os.path.exists(filename):
            raise FileExistsError(
                f"Output file {filename!r} already exists. It is refused "
                "rather than overwritten; remove it or pass another 'filename'."
            )
        _jaxns_save_results(results, filename)

    def load_jaxns_results(self, filename: str):
        """Load results written by :meth:`save_jaxns_results` onto the instance."""
        self._results = _jaxns_load_results(filename)
