"""Bilby interface for the jax-based Whittle likelihood.

This module is kept separate from
:mod:`pespace.inference.interface_bilby` so that importing the taichi
interface does not pull in jax (and, through the waveform interfaces,
ripplegw).  ``LikelihoodBilbyInterfaceJax`` is re-exported lazily from
:mod:`pespace.inference.interface_bilby` for backwards compatibility.
"""

from __future__ import annotations

import jax

try:
    from bilby.core.likelihood import Likelihood
except ImportError as err:
    raise ImportError(
        "bilby is not installed by default, if sampler interface in bilby is needed, "
        "please install bilby manually."
    ) from err

from .likelihood_jax import WhittleLikelihoodJax


class LikelihoodBilbyInterfaceJax(WhittleLikelihoodJax, Likelihood):
    """Bilby-compatible likelihood class using the jax backend.

    This class combines the jax-based Whittle likelihood computation in
    :class:`~pespace.inference.likelihood_jax.WhittleLikelihoodJax` with the
    bilby :class:`bilby.core.likelihood.Likelihood` interface, so that the
    external samplers supported in bilby can be used for parameter estimation
    with the jax backend.

    The parameters are passed to :meth:`log_likelihood` as a dictionary (the
    new bilby parameters-as-argument convention), which is converted to the
    namedtuple expected by
    :meth:`~pespace.inference.likelihood_jax.WhittleLikelihoodJax.log_likelihood`;
    keys that are not required by the waveform/response computation are ignored.

    Parameters
    ----------
    waveform : WaveformRippleBase
        Waveform interface instance for waveform generation. See
        :class:`~pespace.inference.likelihood_jax.WhittleLikelihoodJax`.
    detector : InterferometerAntenna or tuple of InterferometerAntenna
        Detector instance(s) configured with ``backend='jax'``.
    channels : tuple of str
        TDI channels used in the likelihood computation, a subset of
        ('A', 'E', 'T').
    parameter_conversion : callable, optional
        Optional callable that takes a dict of parameters and returns a dict.
        If given, it is applied to the ``parameters`` dict in
        :meth:`log_likelihood` before constructing the namedtuple expected by
        the jax backend.
    """

    def __init__(
        self,
        waveform: WaveformRippleBase,
        detector: InterferometerAntenna | tuple[InterferometerAntenna],
        channels: tuple[str],
        parameter_conversion: Callable[[dict], dict] | None = None,
    ):
        # The two parents have incompatible ``__init__`` signatures, so each
        # is initialized explicitly by class name (instead of cooperative
        # ``super()``).
        WhittleLikelihoodJax.__init__(self, waveform, detector, channels)
        Likelihood.__init__(self)
        self.parameter_conversion = parameter_conversion

        # jit-compile the jax-backend likelihood evaluation; the first call
        # pays the compilation cost and subsequent calls reuse the compiled
        # executable. Bound here (instead of decorating the method) so that
        # other sampler interfaces can still compose vmap/grad/jit freely.
        self._log_likelihood_jit = jax.jit(
            WhittleLikelihoodJax.log_likelihood.__get__(self)
        )

    def log_likelihood(self, parameters) -> float:
        """
        Calculate the log-likelihood value for the given parameters.

        Parameters
        ----------
        parameters : dict
            Dictionary of the parameter names and values. Only the keys
            required by the waveform and the detector response computation are
            used, other keys are ignored. If ``parameter_conversion`` is set,
            it is applied to ``parameters`` first.

        Returns
        -------
        float
            The log-likelihood value, summed over all detectors and channels.
            The evaluation of the underlying jax computation is jit-compiled
            in ``__init__``.
        """
        if self.parameter_conversion is not None:
            parameters = self.parameter_conversion(parameters)

        likelihood_params = self.likelihood_param_type(
            **{name: parameters[name] for name in self.likelihood_param_type._fields}
        )
        return self._log_likelihood_jit(likelihood_params)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..detector.antenna import InterferometerAntenna
    from ..waveform.interface_ripple import WaveformRippleBase
