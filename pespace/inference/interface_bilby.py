"""
Interface for using bilby to perform parameter estimation.

This module provides a bilby-compatible likelihood class enabling the use of various
external samplers supported in bilby.
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

from .likelihood_ti import compute_whittle_likelihood_ti
from .likelihood_jax import WhittleLikelihoodJax


class LikelihoodBilbyInterfaceTaichi(Likelihood):
    """Bilby-compatible likelihood class for sampling.

    TODO:

    - Add support for multiple detectors with shared or different frequency samples,
      observation durations, or cadences
    - Implement phase, time, and distance marginalization
    """

    def __init__(
        self,
        waveform: BaseWaveform | WaveformLALSimulationInterface,
        detector: InterferometerAntenna | tuple[InterferometerAntenna],
        channels: tuple[str],
    ):
        """
        Initialize the Bilby likelihood interface.

        Parameters
        ----------
        waveform : BaseWaveform or WaveformLALSimulationInterface
            Waveform model instance that provides the ``waveform_container`` attribute
            and ``update_waveform`` method for generating gravitational wave signals.
        detector : InterferometerAntenna or tuple of InterferometerAntenna
            Detector instance(s) for computing detector responses. Each detector must
            have ``tdi_data`` and ``update_detector_response`` attributes/methods.
        channels : tuple of str
            TDI channels used in the likelihood computation. Must be a subset of
            ('A', 'E', 'T'). Common choices are ('A', 'E', 'T') or ('A', 'E').

        Raises
        ------
        ValueError
            If any channel in ``channels`` is not one of 'A', 'E', or 'T'.

        Notes
        -----
        The likelihood computation uses the Whittle approximation, which assumes
        Gaussian and stationary noise.
        """
        super().__init__()

        self.waveform = waveform

        if not isinstance(detector, tuple):
            detector = (detector,)
        self.detector = detector

        unsupported_chans = [chan for chan in channels if chan not in ("A", "E", "T")]
        if any(unsupported_chans):
            raise ValueError(
                f"The likelihood compution only support TDI channels of ('A','E','T'). "
                f"{unsupported_chans} are not supported currently."
            )
        self.channels = channels

    def log_likelihood(self, parameters):
        """
        Calculate the log-likelihood value for the given parameters.

        This method updates the waveform model with the parameter values,
        computes the detector response for each detector, and evaluates the
        Whittle likelihood across all specified TDI channels.

        Parameters
        ----------
        parameters : dict
            Dictionary of the parameter names and values used to update the
            waveform model and the detector responses.

        Returns
        -------
        float
            The log-likelihood value summed over all detectors and channels.
        """
        self.waveform.update_waveform(parameters)
        logl = 0.0
        for det in self.detector:
            det.update_detector_response(
                self.waveform.waveform_container,
                parameters["ecliptic_longitude"],
                parameters["ecliptic_latitude"],
                parameters["polarization"],
                parameters["coalescence_time"],
            )

            logl += compute_whittle_likelihood_ti(
                self.channels,
                det.tdi_data.fd_data,
                det.tdi_response,
                det.tdi_data.fd_noise_power_density,
                det.tdi_data.data_info.delta_frequency,
            )

        return logl


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

    from tiwave.waveform.base_waveform import BaseWaveform

    from ..detector.antenna import InterferometerAntenna
    from ..waveform.interface_lalsim import WaveformLALSimulationInterface
    from ..waveform.interface_ripple import WaveformRippleBase
