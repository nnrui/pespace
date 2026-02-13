"""
Interface for using bilby to perform parameter estimation.

This module provides a bilby-compatible likelihood class enabling the use of various 
external samplers supported in bilby.
"""
from __future__ import annotations
import logging

from .common import _compute_whittle_likelihood

try:
    from bilby.core.likelihood import Likelihood
except ImportError:
    logging.error(
        "bilby is not installed by default, if sampler interface in bilby is needed, "
        "please install bilby manually."
    )
    raise


class LikelihoodBilbyInterface(Likelihood):
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
            TDI channels to use for likelihood computation. Must be a subset of
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
        super(LikelihoodBilbyInterface, self).__init__(parameters=dict())

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

    def log_likelihood(self):
        """
        Calculate the log-likelihood value for the current parameters.

        This method updates the waveform model with the current parameter values,
        computes the detector response for each detector, and evaluates the
        Whittle likelihood across all specified TDI channels.

        Returns
        -------
        float
            The log-likelihood value summed over all detectors and channels.
        """
        self.waveform.update_waveform(self.parameters)
        logl = 0.0
        for det in self.detector:
            det.update_detector_response(
                self.waveform.waveform_container,
                self.parameters["ecliptic_longitude"],
                self.parameters["ecliptic_latitude"],
                self.parameters["polarization"],
                self.parameters["coalescence_time"],
            )

            logl += _compute_whittle_likelihood(
                self.channels,
                det.tdi_data.fd_data,
                det.tdi_response,
                det.tdi_data.fd_noise_power_density,
                det.tdi_data.data_info.delta_frequency,
            )

        return logl


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiwave.waveform.base_waveform import BaseWaveform

    from ..detector.antenna import InterferometerAntenna
    from ..waveform.interface_lalsim import WaveformLALSimulationInterface
