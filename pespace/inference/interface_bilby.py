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
    # TODO:
    # - add support for multiple, share or check the same frequency samples, different obs duration or cadance
    # - phase time distance marginalization

    def __init__(
        self,
        waveform: BaseWaveform | WaveformLALSimulationInterface,
        detector: InterferometerAntenna | tuple[InterferometerAntenna],
        channels: tuple[str],
    ):
        """
        create a Likelihood instance which can be used for sampling with bilby

        Parameters
        ==========
        wavefrom: object
            the instance where `waveform_container` is detectors
        detector: object
            see pespace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
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
        Calculates the real part of log-likelihood value

        Returns
        =======
        float: The real part of the log likelihood

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
