import copy
import taichi as ti
import taichi.math as tm

import numpy as np
from bilby.core.likelihood import Likelihood
from tiwave.waveforms import BaseWaveform

from ..constants import *
from ..detector.antenna import SpaceborneInterferometer


@ti.kernel
def _compute_frequency_domain_likelihood(
    channels: ti.template(),
    observed: ti.template(),
    response: ti.template(),
    psd: ti.template(),
    df: float,
) -> float:
    log_l = 0.0
    for i in observed:
        inner_product = 0.0
        for chan in ti.static(channels):
            # AoS is used for StructField, placing the loop for channels inside.
            inner_product += (observed[i][chan] - response[i][chan]).norm_sqr() / psd[
                i
            ][chan]
        ti.atomic_add(log_l, inner_product)
    log_l *= -2 * df

    return log_l


class FrequencyDomainLikelihood(Likelihood):
    # TODO:
    # - add support for multiple, share or check the same frequency samples, different obs duration or cadance
    # - phase time distance marginalization

    def __init__(
        self,
        waveform: BaseWaveform,
        detector: SpaceborneInterferometer | tuple[SpaceborneInterferometer],
    ):
        """
        create a FrequencyDomainLikelihood instance

        Parameters
        ==========
        wavefrom: object
            the instance where `waveform_container` is detectors
        detector: object
            see pespace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
        """
        super(FrequencyDomainLikelihood, self).__init__(parameters=dict())

        self.waveform = waveform

        if not isinstance(detector, tuple):
            detector = (detector,)
        for det in detector:
            channels = set(det.TDI_data.data_info.channels)
            if not (channels == {"A", "E"} or channels == {"A", "E", "T"}):
                raise ValueError(
                    "The likelihood compution expect the TDI channels of {'A','E','T'} or "
                    + f"{{'A','E'}}, while the channels of the detector {det.name} "
                    + f"in the input is {channels}."
                )
        self.detector = detector

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
            det.update_frequency_domain_response(
                self.waveform.waveform_container,
                self.parameters["ecliptic_longitude"],
                self.parameters["ecliptic_latitude"],
                self.parameters["polarization"],
            )

            logl += _compute_frequency_domain_likelihood(
                det.TDI_data.data_info.channels,
                det.TDI_data.frequency_domain_TDI_data,
                det.response_container,
                det.TDI_data.frequency_domain_noise_power_density,
                det.TDI_data.data_info.delta_frequency,
            )

        return logl


class FisherMatrixLikelihood(Likelihood):
    def __init__(self, fiducial_parameters, fixed_parameters, parameters=None):
        super().__init__(parameters)

    def Fisher_matrix(
        self,
    ):
        pass

    def covariance_matrix(
        self,
    ):
        pass

    def generate_posterior_samples(
        self,
    ):
        pass


class WaveletDomainLikelihood(Likelihood):
    pass
