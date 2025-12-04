from dataclasses import dataclass
from abc import ABC, abstractmethod

import taichi as ti
import numpy as np
from numpy import sin, cos
from numpy.typing import NDArray

from ..utils.constants import *


class FrequencyDomainNoiseModel(ABC):
    @abstractmethod
    def power_spectral_density_array(
        self,
        frequencies: NDArray,
        tdi_channels: tuple[str, ...],
        tdi_generation: str,
    ) -> dict[str, NDArray]:
        pass

    def __init_subclass__(cls) -> None:
        cls.__call__ = cls.power_spectral_density_array


@dataclass
class AnalyticPowerSpectralDensity(FrequencyDomainNoiseModel):
    """
    Analytic model for noise power spectral density, the formulae come from
    https://arxiv.org/abs/2108.01167

    OMS_noise_level:
        The noise level for Optical Metrology System, m^2 Hz^-1
    acc_noise_level:
        The noise level for acceleration, m^2 s^-4 Hz^-1
    arm_length_sec:
        The arm length of detector, s

    """

    OMS_noise_level: float
    acc_noise_level: float
    arm_length_sec: float

    def _S_oms(self, frequencies: NDArray) -> NDArray:
        return (
            self.OMS_noise_level
            * (1.0 + (2.0e-3 / frequencies) ** 4)
            * (2.0 * PI * frequencies / C_SI) ** 2
        )

    def _S_acc(self, frequencies: NDArray) -> NDArray:
        return (
            self.acc_noise_level
            * (1.0 + (0.4e-3 / frequencies) ** 2)
            * (1.0 + (frequencies / 8e-3) ** 4)
            / (2 * PI * frequencies * C_SI) ** 2
        )

    def power_spectral_density_array(
        self,
        frequencies: NDArray,
        tdi_channels: tuple[str, ...],
        scaling: bool,
        tdi_generation: str,
    ) -> dict[str, NDArray]:
        """Generate psd array for given frequency array"""

        # Convert displacement noise and acceleration noise to the same dimension of relative frequency
        S_oms = self._S_oms(frequencies)
        S_acc = self._S_acc(frequencies)

        if scaling:
            scaling_factor = (PC_SI / (MRSUN_SI * MTSUN_SI)) ** 2
            S_oms *= scaling_factor
            S_acc *= scaling_factor

        if tdi_generation == "1.5":
            prefactor = 1.0
        elif tdi_generation == "2.0":
            prefactor = 4.0 * sin(4 * PI * frequencies * self.arm_length_sec) ** 2
        else:
            raise Exception("The TDI generation {} is unknown".format(tdi_generation))

        psd_dict = {}
        for chan in tdi_channels:
            if chan in ["X", "Y", "Z"]:
                psd = (
                    16
                    * sin(2 * PI * frequencies * self.arm_length_sec) ** 2
                    * (
                        S_oms
                        + (3 + cos(4 * PI * frequencies * self.arm_length_sec)) * S_acc
                    )
                )
            elif chan in ["A", "E"]:
                psd = (
                    8
                    * sin(2 * PI * frequencies * self.arm_length_sec) ** 2
                    * (
                        (2 + cos(2 * PI * frequencies * self.arm_length_sec)) * S_oms
                        + (
                            6
                            + 4 * cos(2 * PI * frequencies * self.arm_length_sec)
                            + 2 * cos(4 * PI * frequencies * self.arm_length_sec)
                        )
                        * S_acc
                    )
                )
            elif chan == "T":
                psd = (
                    32
                    * sin(2 * PI * frequencies * self.arm_length_sec) ** 2
                    * sin(PI * frequencies * self.arm_length_sec) ** 2
                    * (
                        S_oms
                        + 4 * sin(PI * frequencies * self.arm_length_sec) ** 2 * S_acc
                    )
                )
            else:
                raise ValueError(f"Unknown TDI channel {chan}.")
            psd *= prefactor

            if ti.lang.impl.current_cfg().default_fp.to_string() == "f32":
                data_type = np.float32
            else:
                data_type = np.float64
            psd_dict[chan] = np.astype(psd, data_type)

        return psd_dict


class TianQinAnalyticPowerSpectralDensity(AnalyticPowerSpectralDensity):
    """
    S_oms and S_acc for TianQin detector have the difference of frequency dependency.
    https://arxiv.org/abs/2309.15020
    """

    def _S_oms(self, frequencies: NDArray) -> NDArray:
        return self.OMS_noise_level * (2.0 * PI * frequencies / C_SI) ** 2

    def _S_acc(self, frequencies: NDArray) -> NDArray:
        return (
            self.acc_noise_level
            * (1.0 + 0.1e-3 / frequencies)
            / (2 * PI * frequencies * C_SI) ** 2
        )


available_noise_models = {
    "LISA_SciRDv1": AnalyticPowerSpectralDensity(
        OMS_noise_level=(15.0e-12) ** 2,
        acc_noise_level=(3.0e-15) ** 2,
        arm_length_sec=2.5e9 / C_SI,
    ),  # https://arxiv.org/abs/2108.01167
    "Taiji_TDC": AnalyticPowerSpectralDensity(
        OMS_noise_level=(8.0e-12) ** 2,
        acc_noise_level=(3.0e-15) ** 2,
        arm_length_sec=3.0e9 / C_SI,
    ),  # https://doi.org/10.1038/s41550-019-1008-4
    "Tianqin_GWSpace": TianQinAnalyticPowerSpectralDensity(
        OMS_noise_level=(1.0e-12) ** 2,
        acc_noise_level=(1.0e-15) ** 2,
        arm_length_sec=1.7e8 / C_SI,
    ),  # https://arxiv.org/abs/2309.15020
}
