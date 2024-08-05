from dataclasses import dataclass

import numpy as np
from numpy import sin, cos
from numpy.typing import NDArray

from .constants import *


@dataclass
class AnalysticNoisePSDModel(object):
    """
    Analystic model for noise power spectrum density, the fomulae come from
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

    def __call__(
        self,
        frequencies: NDArray[np.float64],
        TDI_channel: tuple[str, ...],
        TDI_generation: str,
    ):
        """Generate psd array for given frequency array"""

        # Convert displace noise and acceleration noise to the same dimension of relative frequency
        S_oms = (
            self.OMS_noise_level
            * (1.0 + (2.0e-3 / frequencies) ** 4)
            * (2.0 * PI * frequencies / C_SI) ** 2
        )
        S_acc = (
            self.acc_noise_level
            * (1.0 + (0.4e-3 / frequencies) ** 2)
            * (1.0 + (frequencies / 8e-3) ** 4)
            / (2 * PI * frequencies * C_SI) ** 2
        )

        if TDI_generation == "1.5":
            prefactor = 1.0
        elif TDI_generation == "2.0":
            prefactor = 4.0 * sin(4 * PI * frequencies * self.arm_length_sec) ** 2
        else:
            raise Exception("The TDI generation {} is unknown".format(TDI_generation))

        psd_dict = {}
        for chan in TDI_channel:
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
            psd *= prefactor
            psd_dict[chan] = psd

        return psd_dict


noise_models = {
    "LISA_SciRDv1": AnalysticNoisePSDModel(
        OMS_noise_level=(15.0e-12) ** 2,
        acc_noise_level=(3.0e-15) ** 2,
        arm_length_sec=2.5e9 / C_SI,
    ),  # https://arxiv.org/abs/2108.01167
    "Taiji_TDC": AnalysticNoisePSDModel(
        OMS_noise_level=(8.0e-12) ** 2,
        acc_noise_level=(3.0e-15) ** 2,
        arm_length_sec=3.0e9 / C_SI,
    ),  # https://doi.org/10.1038/s41550-019-1008-4
    "Tianqin_Luo2016": AnalysticNoisePSDModel(
        OMS_noise_level=(1.0e-12) ** 2,
        acc_noise_level=(1.0e-15) ** 2,
        arm_length_sec=1.0e8 / C_SI,
    ),  # https://iopscience.iop.org/article/10.1088/0264-9381/33/3/035010
}
