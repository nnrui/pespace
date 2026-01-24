# !remember that if including the precession the ``detectors.LISALike.generate_singlelink_responses`` has to be modifed
# !if higher order included, ``generate_singlelink_responses`` has to be modified

# TODO waveform_arguments


import numpy as np
import taichi as ti
import taichi.math as tm
import bilby
from bilby.core.utils import logger

import lal
import lalsimulation as lalsim

from constants import *


default_waveform_arguments = {
    "neglect_waveform_errors": False,
}


def IMRPhenomD_h22_Amplitude_Phase_tf(
    frequencies,
    waveform,
    parameters,
    length,
    waveform_arguments={},
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
):
    # TODO insert tgr params in laldict
    # TODO add more keyword in waveform_arguments
    waveform_kwargs = default_waveform_arguments.update(waveform_arguments)

    waveform_dictionary = lal.CreateDict()

    try:
        new_parameters, _ = parameter_conversion(parameters)
        amp, phase, tf = lalsim.SimIMRPhenomDFrequencySequenceh22AmpPhasetf(
            frequencies,
            new_parameters["mass_1"],
            new_parameters["mass_2"],
            new_parameters["chi_1"],
            new_parameters["chi_2"],
            new_parameters["luminosity_distance"],
            new_parameters["coalescence_time"],
            new_parameters["coalescence_phase"],
            waveform_dictionary,
        )
    except Exception as e:
        if not waveform_kwargs["neglect_waveform_errors"]:
            raise e
        else:
            if e.args[0] == "Internal function call failed: Input domain error":
                logger.warning(
                    "Evaluating the waveform failed with error: {}\n".format(e)
                    + "The parameters were {}\n".format(new_parameters)
                    + "Likelihood will be set to -inf."
                )
                return FAILURE
            else:
                raise
    amp_data = amp.data
    phase_data = phase.data
    tf_data = tf.data

    # TODO why don't let the lalsim directly return h_cross and h_plus, (could be problematic when including higher modes and interplation)
    h22 = amp_data * np.exp(
        1j * phase_data
    )  # NOTE whether the returned phase should include the minus
    Y22 = lal.SpinWeightedSphericalHarmonic(
        new_parameters["inclination"], new_parameters["coalescence_phase"], -2, 2, 2
    )
    Y2m2star = np.conjugate(
        lal.SpinWeightedSphericalHarmonic(
            new_parameters["inclination"],
            new_parameters["coalescence_phase"],
            -2,
            2,
            -2,
        )
    )
    # NOTE remember that the waform with precession is currently not support
    # TODO double check the convention difference with lalsimulation
    hplus = 0.5 * (Y22 + Y2m2star) * h22
    hcross = 1j * 0.5 * (Y22 - Y2m2star) * h22
    waveform.from_numpy(
        {
            "hplus": hplus.view(dtype=np.float64).reshape(length, 2),
            "hcross": hcross.view(dtype=np.float64).reshape(length, 2),
            "tf": tf_data,
        }
    )

    return SUCCESS
