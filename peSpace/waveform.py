# !remember that if including the precession the ``detectors.LISALike.generate_singlelink_responses`` has to be modifed

import bilby
from bilby.core.utils import logger

import lal
import lalsimulation as lalsim

def IMRPhenomD_h22_Amplitude_Phase_tf(frequency_array, parameters, neglect_waveform_errors=False, parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters, waveform_kwargs=None):
    # TODO insert tgr params in laldict
    waveform_dictionary = lal.CreateDict()

    try:
        new_parameters, _ = parameter_conversion(parameters)
        amp, phase, tf = lalsim.SimIMRPhenomDFrequencySequenceh22AmpPhasetf(frequency_array, new_parameters['mass_1'], new_parameters['mass_2'], new_parameters['chi_1'], new_parameters['chi_2'], new_parameters['luminosity_distance'], new_parameters['coalescence_time'], new_parameters['coalescence_phase'], waveform_dictionary)
    except Exception as e:
        if not neglect_waveform_errors:
            raise e
        else:
            if e.args[0] == 'Internal function call failed: Input domain error':
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(new_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    return {'amplitude': amp.data, 
            'phase': phase.data, 
            'tf': tf.data, 
            'frequencies': frequency_array}