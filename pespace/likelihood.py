import copy
import taichi as ti
import taichi.math as tm

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import component_masses_to_chirp_mass

from .constants import *


@ti.kernel
def _compute_frequency_domain_likelihood(channels: ti.template(),
                                         observed: ti.template(),
                                         response: ti.template(),
                                         psd: ti.template(),
                                         df: ti.f64) -> ti.f64:
    log_l = 0.0
    for i in observed:
        inner_product = 0.0
        for chan in ti.static(channels):
            # AoS is used for StructField, placing the loop for channels inside.
            inner_product += (observed[i][chan] - response[i][chan]).norm_sqr() / psd[i][chan]
        ti.atomic_add(log_l, inner_product)
    log_l *= -2 * df
    
    return log_l


class FrequencyDomainLikelihood(Likelihood):
    # TODO:
    # - add support for multiple

    def __init__(self, waveform, detector):
        '''
        create a FullLikelihood instance

        Parameters
        ==========
        wavefrom: object
            the instance where `waveform_container` is detectors
        detector: object
            see peSpace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
        '''
        super(FrequencyDomainLikelihood, self).__init__(parameters=dict())
        self.waveform = waveform
        if not (set(detector.TDI_data.data_info.channels) == {'A','E'} or 
                set(detector.TDI_data.data_info.channels)== {'A','E','T'}):
            raise Exception(f'Your set detector channels of {detector.TDI_data.channels}, '
                             'while the likelihood compution expect the channels of ("A","E","T") or ("A","E")')
        self.detector = detector

    def log_likelihood(self):
        '''
        Calculates the real part of log-likelihood value

        Returns
        =======
        float: The real part of the log likelihood

        '''
        self.waveform.update_waveform(self.parameters)
        self.detector.update_frequency_domain_response(self.waveform.waveform_container, self.parameters['ecliptic_longitude'], self.parameters['ecliptic_latitude'], self.parameters['polarization'])
        return _compute_frequency_domain_likelihood(self.detector.TDI_data.data_info.channels, 
                                                                 self.detector.TDI_data.frequency_domain_TDI_data, 
                                                                 self.detector.response_container, 
                                                                 self.detector.TDI_data.frequency_domain_noise_power_density, 
                                                                 self.detector.TDI_data.data_info.delta_frequency)



class WaveletDomainLikelihood(Likelihood):
    pass
