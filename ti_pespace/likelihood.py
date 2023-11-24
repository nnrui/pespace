import copy
import taichi as ti
import taichi.math as tm

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import component_masses_to_chirp_mass

from .utilities import inner_product
from .constants import *


@ti.kernel
def _stationary_gaussian_full_likelihood(channels: ti.template(),
                                         channels_data: ti.template(),
                                         strains_FD: ti.template(),
                                         psd: ti.template(),
                                         df: ti.f64) -> ti.f64:
    log_l = 0.0
    for chan in ti.static(channels):
        
        integral = 0.0
        for i in strains_FD[chan]:
            inner_product = (strains_FD[chan][i] - channels_data[chan][i]).norm_sqr() / psd[chan][i]
            ti.atomic_add(integral, inner_product)
        
        log_l += -2 * df * integral

    return log_l

class FullLikelihood(Likelihood):


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
        super(FullLikelihood, self).__init__(parameters=dict())
        self.waveform = waveform
        if sorted(detector.TDI_channels) != ['A','E'] and sorted(detector.TDI_channels)!= ['A','E','T']:
            raise Exception(f'Your set detector channels of {sorted(detector.TDI_channels)}, '
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
        self.detector.updata_TDI_responses(self.parameters)
        return _stationary_gaussian_full_likelihood(self.detector.TDI_channels, self.detector.TDI_data.channels_data, self.detector.strains_FD, self.detector.delta_f)

