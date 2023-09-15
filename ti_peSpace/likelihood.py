import copy
import taichi as ti

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import component_masses_to_chirp_mass

from .utilities import inner_product
from .constants import *


@ti.data_oriented
class BaseLikelihood(Likelihood):

    def __init__(self, waveform_func, detector, neglect_waveform_errors=False):
        '''
        create a BaseLikelihood instance

        Parameters
        ==========
        wavefrom_func: function
            function to return waveform from parameters, see wavefrom.__dir__() for all support funcs
        detector: object
            see peSpace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
        neglect_waveform_errors: bool
            whether raise when failed to call wavefrom_func, raise if False
        '''
        super(BaseLikelihood, self).__init__(dict())
        self.waveform_func = waveform_func
        if detector.TDI_channels != ('A','E') and detector.TDI_channels != ('A','E','T'):
            raise Exception('Your set detector channels of {}, while the likelihood compution expect '
                            'the channels of ("A","E","T") or ("A","E")'.format(detector.TDI_channels))
        self.detector = detector
        self.likelihood_frequency_array = self.create_likelihood_frequency_array()
        self.likelihood_psd_array = self.create_likelihood_psd_array()
        self.likehihodd_strains_FD = self.create_likelihood_strains_FD()
        # TODO: it maybe better to move the neglect_waveform_errors into a dict
        # TODO logically neglect_waveform_errors sould be attribure in wavefrom geration func
        self.neglect_waveform_errors = neglect_waveform_errors

    def create_likelihood_frequency_array(self):
        '''
        the likelihood will be compute on the orginal frequency grid in BaseLikelihood
        return the same frequency_array of the detector

        Returns
        =======
        array: frequency_arry on which the likelihood will be computed
        '''
        return copy.deepcopy(self.detector.frequency_array)
    
    def create_likelihood_psd_array(self):
        '''
        create psd array for likelihood computation, return the same psd_array of the detector

        Returns
        =======
        array: same psd_array of the detector
        '''
        return copy.deepcopy(self.detector.psd_array)

    def create_likelihood_strains_FD(self):
        '''
        return the strains_FD for likelihood computation, has the same shape with the likelihood_frequency_array

        Returns
        =======
        array: has the same shape with the likelihood_frequency_array
        '''
        return copy.deepcopy(self.detector.strains_FD)

    def log_likelihood(self):
        '''
        Calculates the real part of log-likelihood value

        Returns
        =======
        float: The real part of the log likelihood

        '''
        waveform = self.waveform_func(self.likelihood_frequency_array, self.parameters.copy(), self.neglect_waveform_errors)
        if waveform is None:
            return np.nan_to_num(-np.inf)
        GW_signals = self.detector.TDI_responses(waveform, self.parameters)

        log_l = 0
        delta_freq = self.likelihood_frequency_array[1] - self.likelihood_frequency_array[0]
        for chan in self.detector.TDI_channels:
            residual = self.likehihodd_strains_FD[chan] - GW_signals[chan]
            log_l += - 2. * delta_freq * np.vdot(residual, residual/self.likelihood_psd_array[chan]).real

        return log_l

