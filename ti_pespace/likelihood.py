import copy
import taichi as ti

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import component_masses_to_chirp_mass

from .utilities import inner_product
from .constants import *


class FullLikelihood(Likelihood):


    def __init__(self, waveform_func, detector, waveform_arguments=dict()):
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
        super(FullLikelihood, self).__init__(parameters=dict())
        self.waveform_func = waveform_func
        if sorted(detector.TDI_channels) != ['A','E'] and sorted(detector.TDI_channels)!= ['A','E','T']:
            raise Exception(f'Your set detector channels of {sorted(detector.TDI_channels)}, '
                             'while the likelihood compution expect the channels of ("A","E","T") or ("A","E")')
        self.detector = detector
        # TODO: it maybe better to move the neglect_waveform_errors into a dict
        # TODO logically neglect_waveform_errors sould be attribure in wavefrom geration func
        self.wavefrom_arguments = waveform_arguments


    def log_likelihood(self):
        '''
        Calculates the real part of log-likelihood value

        Returns
        =======
        float: The real part of the log likelihood

        '''
        ret = self.waveform_func(self.detector.frequencies, self.detector.waveform_container, self.parameters.copy(), self.detector.data_length, self.wavefrom_arguments)
        if ret == FAILURE:
            return np.nan_to_num(-np.inf)
        self.detector.updata_TDI_responses(self.parameters)
        signal_from_ti = self.TDI_data.TDI_chan_data.to_numpy()

        log_l = 0.0
        for chan in self.detector.TDI_channels:
            residual = self.detector.strains_FD[chan] - signal_from_ti[chan].view(dtype=np.complex128)    # NOTE!!! must use ti.f64 in vec2
            log_l += - 2. * self.detector.delta_f * np.vdot(residual, residual / self.detector.psd_array[chan]).real

        return log_l

