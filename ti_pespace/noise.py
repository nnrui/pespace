import numpy as np
from numpy import sin, cos

from .constants import *


def psd_LISA_SciRDv1(frequencies, TDI_channel='X', TDI_generation='1.5'):
    '''
    LISA Sensitivity and SNR Calculations, https://arxiv.org/abs/2108.01167
    '''
    S_oms = (15.e-12)**2 * (1. + (2.e-3/frequencies)**4) * (2.0*PI*frequencies/C_SI)**2
    S_acc = (3.e-15)**2 * (1.0 + (0.4e-3/frequencies)**2) * (1.0 + (frequencies/8e-3)**4) / (2*PI*frequencies*C_SI)**2

    if TDI_generation == '1.5':
        prefactor = 1.0
    elif TDI_generation == '2.0':
        prefactor = 4.0 * sin(4*PI*frequencies*ARM_LENGTH_LISA_SEC)**2
    else:
        raise Exception('The TDI generation {} is unknown'.format(TDI_generation))

    if TDI_channel in ['X', 'Y', 'Z']:
        psd_array = 16 * sin(2*PI*frequencies*ARM_LENGTH_LISA_SEC)**2 * (S_oms + (3 + cos(4*PI*frequencies*ARM_LENGTH_LISA_SEC))*S_acc)
    elif TDI_channel in ['A', 'E']:
        psd_array = 8 * sin(2*PI*frequencies*ARM_LENGTH_LISA_SEC)**2 * ((2 + cos(2*PI*frequencies*ARM_LENGTH_LISA_SEC))*S_oms 
                                                                        + (6 + 4*cos(2*PI*frequencies*ARM_LENGTH_LISA_SEC) + 2*cos(4*PI*frequencies*ARM_LENGTH_LISA_SEC))*S_acc)
    elif TDI_channel == 'T':
        psd_array = 32 * sin(2*PI*frequencies*ARM_LENGTH_LISA_SEC)**2 * sin(PI*frequencies*ARM_LENGTH_LISA_SEC)**2 * (S_oms + 4*sin(PI*frequencies*ARM_LENGTH_LISA_SEC)**2*S_acc)

    psd_array *= prefactor

    return psd_array


noise_models = {'LISA_SciRDv1': psd_LISA_SciRDv1,
                'Taiji': None,
                'Tianqin': None, 
                }