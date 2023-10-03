import h5py
import taichi as ti
import taichi.math as tm
import numpy as np

import lal
import lalsimulation as lalsim
from bilby.gw.conversion import component_masses_to_symmetric_mass_ratio

from .constants import *


PolarizationStruct = ti.types.struct(plus=tm.mat3, cross=tm.mat3)


def func():
    '''
    description
    
    Parameters
    ==========


    Returns:
    ========

    '''

    return None


@ti.func
def sinc(x: ti.f64) -> ti.f64:
    ret = 0.0
    if x == 0.0:
        ret = 1.0
    else:
        ret = tm.sin(x)/x
    return ret


@ti.func
def polarization_tensor_SSB(lam: ti.f64, beta: ti.f64, psi: ti.f64) -> PolarizationStruct:     # 
    '''
    return the polarization tensor in SSB, symbols follow the convention in LDC Manual: LISA-LCST-SGS-MAN-001

    Parameters
    ==========
    ecliptic_longitude: lambda, 
    ecliptic_latitude: beta, note that beta is (-pi/2, pi/2)
    polarizatione: psi, 
    mode: one of 'plus', 'cross', 'x', 'y', 'breathing', 'longitudinal'

    Returns:
    ========
    array: 3*3 array
    '''
    # todo the constant should compute only once to reduce compution burden.
    p = tm.vec3([tm.sin(lam) * tm.cos(psi) - tm.sin(beta) * tm.cos(lam) * tm.sin(psi), 
                 -(tm.sin(beta) * tm.sin(lam) * tm.sin(psi)) - tm.cos(lam) * tm.cos(psi), 
                 tm.cos(beta) * tm.sin(psi)])
    q = tm.vec3([-tm.cos(lam) * tm.cos(psi) * tm.sin(beta) - tm.sin(lam) * tm.sin(psi), 
                 -tm.cos(psi) * tm.sin(beta) * tm.sin(lam) + tm.cos(lam) * tm.sin(psi),
                 tm.cos(beta) * tm.cos(psi)])
    
    
    return PolarizationStruct(plus = (p.outer_product(p) - q.outer_product(q)), 
                              cross= (p.outer_product(q) + q.outer_product(p)))

@ti.func
def GW_propagation_unit_vector_k(lam: ti.f64, beta: ti.f64) -> tm.vec3:                    # note that beta is (-pi/2, pi/2)
    return tm.vec3([-tm.cos(beta)*tm.cos(lam), -tm.cos(beta)*tm.sin(lam), -tm.sin(beta)])



# def cutoff_frequency_PhenomD(mass_1, mass_2):
#     '''
#     return the high frequency cutoff in Hz, using Mf=0.2 copied form LALSimIMRPhenomD.h, 
#     which could be used in determining the sampling frequency in TD or the frequency bound in FD for SMBH.
    
#     Parameters
#     ==========
#     mass_1: mass of heavier object in Msun
#     mass_2: mass of lighter object in Msun

#     Returns:
#     ========
#     f_cut: in Hz
#     '''
#     total_mass = mass_1 + mass_2
#     M_sec = total_mass * MTSUN_SI
#     f_cut = Mf_CUT_PhenomD/M_sec
#     return f_cut


# def start_frequency():
#     '''
#     description
    
#     Parameters
#     ==========


#     Returns:
#     ========

#     '''
#     return


# def time_in_band_leading_order(mass_1, mass_2, start_frequency, safety_factor=1.1):
#     '''
#     TODO consider the noise not only the start_frequency
#     time to merger from the minimum_frequency
#     note that the minimum_frequency maybe higher than the low frequency cutoff of the detector
#     the returned time is a rough approximation with the lead oder
    
#     Parameters
#     ==========
#     mass_1: mass of heavier object in Msun
#     mass_2: mass of lighter object in Msun
#     start_frequency: in Hz
#     safety_factor: multiplicitive safety factor

#     Returns:
#     ========
#     time_length: in second
#     '''
#     total_mass = mass_1 + mass_2
#     M_sec = total_mass * MTSUN_SI
#     Mf_start = M_sec * start_frequency
#     eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
#     # dimensionless unit
#     time_to_merger = 5/256 / eta * (PI*Mf_start)**(-8/3)
#     # convert to unit of second
#     time_to_merger *= M_sec
#     time_length = time_to_merger*safety_factor
#     return time_length


# def estimate_imr_duration(mass_1, mass_2, chi_1, chi_2, start_frequency, safety_factor=1.1):
#     '''
#     deprecate, do not use this func, have unknown error, return an negtive value for SMBH.
#     '''
#     time_length = lalsim.SimIMRPhenomDChirpTime(mass_1*MSUN_SI, mass_2*MSUN_SI, chi_1, chi_2, start_frequency)
#     time_length *= safety_factor
#     return time_length


# def post_merger_time_SMBH():
#     '''
#     description
    
#     Parameters
#     ==========


#     Returns:
#     ========

#     ''' 
#     return


def noise_weighted_inner_product(aa, bb, psd_array, delta_freq):
    '''
    compute the noise weighted inner product between two arrays on the uniform frequency grid, <aa|bb>

    Parameters
    ==========
    aa: array
        first array to compute inner product
    bb: array
        second array to compute inner product
    psd_array: array
        psd of the noise which is the array have the same shape of aa and bb
    delta_freq: float
        the spacing of two adjacent frequency points

    Returns
    =======
    float
    '''
    integrand = aa * np.conj(bb) / psd_array
    return (4 * delta_freq * np.sum(integrand)).real


def recursively_save_dict_contents_to_group(h5file, path, dic):
    '''
    Recursively save a dictionary to a HDF5 group
    copied from bilby.core.utils.io.recursively_save_dict_contents_to_group

    Parameters
    ==========
    h5file: h5py.File
        Open HDF5 file
    path: str
        Path inside the HDF5 file
    dic: dict
        The dictionary containing the data
    '''
    for key, value in dic.items():
        if isinstance(value, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", value)
        elif isinstance(value, list):
            if len(value) == 0:
                h5file[path + key] = h5py.Empty('f')
            else:
                for idx, item in enumerate(value):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/' + f'item_{idx}' + '/', item)
        elif isinstance(value, np.ndarray):
            h5file[path + key] = value
        elif value is None:
            h5file[path + key] = h5py.Empty('f')
        else:
            raise ValueError(f'Cannot save {key}: {type(value)} type')


def recursively_load_dict_contents_from_group(h5file, path):
    '''
    Recursively load a HDF5 file into a dictionary
    copied from bilby.core.utils.io.recursively_load_dict_contents_from_group

    Parameters
    ==========
    h5file: h5py.File
        Open h5py file object
    path: str
        Path within the HDF5 file

    Returns
    =======
    output: dict
        The contents of the HDF5 file unpacked into the dictionary.
    '''
    output = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            output[key] = item[()]
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict_contents_from_group(h5file, path + key + "/")
    return output