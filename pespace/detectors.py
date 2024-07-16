"""This is the detectors module, mainly for storing strain data and computing responses."""

# TODO:
# improve data latout of TDI_data, waveform_container
# protect SpaceborneInterferometer.TDI_data unchange
import warnings
from typing import Callable, Optional
from dataclasses import dataclass, field

from scipy import signal
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import taichi as ti
import taichi.math as tm

from .utilities import polarization_tensor_SSB, GW_propagation_unit_vector, sinc, \
                       noise_weighted_inner_product, \
                       recursively_save_dict_contents_to_group, recursively_load_dict_contents_from_group, \
                       vec2_complex
from .orbits import available_orbit_models, OrbitVectorStruct
from .constants import *
from .noise import noise_models


SingleLinksStruct = ti.types.struct(link12=vec2_complex, link21=vec2_complex, 
                                    link23=vec2_complex, link32=vec2_complex, 
                                    link31=vec2_complex, link13=vec2_complex)


@ti.kernel
def _add_1d_field_into_TDI_data(TDI_data:ti.template(), input:ti.template()):
    """Add a 1d field into TDI_data. The input field must has same shape and same channels with TDI_data.
    StructField has the default layout of AoS, unrolling the channels in the inner loop for memory accessing efficiency.
    """
    for i in TDI_data:
        for chan in ti.static(TDI_data.keys):
            TDI_data[i][chan] += input[i][chan]


@ti.kernel
def _inject_into_strains_FD(strains_FD: ti.template(), injected: ti.template()):
    for i in strains_FD:
        for chan in ti.static(strains_FD.keys):
            strains_FD[i][chan] += injected[i][chan]


@ti.kernel
def _generate_TDI_responses(TDI_data: ti.template(),   # ti.field  # ti.Struct.field, keep ti.template() point to the same memory address to avoid kernal repeated instantiation
                            waveform: ti.template(),    # ti.Struct.field
                                                        # the computaion is evaluated in the order of frequency point
                                                        # using AoS structure to store data for efficiency
                            orbit_model: ti.template(),                           
                            armL_sec: ti.f64,
                            lam: ti.f64,
                            beta: ti.f64,
                            psi: ti.f64,
                            ):

    pol_tensor = polarization_tensor_SSB(lam, beta, psi)    # tm.mat3
    k = GW_propagation_unit_vector(lam, beta)             # tm.vec3

    for i in TDI_data:

        constellation_vectors = orbit_model(waveform[i].tf)

        n1Hn1 = (constellation_vectors.n1 @ (pol_tensor.plus) @ constellation_vectors.n1) * waveform[i].hplus + (constellation_vectors.n1 @ (pol_tensor.cross) @ constellation_vectors.n1) * waveform[i].hcross    # complex number, vec2_complex   
        n2Hn2 = (constellation_vectors.n2 @ (pol_tensor.plus) @ constellation_vectors.n2) * waveform[i].hplus + (constellation_vectors.n2 @ (pol_tensor.cross) @ constellation_vectors.n2) * waveform[i].hcross    # complex number, vec2_complex   
        n3Hn3 = (constellation_vectors.n3 @ (pol_tensor.plus) @ constellation_vectors.n3) * waveform[i].hplus + (constellation_vectors.n3 @ (pol_tensor.cross) @ constellation_vectors.n3) * waveform[i].hcross    # complex number, vec2_complex   

        kn1 = k@constellation_vectors.n1    # scalar
        kn2 = k@constellation_vectors.n2    # scalar
        kn3 = k@constellation_vectors.n3    # scalar

        kp1Lp2L = k@(constellation_vectors.p1D + constellation_vectors.p2D)    # scalar
        kp2Lp3L = k@(constellation_vectors.p2D + constellation_vectors.p3D)    # scalar
        kp3Lp1L = k@(constellation_vectors.p3D + constellation_vectors.p1D)    # scalar

        kp0 = k@constellation_vectors.p0    # scalar

        common_sinc = PI * TDI_data[i].frequencies * armL_sec    # scalar
        sinc12 = sinc(common_sinc * (1.-kn3))    # scalar
        sinc21 = sinc(common_sinc * (1.+kn3))    # scalar
        sinc23 = sinc(common_sinc * (1.-kn1))    # scalar
        sinc32 = sinc(common_sinc * (1.+kn1))    # scalar
        sinc31 = sinc(common_sinc * (1.-kn2))    # scalar
        sinc13 = sinc(common_sinc * (1.+kn2))    # scalar

        common_exp = -PI * TDI_data[i].frequencies * vec2_complex([0.0, 1.0])    # complex number, vec2_complex
        exp12 = tm.cexp(common_exp*(armL_sec+kp1Lp2L))    # complex number, vec2_complex
        exp23 = tm.cexp(common_exp*(armL_sec+kp2Lp3L))    # complex number, vec2_complex
        exp31 = tm.cexp(common_exp*(armL_sec+kp3Lp1L))    # complex number, vec2_complex

        prefactor = -PI * TDI_data[i].frequencies * armL_sec * vec2_complex([0.0, 1.0])    # complex number, vec2_complex
        expp0 = tm.cexp(-2 * PI * TDI_data[i].frequencies * kp0 * vec2_complex([0.0, 1.0]))    # complex number, vec2_complex
        commonfac = tm.cmul(prefactor, expp0)    # complex number, vec2_complex

        TDI_data[i]['single_links']['link12'] = sinc12 * tm.cmul(tm.cmul(commonfac, n3Hn3), exp12)    # complex, vec2_complex
        TDI_data[i]['single_links']['link21'] = sinc21 * tm.cmul(tm.cmul(commonfac, n3Hn3), exp12)    # complex, vec2_complex
        TDI_data[i]['single_links']['link23'] = sinc23 * tm.cmul(tm.cmul(commonfac, n1Hn1), exp23)    # complex, vec2_complex
        TDI_data[i]['single_links']['link32'] = sinc32 * tm.cmul(tm.cmul(commonfac, n1Hn1), exp23)    # complex, vec2_complex
        TDI_data[i]['single_links']['link31'] = sinc31 * tm.cmul(tm.cmul(commonfac, n2Hn2), exp31)    # complex, vec2_complex
        TDI_data[i]['single_links']['link13'] = sinc13 * tm.cmul(tm.cmul(commonfac, n2Hn2), exp31)    # complex, vec2_complex

        for chan in ti.static(TDI_data.channels_data.keys):
            TDI_data[i]['channels_data'][chan] = tm.cmul(TDI_data[i]['TDI_gen_prefactor'], TDI_combination_funcs[chan](TDI_data[i]['delay_factor'], TDI_data[i]['single_links']))
        

@ti.kernel
def _compute_TDI_prefactor_FD_response(frequency_field: ti.template(),         
                                       delay_factor_field: ti.template(), 
                                       prefactor_field: ti.template(),
                                       armlength_sec: ti.f64,
                                       TDI_gen: ti.u8
                                      ):
    for i in frequency_field:
        z = tm.cexp(- 2.0 * PI * frequency_field[i] * armlength_sec * vec2_complex([0, 1]))
        
        prefactor = vec2_complex(0.0, 0.0)
        if TDI_gen == 1:
            prefactor = vec2_complex(1, 0) - tm.cpow(z, 2)
        elif TDI_gen == 2:
            prefactor = vec2_complex(1, 0) - tm.cpow(z, 2) - tm.cpow(z, 4) + tm.cpow(z, 6)
        
        prefactor_field[i] = prefactor
        delay_factor_field[i] = z


@ti.func
def _TDI_X(z: vec2_complex, singlelink_responses: SingleLinksStruct) -> vec2_complex:
    '''
    function for computing X channel of TDI combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the X channel without the prefactor which is determined by the TDI generation.
    '''
    return singlelink_responses['link31'] + tm.cmul(z, singlelink_responses['link13']) - singlelink_responses['link21'] - tm.cmul(z, singlelink_responses['link12'])
    


@ti.func
def _TDI_Y(z: vec2_complex, singlelink_responses: SingleLinksStruct) -> vec2_complex:
    '''
    function for computing Y channel of TDI combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the Y channel without the prefactor which is determined by the TDI generation.
    '''
    return singlelink_responses['link12'] + tm.cmul(z, singlelink_responses['link21']) - singlelink_responses['link32'] - tm.cmul(z, singlelink_responses['link23'])


@ti.func
def _TDI_Z(z: vec2_complex, singlelink_responses: SingleLinksStruct) -> vec2_complex:
    '''
    function for computing Z channel of TDI combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the Z channel without the prefactor which is determined by the TDI generation.
    '''
    return singlelink_responses['link23'] + tm.cmul(z, singlelink_responses['link32']) - singlelink_responses['link13'] - tm.cmul(z, singlelink_responses['link31'])


@ti.func
def _TDI_A(z: vec2_complex, singlelink_responses: SingleLinksStruct) -> vec2_complex:
    '''
    function for computing A channel of TDI noise-indenpendent combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the A channel without the prefactor which is determined by the TDI generation.
    '''
    return (singlelink_responses['link23'] + tm.cmul(z, singlelink_responses['link32']) 
         + singlelink_responses['link21'] + tm.cmul(z, singlelink_responses['link12'])
         - tm.cmul((vec2_complex(1, 0) + z), (singlelink_responses['link13']) + singlelink_responses['link31'])
         )/tm.sqrt(2)


@ti.func
def _TDI_E(z: vec2_complex, singlelink_responses: SingleLinksStruct) -> vec2_complex:
    '''
    function for computing E channel of TDI noise-indenpendent combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the E channel without the prefactor which is determined by the TDI generation.
    '''
    return (tm.cmul((vec2_complex(1, 0) - z), (singlelink_responses['link31'] - singlelink_responses['link13'])) + 
         tm.cmul((z + vec2_complex(2, 0)), (singlelink_responses['link32'] - singlelink_responses['link12'])) + 
         tm.cmul((vec2_complex(1, 0) + 2*z), (singlelink_responses['link23'] - singlelink_responses['link21']))
        )/tm.sqrt(6)


@ti.func
def _TDI_T(z: vec2_complex, singlelink_responses: SingleLinksStruct) -> vec2_complex:
    '''
    function for computing T channel of TDI noise-indenpendent combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the T channel without the prefactor which is determined by the TDI generation.
    '''
    return (tm.cmul((singlelink_responses['link12'] - singlelink_responses['link21'] + 
                  singlelink_responses['link23'] - singlelink_responses['link32'] +
                  singlelink_responses['link31'] - singlelink_responses['link13']), 
                  (vec2_complex(1, 0) - z)
                )
         )/tm.sqrt(3)


TDI_combination_funcs = {'X': _TDI_X,
                         'Y': _TDI_Y,
                         'Z': _TDI_Z,
                         'A': _TDI_A,
                         'E': _TDI_E,
                         'T': _TDI_T
                         }
implemented_TDI_generations = ('1.5', '2.0')


################################################################################
@dataclass(frozen=True)
class DataInfo(object):
    """Storing TDI channels data information.
    
    Parameters:
    -----------
    channels: TDI channels, choose from ['X', 'Y', 'Z', 'A', 'E', 'T'];
    generation: TDI generation, one of '1.5' or '2.0';
    duration: observing duration of data, in the unit of second;
    cadence: sampling cadence, in the unit of second;
    start_time: the time label for the first time sample, in the unit of second;
    minimum_frequency: the minimum for the limited frequency band;
    maximum_frequency: the maximum for the limited frequency band.
    """
    channels:tuple[str, ...]
    generation:str
    duration:float
    cadence:float
    start_time:float=0.0
    fmin_in:float=1e-5
    fmax_in:float=0.1

    minimum_frequency:float=field(init=False)
    maximum_frequency:float=field(init=False)
    sampling_frequency:float=field(init=False)
    delta_frequency:float=field(init=False) 
    time_series_length:int=field(init=False)
    time_samples_array:NDArray[np.float64]=field(init=False)
    full_frequency_series_length:int=field(init=False)
    full_frequency_samples_array:NDArray[np.float64]=field(init=False)
    frequency_mask_array:NDArray[np.bool_]=field(init=False)
    frequency_samples_array:NDArray[np.float64]=field(init=False)
    frequency_series_length:int=field(init=False)

    def __post_init__(self)->None:
        """Generating useful numbers from the duration and cadence, and setting proper time and frequency samples:
        TODO: 1. describing rules for minimum_frequency, maximum_frequency
              2. describing rules for time samples and frequency samples
        """
        if not all([chan in TDI_combination_funcs.keys() for chan in self.channels]):
            raise ValueError(f"You are setting TDIChannelData with channels of {self.channels}. While current supported channels are only including {TDI_combination_funcs.keys()}")
        if not self.generation in implemented_TDI_generations:
            raise ValueError(f"You are setting TDIChannelData with generation of {self.generation}. While current supported channels are only including {implemented_TDI_generations}")

        sampling_frequency = 1/self.cadence
        delta_frequency = 1/self.duration
        fmax = np.minimum(self.fmax_in, sampling_frequency/2)
        fmin = np.maximum(self.fmin_in, 1/self.duration)

        time_series_length = int(np.round(self.duration/self.cadence) + 1)    # using round rather // to avoid missing the last sample due to possible numerical error
        time_samples_array = np.arange(time_series_length) * self.cadence + self.start_time
        
        full_frequency_series_length = int(time_series_length//2 + 1)
        full_frequency_samples_array = np.arange(full_frequency_series_length) * delta_frequency
        frequency_mask_array = (full_frequency_samples_array >= fmin) * (full_frequency_samples_array <= fmax)
        frequency_samples_array = full_frequency_samples_array[frequency_mask_array]
        frequency_series_length = int(len(frequency_samples_array))

        object.__setattr__(self, 'minimum_frequency', fmin)
        object.__setattr__(self, 'maximum_frequency', fmax)
        object.__setattr__(self, 'sampling_frequency', sampling_frequency)
        object.__setattr__(self, 'delta_frequency', delta_frequency)
        object.__setattr__(self, 'time_series_length', time_series_length)
        object.__setattr__(self, 'time_samples_array', time_samples_array)
        object.__setattr__(self, 'full_frequency_series_length', full_frequency_series_length)
        object.__setattr__(self, 'full_frequency_samples_array', full_frequency_samples_array)
        object.__setattr__(self, 'frequency_mask_array', frequency_mask_array)
        object.__setattr__(self, 'frequency_samples_array', frequency_samples_array)
        object.__setattr__(self, 'frequency_series_length', frequency_series_length)

        return None


class TDIChannelsData(object):
    # TODO: 
    # - check the normalizing factor of the rfft function

    """Storing TDI strain and noise feature, transfering data from different domain."""

    def __init__(self, minimum_frequency:float=1e-5, maximum_frequency:float=0.1) -> None:
        """
        Parameters:
        -----------
        minimum_frequency:
        maximum_frequency:
        """
        self.time_samples = None
        self.frequency_samples = None
        self.wavelet_samples = None

        self.time_domain_TDI_data = None
        self.frequency_domain_TDI_data = None
        self.wavelet_domain_TDI_data = None

        # self.time_domain_noise_correlation_function = None
        self.frequency_domain_noise_power_density = None
        self.wavelet_domain_noise_power_density = None

        self._data_info = None
        self._reset_flag = False
        self._fmin_in = minimum_frequency
        self._fmax_in = maximum_frequency

        return None
    
    def _reset(self)->None:
        self.__init__(self._fmin_in, self._fmax_in)
        return None
    
    @property
    def data_info(self)->Optional[DataInfo]:
        return self._data_info
    
    def set_data_info(self, channels:tuple[str, ...], generation:str, duration:float, cadence:float, start_time:float=0.0)->None:
        if self._reset_flag:
            warnings.warn("You are setting `data_info`, whereas you have set TDI data of current instance previously. \
                           Setting `data_info` along may lead mismatch of data_info and the stored data. \n \
                           Please check whether this is intentional.")
        self._data_info = DataInfo(channels, generation, duration, cadence, start_time, self._fmin_in, self._fmax_in)
        return None
    
    def _initialize_time_domain_data(self)->None:
        """Initializing `ti.field` for `time_samples` and `time_domain_TDI_data`, only for internel calls.
           Call after setting data_info.
           Setting time domain data externally using `set_time_domain_data_from_zero`.
        """
        self.time_samples = ti.field(ti.f64, (self.data_info.time_series_length,))
        self.time_samples.from_numpy(self.data_info.time_samples_array)
        self.time_domain_TDI_data = ti.Struct.field(dict.fromkeys(self.data_info.channels, ti.float64), shape=(self.data_info.time_series_length,))
        return None

    def _initialize_frequency_domain_data(self)->None:
        """Initializing `ti.field` for `frequency_samples` and `frequency_domain_TDI_data`, only for internel calls.
            Call after setting data_info.
           Setting frequency domain data externally using `set_frequency_domain_data_from_zero`.
        """
        self.frequency_samples = ti.field(ti.f64, (self.data_info.frequency_series_length,))
        self.frequency_samples.from_numpy(self.data_info.frequency_samples_array)
        self.frequency_domain_TDI_data = ti.Struct.field(dict.fromkeys(self.data_info.channels, vec2_complex), shape=(self.data_info.frequency_series_length,))
        return None

    def _initialize_wavelet_domain_data(self)->None:
        return None

    def set_time_domain_data_from_input_array(self, channels:tuple[str, ...], generation:str, duration:float, cadence:float, TDI_data_array:NDArray[np.float64], start_time:float=0.0)->None:
        """Set time domain TDI data from input numpy array. 
        
        Parameters:
        ----------
        channels: TDI channels, choose from ['X', 'Y', 'Z', 'A', 'E', 'T'];
        generation: TDI generation, one of '1.5' or '2.0';
        duration: observing duration of data, in the unit of second;
        cadence: sampling cadence, in the unit of second;
        TDI_data_array: array storing TDI data with the shape of (len(channels), time_series_length), the order in channels list must to be same with the TDI_data_array in input array; 
        start_time: the time label for the first time sample, in the unit of second;

        """

        if self._reset_flag:
            warnings.warn("You are setting `time_domain_data` with input array, \
                           whereas you have probably set TDI data of current instance previously. \
                           Please check whether this is intertional. \n \
                           In order to avoid potential errors, current instance is reset. \
                           Please regenerate TDI data of other domian or noise behavior data if needed. ")
            self._reset()
        
        self.set_data_info(channels, generation, duration, cadence, start_time)
        channels_num, samples_num = TDI_data_array.shape
        if not len(channels) == channels_num:
            raise ValueError(f"You set channenls with {channels}, while the length of first dimension of input TDI_data_array is {channels_num}.")
        if not self.data_info.time_series_length == samples_num:
            raise ValueError(f"The length of second dimension of input array is {samples_num} which is different with the `time_series_length={self.data_info.time_series_length}` \
                             set according to the duration and cadence by `time_series_length = int(np.round(duration/cadence) + 1)`. \
                             If them are only different by 1, probably since there is duraion/cadence > N + 0.5 in your input. Check the input values or open an issue.")
        
        self._initialize_time_domain_data()
        self.time_domain_TDI_data.from_numpy(dict(zip(channels, TDI_data_array)))
        
        self._reset_flag = True
        return None
    
    def set_frequency_domain_data_from_input_array(self, channels:tuple[str, ...], generation:str, duration:float, cadence:float, TDI_data_array:NDArray[np.complex128])->None:
        """Note: the order in channels list must to be same with the TDI_data_array in input array 
           the  length of input TDI_data_array need to be cropped with fmax = Min(f_Nyquist, fmax_in) and fmin = Max(1/T, fmin_in)
        """

        if self._reset_flag:
            warnings.warn("You are setting `frequency_domain_data` with input array, \
                           whereas you have probably set TDI data of current instance previously. \
                           Please check whether this is intertional. \n \
                           In order to avoid potential errors, current instance is reset. \
                           Please regenerate TDI data of other domian or noise behavior data if needed. ")
            self._reset()

        self.set_data_info(channels, generation, duration, cadence)
        channels_num, samples_num = TDI_data_array.shape
        if not len(channels) == channels_num:
            raise ValueError(f"You set channenls with {channels}, while the length of first dimension of input array is {channels_num}.")
        if not self.data_info.frequency_series_length == samples_num:
            raise ValueError(f"The length of second dimension of input array is {samples_num} which is different with the `frequency_series_length={self.data_info.frequency_series_length}` \
                             set according to the duration, cadence and the minimum and maximum frequency. \n \
                             You may need to crop the TDI_data_array with `TDIChannelsData.data_info.frequency_mask_array` before input. \
                             Considering check the input values again or open an issue.")

        self._initialize_frequency_domain_data()
        self.frequency_domain_TDI_data.from_numpy(dict(zip(channels, np.stack((TDI_data_array.real, TDI_data_array.imag), axis=-1))))

        self._reset_flag = True
        return None
    
    def set_time_domain_data_with_zero_value(self, channels:tuple[str, ...], generation:str, duration:float, cadence:float, start_time:float=0.0)->None:
        if self._reset_flag:
            warnings.warn("You are setting `time_domain_data` with zero value, \
                           whereas you have probably set TDI data of current instance previously. \
                           Please check whether this is intertional. \n \
                           In order to avoid potential errors, current instance is reset. \
                           Please regenerate TDI data of other domian or noise behavior data if needed. ")
            self._reset()
        
        self.set_data_info(channels, generation, duration, cadence, start_time)
        
        self._initialize_time_domain_data()
        self.time_domain_TDI_data.fill(0.0)

        self._reset_flag = True
        return None
    
    def set_frequency_domain_data_with_zero_value(self, channels:tuple[str, ...], generation, duration:float, cadence:float)->None:
        if self._reset_flag:
            warnings.warn("You are setting `frequency_domain_data` with zero value, \
                           whereas you have probably set TDI data of current instance previously. \
                           Please check whether this is intertional. \n \
                           In order to avoid potential errors, current instance is reset. \
                           Please regenerate TDI data of other domian or noise behavior data if needed. ")
            self._reset()

        self.set_data_info(channels, generation, duration, cadence)

        self._initialize_frequency_domain_data()
        self.frequency_domain_TDI_data.fill(0.0)

        self._reset_flag = True
        return None
    
    def set_wavelet_domain_data_with_zero_value(self)->None:
        return None
    
    def Fourier_transform_time_domain_data_to_frequency_domain(self, window:float|str|tuple[str|float]=('tukey', 0.2))->None:
        """see scipy.signal.get_window for more details about window parameter
           TODO: check the normalizing factor
        """

        if (self.time_domain_TDI_data is None) or (self.frequency_domain_TDI_data is not None):
            warnings.warn("Fourier transform will not be excuted since the `time_domain_TDI_data` is not set or \
                           `frequency_domain_TDI_data` has been set previously")
        else:
            self._initialize_frequency_domain_data()
            weight = signal.get_window(window, self.data_info.time_series_length)
            for chan in self.data_info.channels:
                td_strain = self.time_domain_TDI_data.get_member_field(chan).to_numpy()
                windowed_strain = td_strain * weight
                fd_strain = np.fft.rfft(windowed_strain)
                fd_strain /= self.data_info.sampling_frequency
                fd_strain = fd_strain[self.data_info.frequency_mask_array]
                self.frequency_domain_TDI_data.get_member_field(chan).from_numpy(np.stack((fd_strain.real, fd_strain.imag), axis=-1))

        return None
    
    def inverse_Fourier_transform_frequency_domain_data_to_time_domain(self)->None:
        """ By default, irfft assumes an even output length which puts the last entry at the Nyquist frequency;
        To avoid losing information, the correct length of the real input must be given."""
        return None
    
    def wavelet_transform_time_domain_data_to_wavelet_domain(self)->None:
        return None
    
    def wavelet_transform_frequency_domain_data_to_wavelet_domain(self)->None:
        return None
    
    def set_frequency_domain_noise_power_density_from_time_domain_data(self)->None:
        return None
    
    def set_frequency_domain_noise_power_density_from_analystic_model(self, noise_model:dict[str, Callable[[np.float64], np.float64]])->None:
        if self.data_info is None:
            raise ValueError('The `data_info` has not yet set. Can not obtain `frequency_series_length` to initialize `frequency_domain_noise_power_density` \
                             Please first set TDI_data in any domain or call directly `data_info`.')
        if self.frequency_domain_noise_power_density is not None:
            warnings.warn("You are setting `frequency_domain_noise_power_density` for current instance, whereas it have been set previously. \
                          It will be reset and updated, please make sure the updated noise power density is consistent with the stored TDI data.")
        self.frequency_domain_noise_power_density = ti.Struct.field(dict.fromkeys(self.data_info.channels, ti.f64), shape=(self.data_info.frequency_series_length,))
        for chan in self.data_info.channels:
            self.frequency_domain_noise_power_density.get_member_field(chan).from_numpy(noise_model[chan](self.data_info.frequency_samples_array))
        return None
    
    def set_frequency_domain_noise_power_density_from_input_array(self)->None:
        return None
    
    def generate_realization_from_frequency_domain_noise_power_density(self, seed=None, output_type:str='taichi')->ti.StructField|dict[str, NDArray[np.complex128]]:
        """Generating a noise realization in frequency domian. 
        there is no sanity check
        To avoid directly modifiying the stroed TDI_data internally, which could potentially leading the missmatch among data of different domain, 
        this method only return the generated noise data as `NDArray`. Using `add_into_frequency_domian_data` manually to add the noise realization into the TDI_data externally.
        
        generate a noise realization from psd
        Reference:
        (eq.12) in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.023033
        https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.psd.PowerSpectralDensity.html#bilby.gw.detector.psd.PowerSpectralDensity.get_noise_realisation
        
        Parameters
        ==========
        seed: integer, 
            set the seed for predictable random number sequence, default is None
        """
        rng = np.random.default_rng(seed=seed)
        var = 0.5  / (self.data_info.delta_frequency)**0.5

        noise_strains = {}
        for chan in self.data_info.channels:
            # generate white noise
            re, im = rng.normal(0, var, (2, self.data_info.full_frequency_series_length))
            # set DC component
            re[0] = 0.0
            im[0] = 0.0
            # set Nyquist frequency component for ensuring the Hermitian symmetry
            if np.mod(self.data_info.time_series_length, 2) == 0:
                im[-1] = 0.0
            noise = np.vstack((re[self.data_info.frequency_mask_array], im[self.data_info.frequency_mask_array])) * self.frequency_domain_noise_power_density_numpy_array[chan] ** 0.5
            noise_strains[chan] = noise.T

        if output_type == "taichi":
            ret = ti.Struct.field(dict.fromkeys(self.data_info.channels, vec2_complex), shape=(self.data_info.frequency_series_length, ))
            ret.from_numpy(noise_strains)
        elif output_type == "numpy":
            ret = {}
            for chan, data in noise_strains.items():
                ret[chan] = data[:,0] + 1j*data[:, 1]

        return ret
    
    
    def add_into_time_domian_data(self)->None:
        return None
    

    def add_into_frequency_domian_data(self, input:ti.StructField|dict[str, NDArray[np.complex128]])->None:
        if isinstance(input, ti.StructField):
            if not input.shape == (self.data_info.frequency_series_length, ):
                raise ValueError("Cannot add the input StructField into the `frequency_domian_TDI_data`, since the shape of input is different with the TDI data")
            if not set(input.keys) == set(self.data_info.channels):
                raise ValueError("Cannot add the input StructField into the `frequency_domian_TDI_data`, since the channnels contained by input is different with the TDI data")
            input_field = input

        elif isinstance(input, dict):
            if not all([len(data)==self.data_info.frequency_series_length for _, data in input.items()]):
                raise ValueError("Cannot add the input dict of array into the `frequency_domian_TDI_data`, since there is at least one array in the input dict having different length with the TDI data.")
            if not set(input.keys()) == set(self.data_info.channels):
                raise ValueError('Cannot add the input dict of array into the `frequency_domian_TDI_data`, since the channnels contained by input is different with the TDI data')
            input_field = ti.Struct.field(dict.fromkeys(self.data_info.channels, vec2_complex), shape=(self.data_info.frequency_series_length,))
            input_field.from_numpy(input)
        else:
            raise TypeError("Unsupported type, expect ti.StructField or dict[NDArray]")
        
        _add_1d_field_into_TDI_data(self.frequency_domain_TDI_data, input_field)
        return None

    def add_into_wavelet_domian_data(self)->None:
        return None


    @property
    def time_samples_numpy_array(self)->Optional[NDArray[np.float64]]:
        """Low performance, do not use in MCMC sampling"""
        if self.data_info is not None:
            return self.data_info.time_samples_array
        else:
            return None
    
    @property
    def frequency_samples_numpy_array(self)->Optional[NDArray[np.float64]]:
        """Low performance, do not use in MCMC sampling"""
        if self.data_info is not None:
            return self.data_info.frequency_samples_array
        else:
            return None

    @property
    def time_domain_TDI_data_numpy_array(self)->Optional[dict[str, NDArray[np.float64]]]:
        """Low performance, do not use in MCMC sampling"""
        if self.time_domain_TDI_data is not None:
            return self.time_domain_TDI_data.to_numpy()
        else:
            return None
    
    @property
    def frequency_domain_TDI_data_numpy_array(self)->Optional[dict[str, NDArray[np.complex128]]]:
        """Low performance, do not use in MCMC sampling"""
        if self.frequency_domain_TDI_data is not None:
            array_dict = self.frequency_domain_TDI_data.to_numpy()
            returned_dict = {}
            for chan, data in array_dict.items():
                returned_dict[chan] = data[:,0] + 1j*data[:,1]
            return returned_dict
        else:
            return None
        
    @property
    def frequency_domain_noise_power_density_numpy_array(self)->Optional[dict[str, NDArray[np.complex128]]]:
        """Low performance, do not use in MCMC sampling"""
        if self.frequency_domain_noise_power_density is not None:
            return self.frequency_domain_noise_power_density.to_numpy()
        else:
            return None

    

@ti.data_oriented
class SpaceborneInterferometer(object):

    def __init__(self, name:str, TDI_data:TDIChannelsData, 
                 orbit:str|Callable[[ti.f64], OrbitVectorStruct]='LISA_analytic', armlength:float=ARM_LENGTH_LISA_SI, 
                 TDI_channels:tuple[str, ...]=('A', 'E'), TDI_generation:str='1.5', response_model:str='full')->None:
        '''
        Instantiate an space detector object.

        Parameters
        ==========
        TDI_data: 
        orbit: string
            orbit model of the constellation, see ".orbits.orbit_models.keys()" for all available options
        armlength: float
            armlength in meter
        TDI_channels: tuple
            TDI channels considered, could be any of "X", "Y", "Z", "A", "E", "T"
            but can only be ("A", "E", "T") or ("A", "E") when computing likelihood
        TDI_generation: string
            TDI generation, could be '1.5' or '2.0'
        response_model: string
            one of 'full', 'low-frequency', 'frozen', 'frozen_low-frequency' (only full are implemented currently)
        '''
        self.name = name
        if isinstance(orbit, str):
            if orbit in available_orbit_models.keys():
                self.orbit_vectors_function = available_orbit_models[orbit]
            else:
                raise ValueError(f'{orbit} is not a implemented orbit model. \n \
                                 Current available orbit models including {[*available_orbit_models.keys()]} \n \
                                 Please check the spell or use a callable function instead.')
        elif callable(orbit):
            self.orbit_vectors_function = orbit

        self.armlength = armlength
        self.armlength_sec = armlength/C_SI
        self.TDI_channels = tuple(TDI_channels)
        self.TDI_generation = TDI_generation
        self.response_model = response_model

        self.TDI_data = TDI_data
        self.response_container = None
        self.waveform_container = None
        self._FD_response_assistance = None

    def initialize_response_container_in_time_domain(self)->None:
        return None
    
    def initialize_response_container_in_frequency_domain(self)->None:
        if self.TDI_data.frequency_domain_TDI_data is None:
            raise ValueError("The `frequency_domain_TDI_data` of the passed-in TDI_data is `None`. \
                             Please set it before calling `initilize_response_container_in_frequency_domain`.")
        else:
            self.response_container = ti.Struct.field(dict.fromkeys(self.TDI_data.data_info.channels, vec2_complex), 
                                                      shape=(self.TDI_data.data_info.frequency_series_length,),
                                                      name='frequency domain response container')
            self._FD_response_assistance = ti.Struct.field(dict(delay_factor = vec2_complex,
                                                                TDI_generation_prefactor = vec2_complex,
                                                                single_links = SingleLinksStruct), 
                                                           shape=(self.TDI_data.data_info.frequency_series_length,))
            if self.TDI_data.data_info.generation == '1.5':
                int_TDI_gen = 1
            elif self.TDI_data.data_info.generation == '2.0':
                int_TDI_gen = 2
            else:
                raise ValueError(f"TDI generation {self.TDI_data.data_info.generation} is unknown. \n \
                                 Please choose from '1.5' or '2.0'.")
            _compute_TDI_prefactor_FD_response(self.TDI_data.frequency_samples, 
                                               self._FD_response_assistance.delay_factor, 
                                               self._FD_response_assistance.TDI_generation_prefactor,
                                               self.armlength_sec, 
                                               int_TDI_gen)
        return None

    def initialize_response_container_in_wavelet_domain(self)->None:
        return None
    
    def initialize_waveform_container_in_time_domain(self)->None:
        return None
    
    def initialize_waveform_container_in_frequency_domain(self)->None:
        self.waveform_container = ti.Struct.field({'hf_plus': vec2_complex,
                                                   'hf_cross': vec2_complex,
                                                   'tf': ti.f64}, 
                                                   shape=(self.TDI_data.data_info.frequency_series_length,))
        
        return None

    def initialize_waveform_container_in_wavelet_domain(self)->None:
        return None

    @ti.kernel
    def update_frequency_domain_response(self, waveform:ti.template(), lam:ti.f64, beta:ti.f64, psi:ti.f64):
        pol_tensor = polarization_tensor_SSB(lam, beta, psi)    # tm.mat3
        k = GW_propagation_unit_vector(lam, beta)             # tm.vec3
        
        for i in self.response_container:

            constellation_vectors = self.orbit_vectors_function(waveform[i].tf)

            n1Hn1 = (constellation_vectors.n1 @ (pol_tensor.plus) @ constellation_vectors.n1) * waveform[i].hplus + (constellation_vectors.n1 @ (pol_tensor.cross) @ constellation_vectors.n1) * waveform[i].hcross    # complex number, vec2_complex   
            n2Hn2 = (constellation_vectors.n2 @ (pol_tensor.plus) @ constellation_vectors.n2) * waveform[i].hplus + (constellation_vectors.n2 @ (pol_tensor.cross) @ constellation_vectors.n2) * waveform[i].hcross    # complex number, vec2_complex   
            n3Hn3 = (constellation_vectors.n3 @ (pol_tensor.plus) @ constellation_vectors.n3) * waveform[i].hplus + (constellation_vectors.n3 @ (pol_tensor.cross) @ constellation_vectors.n3) * waveform[i].hcross    # complex number, vec2_complex   

            kn1 = k@constellation_vectors.n1    # scalar
            kn2 = k@constellation_vectors.n2    # scalar
            kn3 = k@constellation_vectors.n3    # scalar

            kp1Lp2L = k@(constellation_vectors.p1D + constellation_vectors.p2D)    # scalar
            kp2Lp3L = k@(constellation_vectors.p2D + constellation_vectors.p3D)    # scalar
            kp3Lp1L = k@(constellation_vectors.p3D + constellation_vectors.p1D)    # scalar

            kp0 = k@constellation_vectors.p0    # scalar

            common_sinc = PI * self.TDI_data.frequency_samples[i] * self.armlength_sec  # scalar
            sinc12 = sinc(common_sinc * (1.-kn3))    # scalar
            sinc21 = sinc(common_sinc * (1.+kn3))    # scalar
            sinc23 = sinc(common_sinc * (1.-kn1))    # scalar
            sinc32 = sinc(common_sinc * (1.+kn1))    # scalar
            sinc31 = sinc(common_sinc * (1.-kn2))    # scalar
            sinc13 = sinc(common_sinc * (1.+kn2))    # scalar

            common_exp = -PI * self.TDI_data.frequency_samples[i] * vec2_complex([0.0, 1.0])    # complex number, vec2_complex
            exp12 = tm.cexp(common_exp*(self.armlength_sec+kp1Lp2L))    # complex number, vec2_complex
            exp23 = tm.cexp(common_exp*(self.armlength_sec+kp2Lp3L))    # complex number, vec2_complex
            exp31 = tm.cexp(common_exp*(self.armlength_sec+kp3Lp1L))    # complex number, vec2_complex

            prefactor = -PI * self.TDI_data.frequency_samples[i] * self.armlength_sec * vec2_complex([0.0, 1.0])    # complex number, vec2_complex
            expp0 = tm.cexp(-2 * PI * self.TDI_data.frequency_samples[i] * kp0 * vec2_complex([0.0, 1.0]))    # complex number, vec2_complex
            commonfac = tm.cmul(prefactor, expp0)    # complex number, vec2_complex

            self._FD_response_assistance[i]['single_links']['link12'] = sinc12 * tm.cmul(tm.cmul(commonfac, n3Hn3), exp12)    # complex, vec2_complex
            self._FD_response_assistance[i]['single_links']['link21'] = sinc21 * tm.cmul(tm.cmul(commonfac, n3Hn3), exp12)    # complex, vec2_complex
            self._FD_response_assistance[i]['single_links']['link23'] = sinc23 * tm.cmul(tm.cmul(commonfac, n1Hn1), exp23)    # complex, vec2_complex
            self._FD_response_assistance[i]['single_links']['link32'] = sinc32 * tm.cmul(tm.cmul(commonfac, n1Hn1), exp23)    # complex, vec2_complex
            self._FD_response_assistance[i]['single_links']['link31'] = sinc31 * tm.cmul(tm.cmul(commonfac, n2Hn2), exp31)    # complex, vec2_complex
            self._FD_response_assistance[i]['single_links']['link13'] = sinc13 * tm.cmul(tm.cmul(commonfac, n2Hn2), exp31)    # complex, vec2_complex

            for chan in ti.static(self.TDI_data.data_info.channels):
                self.response_container[i][chan] = tm.cmul(self._FD_response_assistance[i]['TDI_gen_prefactor'], TDI_combination_funcs[chan](self._FD_response_assistance[i]['delay_factor'], self._FD_response_assistance[i]['single_links']))
    
    def update_wavelet_domain_response(self)->None:
        return None
    
    def inject_time_domain_signal(self)->None:
        pass

    def inject_frequency_domain_signal(self)->None:
        pass

    def inject_wavelet_domain_signal(self)->None:
        pass

    def view_response_as_numpy_array(self):
        pass

    def view_waveform_as_numpy_array(self):
        pass
    
    def plot(self):
        pass

    def initialize_TDI_data(self):
        '''
        set TDI_data field, using AoS structure to store data for efficiency, 
        keep the memory address fixed to avoid repeated repeated instantiation of the computational kernel
        {frequencies: ti.f64, 
         delay_factor: vec2_complex, 
         TDI_gen_prefactor: vec2_complex, 
         single_links: SingleLinksStruct, 
         channels_data: ti.types.struct(TDI_chan_dict)
         }
        '''
        TDI_chan_dict = dict.fromkeys(self.TDI_channels, vec2_complex)
        TDI_chan_struct = ti.types.struct(**TDI_chan_dict)
        
        TDI_data_struct = ti.types.struct(frequencies = ti.f64, 
                                          delay_factor = vec2_complex,
                                          TDI_gen_prefactor = vec2_complex,
                                          single_links = SingleLinksStruct,
                                          channels_data = TDI_chan_struct)
        TDI_data_field = TDI_data_struct.field()
        ti.root.dense(ti.i, self.data_length).place(TDI_data_field)

        # set frequencies field
        TDI_data_field.frequencies.copy_from(self.frequencies)
        # set dalay_factor and TDI_gen_prefactor
        if self.TDI_generation == '1.5':
            int_TDI_gen = 1
        elif self.TDI_generation == '2.0':
            int_TDI_gen = 2
        _compute_TDI_prefactor(TDI_data_field.frequencies, TDI_data_field.delay_factor, TDI_data_field.TDI_gen_prefactor,
                               self.armlength_sec, int_TDI_gen)

        self.TDI_data = TDI_data_field

        return None
    
    
    def initialize_waveform_container(self):
        waveform_field = ti.Struct.field({'hplus': vec2_complex,
                                          'hcross': vec2_complex,
                                          'tf': ti.f64})
        ti.root.dense(ti.i, self.data_length).place(waveform_field)
        self.waveform_container = waveform_field
        return None


    def updata_TDI_responses(self, parameters):
        '''
        compute the strain of TDI channels from given waveform
        
        Parameters
        ==========
        waveform: dict
            contains the keys "amplitude", "phase", "tf", "frequencies"
        parameters: dict
            parameters describes the GW source

        Returns:
        ========
        dict, strains of TDI channels of current instance
        '''
        _generate_TDI_responses(self.TDI_data, self.waveform_container, self._orbit_vectors_func, self.armlength_sec, 
                                parameters['ecliptic_longitude'],  parameters['ecliptic_latitude'],  parameters['polarization'])
        return None

    
    def initialize_strains_FD(self):
        strains_FD_field = ti.Struct.field(dict.fromkeys(self.TDI_channels, vec2_complex))
        ti.root.dense(ti.i, self.data_length).place(strains_FD_field)
        self.strains_FD = strains_FD_field
        return None
    

    def initialize_strains_TD(self):
        self.strains_TD = None
        return None


    def inject_signal_FD(self, parameters, waveform):
        '''
        TODO wavefrom_dictionary
        inject the GW signal into the detector strains

        Parameters
        ==========
        parameters: dict
            parameters describes the GW source
        waveform: waveform object which contains the detector.waveform_container
            
        '''
        waveform.update_waveform(parameters)
        self.updata_TDI_responses(parameters)
        _inject_into_strains_FD(self.strains_FD, self.TDI_data.channels_data)

        injected_signals = ti.Struct.field(dict.fromkeys(self.TDI_channels, vec2_complex), shape=(self.data_length,))
        injected_signals.copy_from(self.TDI_data.channels_data)
        self.signals.append(injected_signals)

        return None
    
    def initialize_PSDs(self):
        self.PSDs = ti.Struct.field(dict.fromkeys(self.TDI_channels, ti.f64), shape=(self.data_length,))
        return None

    def set_PSDs_from_noise_model(self):
        '''
        compute the psd array from the give noise model
        
        Parameters
        ==========
        frequencies: array, 
            default is None which will use the self.frequencies
        
        Returns:
        ========
        dict, psd array of each TDI channels
        '''
        PSDs_array = {}
        for chan in self.TDI_channels:
            PSDs_array[chan] = noise_models[self.psd_model](self._np_array_frequenices, chan, self.TDI_generation)
        self.PSDs.from_numpy(PSDs_array)
        self._np_array_PSDs = PSDs_array
        return None
    
    @property
    def np_array_PSDs(self):
        return self._np_array_PSDs


    def inject_noise_FD_realization_from_psd(self, seed=None):
        '''
        generate a noise realization from psd
        (eq.12) in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.023033

        Parameters
        ==========
        seed: integer, 
            set the seed for predictable random number sequence, default is None
        '''
        rng = np.random.default_rng(seed=seed)
        var = 0.5 * (1. / self.delta_f)**0.5
        noise_strains = {}
        for chan in self.TDI_channels:
            # noise_amp = rng.normal(0, var, num) * (self.psd_array[chan])**0.5
            # random_phase = rng.uniform(0, 2*PI, num)
            # noise_chan = noise_amp * np.exp(1j*random_phase)
            re = rng.normal(0, var, self.data_length) * (self._np_array_PSDs[chan])**0.5
            im = rng.normal(0, var, self.data_length) * (self._np_array_PSDs[chan])**0.5
            noise_strains[chan] = np.vstack((re, im)).T

        noise_strains_field = ti.Struct.field(dict.fromkeys(self.TDI_channels, vec2_complex), shape=(self.data_length, ))
        noise_strains_field.from_numpy(noise_strains)
        _inject_into_strains_FD(self.strains_FD, noise_strains_field)

        return None
    

    # def optimal_snr(self):
    #     '''
    #     compute the optimal SNR of the GW signal of each channels

    #     Returns:
    #     ========
    #     dict, contain snr of each channels, if ("A", "E", "T") or ("A", "E") channels are contained, total SNR also will be returned
    #     '''
    #     if self.signals is None:
    #         raise Exception('the signals in None, set the GW signal before computing SNR')
        
    #     indep_chan = sorted([chan for chan in self.TDI_channels if chan in ['A', 'E', 'T']])
    #     compute_total = (indep_chan == ['A', 'E', 'T'] or indep_chan == ['A', 'E'])
    #     if compute_total:
    #         total_rho2 = 0.0
    #     else:
    #         print(f'TDI channels are set to {self.TDI_channels} which don\'t contain independent channels '
    #                '("A", "E", "T") or ("A", "E") total SNR will not be computed.')

    #     snr_dict = {}
    #     for chan in self.TDI_channels:
    #         rho2_chan = noise_weighted_inner_product(self.signals[chan], self.signals[chan], self.psd_array[chan], self.delta_f)
    #         snr_dict[chan] = rho2_chan**0.5
    #         if chan in indep_chan and compute_total:
    #             total_rho2 += rho2_chan
        
    #     if compute_total:
    #         snr_dict['total'] = total_rho2**0.5

    #     return snr_dict


    # def plot_FD_data_amplitude(self, outdir='.', contents=['strains_FD', 'signals', 'noise', 'psd_array']):
    #     '''
    #     plot the FD data in the instance

    #     Parameters
    #     ==========
    #     outdir: string
    #         outdit for saving the figure
    #     contents: list
    #         contents in the figure, all available contents are ['strains_FD', 'signals', 'noise', 'psd_array']
    #     '''
    #     if ('signals' in contents) and (len(self.signals)==0):
    #         print(f'Warning: You are requiring to plot `signals`, which do not contain any injections and will be neglicted, call `inject_signal_FD` first.')
    #         contents.remove('signals')
    #     for item in contents[:]:    # using the copy of the list to avoid the unexpected result
    #         if getattr(self, item) is None:
    #             print(f'Warning: You are requiring to plot {item}, which do not contained in your detector instance and will be neglicted.')
    #             contents.remove(item)

    #     for chan in self.TDI_channels:
    #         fig, ax = plt.subplots()
    #         ax.set_title(f'channel {chan}; generation {self.TDI_generation}')
    #         if 'noise' in contents:
    #             ax.loglog(self.frequencies, np.abs(self.noise[chan]), color='C2', label='noise realization')
    #         if 'strains_FD' in contents:
    #             ax.loglog(self.frequencies, np.abs(self.strains_FD[chan]), color='C0', label='total strain')
    #         if 'signals' in contents:
    #             for idx, injection in enumerate(self.signals):
    #                 ax.loglog(self.frequencies, np.abs(injection[chan]), color='C1', label=f'injected GW signal {idx}')
    #         if 'psd_array' in contents:
    #             ax.loglog(self.frequencies, 0.5*np.sqrt(self.psd_array[chan])*(self.duration)**0.5, color='C3', label=r'$\frac{1}{2}\sqrt{S_n(f)T}$')
      
    #         ax.grid(True)
    #         ax.set_ylabel(r'Strain $[1/{\rm Hz}]$')
    #         ax.set_xlabel(r'Frequency [Hz]')
    #         ax.legend(loc='best')
    #         fig.tight_layout()
    #         fig.savefig('{}/{}_{}{}_data_FD.png'.format(outdir, self.name, chan, self.TDI_generation))
    #         plt.close(fig)

    #     return None


    # def plot_TD_data(self, contents=['signal', 'noise']):
    #     pass
    #     return None


    # def save_detector_data(self, outdir='.', label=None):
    #     '''
    #     TODO save the parameters of injected signals
    #     save the data in the instance to a hdf5 file, save contents: [signals, noise, strains_FD, strains_TD, 
    #     frequencies, psd_array]
        
    #     Parameters
    #     ==========
    #     outdir: string
    #     '''
    #     contents = ['signals', 'noise', 'strains_FD', 'strains_TD', 'frequencies', 'psd_array']
    #     save_dict = {}
    #     for item in contents:
    #         save_dict[item] = getattr(self, item)

    #     filename = f'{outdir}/{self.name}_detector_data_{label}.hdf5'
    #     with h5py.File(filename, 'w') as file:
    #         recursively_save_dict_contents_to_group(file, '/', save_dict)

    #     return None
    
    # def set_detector_data_from_file(self, filename):
    #     '''
    #     uncompleted !!!
    #     set ['signals', 'noise', 'strains_FD', 'strains_TD', 'frequencies', 'psd_array']
    #     TODO this function is incomplete !!! when use this func the frequencies, psd_array, ... may not have the same shape 
    #     TODO consider the conflict with read-in data and already set data
    #     TODO add supportation of other attribute, 
    #     set the strains_FD from h5py file
        
    #     Parameters
    #     ==========
    #     filename: string
    #         hdf5 file containing the 'strains_FD'
    #     '''
    #     with h5py.File(filename, 'r') as file:
    #         data = recursively_load_dict_contents_from_group(file, '/')
    #     # TODO corresponding frequencies, duration, cadence should be check

    #     self.frequencies = data['frequencies']
    #     self.psd_array =  data['psd_array']
    #     self.strains_TD = data['strains_TD']
    #     self.strains_FD = data['strains_FD']
    #     self.noise = data['noise']
    #     self.signals = list(data['signals'].values())

    #     return None



