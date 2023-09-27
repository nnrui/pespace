import copy

import h5py
import numpy as np
from matplotlib import pyplot as plt
import taichi as ti
import taichi.math as tm

import lal

from .utilities import polarization_tensor_SSB, GW_propagation_unit_vector_k, sinc, inner_product,   \
                       recursively_save_dict_contents_to_group, recursively_load_dict_contents_from_group
from .orbits import available_orbit_models
from .constants import *
from .noise import noise_models


SingleLinksStruct = ti.types.struct(link12=tm.vec2, link21=tm.vec2, 
                                    link23=tm.vec2, link32=tm.vec2, 
                                    link31=tm.vec2, link13=tm.vec2)

@ti.kernal
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
        k = GW_propagation_unit_vector_k(lam, beta)             # tm.vec3


        for i in TDI_data:
            item = TDI_data[i]

            constellation_vectors = orbit_model(time=waveform[i].tf)

            n1Hn1 = (constellation_vectors.n1 @ (pol_tensor.plus) @ constellation_vectors.n1) * waveform[i].hplus + (constellation_vectors.n1 @ (pol_tensor.cross) @ constellation_vectors.n1) * waveform[i].hcross    # complex number, tm.vec2   
            n2Hn2 = (constellation_vectors.n2 @ (pol_tensor.plus) @ constellation_vectors.n2) * waveform[i].hplus + (constellation_vectors.n2 @ (pol_tensor.cross) @ constellation_vectors.n2) * waveform[i].hcross    # complex number, tm.vec2   
            n3Hn3 = (constellation_vectors.n3 @ (pol_tensor.plus) @ constellation_vectors.n3) * waveform[i].hplus + (constellation_vectors.n3 @ (pol_tensor.cross) @ constellation_vectors.n3) * waveform[i].hcross    # complex number, tm.vec2   

            kn1 = k@constellation_vectors.n1    # scalar
            kn2 = k@constellation_vectors.n2    # scalar
            kn3 = k@constellation_vectors.n3    # scalar

            kp1Lp2L = k@(constellation_vectors.p1D + constellation_vectors.p2D)    # scalar
            kp2Lp3L = k@(constellation_vectors.p2D + constellation_vectors.p3D)    # scalar
            kp3Lp1L = k@(constellation_vectors.p3D + constellation_vectors.p1D)    # scalar

            kp0 = k@constellation_vectors.p0    # scalar

            common_sinc = PI * item.frequencies * armL_sec    # scalar
            sinc12 = sinc(common_sinc * (1.-kn3))    # scalar
            sinc21 = sinc(common_sinc * (1.+kn3))    # scalar
            sinc23 = sinc(common_sinc * (1.-kn1))    # scalar
            sinc32 = sinc(common_sinc * (1.+kn1))    # scalar
            sinc31 = sinc(common_sinc * (1.-kn2))    # scalar
            sinc13 = sinc(common_sinc * (1.+kn2))    # scalar

            common_exp = -PI * item.frequencies * tm.vec2([0.0, 1.0])    # complex number, tm.vec2
            exp12 = tm.cexp(common_exp*(armL_sec+kp1Lp2L))    # complex number, tm.vec2
            exp23 = tm.cexp(common_exp*(armL_sec+kp2Lp3L))    # complex number, tm.vec2
            exp31 = tm.cexp(common_exp*(armL_sec+kp3Lp1L))    # complex number, tm.vec2

            prefactor = -PI * item.frequencies * armL_sec * tm.vec2([0.0, 1.0])    # complex number, tm.vec2
            expp0 = tm.cexp(-2 * PI * item.frequencies * kp0 * tm.vec2([0.0, 1.0]))    # complex number, tm.vec2
            commonfac = tm.cmul(prefactor, expp0)    # complex number, tm.vec2

            item['single_links']['link12'] = sinc12 * tm.cmul(tm.cmul(commonfac, n3Hn3), exp12)    # complex, tm.vec2
            item['single_links']['link21'] = sinc21 * tm.cmul(tm.cmul(commonfac, n3Hn3), exp12)    # complex, tm.vec2
            item['single_links']['link23'] = sinc23 * tm.cmul(tm.cmul(commonfac, n1Hn1), exp23)    # complex, tm.vec2
            item['single_links']['link32'] = sinc32 * tm.cmul(tm.cmul(commonfac, n1Hn1), exp23)    # complex, tm.vec2
            item['single_links']['link31'] = sinc31 * tm.cmul(tm.cmul(commonfac, n2Hn2), exp31)    # complex, tm.vec2
            item['single_links']['link13'] = sinc13 * tm.cmul(tm.cmul(commonfac, n2Hn2), exp31)    # complex, tm.vec2

            for chan in ti.static(TDI_data.TDI_chan_data.keys):
                item['TDI_chan_data'][chan] = tm.cmul(item['TDI_gen_prefactor'], TDI_combination_funcs[chan](item['delay_factor'], item['signle_links']))


@ti.kernel
def _compute_TDI_prefactor(frequencies: ti.template(),         
                           z_field: ti.template(), 
                           prefactor_field: ti.template(),
                           armlength_sec: ti.f64,
                           TDI_gen: ti.u8
                          ):
    for i in frequencies:
        z = tm.cexp(- 2.0 * PI * frequencies[i] * armlength_sec * tm.vec2([0, 1]))
        
        prefactor = tm.vec2(0.0, 0.0)
        if TDI_gen == 1:
            prefactor = tm.vec2(1, 0) - tm.cpow(z, 2)
        elif TDI_gen == 2:
            prefactor = tm.vec2(1, 0) - tm.cpow(z, 2) - tm.cpow(z, 4) + tm.cpow(z, 6)
        
        prefactor_field[i] = prefactor
        z_field[i] = z


@ti.func
def _TDI_X(z: tm.vec2, singlelink_responses: SingleLinksStruct) -> tm.vec2:
    '''
    function for computing X channel of TDI combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*arm_length_LISA_sec)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the X channel without the prefactor which is determined by the TDI generation.
    '''
    return singlelink_responses['link31'] + tm.cmul(z, singlelink_responses['link13']) - singlelink_responses['link21'] - tm.cmul(z, singlelink_responses['link12'])
    


@ti.func
def _TDI_Y(z: tm.vec2, singlelink_responses: SingleLinksStruct) -> tm.vec2:
    '''
    function for computing Y channel of TDI combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*arm_length_LISA_sec)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the Y channel without the prefactor which is determined by the TDI generation.
    '''
    return singlelink_responses['link12'] + tm.cmul(z, singlelink_responses['link21']) - singlelink_responses['link32'] - tm.cmul(z, singlelink_responses['link23'])


@ti.func
def _TDI_Z(z: tm.vec2, singlelink_responses: SingleLinksStruct) -> tm.vec2:
    '''
    function for computing Z channel of TDI combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*arm_length_LISA_sec)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the Z channel without the prefactor which is determined by the TDI generation.
    '''
    return singlelink_responses['link23'] + tm.cmul(z, singlelink_responses['link32']) - singlelink_responses['link13'] - tm.cmul(z, singlelink_responses['link31'])


@ti.func
def _TDI_A(z: tm.vec2, singlelink_responses: SingleLinksStruct) -> tm.vec2:
    '''
    function for computing A channel of TDI noise-indenpendent combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*arm_length_LISA_sec)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the A channel without the prefactor which is determined by the TDI generation.
    '''
    return (singlelink_responses['link23'] + tm.cmul(z, singlelink_responses['link32']) 
         + singlelink_responses['link21'] + tm.cmul(z, singlelink_responses['link12'])
         - tm.cmul((tm.vec2(1, 0) + z), (singlelink_responses['link13']) + singlelink_responses['link31'])
         )/tm.sqrt(2)


@ti.func
def _TDI_E(z: tm.vec2, singlelink_responses: SingleLinksStruct) -> tm.vec2:
    '''
    function for computing E channel of TDI noise-indenpendent combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*arm_length_LISA_sec)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the E channel without the prefactor which is determined by the TDI generation.
    '''
    return (tm.cmul((tm.vec2(1, 0) - z), (singlelink_responses['link31'] - singlelink_responses['link13'])) + 
         tm.cmul((z + tm.vec2(2, 0)), (singlelink_responses['link32'] - singlelink_responses['link12'])) + 
         tm.cmul((tm.vec2(1, 0) + 2*z), (singlelink_responses['link23'] - singlelink_responses['link21']))
        )/tm.sqrt(6)


@ti.func
def _TDI_T(z: tm.vec2, singlelink_responses: SingleLinksStruct) -> tm.vec2:
    '''
    function for computing T channel of TDI noise-indenpendent combination

    Parameters
    ==========
    z: array
        delay factor, exp(-1j*2*PI*frequencies*arm_length_LISA_sec)
    singlelink_responses: dict
        contains the 6 array which are GW responses of each link

    Returns:
    ========
    array, the T channel without the prefactor which is determined by the TDI generation.
    '''
    return (tm.cmul((singlelink_responses['link12'] - singlelink_responses['link21'] + 
                  singlelink_responses['link23'] - singlelink_responses['link32'] +
                  singlelink_responses['link31'] - singlelink_responses['link13']), 
                  (tm.vec2(1, 0) - z)
                )
         )/tm.sqrt(3)


TDI_combination_funcs = {'X': _TDI_X,
                         'Y': _TDI_Y,
                         'Z': _TDI_Z,
                         'A': _TDI_A,
                         'E': _TDI_E,
                         'T': _TDI_T
                         }

################################################################################
class LISALike(object):


    def __init__(self, name, duration, cadance, start_time=0.0, minimum_frequency=1.0e-4, maximum_frequency=0.1, 
                 psd_model='LISA_SciRDv1', orbit='LISA_analytic', armlength=ARM_LENGTH_LISA_SI, TDI_channels=('A', 'E'), 
                 TDI_generation='1.5', response_model='full', strains_TD=None, strains_FD=None):
        '''
        Instantiate an space detector object.

        Parameters
        ==========
        duration: float
            duration of the data, in the unit of second
        cadance: float
            cadance of the sampling, in the unit of second
        minimum_frequency: float
            minimum frequency to analyse for detector.        
        maximum_frequency: float
            minimum frequency to analyse for detector.
        psd_model: string
            power spectral density model, see "noise_models.keys()" for all available options
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
            one of 'full', 'low-frequency', 'frozen', 'frozen_low-frequency'
        strains_TD: array (TODO add the check and set function)
            time domain strain, the shape must be compatible with the duration and cadance
        strains_FD: array (TODO add the check and set function)
            frequency domain strain, the shape must be compatible with the duration, cadance and the frequency bound
        '''
        # TODO protect the attribute which should not be changed in outside
        # vars in python scope, keep static
        self.name = name
        self.duration = duration
        self.cadance = cadance
        self.start_time = start_time
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.psd_model = psd_model
        self.orbit = orbit
        self._orbit_vectors_func = available_orbit_models[orbit]
        self.armlength = armlength
        self.armlength_sec = armlength/C_SI
        self.TDI_channels = tuple(TDI_channels)
        self.TDI_generation = TDI_generation
        # TODO different response model: full, frozen, low-f, frozen and low-f
        self.response_model = response_model
        self.strains_TD = strains_TD
        self.strains_FD = strains_FD
        # var in global scope, can be modified
        self.set_frequencies()
        self.initialize_TDI_data()
        self.initialize_waveform_container()

        # self.psd_array = self.get_psd_array()
        # TODO check the strains_TD and strains_FD before set
        # TODO use the specific method to set
        self.signals = None
        self.noise = None


    def set_frequencies(self):
        '''
        TODO:
        1. protect the frequencies to avoide accidental modification; 
        2. access frequencies by numpy.ndarray

        set the frequency array, except the given frequency bound, the Nyquist frequency and the duration of 
        the data also need to be considered.

        set frequencies, delta_f, length
        '''
        frequencies = np.arange(0, 1.0/(2*self.cadance), 1.0/self.duration)
        bound = ((frequencies >= self.minimum_frequency) * (frequencies <= self.maximum_frequency))
        frequencies = frequencies[bound]
        
        self.frequencies = frequencies
        self.delta_f = 1.0/self.duration
        self.length = len(frequencies)

        ti_frequencies = ti.field(ti.f64, (self.length,))
        ti_frequencies.from_numpy(frequencies)
        self._ti_frequencies = ti_frequencies    # for convenient and efficient when frequenies are used in ti scope

        return None


    def initialize_TDI_data(self):
        '''
        set TDI_data field, using AoS structure to store data for efficiency, 
        keep the memory address fixed to avoid repeated repeated instantiation of the computational kernel
        {frequencies: ti.f64, 
         delay_factor: tm.vec2, 
         TDI_gen_prefactor: tm.vec2, 
         single_links: SingleLinksStruct, 
         TDI_chan_data: ti.types.struct(TDI_chan_dict)
         }
        '''
        TDI_chan_dict = dict.fromkeys(self.TDI_channels, tm.vec2)
        TDI_chan_struct = ti.types.struct(TDI_chan_dict)
        
        TDI_data_struct = ti.types.struct(frequencies = ti.f64, 
                                           delay_factor = tm.vec2,
                                           TDI_gen_prefactor = tm.vec2,
                                           single_links = SingleLinksStruct,
                                           TDI_chan_data = TDI_chan_struct)
        TDI_data_field = TDI_data_struct.field()
        ti.root.dense(ti.i, self.length).place(TDI_data_field)

        # set frequencies field
        TDI_data_field.frequencies.copy_from(self._ti_frequencies)
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
        waveform_field = ti.Struct.field({'hplus': tm.vec2,
                                          'hcross': tm.vec2,
                                          'tf': ti.f64})
        ti.root.dense(ti.i, self.length).place(waveform_field)
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

    
    # def inject_signal_FD(self, parameters, waveform_func):
    #     '''
    #     TODO wavefrom_dictionary
    #     inject the GW signal into the detector strains

    #     Parameters
    #     ==========
    #     parameters: dict
    #         parameters describes the GW source
    #     waveform_func: function
    #         see wavefrom.__dir__() for all support wavefrom
    #     '''
    #     if self.strains_FD is None:
    #         self.strains_FD = dict.fromkeys(self.TDI_channels, np.zeros(len(self.frequency_array), dtype=np.complex128))
    #     strains_FD = copy.deepcopy(self.strains_FD)     # remember using deepcopy to avoid error when plus the signals

    #     waveform = waveform_func(self.frequency_array, parameters.copy(), neglect_waveform_errors=False)
    #     signals = self.TDI_responses(waveform, parameters)
    #     self.signals = signals
    #     for chan in self.TDI_channels:
    #         self.strains_FD[chan] = strains_FD[chan] + signals[chan]  # remember the deepcopy of self.strains_FD rather than ifself
        
    #     return None
    
    # def get_psd_array(self, frequencies=None):
    #     '''
    #     compute the psd array from the give noise model
        
    #     Parameters
    #     ==========
    #     frequencies: array, 
    #         default is None which will use the self.frequency_array
        
    #     Returns:
    #     ========
    #     dict, psd array of each TDI channels
    #     '''
    #     if frequencies == None:
    #         frequencies = self.frequency_array
        
    #     noise_dict = {}
    #     for chan in self.TDI_channels:
    #         noise_dict[chan] = noise_models[self.psd_model](frequencies, chan, self.TDI_generation)
        
    #     return noise_dict
    
    # def generate_FD_noise_realization_from_psd(self, seed=None):
    #     '''
    #     generate a noise realization from psd
    #     (eq.12) in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.023033

    #     Parameters
    #     ==========
    #     seed: integer, 
    #         set the seed for predictable random number sequence, default is None
    #     '''
    #     num = len(self.frequency_array)
    #     rng = np.random.default_rng(seed=seed)
    #     var = 0.5 * (1. / self.delta_freq)**0.5
    #     noise_strain = {}
    #     for chan in self.TDI_channels:
    #         # noise_amp = rng.normal(0, var, num) * (self.psd_array[chan])**0.5
    #         # random_phase = rng.uniform(0, 2*PI, num)
    #         # noise_chan = noise_amp * np.exp(1j*random_phase)
    #         re = rng.normal(0, var, num)
    #         im = rng.normal(0, var, num)
    #         noise_chan =(re + 1j*im) * (self.psd_array[chan])**0.5

    #         noise_strain[chan] = noise_chan
    #     self.noise = noise_strain

    #     if self.strains_FD is not None:
    #         print('Warning: the strains_FD of the detector instance is not None, please make sure you actually want to overwrite the original signal with the new noise')
    #     self.strains_FD = copy.deepcopy(noise_strain)    # remember using the deepcopy when assignment

    #     return None
    
    # def optimal_snr(self):
    #     '''
    #     compute the optimal SNR of the GW signal of each channels

    #     Returns:
    #     ========
    #     dict, contain snr of each channels, if ("A", "E", "T") or ("A", "E") channels are contained, total SNR also will be returned
    #     '''
    #     if self.signals is None:
    #         raise Exception('the signals in None, set the GW signal before computing SNR')
        
    #     indep_chan = [chan for chan in self.TDI_channels if chan in ['A', 'E', 'T']]
    #     compute_total = (indep_chan == ['A', 'E', 'T'] or indep_chan == ['A', 'E'])
    #     if compute_total:
    #         total_rho2 = 0.0
    #     else:
    #         print(f'TDI channels are set to {self.TDI_channels} which don\'t contain independent channels '
    #                '("A", "E", "T") or ("A", "E") total SNR will not be computed.')

    #     snr_dict = {}
    #     for chan in self.TDI_channels:
    #         rho2_chan = inner_product(self.signals[chan], self.signals[chan], self.psd_array[chan], 1./self.duration)
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
    #     for item in contents[:]:    # using the copy of the list to avoid the unexpected result
    #         if getattr(self, item) is None:
    #             print(f'Warning: You are requiring to plot {item}, which do not contained in your detector instance and will be neglicted.')
    #             contents.remove(item)

    #     for chan in self.TDI_channels:
    #         fig, ax = plt.subplots()
    #         ax.set_title(f'channel {chan}; generation {self.TDI_generation}')
    #         if 'noise' in contents:
    #             ax.loglog(self.frequency_array, np.abs(self.noise[chan]), color='C2', label='noise realization')
    #         if 'strains_FD' in contents:
    #             ax.loglog(self.frequency_array, np.abs(self.strains_FD[chan]), color='C0', label='total strain')
    #         if 'signals' in contents:
    #             ax.loglog(self.frequency_array, np.abs(self.signals[chan]), color='C1', label='injected GW signal')
    #         if 'psd_array' in contents:
    #             ax.loglog(self.frequency_array, 0.5*np.sqrt(self.psd_array[chan])*(self.duration)**0.5, color='C3', label=r'$\frac{1}{2}\sqrt{S_n(f)T}$')
      
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
    
    # def save_data(self, outdir='.', label=None):
    #     '''
    #     TODO save the parameters of injected signals
    #     save the data in the instance to a .json file, save contents: [signals, noise, strains_FD, strains_TD, 
    #     frequency_array, psd_array]
        
    #     Parameters
    #     ==========
    #     outdir: string
    #     '''
    #     contents = ['signals', 'noise', 'strains_FD', 'strains_TD', 'frequency_array', 'psd_array']
    #     save_dict = {}
    #     for item in contents:
    #         save_dict[item] = getattr(self, item)

    #     filename = f'{outdir}/{self.name}_detector_data_{label}.hdf5'
    #     with h5py.File(filename, 'w') as file:
    #         recursively_save_dict_contents_to_group(file, '/', save_dict)

    #     return None
    
    # def set_strains_FD_from_file(self, filename):
    #     '''
    #     TODO this function is incomplete !!! when use this func the frequency_array, psd_array, ... may not have the same shape 
    #     TODO consider the conflict with read-in data and already set data
    #     TODO add supportation of other attribute
    #     set the strains_FD from h5py file
        
    #     Parameters
    #     ==========
    #     filename: string
    #         hdf5 file containing the 'strains_FD'
    #     '''
    #     with h5py.File(filename, 'r') as file:
    #         data = recursively_load_dict_contents_from_group(file, '/')
    #     # TODO corresponding frequency_array, duration, cadance should be check
    #     self.strains_FD = data['strains_FD']
    #     return None

    # def set_strains_TD_from_strains_FD(self):
    #     pass
    #     return None

