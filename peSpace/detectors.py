import copy

import h5py
import numpy as np
from matplotlib import pyplot as plt

import lal

from .utilities import polarization_tensor_SSB, GW_propagation_unit_vector_k, sinc, inner_product,   \
                       recursively_save_dict_contents_to_group, recursively_load_dict_contents_from_group
from .orbits import get_constellation_vectors_from_orbits
from .constants import *
from .noise import noise_models


def TDI_X(z, singlelink_responses):
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
    X = singlelink_responses['link31'] + z*singlelink_responses['link13'] - singlelink_responses['link21'] - z*singlelink_responses['link12']
    return X

def TDI_Y(z, singlelink_responses):
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
    Y = singlelink_responses['link12'] + z*singlelink_responses['link21'] - singlelink_responses['link32'] - z*singlelink_responses['link23']
    return Y

def TDI_Z(z, singlelink_responses):
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
    Z = singlelink_responses['link23'] + z*singlelink_responses['link32'] - singlelink_responses['link13'] - z*singlelink_responses['link31']
    return Z

def TDI_A(z, singlelink_responses):
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
    A = (singlelink_responses['link23'] + singlelink_responses['link32']*z 
         + singlelink_responses['link21'] + singlelink_responses['link12']*z 
         - (1+z)*(singlelink_responses['link13'] + singlelink_responses['link31'])
         )/np.sqrt(2)
    return A

def TDI_E(z, singlelink_responses):
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
    E = ((1-z)*(singlelink_responses['link31'] - singlelink_responses['link13'])
         + (z+2)*(singlelink_responses['link32'] - singlelink_responses['link12'])
         + (1+2*z)*(singlelink_responses['link23'] - singlelink_responses['link21'])
         )/np.sqrt(6)
    return E

def TDI_T(z, singlelink_responses):
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
    T = ((singlelink_responses['link12'] - singlelink_responses['link21'] + 
          singlelink_responses['link23'] - singlelink_responses['link32'] +
          singlelink_responses['link31'] - singlelink_responses['link13'])*(1-z)
         )/np.sqrt(3)
    return T

TDI_combination = {'X': TDI_X,
                   'Y': TDI_Y,
                   'Z': TDI_Z,
                   'A': TDI_A,
                   'E': TDI_E,
                   'T': TDI_T,
                  }


class LISALike(object):

    def __init__(self, name, duration, cadance, start_time=0.0, minimum_frequency=1.0e-4, maximum_frequency=0.1, 
                 psd_model='LISA_SciRDv1', orbit='LISA_analytic', TDI_channels=('A', 'E'), TDI_generation='1.5',
                 response_model='full', strains_TD=None, strains_FD=None):
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
        self.name = name
        self.duration = duration
        self.cadance = cadance
        self.start_time = start_time
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.psd_model = psd_model
        self.orbit = orbit
        self.TDI_channels = tuple(TDI_channels)
        self.TDI_generation = TDI_generation
        self.response_model = response_model
        self.strains_TD = strains_TD
        self.strains_FD = strains_FD

        self.frequency_array = self.set_frequency_array()
        self.delta_freq = self.frequency_array[1] - self.frequency_array[0]
        self.frequency_length = len(self.frequency_array)
        self.psd_array = self.get_psd_array()
        # TODO check the strains_TD and strains_FD before set
        # TODO use the specific method to set
        self.signals = None
        self.noise = None

    def set_frequency_array(self):
        '''
        TODO protect the frequency_array to avoide accidental modification
        set the frequency array, except the given frequency bound, the Nyquist frequency and the duration of 
        the data also need to be considered.

        Returns:
        ========
        array, the frequency array on which the simulated FD strain will be generated and analysised
        '''
        f_low = np.maximum(self.minimum_frequency, 1.0/self.duration)
        f_high = np.minimum(self.maximum_frequency, 1.0/(2*self.cadance))
        frequencies = np.arange(f_low, f_high, 1.0/self.duration)
        return frequencies
    
    def generate_singlelink_responses(self, waveform, parameters):
        '''
        TODO: different response models
        computing the responses of each laser link to the GW
        https://lisa-ldc.lal.in2p3.fr/static/data/pdf/LDC-manual-002.pdf

        Parameters
        ==========
        waveform: dict
            contains the keys "amplitude", "phase", "tf", "frequencies"
        parameters: dict
            parameters describes the GW source

        Returns:
        ========
        dict, responses of 6 links
        '''
        amp = waveform['amplitude']
        phase = waveform['phase']
        tf = waveform['tf']
        frequencies = waveform['frequencies']
        num = len(frequencies)

        lam = parameters['ecliptic_longitude']
        beta = parameters['ecliptic_latitude']
        psi = parameters['polarization']
        inc = parameters['inclination']
        phi0 = parameters['coalescence_phase']

        # TODO why don't let the lalsim directly return h_cross and h_plus, (could be problematic when including higher modes and interplation)
        h22 = amp*np.exp(1j*phase) # NOTE whether the returned phase should include the minus
        Y22 = lal.SpinWeightedSphericalHarmonic(inc, phi0, -2, 2, 2)
        Y2m2star = np.conjugate(lal.SpinWeightedSphericalHarmonic(inc, phi0, -2, 2, -2))
        # NOTE remember that the waform with precession is currently not support
        # TODO double check the convention difference with lalsimulation
        hplus = 0.5*(Y22 + Y2m2star) * h22
        hcross = 1j*0.5*(Y22 - Y2m2star) * h22

        pol_tensor_plus = polarization_tensor_SSB(lam, beta, psi, 'plus')
        pol_tensor_cross = polarization_tensor_SSB(lam, beta, psi, 'cross')
        k = GW_propagation_unit_vector_k(lam, beta)

        # allocating output array
        link12 = np.zeros(num, dtype=np.complex128)
        link21 = np.zeros(num, dtype=np.complex128)
        link23 = np.zeros(num, dtype=np.complex128)
        link32 = np.zeros(num, dtype=np.complex128)
        link31 = np.zeros(num, dtype=np.complex128)
        link13 = np.zeros(num, dtype=np.complex128)

        for i in range(num):
            constellation_vectors = get_constellation_vectors_from_orbits(time=tf[i], orbit=self.orbit)

            n1Hn1 = constellation_vectors['n1']@(pol_tensor_plus*hplus[i] + pol_tensor_cross*hcross[i])@constellation_vectors['n1_T']
            n2Hn2 = constellation_vectors['n2']@(pol_tensor_plus*hplus[i] + pol_tensor_cross*hcross[i])@constellation_vectors['n2_T']
            n3Hn3 = constellation_vectors['n3']@(pol_tensor_plus*hplus[i] + pol_tensor_cross*hcross[i])@constellation_vectors['n3_T']

            kn1 = k@constellation_vectors['n1_T']
            kn2 = k@constellation_vectors['n2_T']
            kn3 = k@constellation_vectors['n3_T']

            kp1Lp2L = k@((constellation_vectors['p1L'] + constellation_vectors['p2L']).T)
            kp2Lp3L = k@((constellation_vectors['p2L'] + constellation_vectors['p3L']).T)
            kp3Lp1L = k@((constellation_vectors['p3L'] + constellation_vectors['p1L']).T)

            kp0 = k@(constellation_vectors['p0'].T)

            common_sinc = PI * frequencies[i] * ARM_LENGTH_LISA_SEC
            sinc12 = sinc(common_sinc * (1.-kn3))
            sinc21 = sinc(common_sinc * (1.+kn3))
            sinc23 = sinc(common_sinc * (1.-kn1))
            sinc32 = sinc(common_sinc * (1.+kn1))
            sinc31 = sinc(common_sinc * (1.-kn2))
            sinc13 = sinc(common_sinc * (1.+kn2))

            common_exp = -1j*PI*frequencies[i]
            exp12 = np.exp(common_exp*(ARM_LENGTH_LISA_SEC+kp1Lp2L))
            exp23 = np.exp(common_exp*(ARM_LENGTH_LISA_SEC+kp2Lp3L))
            exp31 = np.exp(common_exp*(ARM_LENGTH_LISA_SEC+kp3Lp1L))

            prefactor = -1j * PI * frequencies[i] * ARM_LENGTH_LISA_SEC
            expp0 = np.exp(-1j * 2 * PI * frequencies[i] * kp0)
            commonfac = prefactor * expp0

            link12[i] = commonfac * n3Hn3 * sinc12 * exp12
            link21[i] = commonfac * n3Hn3 * sinc21 * exp12
            link23[i] = commonfac * n1Hn1 * sinc23 * exp23
            link32[i] = commonfac * n1Hn1 * sinc32 * exp23
            link31[i] = commonfac * n2Hn2 * sinc31 * exp31
            link13[i] = commonfac * n2Hn2 * sinc13 * exp31

        return {'link12': link12, 
                'link21': link21,
                'link23': link23,
                'link32': link32,
                'link31': link31,
                'link13': link13,}
    
    def TDI_responses(self, waveform, parameters):
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
        frequencies = waveform['frequencies']
        singlelink_responses = self.generate_singlelink_responses(waveform, parameters)
        z = np.exp(-1j*2*PI*frequencies*ARM_LENGTH_LISA_SEC)    # delay factor

        if self.TDI_generation == '1.5':
            prefactor = (1 - z**2)
        elif self.TDI_generation == '2.0':
            prefactor = (1 - z**2 - z**4 + z**6)
        
        TDI_dict  = {}
        for chan in self.TDI_channels:
            TDI_dict[chan] = prefactor * TDI_combination[chan](z, singlelink_responses)

        return TDI_dict
    
    def inject_signal_FD(self, parameters, waveform_func):
        '''
        TODO wavefrom_dictionary
        inject the GW signal into the detector strains

        Parameters
        ==========
        parameters: dict
            parameters describes the GW source
        waveform_func: function
            see wavefrom.__dir__() for all support wavefrom
        '''
        if self.strains_FD is None:
            self.strains_FD = dict.fromkeys(self.TDI_channels, np.zeros(len(self.frequency_array), dtype=np.complex128))
        strains_FD = copy.deepcopy(self.strains_FD)     # remember using deepcopy to avoid error when plus the signals

        waveform = waveform_func(self.frequency_array, parameters.copy(), neglect_waveform_errors=False)
        signals = self.TDI_responses(waveform, parameters)
        self.signals = signals
        for chan in self.TDI_channels:
            self.strains_FD[chan] = strains_FD[chan] + signals[chan]  # remember the deepcopy of self.strains_FD rather than ifself
        
        return None
    
    def get_psd_array(self, frequencies=None):
        '''
        compute the psd array from the give noise model
        
        Parameters
        ==========
        frequencies: array, 
            default is None which will use the self.frequency_array
        
        Returns:
        ========
        dict, psd array of each TDI channels
        '''
        if frequencies == None:
            frequencies = self.frequency_array
        
        noise_dict = {}
        for chan in self.TDI_channels:
            noise_dict[chan] = noise_models[self.psd_model](frequencies, chan, self.TDI_generation)
        
        return noise_dict
    
    def generate_FD_noise_realization_from_psd(self, seed=None):
        '''
        generate a noise realization from psd
        (eq.12) in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.023033

        Parameters
        ==========
        seed: integer, 
            set the seed for predictable random number sequence, default is None
        '''
        num = len(self.frequency_array)
        rng = np.random.default_rng(seed=seed)
        var = 0.5 * (1. / self.delta_freq)**0.5
        noise_strain = {}
        for chan in self.TDI_channels:
            # noise_amp = rng.normal(0, var, num) * (self.psd_array[chan])**0.5
            # random_phase = rng.uniform(0, 2*PI, num)
            # noise_chan = noise_amp * np.exp(1j*random_phase)
            re = rng.normal(0, var, num)
            im = rng.normal(0, var, num)
            noise_chan =(re + 1j*im) * (self.psd_array[chan])**0.5

            noise_strain[chan] = noise_chan
        self.noise = noise_strain

        if self.strains_FD is not None:
            print('Warning: the strains_FD of the detector instance is not None, please make sure you actually want to overwrite the original signal with the new noise')
        self.strains_FD = copy.deepcopy(noise_strain)    # remember using the deepcopy when assignment

        return None
    
    def optimal_snr(self):
        '''
        compute the optimal SNR of the GW signal of each channels

        Returns:
        ========
        dict, contain snr of each channels, if ("A", "E", "T") or ("A", "E") channels are contained, total SNR also will be returned
        '''
        if self.signals is None:
            raise Exception('the signals in None, set the GW signal before computing SNR')
        
        indep_chan = [chan for chan in self.TDI_channels if chan in ['A', 'E', 'T']]
        compute_total = (indep_chan == ['A', 'E', 'T'] or indep_chan == ['A', 'E'])
        if compute_total:
            total_rho2 = 0.0
        else:
            print(f'TDI channels are set to {self.TDI_channels} which don\'t contain independent channels '
                   '("A", "E", "T") or ("A", "E") total SNR will not be computed.')

        snr_dict = {}
        for chan in self.TDI_channels:
            rho2_chan = inner_product(self.signals[chan], self.signals[chan], self.psd_array[chan], 1./self.duration)
            snr_dict[chan] = rho2_chan**0.5
            if chan in indep_chan and compute_total:
                total_rho2 += rho2_chan
        
        if compute_total:
            snr_dict['total'] = total_rho2**0.5

        return snr_dict

    def plot_FD_data_amplitude(self, outdir='.', contents=['strains_FD', 'signals', 'noise', 'psd_array']):
        '''
        plot the FD data in the instance

        Parameters
        ==========
        outdir: string
            outdit for saving the figure
        contents: list
            contents in the figure, all available contents are ['strains_FD', 'signals', 'noise', 'psd_array']
        '''
        for item in contents[:]:    # using the copy of the list to avoid the unexpected result
            if getattr(self, item) is None:
                print(f'Warning: You are requiring to plot {item}, which do not contained in your detector instance and will be neglicted.')
                contents.remove(item)

        for chan in self.TDI_channels:
            fig, ax = plt.subplots()
            ax.set_title(f'channel {chan}; generation {self.TDI_generation}')
            if 'noise' in contents:
                ax.loglog(self.frequency_array, np.abs(self.noise[chan]), color='C2', label='noise realization')
            if 'strains_FD' in contents:
                ax.loglog(self.frequency_array, np.abs(self.strains_FD[chan]), color='C0', label='total strain')
            if 'signals' in contents:
                ax.loglog(self.frequency_array, np.abs(self.signals[chan]), color='C1', label='injected GW signal')
            if 'psd_array' in contents:
                ax.loglog(self.frequency_array, 0.5*np.sqrt(self.psd_array[chan])*(self.duration)**0.5, color='C3', label=r'$\frac{1}{2}\sqrt{S_n(f)T}$')
      
            ax.grid(True)
            ax.set_ylabel(r'Strain $[1/{\rm Hz}]$')
            ax.set_xlabel(r'Frequency [Hz]')
            ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig('{}/{}_{}{}_data_FD.png'.format(outdir, self.name, chan, self.TDI_generation))
            plt.close(fig)

        return None

    def plot_TD_data(self, contents=['signal', 'noise']):
        pass
        return None
    
    def save_data(self, outdir='.', label=None):
        '''
        TODO save the parameters of injected signals
        save the data in the instance to a .json file, save contents: [signals, noise, strains_FD, strains_TD, 
        frequency_array, psd_array]
        
        Parameters
        ==========
        outdir: string
        '''
        contents = ['signals', 'noise', 'strains_FD', 'strains_TD', 'frequency_array', 'psd_array']
        save_dict = {}
        for item in contents:
            save_dict[item] = getattr(self, item)

        filename = f'{outdir}/{self.name}_detector_data_{label}.hdf5'
        with h5py.File(filename, 'w') as file:
            recursively_save_dict_contents_to_group(file, '/', save_dict)

        return None
    
    def set_strains_FD_from_file(self, filename):
        '''
        TODO this function is incomplete !!! when use this func the frequency_array, psd_array, ... may not have the same shape 
        TODO consider the conflict with read-in data and already set data
        TODO add supportation of other attribute
        set the strains_FD from h5py file
        
        Parameters
        ==========
        filename: string
            hdf5 file containing the 'strains_FD'
        '''
        with h5py.File(filename, 'r') as file:
            data = recursively_load_dict_contents_from_group(file, '/')
        # TODO corresponding frequency_array, duration, cadance should be check
        self.strains_FD = data['strains_FD']
        return None

    def set_strains_TD_from_strains_FD(self):
        pass
        return None

















    # def get_frequency_domain_noise_strain():

    #     return {'LISA_A':, 'LISA_E':, 'LISA_T':}

    # def get_frequency_domain_noise_strain_with_injected_signal(self, parameters, frequencies=None):

    #     return {'LISA_A':, 'LISA_E':, 'LISA_T':}

    # def get_detector_response(self, waveform, parameters, frequencies=None):

    #     return {'LISA_A':, 'LISA_E':, 'LISA_T':}


# def IMRPhenomD_h22_Amp_Phase_tf(frequency_array, mass_1, mass_2, chi_1, chi_2,
#                                 luminosity_distance, iota, phi_ref, **kwargs):
#     '''return waveform h22 of model IMRPhenomD at given frequency points. 
    
#     Parameters:
#     ===========
#     frequency_array: 1D array_like
#         frequencies at which waveforms are evaluated
#     mass_1 (solar mass): float
#         The mass of the heavier object in solar masses
#     mass_2 (solar mass): float
#         The mass of the lighter object in solar masses
#     chi_1: float
#         primary spin magnitude in the direction perpendicular with orbital plane
#     chi_2: float
#         secondary spin magnitude in the direction perpendicular with orbital plane
#     luminosity_distance (Mpc): float
#         The luminosity distance in megaparsec
#     iota: float
#         Angle between the total binary angular momentum and the line of sight
#     phi_ref: float
#         The phase at reference frequency
#     **kwargs:
#         - reference_frequency
#         TODO: 
#         - waveform_approximant
#         - pn_spin_order
#         - pn_tidal_order
#         - pn_phase_order
#         - pn_amplitude_order
#         - modes_array

#     Returns:
#     ========
#     dict: A dictionary with Amp, phase, tf
#     '''
#     from lal import CreateDict, REAL8Vector, CreateREAL8Vector
#     import lalsimulation as lalsim

#     reference_frequency = kwargs['reference_frequency']
#     pn_spin_order=-1
#     pn_tidal_order=-1
#     pn_phase_order=-1
#     pn_amplitude_order=0
#     waveform_dictionary = CreateDict()

#     lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(
#         waveform_dictionary, int(pn_spin_order))
#     lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(
#         waveform_dictionary, int(pn_tidal_order))
#     lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
#         waveform_dictionary, int(pn_phase_order))
#     lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
#         waveform_dictionary, int(pn_amplitude_order))

#     for key, value in kwargs.items():
#         func = getattr(lalsim, "SimInspiralWaveformParamsInsert" + key, None)
#         if func is not None:
#             func(waveform_dictionary, value)

#     [mass_1, mass_2, chi_1, chi_2, luminosity_distance, iota, phi_ref, 
#         reference_frequency] = convert_args_list_to_float(mass_1, mass_2, 
#         chi_1, chi_2, luminosity_distance, iota, phi_ref, reference_frequency)

#     if not isinstance(frequency_array, REAL8Vector):
#         old_frequency_array = frequency_array.copy()
#         frequency_array = CreateREAL8Vector(len(old_frequency_array))
#         frequency_array.data = old_frequency_array

#     try:
#         amp, phase, tf = lalsim.SimIMRPhenomDFrequencySequenceh22AmpPhasetf(
#             frequency_array, mass_1, mass_2, chi_1, chi_2, luminosity_distance, 
#             iota, phi_ref, reference_frequency, waveform_dictionary)
#     except Exception as e:
#         EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
#         if EDOM:
#             failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
#                                      chi_1=chi_1, chi_2=chi_2,
#                                      luminosity_distance=luminosity_distance,
#                                      iota=iota, phi_ref=phi_ref)
#             logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
#                            "The parameters were {}\n".format(failed_parameters) +
#                            "Likelihood will be set to -inf.")
#             return None
#         else:
#             raise

#     return {'amp':amp.data.data, 'phase':phase.data.data, 'tf':tf.data.data}


# class LISA_Likelihood(Likelihood):

#     def __init__(self, channels, waveform_generator):
#         '''
#         Parameters
#         ==========
#         channels: list
#             A list of 3 channels (A, E, T) of LISA, contain FD data and 
#             power spectral densities, and can have method to compute response 
#             of given GW waveform.
#         waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
#             An object which computes the frequency-domain strain of the signal
#             given some set of parameters, as well as the t(f) array in order to 
#             compuate full response of space detectors.
#         '''
#         super(LISA_Likelihood, self).__init__()
#         self.channels = channels
#         self.waveform_generator = waveform_generator

#     def __repr__(self):
#         return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})' \
#             .format(self.channels, self.waveform_generator)

#     def log_likelihood(self):
#         '''
#         Calculates the real part of log-likelihood value

#         Returns
#         =======
#         float: The real part of the log likelihood
#         '''
#         log_l = 0
#         waveform_polarizations = \
#             self.waveform_generator.frequency_domain_strain(
#                 self.parameters.copy())
#         if waveform_polarizations is None:
#             return np.nan_to_num(-np.inf)
#         for interferometer in self.interferometers:
#             log_l += self.log_likelihood_interferometer(
#                 waveform_polarizations, interferometer)
#         return log_l.real
