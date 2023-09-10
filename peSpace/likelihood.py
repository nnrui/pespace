import copy

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import component_masses_to_chirp_mass

from .utilities import inner_product
from .constants import *


class FullLikelihood(Likelihood):

    def __init__(self, waveform_func, detector, neglect_waveform_errors=False):
        '''
        create a Fulllikelihood instance

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
        super(FullLikelihood, self).__init__(dict())
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
        the likelihood will be compute on the orginal frequency grid in FullLikelihood
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


class SparseLikelihood(Likelihood):

    def __init__(self, sparse_ratio, waveform_func, detector, neglect_waveform_errors=False):
        '''
        create a SparseLikelihood instance which simplily use the frequency array with delta_f*sparse_ratio
        when sparse_ratio=1, it will be same with FullLikelihood

        Parameters
        ==========
        spares_ratio: int
            when 1, use the same frequency_array of the detector
        wavefrom_func: function
            function to return waveform from parameters, see wavefrom.__dir__() for all support funcs
        detector: object
            see peSpace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
        neglect_waveform_errors: bool
            whether raise when failed to call wavefrom_func, raise if False
        '''
        super(SparseLikelihood, self).__init__(dict())
        self.waveform_func = waveform_func
        if detector.TDI_channels != ('A','E') and detector.TDI_channels != ('A','E','T'):
            raise Exception('Your set detector channels of {}, while the likelihood compution expect '
                            'the channels of ("A","E","T") or ("A","E")'.format(detector.TDI_channels))

        self.sparse_ratio = sparse_ratio
        origin_num = len(detector.frequency_array)
        sparse_ratio = int(sparse_ratio)
        sparse_index = range(0, origin_num, sparse_ratio)
        self.sparse_index = sparse_index

        self.detector = detector
        self.likelihood_frequency_array = self.create_likelihood_frequency_array()
        self.likelihood_psd_array = self.create_likelihood_psd_array()
        self.likehihodd_strains_FD = self.create_likelihood_strains_FD()
        # TODO: it maybe better to move the neglect_waveform_errors into a dict
        # TODO logically neglect_waveform_errors sould be attribure in wavefrom geration func
        self.neglect_waveform_errors = neglect_waveform_errors

    def create_likelihood_frequency_array(self):
        '''
        the likelihood will be compute on the sparsed frequency grid

        Returns
        =======
        array: frequency_arry on which the likelihood will be computed
        '''
        return self.detector.frequency_array[self.sparse_index]
    
    def create_likelihood_psd_array(self):
        '''
        create psd array on the sparsed frequency grid

        Returns
        =======
        array: same psd_array of the detector
        '''
        likelihood_psd_array = {}
        for chan in self.detector.TDI_channels:
            likelihood_psd_array[chan] = self.detector.psd_array[chan][self.sparse_index]
        return likelihood_psd_array

    def create_likelihood_strains_FD(self):
        '''
        return the strains_FD for likelihood computation, has the same shape with the likelihood_frequency_array

        Returns
        =======
        array: has the same shape with the likelihood_frequency_array
        '''
        likelihood_strains_FD = {}
        for chan in self.detector.TDI_channels:
            likelihood_strains_FD[chan] = self.detector.strains_FD[chan][self.sparse_index]
        return likelihood_strains_FD
    
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



class HeterodynedLikelihood(Likelihood):

    def __init__(self, waveform_func, detector, fiducial_parameters, FFT_points=4096, dT=3e5, neglect_waveform_errors=False):
        '''
        create a HeterodynedLikelihood instance, 
        reference: Cornish2020(https://doi.org/10.1103/PhysRevD.101.124008)

        Parameters
        ==========
        wavefrom_func: function
            function to return waveform from parameters, see wavefrom.__dir__() for all supported funcs
        detector: object
            see peSpace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
        fiducial_parameters: dict
            parameters describes the GW source
        FFT_points: integer
            control the number of points to compute the term rdh
        dT: float
            control the frequency spacing when computing the term dhdh
        neglect_waveform_errors: bool
            whether raise when failed to call wavefrom_func, raise if False
        '''
        super(HeterodynedLikelihood, self).__init__(dict())
        self.waveform_func = waveform_func
        if detector.TDI_channels != ('A','E') and detector.TDI_channels != ('A','E','T'):
            raise Exception('Your set detector channels of {}, while the likelihood compution expect '
                            'the channels of ("A","E","T") or ("A","E")'.format(detector.TDI_channels))
        self.detector = detector
        self.fiducial_parameters = fiducial_parameters
        self.FFT_points = FFT_points
        self.dT = dT
        self.neglect_waveform_errors = neglect_waveform_errors

        # self.fiducial_strains_FD = self.set_fiducial_strains_FD()
        # self.full_r_array = self.set_full_r_array()
        # self.term_rr = self.compute_term_rr()
        # self.rdh_precaculate_info = self.set_rdh_precaculate_info()

    def set_rdh_precaculate_info(self):
        '''
        Precaculate all array used in computation of the term r \delta h^*.
        The orignal frequency_array of detector will be saprsely selected according to the setting of `FFT_points`.
        The returned dict including the sparse_index and corresponding frequency_array, r_array, psd_array.

        Returns
        =======
        dict, 
        '''
        if self.FFT_points%2 != 0:
            self.FFT_points += 1
        num = int(self.FFT_points/2)
        length_full = self.detector.frequency_length
        part = int(length_full//num)
        index = [not bool(x%part) for x in range(length_full)]
        f_array = self.detector.frequency_array[index]
        delta_freq = f_array[1] - f_array[0]

        if DEBUG:
            waveform = self.waveform_func(f_array, self.parameters.copy(), self.neglect_waveform_errors)
            if waveform is None:
                return np.nan_to_num(-np.inf)
            GW_signals = self.detector.TDI_responses(waveform, self.parameters)

            waveform_0 = self.waveform_func(f_array, self.fiducial_parameters.copy(), self.neglect_waveform_errors)
            if waveform_0 is None:
                return np.nan_to_num(-np.inf)
            GW_signals_0 = self.detector.TDI_responses(waveform, self.fiducial_parameters)

            chan = 'A'
            h0 = GW_signals_0[chan]
            r = self.detector.strains_FD[chan][index] - h0
            dh = h0 - GW_signals[chan]
            rdh = delta_freq * np.vdot(r, dh/self.detector.psd_array[chan][index]).real
            return {'rdh': rdh,
                    'h0': h0,
                    'r': r,
                    'dh': dh,
                    'f_array': f_array}


        r_array = {}
        psd_array = {}
        h0_array = {}
        for chan in self.detector.TDI_channels:
            r_array[chan] = self.full_r_array[chan][index]
            psd_array[chan] = self.detector.psd_array[chan][index]
            h0_array[chan] = self.fiducial_strains_FD[chan][index]

        return {'rdh_sparse_index': index, 
                'rdh_frequency_array': f_array,
                'rdh_delta_freq' : delta_freq,
                'rdh_h0_array': h0_array,
                'rdh_r_array': r_array,
                'rdh_psd_array': psd_array}

    def set_full_r_array(self):
        '''
        r = d - h_0, at the full orignal frequency_array of `detector`

        Returns
        =======
        dict, contains the full_r_array of each channel
        '''
        r = {}
        for chan in self.detector.TDI_channels:
            r_chan = self.detector.strains_FD[chan] - self.fiducial_strains_FD[chan]
            r[chan] = r_chan        
        return r

    def set_fiducial_strains_FD(self):
        '''
        Computing the fiducial strains h_0.
        Compution is performed on the orignal full frequency grid of `detector` object.

        Returns
        =======
        dict, fiducial strains of each channles
        '''
        fiducial_waveform = self.waveform_func(self.detector.frequency_array, self.fiducial_parameters.copy(), self.neglect_waveform_errors)
        if fiducial_waveform is None:
            raise Exception(f'waveform function call failed with the inputed fiducial parameters: {self.fiducial_parameters}')
        fiducial_signals = self.detector.TDI_responses(fiducial_waveform, self.fiducial_parameters)
        return fiducial_signals

    def compute_term_rr(self):
        '''
        compute the term rr^* which is independent with h(f) and can be evaluated once and stored before sampling. 
        this term will be computed at the orignal full frequency grid in the `self.detector`.
        
        Returns
        =======
        dict: contains the term rr^* of each channel
        '''
        rr = {}
        for chan in self.detector.TDI_channels:
            r = self.full_r_array[chan]
            rr_chan = self.detector.delta_freq * np.vdot(r, r/self.detector.psd_array[chan]).real
            rr[chan] = rr_chan
        return rr
    
    def compute_term_rdh(self, channel, h):
        '''
        Computing the term (r \delta h^* + r^* \delta h)
        `h` is on the grid of `self.rdh_precaculate_info['rdh_frequency_array']`
        This function only for one TDI channel

        Parameters
        ==========
        channel: string
            the name of the TDI channel
        h: array
            strains of the TDI channel for the sampled parameters

        Returns
        =======
        dict, the term of rdh for each TDI channels
        '''
        dh = self.rdh_precaculate_info['rdh_h0_array'][channel] - h    
        rdh = self.rdh_precaculate_info['rdh_delta_freq'] * np.vdot(self.rdh_precaculate_info['rdh_r_array'][channel], dh/self.rdh_precaculate_info['rdh_psd_array'][channel]).real
        return rdh

    def compute_term_dhdh(self, channel, index_freq, h):
        '''
        Computing the term (\delta h \delta h^*)
        `h` is on a irregular frequency grid. `h` only for one TDI channel
        The integrand will be interpolate.

        Parameters
        ==========
        channel: string
            the name of the TDI channel
        index_freq: list
            list of bool, have the length of full frequency array, used to index the `self.detector.frequency_array`
        h: dict
            strains of the TDI channels for the sampled parameters

        Returns
        =======
        dict, the term of rdh for each TDI channels
        '''
        dh = self.fiducial_strains_FD[channel][index_freq] - h
        dhdh = np.abs(dh)**2
        dhdh_full = np.interp(self.detector.frequency_array, self.detector.frequency_array[index_freq], dhdh)
        integ_dhdh = self.detector.delta_freq * np.sum(dhdh_full / self.detector.psd_array[channel])
        return integ_dhdh

    def generate_dynamic_frequency_idx_dhdh(self):
        '''
        Generate the frequency index for computation of term dhdh.
        Returned list have the length of `self.detector.freqency_array`.
        Based on the Eq.(C10) in Cornish2020 (added the f**(11/3)).

        Returns
        =======
        list
        '''
        delta_f = self.detector.delta_freq
        idx = [False] * self.detector.frequency_length

        chirp_mass = component_masses_to_chirp_mass(self.parameters['mass_1'], self.parameters['mass_2'])
        total_mass = self.parameters['mass_1'] + self.parameters['mass_2']
        f_cut = Mf_CUT_PhenomD / (total_mass * MTSUN_SI)
        df_max = f_cut / 100      # following Cornish2020
        di_max = int(df_max//delta_f)

        i = 0
        while i < self.detector.frequency_length:
            idx[i] = True
            f = self.detector.frequency_array[i]
            df = self.dT * (96/5 * PI**(8/3) * (MTSUN_SI*chirp_mass)**(5/3) * f**(11/3))    # leading order of \dot{f}*dT
            if df < delta_f:
                i += 1
            elif df > df_max:
                i += di_max
            else:
                i += int(df//delta_f)
        idx[-1] = True

        return idx

    def log_likelihood(self):
        '''
        Calculates the real part of log-likelihood value

        Returns
        =======
        float: The real part of the log likelihood
        '''        
        idx_dhdh_full = np.array(self.generate_dynamic_frequency_idx_dhdh())
        idx_rdh_full = np.array(self.rdh_precaculate_info['rdh_sparse_index'])
        idx_full = idx_dhdh_full + idx_rdh_full
        idx_dhdh_signal = idx_dhdh_full[idx_full]
        idx_rdh_signal = idx_rdh_full[idx_full]
        # if DEBUG:
        #     print(len(idx_dhdh_full))
        #     print(len(self.rdh_precaculate_info['rdh_sparse_index']))
        #     print(len(idx_full))
        #     print(len(idx_dhdh_signal))
        #     print(len(idx_rdh_signal))
        #     print(np.sum(np.ones(self.detector.frequency_length)[idx_full]))
        #     print(np.sum(np.ones(self.detector.frequency_length)[idx_dhdh_full]))
        #     print(np.sum(np.ones(self.detector.frequency_length)[self.rdh_precaculate_info['rdh_sparse_index']]))
        #     print(len(self.rdh_precaculate_info['rdh_frequency_array']))
        #     print(idx_full[0:100])
        #     print(idx_dhdh_full[0:100])
        #     print(self.rdh_precaculate_info['rdh_sparse_index'][0:100])

        frequencies = self.detector.frequency_array[idx_full]
        waveform = self.waveform_func(frequencies, self.parameters.copy(), self.neglect_waveform_errors)
        if waveform is None:
            return np.nan_to_num(-np.inf)
        GW_signals = self.detector.TDI_responses(waveform, self.parameters)
        
        # if DEBUG:
        #     waveform_full = self.waveform_func(self.detector.frequency_array, self.parameters.copy(), self.neglect_waveform_errors)
        #     if waveform is None:
        #         return np.nan_to_num(-np.inf)
        #     GW_signals_full = self.detector.TDI_responses(waveform_full, self.parameters)
            
        #     frequencies_heterodyned = self.detector.frequency_array[idx_dhdh_full]
        #     dhdh = {}
        #     for chan in self.detector.TDI_channels:
        #         dh_heterodyned = self.fiducial_strains_FD[chan][idx_dhdh_full] - GW_signals[chan][idx_dhdh_signal]
        #         dhdh_heterodyned = np.abs(dh_heterodyned)**2 * frequencies_heterodyned / self.detector.psd_array[chan][idx_dhdh_full]

        #         dh_full = self.fiducial_strains_FD[chan] - GW_signals_full[chan]
        #         dhdh_full = np.abs(dh_full)**2 * self.detector.frequency_array / self.detector.psd_array[chan]

        #         dhdh[chan] = {'dhdh_heterodyned': dhdh_heterodyned,
        #                       'dhdh_full': dhdh_full}
            
        #     return {'dhdh': dhdh,
        #             'frequencies_heterodyned': frequencies_heterodyned}

        log_l = 0.0
        for chan in self.detector.TDI_channels:
            rdh_chan = self.compute_term_rdh(chan, GW_signals[chan][idx_rdh_signal])
            dhdh_chan = self.compute_term_dhdh(chan, idx_dhdh_full, GW_signals[chan][idx_dhdh_signal])
            log_l_chan = self.term_rr[chan] + dhdh_chan + 2*rdh_chan
            log_l += log_l_chan

        return -2*log_l
    



class MultibandLikelihood(Likelihood):
    '''
    Multiband method for accelerating likelihood evaluation
    Reference Morisaki2021(https://doi.org/10.1103/PhysRevD.104.044062)
    Based on the implementation in `bilby.gw.likelihood.multiband.MBGravitationalWaveTransient`
    '''

    def __init__(self, waveform_func, detector, highest_mode=2, neglect_waveform_errors=False, 
                 priors=None, minimum_band_duration=None, reference_chirp_mass=None, 
                 maximum_band_frequency=None, 
                 ):
        '''
        create a MultibandLikelihood instance

        Parameters
        ==========
        wavefrom_func: function
            function to return waveform from parameters, see wavefrom.__dir__() for all support funcs
        detector: object
            see peSpace.detectors for all supported detector class. for likelihood evaluation, the TDI channels
            must be set as ("A","E","T") or ("A","E").
        highest_mode: int
            The maximum magnetic number of gravitational-wave moments. Default is 2
        neglect_waveform_errors: bool
            whether raise when failed to call wavefrom_func, raise if False
        priors: dict or bilby.core.prior.PriorDict
            Used to determin the minimum_band_duration, reference_chirp_mass when they are not given. Need to contain `coalescence_time`, `chirp_mass`.
        minimum_band_duration: float
            T, T/2. T/4, ..., until less than minimum_band_duration
        reference_chirp_mass: float, in solar mass
            If not given, the minimum of `chirp_mass` prior will be used.
        maximum_band_frequency: float
            The upper limit on a starting frequency of a band, f^{B-1}
        '''
        super(MultibandLikelihood, self).__init__(dict())
        self.waveform_func = waveform_func
        if detector.TDI_channels != ('A','E') and detector.TDI_channels != ('A','E','T'):
            raise Exception('Your set detector channels of {}, while the likelihood compution expect '
                            'the channels of ("A","E","T") or ("A","E")'.format(detector.TDI_channels))
        self.detector = detector
        # TODO: it maybe better to move the neglect_waveform_errors into a dict
        # TODO logically neglect_waveform_errors sould be attribure in wavefrom geration func
        self.neglect_waveform_errors = neglect_waveform_errors
        self.highest_mode = highest_mode
        '''
        set minimum_band_duration
        determined by `coalescence_time` if not given
        '''
        if minimum_band_duration is not None:
            self.minimum_band_duration = minimum_band_duration
        elif priors is not None and 'coalescence_time' in priors:
            # neglect the time from heliocenter to detector
            self.minimum_band_duration = detector.start_time + detector.duration - priors['coalescence_time'].minimum
        else:
            raise Exception('can not set minimum_band_duration, either set minimum_band_duration '
                            'or pass in `priors` including `coalescence_time`.')      
        '''
        set reference_chirp_mass
        If not given, the minimum of `chirp_mass` prior will be used
        '''
        if reference_chirp_mass is not None:
            self.reference_chirp_mass = reference_chirp_mass
        elif priors is not None and 'chirp_mass' in priors:
            self.reference_chirp_mass = priors['chirp_mass'].minimum
        else:
            raise Exception('can not set reference_chirp_mass, either set reference_chirp_mass '
                            'or pass in `priors` including `chirp_mass`.')
        '''
        set maximum_band_frequency
        default balue is the frequency at which f - 1 / \sqrt(- d\tau / df) starts to decrease.
        The user-specified frequency is used if it is lower than that frequency.
        '''
        fmax = (15/968)**(3/5) * (highest_mode/(2*np.pi))**(8/5) / (self.reference_chirp_mass*MTSUN_SI)
        if maximum_band_frequency is None:
            self.maximum_band_frequency = fmax
        elif maximum_band_frequency > fmax:
            print(f'Warning: the given maximum_band_freqency is larger than default {fmax} Hz. '
                  f'It is set to be the default value {fmax} Hz')
            self.maximum_band_frequency = fmax
        else:
            self.maximum_band_frequency = maximum_band_frequency



        self.precaculate_info = self.set_precaculate_info()

        self.noise_log_likelihood_value = self.noise_log_likelihood()


    def set_precaculate_info(self):
        '''
        Computing quantities which could be done before sampling
        
        Returns:
        ========
        dict, have the keys band_durations: array, durations of each band, T^b;
                            band_frequencies: array, corresponding frequency of each band, f^b;
                            band_delta_f: array, length of roll-off of the window function of each band, \Delta f^b;
                            number_of_bands: int;

        '''
        
        precaculate_info = {}

        # Set frequency bands. 
        # Corresponding band duration is T, T/2. T/4, ..., until less than `minimum_band_duration`
        band_durations = [self.detector.duration]
        band_frequencies = [self.detector.minimum_frequency]
        band_delta_f = [0.0]
        dnext = self.detector.duration / 2
        while dnext > self.minimum_band_duration:
            fnow = band_frequencies[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, fnow)
            if fnext is not None and fnext < min(self.maximum_frequency, self.maximum_band_frequency):
                band_durations.append(dnext)
                band_frequencies.append(fnext)
                band_delta_f.append(dfnext)
                dnext /= 2
            else:
                break
        band_frequencies.append(self.detector.maximum_frequency + self.delta_f_end)
        band_delta_f.append(self.delta_f_end)
        precaculate_info['band_durations'] = np.array(band_durations)
        precaculate_info['band_frequencies'] = np.array(band_frequencies)
        precaculate_info['band_delta_f'] = np.array(band_delta_f)
        precaculate_info['number_of_bands'] = len(band_durations)
        print('The total frequency range is divided into {} bands with frequency intervals of {}.'.format(
            precaculate_info['number_of_bands'], ', '.join(["1/{} Hz".format(d) for d in band_durations])))

        


        
        
        return precaculate_info
    


    def log_likelihood(self):
        return self.log_likelihood_ratio() + self.noise_log_likelihood_value
    
    def noise_log_likelihood(self):
        noise_logL = 0
        for chan in self.detector.TDI_channels:
            noise_logL += -2 * self.detector.delta_freq * np.vdot(self.detector.strains_FD[chan], self.detector.strains_FD[chan]/self.detector.psd_array[chan])
        return float(np.real(noise_logL))
    
    def log_likelihood_ratio(self):
        waveform = self.waveform_func(self.likelihood_frequency_array, self.parameters.copy(), self.neglect_waveform_errors)
        if waveform is None:
            return np.nan_to_num(-np.inf)
        GW_signals = self.detector.TDI_responses(waveform, self.parameters)






        d_inner_h = 0.0
        for chan in self.detector.TDI_channels:
            d_inner_h += np.dot(GW_signals[chan], self.precaculate_info['dh_coeffs'][chan])






        log_l_ratio = -2 * (optimal_snr - 2*d_inner_h)

        return float(log_l_ratio)
