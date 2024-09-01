import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import time
from matplotlib import pyplot as plt
import bilby
import os
MTSUN_SI = 4.925490947641266978197229498498379006e-6
PI = 3.141592653589793
PC_SI = 3.085677581491367e+16

# local_rank_id = int(os.environ['MPI_LOCALRANKID'])
# gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
# selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
# os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu


num_tests = 20
minimum_frequency = 1e-5
maximum_frequency = 1e-1
cadence = 10
f_cut = 0.2
max_mass = maximum_frequency/f_cut/MTSUN_SI
# print(max_mass)

duration = 31536000    # 1 year observation
cadence = 10
TDI_chans = ("A", "E", "T")
TDI_gen = '1.5'
inj_parameters = dict(
    total_mass = 3089053.9,
    mass_ratio = 0.11,
    chi_1 = 0.3986046314480332,
    chi_2 = 0.5465882372512956,
    luminosity_distance = 60000.42017466175677,
    inclination = 0.30782038099413395,
    reference_phase = 0.0,
    coalescence_time = 0.0,
    ecliptic_latitude = -0.5256036732051035,
    ecliptic_longitude = 1.1637,
    polarization = 1.30782038099413395,
)

rng = np.random.default_rng()
parameters = {}
parameters['total_mass'] = rng.uniform(1e3, max_mass, num_tests)
parameters['mass_ratio'] = rng.uniform(0.2, 1.0, num_tests)
parameters['chi_1'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['chi_2'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['luminosity_distance'] = rng.uniform(1000.0, 10000.0, num_tests)
parameters['inclination'] = rng.uniform(0, np.pi, num_tests)
parameters['reference_phase'] = rng.uniform(0, 2*np.pi, num_tests)
parameters['ecliptic_longitude'] = np.ones(num_tests) * inj_parameters["ecliptic_longitude"]
parameters['ecliptic_latitude'] = np.ones(num_tests) * inj_parameters["ecliptic_latitude"]
parameters['polarization'] = np.ones(num_tests) * inj_parameters["polarization"]
parameters['coalescence_time'] = rng.uniform(-3600*24, -3600*24, num_tests)
parameters['log_likelihood'] = np.zeros(num_tests)
# parameters['luminosity_distance'] = rng.uniform(1000.0, 10000.0, num_tests)
# parameters['inclination'] = rng.uniform(0, np.pi, num_tests)
# parameters['reference_phase'] = rng.uniform(0, 2*np.pi, num_tests)
# parameters['ecliptic_longitude'] = rng.uniform(-PI, PI, num_tests)
# parameters['ecliptic_latitude'] = rng.uniform(-PI/2, PI/2, num_tests)
# parameters['polarization'] = rng.uniform(0, 2*PI, num_tests)
# parameters['coalescence_time'] = np.zeros(num_tests)
parameters = bilby.gw.conversion.generate_mass_parameters(parameters)
parameters = pd.DataFrame(parameters)


import sys 
import sys 
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/pespace")
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/tiwave")
sys.path.append('/home/hydrogen/workspace/Space_GW/pespace')
sys.path.append('/home/hydrogen/workspace/Space_GW/tiwave')
import taichi as ti
ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False)
import numpy as np
from matplotlib import pyplot as plt


from pespace.detectors import TDIChannelsData, SpaceborneInterferometer
from pespace.noise import available_noise_models
from pespace.likelihood import FrequencyDomainLikelihood
from pespace.orbits import KaplerianHeliocentric
from tiwave.waveforms import IMRPhenomD


mbhb = TDIChannelsData(label="injection_individual_MBHB")
mbhb.set_frequency_domain_data_with_zero_value(channels=TDI_chans, generation=TDI_gen, duration=duration, cadence=cadence)
mbhb.set_frequency_domain_noise_power_density_from_model(available_noise_models['LISA_SciRDv1'])
noise_realization = mbhb.generate_realization_from_frequency_domain_noise_power_density()
mbhb.add_into_frequency_domian_data(noise_realization)
lisa = SpaceborneInterferometer(name='LISA', TDI_data=mbhb, orbit=KaplerianHeliocentric(2.5e9, 0.0, 0.0))
lisa.initialize_response_container_in_frequency_domain()

wf = IMRPhenomD(lisa.TDI_data.frequency_samples)
wf.update_waveform(inj_parameters)
lisa.inject_frequency_domain_signal(wf.waveform_container, inj_parameters['ecliptic_longitude'], inj_parameters['ecliptic_latitude'], inj_parameters['polarization'])

likelihood = FrequencyDomainLikelihood(wf, lisa)
st = time.perf_counter()
for i in range(num_tests):
    likelihood.parameters.update(parameters.iloc[i])
    parameters["log_likelihood"][i] = likelihood.log_likelihood()
ed = time.perf_counter()
print('ti time consuming: ', (ed-st)/num_tests)

print(parameters)

# import taichi as ti
# ti.init(arch=ti.cuda, default_fp=ti.f64, device_memory_fraction=0.9, kernel_profiler=True)



# det15 = LISALike(name='LISA', duration=3600*24*30*12*5, cadence=cadence, minimum_frequency=minimum_frequency, 
#                  maximum_frequency=maximum_frequency, TDI_channels=('A', 'E', 'T'), 
#                  TDI_generation='1.5')
# wf15 = ti_IMRPhenomD(det15.frequencies, det15.waveform_container)
# det20 = LISALike(name='LISA', duration=3600*24*30*12*5, cadence=cadence, minimum_frequency=minimum_frequency, 
#                  maximum_frequency=maximum_frequency, TDI_channels=('A', 'E', 'T'), 
#                  TDI_generation='2.0')
# wf20 = ti_IMRPhenomD(det20.frequencies, det20.waveform_container)


# det15.set_PSDs_from_noise_model()
# det15.inject_noise_FD_realization_from_psd()
# det15.inject_signal_FD(parameters.iloc[0], wf15)
# likelihood15 = FullLikelihood(wf15, det15)
# det20.set_PSDs_from_noise_model()
# det20.inject_noise_FD_realization_from_psd()
# det20.inject_signal_FD(parameters.iloc[0], wf20)
# likelihood20 = FullLikelihood(wf20, det20)
# likelihood15.parameters.update(parameters.iloc[1])
# likelihood15.log_likelihood()
# likelihood20.parameters.update(parameters.iloc[1])
# likelihood20.log_likelihood()
# st = time.perf_counter()
# likelihood15.parameters.update(parameters.iloc[1])
# likelihood15.log_likelihood()
# likelihood20.parameters.update(parameters.iloc[1])
# likelihood20.log_likelihood()
# ed = time.perf_counter()
# print('ti time consuming: ', (ed-st))

# ti.profiler.print_kernel_profiler_info()


# for p in powers_of_2:
#     duration = 2**p
#     det15 = LISALike(name='LISA', duration=duration, cadence=cadence, minimum_frequency=minimum_frequency, 
#                      maximum_frequency=maximum_frequency, TDI_channels=('A', 'E'), 
#                      TDI_generation='1.5')
#     wf15 = ti_IMRPhenomD(det15.frequencies, det15.waveform_container)

#     det15.set_PSDs_from_noise_model()
#     det15.inject_noise_FD_realization_from_psd()
#     det15.inject_signal_FD(parameters.iloc[0], wf15)

#     likelihood = FullLikelihood(wf15, det15)

#     st = time.perf_counter()
#     for i in range(num_tests):
#         likelihood.parameters.update(parameters.iloc[i])
#         likelihood.log_likelihood()
#     ed = time.perf_counter()
#     print('ti time consuming: ', (ed-st)/num_tests)

# ######## generate data
# from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
# from bbhx.waveformbuild import BBHWaveformFD
# from bbhx.likelihood import Likelihood as bbhx_Likelihood
# waveform_keywords = dict(modes=[(2,2)], direct=True)
# bbhx_wf = BBHWaveformFD(use_gpu=True)

# for p in powers_of_2:
#     duration = 2**p
#     f_array = np.arange(0, 1.0/(2*cadence), 1.0/duration)
#     bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
#     f_array = f_array[bound]
#     data_channels = bbhx_wf(parameters.iloc[0]['mass_1'],
#                             parameters.iloc[0]['mass_2'],
#                             parameters.iloc[0]['chi_1'],
#                             parameters.iloc[0]['chi_2'],
#                             parameters.iloc[0]['luminosity_distance']*1e6*PC_SI,
#                             0.0,
#                             0.0,
#                             parameters.iloc[0]['inclination'],
#                             parameters.iloc[0]['ecliptic_longitude'],
#                             parameters.iloc[0]['ecliptic_latitude'],
#                             parameters.iloc[0]['polarization'],
#                             parameters.iloc[0]['coalescence_time'],
#                             length=len(f_array),
#                             freqs=f_array,
#                             **waveform_keywords)[0]
#     PSD_A = np.ones(len(f_array))
#     PSD_E = np.ones(len(f_array))
#     PSD_T = np.ones(len(f_array))
#     psd = np.array([PSD_A, PSD_E, PSD_T])
#     like = bbhx_Likelihood(bbhx_wf,
#                            f_array,
#                            data_channels,
#                            psd,
#                            use_gpu=True)
    
#     st = time.perf_counter()
#     for i in range(num_tests):
#         params =np.array([parameters.iloc[i]['mass_1'],
#                 parameters.iloc[i]['mass_2'],
#                 parameters.iloc[i]['chi_1'],
#                 parameters.iloc[i]['chi_2'],
#                 parameters.iloc[i]['luminosity_distance']*1e6*PC_SI,
#                 0.0,
#                 0.0,
#                 parameters.iloc[0]['inclination'],
#                 parameters.iloc[0]['ecliptic_longitude'],
#                 parameters.iloc[0]['ecliptic_latitude'],
#                 parameters.iloc[0]['polarization'],
#                 parameters.iloc[i]['coalescence_time']
#                 ])
        
#         like.get_ll(params, direct=True)
#     ed = time.perf_counter()
#     time_consuming = (ed - st)/num_tests
#     print(f'bbhx {p}th, time:{time_consuming}')


# print(det.frequencies)
# print(det.TDI_data)
# print(det.waveform_container)
# print('waveform time consuming: ', ed1-st)
# print('response time consuming: ', ed2-ed1)

# TDI_X15_array = np.zeros(det15.data_length)
# TDI_Y15_array = np.zeros(det15.data_length)
# TDI_Z15_array = np.zeros(det15.data_length)
# TDI_A15_array = np.zeros(det15.data_length)
# TDI_E15_array = np.zeros(det15.data_length)
# TDI_T15_array = np.zeros(det15.data_length)

# TDI_X20_array = np.zeros(det20.data_length)
# TDI_Y20_array = np.zeros(det20.data_length)
# TDI_Z20_array = np.zeros(det20.data_length)
# TDI_A20_array = np.zeros(det20.data_length)
# TDI_E20_array = np.zeros(det20.data_length)
# TDI_T20_array = np.zeros(det20.data_length)

# for i in range(det15.data_length):
#     TDI_X15_array[i] = (det15.TDI_data[i].channels_data.X.norm())
#     TDI_Y15_array[i] = (det15.TDI_data[i].channels_data.Y.norm())
#     TDI_Z15_array[i] = (det15.TDI_data[i].channels_data.Z.norm())
#     TDI_A15_array[i] = (det15.TDI_data[i].channels_data.A.norm())
#     TDI_E15_array[i] = (det15.TDI_data[i].channels_data.E.norm())
#     TDI_T15_array[i] = (det15.TDI_data[i].channels_data.T.norm())
# for i in range(det20.data_length):
#     TDI_X20_array[i] = (det20.TDI_data[i].channels_data.X.norm())
#     TDI_Y20_array[i] = (det20.TDI_data[i].channels_data.Y.norm())
#     TDI_Z20_array[i] = (det20.TDI_data[i].channels_data.Z.norm())
#     TDI_A20_array[i] = (det20.TDI_data[i].channels_data.A.norm())
#     TDI_E20_array[i] = (det20.TDI_data[i].channels_data.E.norm())
#     TDI_T20_array[i] = (det20.TDI_data[i].channels_data.T.norm())

# ti_strain = {'X15': TDI_X15_array,
#              'Y15': TDI_Y15_array,
#              'Z15': TDI_Z15_array,
#              'A15': TDI_A15_array,
#              'E15': TDI_E15_array,
#              'T15': TDI_T15_array,
#              'X20': TDI_X20_array,
#              'Y20': TDI_Y20_array,
#              'Z20': TDI_Z20_array,
#              'A20': TDI_A20_array,
#              'E20': TDI_E20_array,
#              'T20': TDI_T20_array
#             }



# from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf as pyIMRPhenomD_h22_Amplitude_Phase_tf
# from peSpace.detectors import LISALike as pyLISALike

# pydet15 = pyLISALike(name='LISA', duration=864000, cadence=10, minimum_frequency=1e-4, 
#                maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
#                TDI_generation='1.5')
# pydet20 = pyLISALike(name='LISA', duration=864000, cadence=10, minimum_frequency=1e-4, 
#                maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
#                TDI_generation='2.0')

# st = time.perf_counter()
# pydet15.inject_signal_FD(parameters, pyIMRPhenomD_h22_Amplitude_Phase_tf)
# pydet20.inject_signal_FD(parameters, pyIMRPhenomD_h22_Amplitude_Phase_tf)
# ed = time.perf_counter()
# print('py time consuming: ', ed-st)

# for chan in ['X', 'Y', 'Z', 'A', 'E', 'T']:
#     fig, ax = plt.subplots()
#     ax.loglog(det15.frequencies, ti_strain[f'{chan}15'], label=f'ti_{chan}_15')
#     ax.loglog(pydet15.frequency_array,  np.abs(pydet15.signals[chan]), label=f'py_{chan}_15', linestyle='dashed')
#     ax.loglog(det20.frequencies, ti_strain[f'{chan}20'], label=f'ti_{chan}_20')
#     ax.loglog(pydet20.frequency_array,  np.abs(pydet20.signals[chan]), label=f'py_{chan}_20', linestyle='dashed')
#     ax.legend()
#     ax.set_title(chan)
#     fig.savefig(f'{chan}_norm.png')
#     plt.close()



# ####################################################################################
# TDI_X15_array = np.zeros(det15.data_length)
# TDI_Y15_array = np.zeros(det15.data_length)
# TDI_Z15_array = np.zeros(det15.data_length)
# TDI_A15_array = np.zeros(det15.data_length)
# TDI_E15_array = np.zeros(det15.data_length)
# TDI_T15_array = np.zeros(det15.data_length)

# TDI_X20_array = np.zeros(det20.data_length)
# TDI_Y20_array = np.zeros(det20.data_length)
# TDI_Z20_array = np.zeros(det20.data_length)
# TDI_A20_array = np.zeros(det20.data_length)
# TDI_E20_array = np.zeros(det20.data_length)
# TDI_T20_array = np.zeros(det20.data_length)

# for i in range(det15.data_length):
#     TDI_X15_array[i] = (det15.TDI_data[i].channels_data.X[0])
#     TDI_Y15_array[i] = (det15.TDI_data[i].channels_data.Y[0])
#     TDI_Z15_array[i] = (det15.TDI_data[i].channels_data.Z[0])
#     TDI_A15_array[i] = (det15.TDI_data[i].channels_data.A[0])
#     TDI_E15_array[i] = (det15.TDI_data[i].channels_data.E[0])
#     TDI_T15_array[i] = (det15.TDI_data[i].channels_data.T[0])
# for i in range(det20.data_length):
#     TDI_X20_array[i] = (det20.TDI_data[i].channels_data.X[0])
#     TDI_Y20_array[i] = (det20.TDI_data[i].channels_data.Y[0])
#     TDI_Z20_array[i] = (det20.TDI_data[i].channels_data.Z[0])
#     TDI_A20_array[i] = (det20.TDI_data[i].channels_data.A[0])
#     TDI_E20_array[i] = (det20.TDI_data[i].channels_data.E[0])
#     TDI_T20_array[i] = (det20.TDI_data[i].channels_data.T[0])

# ti_strain = {'X15': TDI_X15_array,
#              'Y15': TDI_Y15_array,
#              'Z15': TDI_Z15_array,
#              'A15': TDI_A15_array,
#              'E15': TDI_E15_array,
#              'T15': TDI_T15_array,
#              'X20': TDI_X20_array,
#              'Y20': TDI_Y20_array,
#              'Z20': TDI_Z20_array,
#              'A20': TDI_A20_array,
#              'E20': TDI_E20_array,
#              'T20': TDI_T20_array
#             }

# for chan in ['X', 'Y', 'Z', 'A', 'E', 'T']:
#     fig, ax = plt.subplots()
#     ax.semilogx(det15.frequencies, ti_strain[f'{chan}15'], label=f'ti_{chan}_15')
#     ax.semilogx(pydet15.frequency_array,  (pydet15.signals[chan]).real, label=f'py_{chan}_15', linestyle='dashed')
#     ax.semilogx(det20.frequencies, ti_strain[f'{chan}20'], label=f'ti_{chan}_20')
#     ax.semilogx(pydet20.frequency_array,  (pydet20.signals[chan]).real, label=f'py_{chan}_20', linestyle='dashed')
#     ax.legend()
#     ax.set_title(chan)
#     fig.savefig(f'{chan}_real.png')
#     plt.close()


# ####################################################################################
# TDI_X15_array = np.zeros(det15.data_length)
# TDI_Y15_array = np.zeros(det15.data_length)
# TDI_Z15_array = np.zeros(det15.data_length)
# TDI_A15_array = np.zeros(det15.data_length)
# TDI_E15_array = np.zeros(det15.data_length)
# TDI_T15_array = np.zeros(det15.data_length)

# TDI_X20_array = np.zeros(det20.data_length)
# TDI_Y20_array = np.zeros(det20.data_length)
# TDI_Z20_array = np.zeros(det20.data_length)
# TDI_A20_array = np.zeros(det20.data_length)
# TDI_E20_array = np.zeros(det20.data_length)
# TDI_T20_array = np.zeros(det20.data_length)

# for i in range(det15.data_length):
#     TDI_X15_array[i] = (det15.TDI_data[i].channels_data.X[1])
#     TDI_Y15_array[i] = (det15.TDI_data[i].channels_data.Y[1])
#     TDI_Z15_array[i] = (det15.TDI_data[i].channels_data.Z[1])
#     TDI_A15_array[i] = (det15.TDI_data[i].channels_data.A[1])
#     TDI_E15_array[i] = (det15.TDI_data[i].channels_data.E[1])
#     TDI_T15_array[i] = (det15.TDI_data[i].channels_data.T[1])
# for i in range(det20.data_length):
#     TDI_X20_array[i] = (det20.TDI_data[i].channels_data.X[1])
#     TDI_Y20_array[i] = (det20.TDI_data[i].channels_data.Y[1])
#     TDI_Z20_array[i] = (det20.TDI_data[i].channels_data.Z[1])
#     TDI_A20_array[i] = (det20.TDI_data[i].channels_data.A[1])
#     TDI_E20_array[i] = (det20.TDI_data[i].channels_data.E[1])
#     TDI_T20_array[i] = (det20.TDI_data[i].channels_data.T[1])

# ti_strain = {'X15': TDI_X15_array,
#              'Y15': TDI_Y15_array,
#              'Z15': TDI_Z15_array,
#              'A15': TDI_A15_array,
#              'E15': TDI_E15_array,
#              'T15': TDI_T15_array,
#              'X20': TDI_X20_array,
#              'Y20': TDI_Y20_array,
#              'Z20': TDI_Z20_array,
#              'A20': TDI_A20_array,
#              'E20': TDI_E20_array,
#              'T20': TDI_T20_array
#             }

# for chan in ['X', 'Y', 'Z', 'A', 'E', 'T']:
#     fig, ax = plt.subplots()
#     ax.semilogx(det15.frequencies, ti_strain[f'{chan}15'], label=f'ti_{chan}_15')
#     ax.semilogx(pydet15.frequency_array,  (pydet15.signals[chan]).imag, label=f'py_{chan}_15', linestyle='dashed')
#     ax.semilogx(det20.frequencies, ti_strain[f'{chan}20'], label=f'ti_{chan}_20')
#     ax.semilogx(pydet20.frequency_array,  (pydet20.signals[chan]).imag, label=f'py_{chan}_20', linestyle='dashed')
#     ax.legend()
#     ax.set_title(chan)
#     fig.savefig(f'{chan}_imag.png')
#     plt.close()


# ####################################################################################



