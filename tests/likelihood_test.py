import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import bilby
MTSUN_SI = 4.925490947641266978197229498498379006e-6
PI = 3.141592653589793
PC_SI = 3.085677581491367e+16


import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/pespace')
sys.path.append('/home/hydrogen/workspace/Space_GW/wf4ti')
sys.path.append('/home/changfenggroup/nrui/works/codes/gw_space/pespace')
sys.path.append('/home/changfenggroup/nrui/works/codes/gw_space/wf4ti')


powers_of_2 = range(10, 12)
num_tests = 10

time_consuming_bbhx = []
time_consuming_pespace = []
data_length_list = []

minimum_frequency = 1e-5
maximum_frequency = 1e-1
cadence = 5
f_cut = 0.2
max_mass = maximum_frequency/f_cut/MTSUN_SI
# print(max_mass)

rng = np.random.default_rng()
parameters = {}
parameters['total_mass'] = rng.uniform(1e3, max_mass, num_tests)
parameters['mass_ratio'] = rng.uniform(0.2, 1.0, num_tests)
parameters['chi_1'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['chi_2'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['luminosity_distance'] = rng.uniform(1000.0, 10000.0, num_tests)
parameters['inclination'] = rng.uniform(0, np.pi, num_tests)
parameters['reference_phase'] = rng.uniform(0, 2*np.pi, num_tests)
parameters['ecliptic_longitude'] = rng.uniform(-PI, PI, num_tests)
parameters['ecliptic_latitude'] = rng.uniform(-PI/2, PI/2, num_tests)
parameters['polarization'] = rng.uniform(0, 2*PI, num_tests)
parameters['coalescence_time'] = np.zeros(num_tests)
parameters = bilby.gw.conversion.generate_mass_parameters(parameters)
parameters = pd.DataFrame(parameters)

######################################################################################
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.likelihood import Likelihood 
from bbhx.utils.constants import *

# duration = 3600*24
# n = int(duration / cadence)
# f_array = np.fft.rfftfreq(n, cadence)
# bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
# f_array = f_array[bound]
# # print(len(f_array))
# # print(f_array)
# # f_array = np.arange(0, 1.0/(2*cadence), 1.0/duration)
# # bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
# # f_array = f_array[bound]
# # print(len(f_array))
# # print(f_array)
# waveform_kwargs = dict(modes=[(2,2)], direct=True, length=len(f_array))
# wave_gen = BBHWaveformFD(amp_phase_kwargs={'run_phenomd': True,
#                                           }, 
#                         response_kwargs={'TDItag':'AET', 
#                                          }, 
#                         interp_kwargs={},
#                         use_gpu=True)
# data_channels = wave_gen(parameters.iloc[0]['mass_1'],
#                          parameters.iloc[0]['mass_2'],
#                          parameters.iloc[0]['chi_1'],
#                          parameters.iloc[0]['chi_2'],
#                          parameters.iloc[0]['luminosity_distance']*1e6*PC_SI,
#                          0.0,
#                          f_array[0],
#                          parameters.iloc[0]['inclination'],
#                          parameters.iloc[0]['ecliptic_longitude'],
#                          parameters.iloc[0]['ecliptic_latitude'],
#                          parameters.iloc[0]['polarization'],
#                          0.0,
#                          freqs=f_array,
#                          direct=True)
# PSD_A = np.ones(len(f_array))
# PSD_E = np.ones(len(f_array))
# PSD_T = np.ones(len(f_array))
# psd = np.array([PSD_A, PSD_E, PSD_T])

# like = Likelihood(wave_gen,
#                   f_array,
#                   data_channels,
#                   psd,
#                   use_gpu=True
#                   )
# print(like)
# params =[parameters.iloc[0]['mass_1'],
#                   parameters.iloc[0]['mass_2'],
#                   parameters.iloc[0]['chi_1'],
#                   parameters.iloc[0]['chi_2'],
#                   parameters.iloc[0]['luminosity_distance']*1e6*PC_SI,
#                   0.0,
#                   f_array[0],
#                   parameters.iloc[0]['inclination'],
#                   parameters.iloc[0]['ecliptic_longitude'],
#                   parameters.iloc[0]['ecliptic_latitude'],
#                   parameters.iloc[0]['polarization'],
#                   0.0,]
# st = time.perf_counter()
# print(st)
# ll = like.get_ll(params, **waveform_kwargs)
# print(ll)
# ed = time.perf_counter()
# print(ed)
# time_consuming = (ed - st)/num_tests
# print(f'bbhx, time:{time_consuming}')

######## generate data
# set parameters
f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 0.0 # phase at f_ref
m1 = 1e6
m2 = 5e5
a1 = 0.2
a2 = 0.4
dist = 18e3  * PC_SI * 1e6 # 3e3 in Mpc
inc = np.pi/3.
beta = np.pi/4.  # ecliptic latitude
lam = np.pi/5.  # ecliptic longitude
psi = np.pi/6.  # polarization angle
t_ref = 1.0 * YRSID_SI  # t_ref  (in the SSB reference frame)

T_obs = 1.2 # years
dt = 10.0

n = int(T_obs * YRSID_SI / dt)
data_freqs = np.fft.rfftfreq(n, dt)[1:] # remove DC

# frequencies to interpolate to
modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
waveform_kwargs = dict(modes=modes, direct=False, fill=True, squeeze=True, length=1024)
wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
data_channels = wave_gen(m1, m2, a1, a2,
                          dist, phi_ref, f_ref, inc, lam,
                          beta, psi, t_ref, freqs=data_freqs,
                          **waveform_kwargs)[0]

######## get noise information (need lisatools)
PSD_A = np.ones(len(data_freqs))
PSD_E = np.ones(len(data_freqs))
PSD_T = np.ones(len(data_freqs))
df = data_freqs[1] - data_freqs[0]

psd = np.array([PSD_A, PSD_E, PSD_T])
df = data_freqs[1] - data_freqs[0]

psd = np.array([PSD_A, PSD_E, PSD_T])

# initialize Likelihood
like = Likelihood(
    wave_gen,
    data_freqs,
    data_channels,
    psd,
    use_gpu=True,
)

# get params
num_bins = 10
params_in = np.tile(np.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref]), (num_bins, 1))

# change masses for test
params_in[:, 0] *= (1 + 1e-4 * np.random.randn(num_bins))

# get_ll and not __call__ to work with lisatools
st = time.perf_counter()
ll = like.get_ll(params_in.T, **waveform_kwargs)
ed = time.perf_counter()
time_consuming = (ed - st)/num_tests
print(f'bbhx, time:{time_consuming}')
print(ll, like.d_h)





# bbhx_wf = BBHWaveformFD(amp_phase_kwargs={'run_phenomd': True,
#                                           }, 
#                         response_kwargs={'TDItag':'AET', 
#                                          }, 
#                         interp_kwargs={},
#                         use_gpu=True)

# # duration = 3600*24*30*12*5
# duration = 3600
# f_array = np.arange(0, 1.0/(2*cadence), 1.0/duration)
# bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
# f_array = f_array[bound]
# data_channels = bbhx_wf(parameters.iloc[0]['mass_1'],
#                         parameters.iloc[0]['mass_2'],
#                         parameters.iloc[0]['chi_1'],
#                         parameters.iloc[0]['chi_2'],
#                         parameters.iloc[0]['luminosity_distance']*1e6*PC_SI,
#                         0.0,
#                         f_array[0],
#                         parameters.iloc[0]['inclination'],
#                         parameters.iloc[0]['ecliptic_longitude'],
#                         parameters.iloc[0]['ecliptic_latitude'],
#                         parameters.iloc[0]['polarization'],
#                         0.0,
#                         freqs=f_array,
#                         length=len(f_array),
#                         direct=True)[0]
# # print(type(data_channels))
# # print(data_channels)

# PSD_A = np.ones(len(f_array))
# PSD_E = np.ones(len(f_array))
# PSD_T = np.ones(len(f_array))
# psd = np.array([PSD_A, PSD_E, PSD_T])
# like = bbhx_Likelihood(bbhx_wf,
#                        f_array,
#                        data_channels,
#                        psd,
#                        use_gpu=True)


# params =np.array([parameters.iloc[0]['mass_1'],
#                   parameters.iloc[0]['mass_2'],
#                   parameters.iloc[0]['chi_1'],
#                   parameters.iloc[0]['chi_2'],
#                   parameters.iloc[0]['luminosity_distance']*1e6*PC_SI,
#                   0.0,
#                   f_array[0],
#                   parameters.iloc[0]['inclination'],
#                   parameters.iloc[0]['ecliptic_longitude'],
#                   parameters.iloc[0]['ecliptic_latitude'],
#                   parameters.iloc[0]['polarization'],
#                   0.0
#                  ])
# print(bbhx_wf(parameters.iloc[0]['mass_1'],
#                         parameters.iloc[0]['mass_2'],
#                         parameters.iloc[0]['chi_1'],
#                         parameters.iloc[0]['chi_2'],
#                         parameters.iloc[0]['luminosity_distance']*1e6*PC_SI,
#                         0.0,
#                         f_array[0],
#                         parameters.iloc[0]['inclination'],
#                         parameters.iloc[0]['ecliptic_longitude'],
#                         parameters.iloc[0]['ecliptic_latitude'],
#                         parameters.iloc[0]['polarization'],
#                         0.0,
#                         freqs=f_array,
#                         length=len(f_array),
#                         direct=True))
# print(like.get_ll(params, length=len(f_array), modes=[(2,2)]))

# st = time.perf_counter()
# for i in range(num_tests):
#     params =np.array([parameters.iloc[i]['mass_1'],
#             parameters.iloc[i]['mass_2'],
#             parameters.iloc[i]['chi_1'],
#             parameters.iloc[i]['chi_2'],
#             parameters.iloc[i]['luminosity_distance']*1e6*PC_SI,
#             0.0,
#             0.0,
#             parameters.iloc[0]['inclination'],
#             parameters.iloc[0]['ecliptic_longitude'],
#             parameters.iloc[0]['ecliptic_latitude'],
#             parameters.iloc[0]['polarization'],
#             parameters.iloc[i]['coalescence_time']
#             ])
    
#     like.get_ll(params, direct=True)
# ed = time.perf_counter()
# time_consuming = (ed - st)/num_tests
# print(f'bbhx {p}th, time:{time_consuming}')



# from ti_pespace.detectors import LISALike
# from ti_pespace.likelihood import FullLikelihood
# from wf4ti.waveforms.IMRPhenomD import IMRPhenomD as ti_IMRPhenomD
# from wf4ti.constants import *

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



