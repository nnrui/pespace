import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')
import time
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt

from peSpace.detectors import LISALike
from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from peSpace.likelihood import FullLikelihood, HeterodynedLikelihood

det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=0.1, TDI_channels=('A', 'E', 'T'), 
               TDI_generation='1.5')

parameters = dict(mass_1 = 1e6,
                  mass_2 = 5e5,
                  luminosity_distance=36594.3,
                  chi_1 = 0.2,
                  chi_2 = 0.4,
                  coalescence_time=0.0,
                  ecliptic_longitude = 3.335,
                  ecliptic_latitude = 1.468,
                  polarization = 2.237,
                  inclination = 3.14/3.0,
                  coalescence_phase = 0.0,)

det.generate_FD_noise_realization_from_psd(seed=0)
det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
print(det.optimal_snr())
det.save_data('/home/hydrogen/workspace/Space_GW/peSpace/data/')

det.set_strains_FD_from_file('/home/hydrogen/workspace/Space_GW/peSpace/data/LISA_detector_data_None.hdf5')




rng = np.random.default_rng(seed=0)
parameters_2 = parameters.copy()
# for key in ['mass_1', 'mass_2', 'luminosity_distance', 'chi_1', 'chi_2', 'coalescence_time', 'ecliptic_longitude', 'ecliptic_latitude', 'polarization', 'inclination', 'coalescence_phase']:
# for key in ['mass_1', 'mass_2',                          'chi_1', 'chi_2',                      'ecliptic_longitude', 'ecliptic_latitude', 'polarization', 'inclination']:
# for key in [                                                                                      'ecliptic_longitude', 'ecliptic_latitude',                              ]:
# for key in ['mass_1', 'mass_2',                          'chi_1', 'chi_2',                                                                                       ]:
for key in ['mass_1', 'mass_2',                                                                                                                                  ]:
    parameters_2[key] = parameters[key] + rng.uniform(-0.1, 0.1)*parameters[key]
parameters_3 = parameters.copy()
for key in ['mass_1', 'mass_2',                          'chi_1', 'chi_2',                      'ecliptic_longitude', 'ecliptic_latitude', 'polarization', 'inclination']:
    parameters_3[key] = parameters[key] + rng.uniform(-0.1, 0.1)*parameters[key]


def rdh_for_map(tau):
    likelihood_heterodyned = HeterodynedLikelihood(waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
                                   detector=det,
                                   fiducial_parameters=parameters_2,
                                   FFT_points=tau
                                #    dT=1e4
                                   )
    likelihood_heterodyned.parameters.update(parameters_3)
    return likelihood_heterodyned.set_rdh_precaculate_info()['rdh']


tau = np.arange(100, 10000)
tau_list = (2*tau).tolist()

st = time.perf_counter()
with Pool(processes=72) as p:
    rdh = p.map(rdh_for_map, tau_list)
ed = time.perf_counter()
print(ed - st)

fig, ax = plt.subplots()
ax.plot(tau_list, rdh)
fig.savefig('test_rdh_2.png')


# likelihood_full = FullLikelihood(waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
#                                  detector=det)
# likelihood_heterodyned.parameters.update(parameters_2)
# likelihood_full.parameters.update(parameters_2)



# st = time.perf_counter()
# log_l_heterodyned = likelihood_heterodyned.log_likelihood()
# ed = time.perf_counter()
# print('heterodyned likelihood, time consuming: ', ed-st)
# st = time.perf_counter()
# log_l_full = likelihood_full.log_likelihood()
# ed = time.perf_counter()
# print('full likelihood, time consuming: ', ed-st)
# print('log_l diff: ', 2*(log_l_full-log_l_heterodyned)/(log_l_full+log_l_heterodyned))


# test_hete = likelihood_heterodyned.log_likelihood()
# fig, ax = plt.subplots()
# ax.semilogx(likelihood_heterodyned.detector.frequency_array, test_hete['dhdh']['A']['dhdh_full'], label='full')
# ax.semilogx(test_hete['frequencies_heterodyned'], test_hete['dhdh']['A']['dhdh_heterodyned'], marker='+', label='heterodyned')
# ax.legend()
# fig.savefig('dhdh_heterodyned.png')




# likelihood_heterodyned = HeterodynedLikelihood(waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
#                                detector=det,
#                                fiducial_parameters=parameters_2,
#                                FFT_points=4096
#                             #    dT=1e4
#                                )
# likelihood_heterodyned.parameters.update(parameters_3)
# test_rdh = likelihood_heterodyned.set_rdh_precaculate_info()

# print(test_rdh['rdh'])

# fig, ax = plt.subplots()
# ax.semilogx(test_rdh['f_array'], test_rdh['h0'], label='h0')
# ax.semilogx(test_rdh['f_array'], test_rdh['r'], label='r')
# ax.semilogx(test_rdh['f_array'], test_rdh['dh'], label='dh')
# # ax.loglog(test_rdh['f_array'], np.abs(test_rdh['dh']), label='dh')
# ax.legend()
# fig.savefig('test_rdh.png')


