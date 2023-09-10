import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')
import time
import copy

import numpy as np
from matplotlib import pyplot as plt

from peSpace.detectors import LISALike
from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from peSpace.likelihood import FullLikelihood, SparseLikelihood

det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=1, TDI_channels=('A', 'E', 'T'), 
               TDI_generation='2.0')

parameters_1 = dict(mass_1 = 1.5e6,
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

rng = np.random.default_rng()

parameters_2 = parameters_1.copy()
# for key in ['mass_1', 'mass_2', 'luminosity_distance', 'chi_1', 'chi_2', 'coalescence_time', 'ecliptic_longitude', 'ecliptic_latitude', 'polarization', 'inclination', 'coalescence_phase']:
# for key in ['mass_1', 'mass_2',                          'chi_1', 'chi_2',                      'ecliptic_longitude', 'ecliptic_latitude', 'polarization', 'inclination']:
# for key in [                                                                                      'ecliptic_longitude', 'ecliptic_latitude',                              ]:
for key in ['mass_1', 'mass_2',                          'chi_1', 'chi_2',                                                                                       ]:
    parameters_2[key] = parameters_1[key] + rng.uniform(-0.1, 0.1)*parameters_1[key]

waveform_1 = IMRPhenomD_h22_Amplitude_Phase_tf(det.frequency_array, parameters_1)
res_1 = det.TDI_responses(waveform_1, parameters_1)
waveform_2 = IMRPhenomD_h22_Amplitude_Phase_tf(det.frequency_array, parameters_2)
res_2 = det.TDI_responses(waveform_2, parameters_2)



fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, np.unwrap(np.angle(res_2['A']/res_1['A']), discont=np.pi, period=2*np.pi))
fig.savefig('comparision_response_A_phase.png')

fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, np.abs(res_2['A']/res_1['A']))
fig.savefig('comparision_response_A_abs.png')

# fig, ax = plt.subplots()
# ax.loglog(det.frequency_array, (res_2['A'].real)/(res_1['A'].real))
# fig.savefig('comparision_response_A_real_ratio.png')


fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, (res_2['A']/res_1['A']).real)
fig.savefig('comparision_response_A_real.png')


fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, (res_2['A']/res_1['A']).imag)
fig.savefig('comparision_response_A_imag.png')


fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, (res_2['E']/res_1['E']).real)
# ax.set_xlim(1e-4, 1e-3)
fig.savefig('comparision_response_E_real.png')



fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, (res_2['E']/res_1['E']).imag)
fig.savefig('comparision_response_E_imag.png')


fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, (res_2['T']/res_1['T']).real)
fig.savefig('comparision_response_T_real.png')


fig, ax = plt.subplots()
ax.semilogx(det.frequency_array, (res_2['T']/res_1['T']).imag)
fig.savefig('comparision_response_T_imag.png')