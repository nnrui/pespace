import numpy as np
import time
from matplotlib import pyplot as plt

import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, default_fp=ti.f64)

from ti_peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from ti_peSpace.detectors import LISALike, _generate_TDI_responses


parameters = dict(total_mass=4e6,
                  mass_ratio=1/3,
                  luminosity_distance=36594.3,
                  chi_1 = 0.2,
                  chi_2 = 0.4,
                  coalescence_time=0.0,
                  ecliptic_longitude = 3.335,
                  ecliptic_latitude = 1.468,
                  polarization = 2.237,
                  inclination = 1.047,
                  coalescence_phase = 0.0,)


det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=0.1, TDI_channels=('A', 'E'), 
               TDI_generation='2.0')

# print(det._ti_frequencies)
# print(det.TDI_data)
# print(det.waveform_container)

IMRPhenomD_h22_Amplitude_Phase_tf(det.frequencies, det.waveform_container, parameters, det.data_length)

det.updata_TDI_responses(parameters)

# print(det._ti_frequencies)
# print(det.TDI_data)
# print(det.waveform_container)
# print('waveform time consuming: ', ed1-st)
# print('response time consuming: ', ed2-ed1)

TDI_A_array = np.zeros(det.data_length)
for i in range(det.data_length):
    TDI_A_array[i] = (det.TDI_data[i].TDI_chan_data.A.norm())

fig, ax = plt.subplots()
ax.loglog(det.frequencies, TDI_A_array)
fig.savefig('A.png')






