import numpy as np
import time

import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, default_fp=ti.f64)

from ti_peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf

cadence = 10
duration = 3600*24*30
minimum_frequency = 1e-4
maximum_frequency = 1e-1

f = np.arange(0, 1.0/(2*cadence), 1.0/duration)
bound = ((f >= minimum_frequency) * (f <= maximum_frequency))
frequencies = f[bound]
length = len(frequencies)

waveform_field = ti.Struct.field({'hplus': tm.vec2,
                                  'hcross': tm.vec2,
                                  'tf': ti.f64})
ti.root.dense(ti.i, length).place(waveform_field)
waveform_container = waveform_field

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

st = time.perf_counter()
IMRPhenomD_h22_Amplitude_Phase_tf(frequencies, waveform_container, parameters, length)
ed = time.perf_counter()

print(waveform_container)
print('time consuming: ', ed-st)



