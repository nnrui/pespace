import numpy as np
import time
from matplotlib import pyplot as plt

import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)

from ti_peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from ti_peSpace.detectors import LISALike


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

det15 = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
               TDI_generation='1.5')
det20 = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
               TDI_generation='2.0')

# print(det._ti_frequencies)
# print(det.TDI_data)
# print(det.waveform_container)

st = time.perf_counter()
IMRPhenomD_h22_Amplitude_Phase_tf(det15.frequencies, det15.waveform_container, parameters, det15.data_length)
IMRPhenomD_h22_Amplitude_Phase_tf(det20.frequencies, det20.waveform_container, parameters, det20.data_length)
det15.updata_TDI_responses(parameters)
det20.updata_TDI_responses(parameters)
ed = time.perf_counter()
print('ti time consuming: ', ed-st)
# print(det._ti_frequencies)
# print(det.TDI_data)
# print(det.waveform_container)
# print('waveform time consuming: ', ed1-st)
# print('response time consuming: ', ed2-ed1)

TDI_X15_array = np.zeros(det15.data_length)
TDI_Y15_array = np.zeros(det15.data_length)
TDI_Z15_array = np.zeros(det15.data_length)
TDI_A15_array = np.zeros(det15.data_length)
TDI_E15_array = np.zeros(det15.data_length)
TDI_T15_array = np.zeros(det15.data_length)

TDI_X20_array = np.zeros(det20.data_length)
TDI_Y20_array = np.zeros(det20.data_length)
TDI_Z20_array = np.zeros(det20.data_length)
TDI_A20_array = np.zeros(det20.data_length)
TDI_E20_array = np.zeros(det20.data_length)
TDI_T20_array = np.zeros(det20.data_length)

for i in range(det15.data_length):
    TDI_X15_array[i] = (det15.TDI_data[i].TDI_chan_data.X.norm())
    TDI_Y15_array[i] = (det15.TDI_data[i].TDI_chan_data.Y.norm())
    TDI_Z15_array[i] = (det15.TDI_data[i].TDI_chan_data.Z.norm())
    TDI_A15_array[i] = (det15.TDI_data[i].TDI_chan_data.A.norm())
    TDI_E15_array[i] = (det15.TDI_data[i].TDI_chan_data.E.norm())
    TDI_T15_array[i] = (det15.TDI_data[i].TDI_chan_data.T.norm())
for i in range(det20.data_length):
    TDI_X20_array[i] = (det20.TDI_data[i].TDI_chan_data.X.norm())
    TDI_Y20_array[i] = (det20.TDI_data[i].TDI_chan_data.Y.norm())
    TDI_Z20_array[i] = (det20.TDI_data[i].TDI_chan_data.Z.norm())
    TDI_A20_array[i] = (det20.TDI_data[i].TDI_chan_data.A.norm())
    TDI_E20_array[i] = (det20.TDI_data[i].TDI_chan_data.E.norm())
    TDI_T20_array[i] = (det20.TDI_data[i].TDI_chan_data.T.norm())

ti_strain = {'X15': TDI_X15_array,
             'Y15': TDI_Y15_array,
             'Z15': TDI_Z15_array,
             'A15': TDI_A15_array,
             'E15': TDI_E15_array,
             'T15': TDI_T15_array,
             'X20': TDI_X20_array,
             'Y20': TDI_Y20_array,
             'Z20': TDI_Z20_array,
             'A20': TDI_A20_array,
             'E20': TDI_E20_array,
             'T20': TDI_T20_array
            }



from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf as pyIMRPhenomD_h22_Amplitude_Phase_tf
from peSpace.detectors import LISALike as pyLISALike

pydet15 = pyLISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
               TDI_generation='1.5')
pydet20 = pyLISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
               TDI_generation='2.0')

st = time.perf_counter()
pydet15.inject_signal_FD(parameters, pyIMRPhenomD_h22_Amplitude_Phase_tf)
pydet20.inject_signal_FD(parameters, pyIMRPhenomD_h22_Amplitude_Phase_tf)
ed = time.perf_counter()
print('py time consuming: ', ed-st)

for chan in ['X', 'Y', 'Z', 'A', 'E', 'T']:
    fig, ax = plt.subplots()
    ax.loglog(det15.frequencies, ti_strain[f'{chan}15'], label=f'ti_{chan}_15')
    ax.loglog(pydet15.frequency_array,  np.abs(pydet15.signals[chan]), label=f'py_{chan}_15', linestyle='dashed')
    ax.loglog(det20.frequencies, ti_strain[f'{chan}20'], label=f'ti_{chan}_20')
    ax.loglog(pydet20.frequency_array,  np.abs(pydet20.signals[chan]), label=f'py_{chan}_20', linestyle='dashed')
    ax.legend()
    ax.set_title(chan)
    fig.savefig(f'{chan}_norm.png')
    plt.close()



####################################################################################
TDI_X15_array = np.zeros(det15.data_length)
TDI_Y15_array = np.zeros(det15.data_length)
TDI_Z15_array = np.zeros(det15.data_length)
TDI_A15_array = np.zeros(det15.data_length)
TDI_E15_array = np.zeros(det15.data_length)
TDI_T15_array = np.zeros(det15.data_length)

TDI_X20_array = np.zeros(det20.data_length)
TDI_Y20_array = np.zeros(det20.data_length)
TDI_Z20_array = np.zeros(det20.data_length)
TDI_A20_array = np.zeros(det20.data_length)
TDI_E20_array = np.zeros(det20.data_length)
TDI_T20_array = np.zeros(det20.data_length)

for i in range(det15.data_length):
    TDI_X15_array[i] = (det15.TDI_data[i].TDI_chan_data.X[0])
    TDI_Y15_array[i] = (det15.TDI_data[i].TDI_chan_data.Y[0])
    TDI_Z15_array[i] = (det15.TDI_data[i].TDI_chan_data.Z[0])
    TDI_A15_array[i] = (det15.TDI_data[i].TDI_chan_data.A[0])
    TDI_E15_array[i] = (det15.TDI_data[i].TDI_chan_data.E[0])
    TDI_T15_array[i] = (det15.TDI_data[i].TDI_chan_data.T[0])
for i in range(det20.data_length):
    TDI_X20_array[i] = (det20.TDI_data[i].TDI_chan_data.X[0])
    TDI_Y20_array[i] = (det20.TDI_data[i].TDI_chan_data.Y[0])
    TDI_Z20_array[i] = (det20.TDI_data[i].TDI_chan_data.Z[0])
    TDI_A20_array[i] = (det20.TDI_data[i].TDI_chan_data.A[0])
    TDI_E20_array[i] = (det20.TDI_data[i].TDI_chan_data.E[0])
    TDI_T20_array[i] = (det20.TDI_data[i].TDI_chan_data.T[0])

ti_strain = {'X15': TDI_X15_array,
             'Y15': TDI_Y15_array,
             'Z15': TDI_Z15_array,
             'A15': TDI_A15_array,
             'E15': TDI_E15_array,
             'T15': TDI_T15_array,
             'X20': TDI_X20_array,
             'Y20': TDI_Y20_array,
             'Z20': TDI_Z20_array,
             'A20': TDI_A20_array,
             'E20': TDI_E20_array,
             'T20': TDI_T20_array
            }

for chan in ['X', 'Y', 'Z', 'A', 'E', 'T']:
    fig, ax = plt.subplots()
    ax.semilogx(det15.frequencies, ti_strain[f'{chan}15'], label=f'ti_{chan}_15')
    ax.semilogx(pydet15.frequency_array,  (pydet15.signals[chan]).real, label=f'py_{chan}_15', linestyle='dashed')
    ax.semilogx(det20.frequencies, ti_strain[f'{chan}20'], label=f'ti_{chan}_20')
    ax.semilogx(pydet20.frequency_array,  (pydet20.signals[chan]).real, label=f'py_{chan}_20', linestyle='dashed')
    ax.legend()
    ax.set_title(chan)
    fig.savefig(f'{chan}_real.png')
    plt.close()


####################################################################################
TDI_X15_array = np.zeros(det15.data_length)
TDI_Y15_array = np.zeros(det15.data_length)
TDI_Z15_array = np.zeros(det15.data_length)
TDI_A15_array = np.zeros(det15.data_length)
TDI_E15_array = np.zeros(det15.data_length)
TDI_T15_array = np.zeros(det15.data_length)

TDI_X20_array = np.zeros(det20.data_length)
TDI_Y20_array = np.zeros(det20.data_length)
TDI_Z20_array = np.zeros(det20.data_length)
TDI_A20_array = np.zeros(det20.data_length)
TDI_E20_array = np.zeros(det20.data_length)
TDI_T20_array = np.zeros(det20.data_length)

for i in range(det15.data_length):
    TDI_X15_array[i] = (det15.TDI_data[i].TDI_chan_data.X[1])
    TDI_Y15_array[i] = (det15.TDI_data[i].TDI_chan_data.Y[1])
    TDI_Z15_array[i] = (det15.TDI_data[i].TDI_chan_data.Z[1])
    TDI_A15_array[i] = (det15.TDI_data[i].TDI_chan_data.A[1])
    TDI_E15_array[i] = (det15.TDI_data[i].TDI_chan_data.E[1])
    TDI_T15_array[i] = (det15.TDI_data[i].TDI_chan_data.T[1])
for i in range(det20.data_length):
    TDI_X20_array[i] = (det20.TDI_data[i].TDI_chan_data.X[1])
    TDI_Y20_array[i] = (det20.TDI_data[i].TDI_chan_data.Y[1])
    TDI_Z20_array[i] = (det20.TDI_data[i].TDI_chan_data.Z[1])
    TDI_A20_array[i] = (det20.TDI_data[i].TDI_chan_data.A[1])
    TDI_E20_array[i] = (det20.TDI_data[i].TDI_chan_data.E[1])
    TDI_T20_array[i] = (det20.TDI_data[i].TDI_chan_data.T[1])

ti_strain = {'X15': TDI_X15_array,
             'Y15': TDI_Y15_array,
             'Z15': TDI_Z15_array,
             'A15': TDI_A15_array,
             'E15': TDI_E15_array,
             'T15': TDI_T15_array,
             'X20': TDI_X20_array,
             'Y20': TDI_Y20_array,
             'Z20': TDI_Z20_array,
             'A20': TDI_A20_array,
             'E20': TDI_E20_array,
             'T20': TDI_T20_array
            }

for chan in ['X', 'Y', 'Z', 'A', 'E', 'T']:
    fig, ax = plt.subplots()
    ax.semilogx(det15.frequencies, ti_strain[f'{chan}15'], label=f'ti_{chan}_15')
    ax.semilogx(pydet15.frequency_array,  (pydet15.signals[chan]).imag, label=f'py_{chan}_15', linestyle='dashed')
    ax.semilogx(det20.frequencies, ti_strain[f'{chan}20'], label=f'ti_{chan}_20')
    ax.semilogx(pydet20.frequency_array,  (pydet20.signals[chan]).imag, label=f'py_{chan}_20', linestyle='dashed')
    ax.legend()
    ax.set_title(chan)
    fig.savefig(f'{chan}_imag.png')
    plt.close()


####################################################################################



