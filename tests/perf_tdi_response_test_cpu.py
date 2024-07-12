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


powers_of_2 = range(10, 28)
num_tests = 50

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
from ti_pespace.detectors import LISALike
from ti_pespace.likelihood import FullLikelihood
from wf4ti.waveforms.IMRPhenomD import IMRPhenomD as ti_IMRPhenomD
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)

from bbhx.waveformbuild import BBHWaveformFD


for p in powers_of_2:
    duration = 2**p
    n = int(duration / cadence)
    f_array = np.fft.rfftfreq(n, cadence)
    bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
    f_array = f_array[bound]
    data_length_list.append(len(f_array))
    ####################################
    wave_gen = BBHWaveformFD(amp_phase_kwargs={'run_phenomd': True,
                                              }, 
                            response_kwargs={'TDItag':'AET', 
                                             }, 
                            interp_kwargs={},
                            use_gpu=False)
    st =  time.perf_counter()
    for i in range(num_tests):
        data_channels = wave_gen(parameters.iloc[i]['mass_1'],
                                 parameters.iloc[i]['mass_2'],
                                 parameters.iloc[i]['chi_1'],
                                 parameters.iloc[i]['chi_2'],
                                 parameters.iloc[i]['luminosity_distance']*1e6*PC_SI,
                                 0.0,
                                 f_array[0],
                                 parameters.iloc[i]['inclination'],
                                 parameters.iloc[i]['ecliptic_longitude'],
                                 parameters.iloc[i]['ecliptic_latitude'],
                                 parameters.iloc[i]['polarization'],
                                 0.0,
                                 freqs=f_array,
                                 direct=True)
    ed = time.perf_counter()
    time_consuming = (ed - st)/num_tests
    time_consuming_bbhx.append(time_consuming)
    print(f'bbhx, time:{time_consuming}')
    ####################################
    det15 = LISALike(name='LISA', duration=duration, cadence=cadence, minimum_frequency=minimum_frequency, 
                     maximum_frequency=maximum_frequency, TDI_channels=('A', 'E', 'T'), 
                     TDI_generation='1.5')
    wf15 = ti_IMRPhenomD(det15.frequencies, det15.waveform_container)
    wf15.update_waveform(parameters.iloc[0])
    det15.updata_TDI_responses(parameters.iloc[0])
    st =  time.perf_counter()
    for i in range(num_tests):
        wf15.update_waveform(parameters.iloc[i])
        det15.updata_TDI_responses(parameters.iloc[i])
    ed = time.perf_counter()
    time_consuming = (ed - st)/num_tests
    time_consuming_pespace.append(time_consuming)
    print(f'pespace, time:{time_consuming}')


save_data = {'data_length': data_length_list,
             'bbhx': time_consuming_bbhx,
             'pespace': time_consuming_pespace,
            }
import json
with open('time_consuming_gpu_tdi_response_cpu.json', 'w') as f:
    json.dump(save_data, f)


day = 3600*24
week = 7*day
month = 4*week
year =  12*month
labeled_duration = [day, week, month, year]
labeled_data_length_list = []
labeled_duration_text = ['day', 'week', 'month', 'year']
for t in labeled_duration:
    n = int(t / cadence)
    f_array = np.fft.rfftfreq(n, cadence)
    bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
    f_array = f_array[bound]
    data_length = len(f_array)
    labeled_data_length_list.append(data_length)
    

with open('time_consuming_gpu_tdi_response_cpu.json', 'r') as f:
    data_tdi_response = json.load(f)
fig, ax = plt.subplots()
ax.loglog(data_tdi_response['data_length'], data_tdi_response['pespace'], label='pespace')
ax.loglog(data_tdi_response['data_length'], data_tdi_response['bbhx'],    label='bbhx')
ax.set_xlabel('data length')
ax.set_ylabel('time consuming (Sec)')
for idx, t in enumerate(labeled_data_length_list):
    ax.axvline(t, linestyle='dashed', color='tab:gray')
    ax.text(t, 2.0, labeled_duration_text[idx])
ax.legend()
fig.savefig('perf_tdi_response_cpu.png')


