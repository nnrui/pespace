import time
import os
import sys 
sys.path.append('/home/changfenggroup/nrui/works/codes/gw_space/pespace')
sys.path.append('/home/changfenggroup/nrui/works/codes/gw_space/tiwave')
# sys.path.append('/home/hydrogen/workspace/Space_GW/pespace')
# sys.path.append('/home/hydrogen/workspace/Space_GW/tiwave')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import bilby
import taichi as ti
from mpi4py import MPI

from pespace.constants import *
from pespace.detectors import TDIChannelsData, SpaceborneInterferometer
from pespace.noise import available_noise_models
from pespace.likelihood import FrequencyDomainLikelihood
from tiwave.waveforms import IMRPhenomD


local_rank_id = int(os.environ['MPI_LOCALRANKID'])
gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False, device_memory_GB=2)

duration = 30*DAY_SI    # 1 month observation
cadence = 10
TDI_chans = ("A", "E")
TDI_gen = '1.5'
inj_parameters = dict(
    total_mass = 3089053.9,
    mass_ratio = 0.11,
    chi_1 = 0.3986046314480332,
    chi_2 = 0.5465882372512956,
    luminosity_distance = 6000.42017466175677,
    inclination = 0.30782038099413395,
    reference_phase = 0.0,
    coalescence_time = 0.0,
    ecliptic_latitude = -0.5256036732051035,
    ecliptic_longitude = 1.1637,
    polarization = 1.30782038099413395,
)

mbhb = TDIChannelsData(label="injection_individual_MBHB")
mbhb.set_frequency_domain_data_with_zero_value(channels=TDI_chans, generation=TDI_gen, duration=duration, cadence=cadence)
mbhb.set_frequency_domain_noise_power_density_from_model(available_noise_models['LISA_SciRDv1'])
noise_realization = mbhb.generate_realization_from_frequency_domain_noise_power_density()
mbhb.add_into_frequency_domian_data(noise_realization)
lisa = SpaceborneInterferometer(name='LISA', TDI_data=mbhb, orbit='LISA_analytic')
lisa.initialize_response_container_in_frequency_domain()
wf = IMRPhenomD(lisa.TDI_data.frequency_samples)
wf.update_waveform(inj_parameters)
lisa.inject_frequency_domain_signal(wf.waveform_container, inj_parameters['ecliptic_longitude'], inj_parameters['ecliptic_latitude'], inj_parameters['polarization'])
likelihood = FrequencyDomainLikelihood(wf, lisa)

num_tests = 10000
minimum_frequency = 1e-5
maximum_frequency = 1e-1
cadence = 5
f_cut = 0.2
max_mass = maximum_frequency/f_cut/MTSUN_SI
# print(max_mass)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

chunk_size = num_tests // size
remain_size = num_tests % size
if rank < remain_size:
    start = rank*(chunk_size+1)
    end = start + (chunk_size+1)
elif rank == remain_size:
    start = rank*(chunk_size+1)
    end = start + chunk_size
else:
    start = rank*chunk_size+remain_size
    end = start + chunk_size

print(f"MPI worker {rank}/{size} working with start idx {start} to end idx {end}")
if comm.rank == 0:
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
    parameters['likelihood'] = np.zeros(num_tests)
    parameters = bilby.gw.conversion.generate_mass_parameters(parameters)
    parameters = pd.DataFrame(parameters)
else:
    parameters = None
parameters = comm.bcast(parameters, root=0)

ret_send = np.zeros(end-start)
ret_recv = np.zeros(num_tests)

st = time.perf_counter()
for i in range(start, end):
    likelihood.parameters.update(parameters.iloc[i])
    ret_send[i-start] = likelihood.log_likelihood()

comm.Gatherv(ret_send, ret_recv, root=0)
if rank==0:
    parameters['likelihood'] = ret_recv

ed = time.perf_counter()

ll_per_sec = num_tests/(ed-st)
if rank==0:
    print('likelihood evaluation per second: ', ll_per_sec)
    print("result: ", parameters)
    with open("../output/multinode_test_output.dat", 'a', encoding="utf_8") as f:
        f.write(f'GPU {len(gpus_list)} cpu {size} ll_per_sec {int(ll_per_sec)};\n')


