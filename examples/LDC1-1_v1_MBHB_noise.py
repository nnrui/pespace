import sys
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/pespace")
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/tiwave")
sys.path.append('/home/hydrogen/workspace/Space_GW/pespace')
sys.path.append('/home/hydrogen/workspace/Space_GW/tiwave')
import os
from pathlib import Path

import numpy as np
import taichi as ti
import bilby
import h5py
from matplotlib import pyplot as plt

from pespace.constants import DAY_SI, PI
from pespace.detectors import TDIChannelsData, SpaceborneInterferometer
from pespace.likelihood import FrequencyDomainLikelihood
from pespace.orbits import KaplerianHeliocentric
from tiwave.waveforms import IMRPhenomD
from tiwave.constants import PC_SI

# local_rank_id = int(os.environ['MPI_LOCALRANKID'])
# gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
# selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
# os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False)


label = "LDC1-1_v1_MBHB"
outdir = f"output/outdir_{label}"
Path(outdir).mkdir(exist_ok=True)


########################################################################################
# Read in the data and print data info
dataname = "LDC1-1_MBHB_v1_1_TD"
file_path = f"/home/hydrogen/workspace/Space_GW/LDC/LDC_data/{dataname}.hdf5"
with h5py.File(file_path, "r") as f:
    f.visit(lambda name: print(name))

    source_parameters = {}
    print('Source information: ')
    for parameter, value in f["H5LISA/GWSources/MBHB-0"].items():
        print(f"{parameter}: {value[()]}")
        source_parameters[parameter] = value[()]

    print(f["H5LISA/PreProcess/TDIGenerator"][()])
    TDI_data = f["H5LISA/PreProcess/TDIdata"][()]
    print(TDI_data)
    print(TDI_data.shape)
# Source information:
# Approximant: b'IMRPhenomD'
# AzimuthalAngleOfSpin1: 0.6171792478977071
# AzimuthalAngleOfSpin2: 4.75979656623224
# Cadence: 10.0
# CoalescenceTime: 25135000.0
# Distance: 60.42017466175677
# EclipticLatitude: -0.5256036732051035
# EclipticLongitude: 1.1637
# InitialAzimuthalAngleL: 0.30782038099413395
# InitialPolarAngleL: 1.2498
# Mass1: 2803843.776
# Mass2: 285210.246
# ObservationDuration: 41943040.0
# PhaseAtCoalescence: 2.596553404898615
# PolarAngleOfSpin1: 0.0
# PolarAngleOfSpin2: 0.0
# Redshift: 6.1178
# Spin1: 0.8986046314480332
# Spin2: 0.9465882372512956
# b'X,Y,Z'
# (4194304, 4)

tc = source_parameters["CoalescenceTime"]
time_array, X_array, Y_array, Z_array = TDI_data.T
A_array = (Z_array - X_array)/np.sqrt(2)
E_array = (X_array - 2*Y_array + Z_array)/np.sqrt(6)
T_array = (X_array + Y_array + Z_array)/np.sqrt(3)

# plt.figure()
# plt.plot(time_array, X_array, label='X')
# plt.plot(time_array, Y_array, label='Y')
# plt.plot(time_array, Z_array, label='Z')
# plt.xlim(tc-0.1*DAY_SI, tc+0.1*DAY_SI)
# plt.legend(loc='lower right')
# plt.savefig(f"{outdir}/{dataname}_raw_data_XYZ.png")

# plt.figure()
# plt.plot(time_array, A_array, label='A')
# plt.plot(time_array, E_array, label='E')
# plt.plot(time_array, T_array, label='T')
# plt.xlim(tc-0.1*DAY_SI, tc+0.1*DAY_SI)
# plt.legend(loc='lower right')
# plt.savefig(f"{outdir}/{dataname}_raw_data_AET.png")

cadence = source_parameters["Cadence"]
duration = source_parameters["CoalescenceTime"] - time_array[0] + DAY_SI
end_idx = int((duration + time_array[0]) // cadence)
input_TDI_data_array = np.array([A_array[:end_idx], E_array[:end_idx], T_array[:end_idx]])
# print(end_idx)
# print(time_array[:end_idx])
# print(source_parameters["CoalescenceTime"] + DAY_SI)
# print(A_array[:end_idx])
# print(len(A_array[:end_idx]))

TDI_chans = ("A", "E", "T")
TDI_gen = "1.5"
LDC_mbhb = TDIChannelsData(label="LDC1-1_MBHB", minimum_frequency=1e-5, maximum_frequency=0.1)
LDC_mbhb.set_time_domain_data_from_input_array(channels=TDI_chans, 
                                               generation=TDI_gen, 
                                               duration=duration, 
                                               cadence=cadence,
                                               TDI_data_array=input_TDI_data_array,
                                               start_time=time_array[0])
# print(LDC_mbhb.data_info.time_series_length)
# print(LDC_mbhb.data_info.time_samples_array)
LDC_mbhb.Fourier_transform_time_domain_data_to_frequency_domain(window=("tukey", 0.0))
LDC_mbhb.set_frequency_domain_noise_power_density_from_model("LISA_SciRDv1")
# plt.figure()
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['A']),
#            label="A")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['E']),
#            label="E")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['T']),
#            label="T")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_td_to_fd.png")


# dataname_noiseless = "LDC1-1_MBHB_v1_1_TD_noiseless"
# file_path_noiseless = f"/home/hydrogen/workspace/Space_GW/LDC/LDC_data/{dataname_noiseless}.hdf5"
# with h5py.File(file_path_noiseless, "r") as f:
#     TDI_data_noiseless = f["H5LISA/PreProcess/TDIdata"][()]

# _, X_array_noiseless, Y_array_noiseless, Z_array_noiseless = TDI_data_noiseless.T
# A_array_noiseless = (Z_array_noiseless - X_array_noiseless)/np.sqrt(2)
# E_array_noiseless = (X_array_noiseless - 2*Y_array_noiseless + Z_array_noiseless)/np.sqrt(6)
# T_array_noiseless = (X_array_noiseless + Y_array_noiseless + Z_array_noiseless)/np.sqrt(3)
# input_TDI_data_array_noiseless = np.array([A_array_noiseless[:end_idx], E_array_noiseless[:end_idx], T_array_noiseless[:end_idx]])

# LDC_mbhb_noiseless = TDIChannelsData(label="LDC1-1_MBHB_noiseless", minimum_frequency=1e-5, maximum_frequency=0.1)
# LDC_mbhb_noiseless.set_time_domain_data_from_input_array(channels=TDI_chans, 
#                                                          generation=TDI_gen, 
#                                                          duration=duration, 
#                                                          cadence=cadence,
#                                                          TDI_data_array=input_TDI_data_array_noiseless,
#                                                          start_time=time_array[0])
# LDC_mbhb_noiseless.Fourier_transform_time_domain_data_to_frequency_domain(window=("tukey", 0.0))
# plt.figure()
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['A']),
#            label="A")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['E']),
#            label="E")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['T']),
#            label="T")
# plt.loglog(LDC_mbhb_noiseless.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb_noiseless.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_noiseless")
# plt.loglog(LDC_mbhb_noiseless.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb_noiseless.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_noiseless")
# plt.loglog(LDC_mbhb_noiseless.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb_noiseless.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_noiseless")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_td_to_fd_with_signal.png")



# pespace_mbhb = TDIChannelsData(label="pespace_MBHB", minimum_frequency=1e-5, maximum_frequency=0.1)
# pespace_mbhb.set_frequency_domain_data_with_zero_value(channels=TDI_chans, 
#                                                        generation=TDI_gen, 
#                                                        duration=duration, 
#                                                        cadence=cadence,)
# pespace_mbhb.set_frequency_domain_noise_power_density_from_model("LISA_SciRDv1")
# noise_realization = pespace_mbhb.generate_realization_from_frequency_domain_noise_power_density()
# pespace_mbhb.add_into_frequency_domian_data(noise_realization)
# plt.figure()
# plt.loglog(pespace_mbhb.data_info.frequency_samples_array, 
#            np.abs(pespace_mbhb.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_pespace")
# plt.loglog(pespace_mbhb.data_info.frequency_samples_array, 
#            np.abs(pespace_mbhb.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_pespace")
# plt.loglog(pespace_mbhb.data_info.frequency_samples_array, 
#            np.abs(pespace_mbhb.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_pespace")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_LDC_fft")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_LDC_fft")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_LDC_fft")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_simulated_noise.png")


params_tiwave= dict(
    mass_1 = source_parameters["Mass1"],
    mass_2 = source_parameters["Mass2"],
    mass_ratio = source_parameters["Mass2"]/source_parameters["Mass1"],
    chi_1 = source_parameters["Spin1"],
    chi_2 = source_parameters["Spin2"],
    luminosity_distance = source_parameters["Distance"]*1000,
    inclination = source_parameters["InitialPolarAngleL"],
    reference_phase = source_parameters["PhaseAtCoalescence"], # remember setting the reference_frequency to coalescence frequency
    coalescence_time = source_parameters["CoalescenceTime"],
    ecliptic_latitude = source_parameters["EclipticLatitude"],
    ecliptic_longitude = source_parameters["EclipticLongitude"],
    polarization = source_parameters["InitialAzimuthalAngleL"],
)
params_tiwave = bilby.gw.conversion.generate_mass_parameters(params_tiwave)
wf = IMRPhenomD(LDC_mbhb.frequency_samples)
wf.update_waveform(params_tiwave)
f_peak = wf.amplitude_coefficients[None].f_peak/wf.source_parameters.M_sec
# to match the PhaseAtCoalescence in LDC, setting reference_frequency around coalescence frequency
wf = IMRPhenomD(LDC_mbhb.frequency_samples, reference_frequency=f_peak)
wf.update_waveform(params_tiwave)

lisa = SpaceborneInterferometer(name='LISA', TDI_data=LDC_mbhb, orbit=KaplerianHeliocentric(2.5e9, 0.0, 0.0))
lisa.initialize_response_container_in_frequency_domain()
lisa.inject_frequency_domain_signal(wf.waveform_container, 
                                    params_tiwave['ecliptic_longitude'], 
                                    params_tiwave['ecliptic_latitude'], 
                                    params_tiwave['polarization'])
response = lisa.response_container.to_numpy()
likelihood_pespace = FrequencyDomainLikelihood(wf, lisa)
likelihood_pespace.parameters.updata(params_tiwave)
print(f"pespace likelihood: {likelihood_pespace.log_likelihood()}")
# plt.figure()
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(response['A'][:,0] + response['A'][:,1]*1j),
#            label="A_response")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(response['E'][:,0] + response['E'][:,1]*1j),
#            label="E_response")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(response['T'][:,0] + response['T'][:,1]*1j),
#            label="T_response")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_simulated_response.png")


from bbhx.waveformbuild import BBHWaveformFD
from bbhx.likelihood import Likelihood
wave_gen = BBHWaveformFD(amp_phase_kwargs={'run_phenomd': True,}, 
                        response_kwargs={'TDItag':'AET',}, 
                        interp_kwargs={},
                        use_gpu=False)
# tdi_bbhx = wave_gen(params_tiwave['mass_1'],
#                     params_tiwave['mass_2'],
#                     params_tiwave['chi_1'],
#                     params_tiwave['chi_2'],
#                     params_tiwave['luminosity_distance']*1e6*PC_SI,
#                     params_tiwave["reference_phase"],
#                     f_peak,
#                     params_tiwave['inclination'],
#                     params_tiwave['ecliptic_longitude'],
#                     params_tiwave['ecliptic_latitude'],
#                     params_tiwave['polarization'],
#                     0.0,
#                     freqs=LDC_mbhb.frequency_samples_numpy_array,
#                     direct=True)[0]
# A_bbhx, E_bbhx, T_bbhx = tdi_bbhx

psd = LDC_mbhb.frequency_domain_noise_power_density_numpy_array()

psd_bbhx = np.array([psd["A"], psd["E"], psd["T"]])

# initialize Likelihood
like = Likelihood(
    wave_gen,
    LDC_mbhb.frequency_samples_numpy_array,
    input_TDI_data_array,
    psd,
    use_gpu=False,
)

# # get params
# num_bins = 10
# params_in = np.tile(np.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref]), (num_bins, 1))

# # change masses for test
# params_in[:, 0] *= (1 + 1e-4 * np.random.randn(num_bins))

# # get_ll and not __call__ to work with lisatools
# ll = like.get_ll(params_in.T, **waveform_kwargs)

# print(ll, like.d_h)






# params_tiwave= dict(
#     mass_1 = source_parameters["Mass1"],
#     mass_2 = source_parameters["Mass2"],
#     mass_ratio = source_parameters["Mass2"]/source_parameters["Mass1"],
#     chi_1 = source_parameters["Spin1"],
#     chi_2 = source_parameters["Spin2"],
#     luminosity_distance = source_parameters["Distance"]*1000,
#     inclination = source_parameters["InitialPolarAngleL"],
#     reference_phase = source_parameters["PhaseAtCoalescence"], # remember setting the reference_frequency to coalescence frequency
#     coalescence_time = source_parameters["CoalescenceTime"],
#     ecliptic_latitude = source_parameters["EclipticLatitude"],
#     ecliptic_longitude = source_parameters["EclipticLongitude"],
#     polarization = source_parameters["InitialAzimuthalAngleL"],
# )
# params_tiwave = bilby.gw.conversion.generate_mass_parameters(params_tiwave)
# wf = IMRPhenomD(inj_mbhb.frequency_samples)
# wf.update_waveform(params_tiwave)
# f_peak = wf.amplitude_coefficients[None].f_peak/wf.source_parameters.M_sec
# # to match the PhaseAtCoalescence in LDC, setting reference_frequency around coalescence frequency
# wf = IMRPhenomD(inj_mbhb.frequency_samples, reference_frequency=f_peak)
# wf.update_waveform(params_tiwave)

# lisa = SpaceborneInterferometer(name='LISA', TDI_data=inj_mbhb, orbit=KaplerianHeliocentric(2.5e9, 0.0, 0.0))
# lisa.initialize_response_container_in_frequency_domain()
# lisa.inject_frequency_domain_signal(wf.waveform_container, 
#                                     params_tiwave['ecliptic_longitude'], 
#                                     params_tiwave['ecliptic_latitude'], 
#                                     params_tiwave['polarization'])
# plt.figure()
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_pespace")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_LDC_fft")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_LDC_fft")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_LDC_fft")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_pespace_LDC_fd.png")

# from bbhx.waveformbuild import BBHWaveformFD
# wave_gen = BBHWaveformFD(amp_phase_kwargs={'run_phenomd': True,}, 
#                         response_kwargs={'TDItag':'AET',}, 
#                         interp_kwargs={},
#                         use_gpu=False)
# tdi_bbhx = wave_gen(params_tiwave['mass_1'],
#                     params_tiwave['mass_2'],
#                     params_tiwave['chi_1'],
#                     params_tiwave['chi_2'],
#                     params_tiwave['luminosity_distance']*1e6*PC_SI,
#                     params_tiwave["reference_phase"],
#                     f_peak,
#                     params_tiwave['inclination'],
#                     params_tiwave['ecliptic_longitude'],
#                     params_tiwave['ecliptic_latitude'],
#                     params_tiwave['polarization'],
#                     0.0,
#                     freqs=inj_mbhb.frequency_samples_numpy_array,
#                     direct=True)[0]
# A_bbhx, E_bbhx, T_bbhx = tdi_bbhx
# plt.figure()
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['A']),
#            label="A_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['E']),
#            label="E_pespace")
# plt.loglog(lisa.TDI_data.data_info.frequency_samples_array, 
#            np.abs(lisa.TDI_data.frequency_domain_TDI_data_numpy_array['T']),
#            label="T_pespace")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(A_bbhx),
#            label="A_bbhx")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(E_bbhx),
#            label="E_bbhx")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(T_bbhx),
#            label="T_bbhx")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_pespace_bbhx_fd.png")

# lisa = SpaceborneInterferometer(ame='LISA', TDI_data=inj_mbhb, orbit='LISA_analytic')
# LDC_mbhb.Fourier_transform_time_domain_data_to_frequency_domain(window=("tukey", 0.0))
# plt.figure()
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['A']),
#            label="A")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['E']),
#            label="E")
# plt.loglog(LDC_mbhb.data_info.frequency_samples_array, 
#            np.abs(LDC_mbhb.frequency_domain_TDI_data_numpy_array['T']),
#            label="T")
# plt.legend()
# plt.savefig(f"{outdir}/{dataname}_to_fd.png")





# # duration = 30*DAY_SI  # 1 month observation
# # cadence = 10
# TDI_chans = ("A", "E")
# TDI_gen = "1.5"
# parameters = dict(
#     total_mass=3089053.9,
#     mass_ratio=0.11,
#     chi_1=0.3986046314480332,
#     chi_2=0.5465882372512956,
#     luminosity_distance=6000.42017466175677,
#     inclination=0.30782038099413395,
#     reference_phase=1.02,
#     coalescence_time=0.0,
#     ecliptic_latitude=-0.5256036732051035,
#     ecliptic_longitude=1.1637,
#     polarization=1.30782038099413395,
# )

# lisa_mbhb = TDIChannelsData(label="inj_MBHB_lisa", minimum_frequency=1e-5, maximum_frequency=0.1)
# lisa_mbhb.set_frequency_domain_data_with_zero_value(channels=TDI_chans, generation=TDI_gen, duration=duration, cadence=cadence)
# lisa_mbhb.set_frequency_domain_noise_power_density_from_model("LISA_SciRDv1")
# lisa_noise = lisa_mbhb.generate_realization_from_frequency_domain_noise_power_density()
# lisa_mbhb.add_into_frequency_domian_data(lisa_noise)

# lisa = SpaceborneInterferometer(name='LISA', TDI_data=lisa_mbhb, orbit="LISA_analytic")
# lisa.initialize_response_container_in_frequency_domain()
# wf = IMRPhenomD(lisa.TDI_data.frequency_samples)
# wf.update_waveform(parameters)
# lisa.inject_frequency_domain_signal(wf.waveform_container, parameters['ecliptic_longitude'], parameters['ecliptic_latitude'], parameters['polarization'])

# priors = {}
# # priors['chi_1'] = parameters["chi_1"]
# # priors['chi_2'] = parameters["chi_2"]
# # priors['ecliptic_longitude'] = parameters["ecliptic_longitude"]
# # priors['ecliptic_latitude'] = parameters["ecliptic_latitude"]
# # priors['inclination'] = parameters["inclination"]
# # priors['polarization'] = parameters["polarization"]
# # priors['reference_phase'] = parameters["reference_phase"]
# # priors['coalescence_time'] = parameters["coalescence_time"]

# priors['chi_1'] = bilby.core.prior.Uniform(name='chi_1', minimum=-0.99, maximum=0.99)
# priors['chi_2'] = bilby.core.prior.Uniform(name='chi_2', minimum=-0.99, maximum=0.99)
# priors['ecliptic_longitude'] = bilby.core.prior.Uniform(name='ecliptic_longitude', minimum=0, maximum=2 * PI, boundary='periodic')
# priors['ecliptic_latitude'] = bilby.core.prior.Cosine(name='ecliptic_latitude')
# priors['inclination'] = bilby.core.prior.Sine(name='inclination')
# priors['polarization'] = bilby.core.prior.Uniform(name='polarization', minimum=0, maximum=PI, boundary='periodic')
# priors['reference_phase'] = bilby.core.prior.Uniform(name='reference_phase', minimum=0, maximum=2 * PI, boundary='periodic')
# priors['coalescence_time'] = bilby.core.prior.Uniform(name='coalescence_time', minimum=-DAY_SI, maximum=DAY_SI)

# priors['total_mass'] = bilby.core.prior.LogUniform(name='total_mass', minimum=1e3, maximum=1e9)
# priors['mass_ratio'] = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.05, maximum=0.99)
# priors['luminosity_distance'] = bilby.gw.prior.Uniform(name='luminosity_distance', minimum=1000, maximum=10000)

# label = "LISA_MBHB_injection_full"
# outdir = f"output/outdir_{label}"
# result = bilby.run_sampler(
#     likelihood=FrequencyDomainLikelihood(wf, lisa),
#     priors=priors,
#     outdir=outdir,
#     label=label,
#     sampler='pymultinest',
#     npoints=2048)

# # PLOT RESULT
# # result = bilby.core.result.read_in_result(f"output/outdir_{label}/LISA_MBHB_injection_full_result.json")
# result.plot_corner(parameters=parameters, 
#                    quantiles=[0.05, 0.95])
