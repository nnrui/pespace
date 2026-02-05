import bilby
import lal
import numpy as np
import taichi as ti
# local_rank_id = int(os.environ['MPI_LOCALRANKID'])
# gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
# selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
# os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
# ti.init(arch=ti.gpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

from pespace.detector.antenna import InterferometerAntenna, FDResponseModelMarset2018
from pespace.detector.tdi import TDIChannelData, FDMichelsonConstantEqualArm
from pespace.detector.orbit import available_orbit_models
from pespace.inference.interface_bilby import LikelihoodBilbyInterface
from tiwave.waveforms import IMRPhenomXHM

########################################################################################
# common settings
# set injection
tdi_gen = "2.0"
tdi_chan = ("A", "E", "T")

dt = 10
f_min = 1e-4
f_max = 0.5*(1/dt)
f_ref = f_min
t_start = 0.0

num_tsamples = 2**np.ceil(np.log2(7*lal.DAYJUL_SI/dt))
duration = num_tsamples * dt
before_tc = 0.8 * duration
after_tc = 0.2 * duration
tc = t_start + before_tc

params = dict(
    total_mass=3e6,
    mass_ratio=0.6,
    chi_1=0.75,
    chi_2=0.62,
    luminosity_distance=56000.0,
    inclination=0.4,
    reference_phase=1.3,
    ecliptic_longitude=1.375,
    ecliptic_latitude=-1.2108,
    polarization=2.659,
    coalescence_time=tc,
)

# detector settings
tdi_data_lisa = TDIChannelData()
tdi_data_lisa.set_fd_data_from_zero(
    channels=tdi_chan, 
    duration=duration, 
    delta_time=dt,
    start_time=t_start,
    minimum_frequency=f_min,
    maximum_frequency=f_max,
    )
tdi_data_lisa.set_fd_noise_power_density_from_model("LISA_SciRDv1", tdi_generation=tdi_gen)
noise_lisa = tdi_data_lisa.get_fd_noise_realization()
tdi_data_lisa.add_into_fd_data(noise_lisa)
lisa = InterferometerAntenna(
    name="lisa",
    tdi_data=tdi_data_lisa,
    orbit_model=available_orbit_models['LISA_analytic'],
    response_model=FDResponseModelMarset2018(),
    tdi_combination=FDMichelsonConstantEqualArm(generation=tdi_gen, orthogonal=True),
)

tdi_data_taiji = TDIChannelData()
tdi_data_taiji.set_fd_data_from_zero(
    channels=tdi_chan, 
    duration=duration, 
    delta_time=dt,
    start_time=t_start,
    minimum_frequency=f_min,
    maximum_frequency=f_max,
    )
tdi_data_taiji.set_fd_noise_power_density_from_model("Taiji_TDC", tdi_generation=tdi_gen)
noise_taiji = tdi_data_taiji.get_fd_noise_realization()
tdi_data_taiji.add_into_fd_data(noise_taiji)
taiji = InterferometerAntenna(
    name="taiji",
    tdi_data=tdi_data_taiji,
    orbit_model=available_orbit_models['Taiji_analytic'],
    response_model=FDResponseModelMarset2018(),
    tdi_combination=FDMichelsonConstantEqualArm(generation=tdi_gen, orthogonal=True),
)

tdi_data_tianqin = TDIChannelData()
tdi_data_tianqin.set_fd_data_from_zero(
    channels=tdi_chan, 
    duration=duration, 
    delta_time=dt,
    start_time=t_start,
    minimum_frequency=f_min,
    maximum_frequency=f_max,
    )
tdi_data_tianqin.set_fd_noise_power_density_from_model("Tianqin_GWSpace", tdi_generation=tdi_gen)
noise_tianqin = tdi_data_tianqin.get_fd_noise_realization()
tdi_data_tianqin.add_into_fd_data(noise_tianqin)
tianqin = InterferometerAntenna(
    name="tianqin",
    tdi_data=tdi_data_tianqin,
    orbit_model=available_orbit_models['Tianqin_analytic'],
    response_model=FDResponseModelMarset2018(),
    tdi_combination=FDMichelsonConstantEqualArm(generation=tdi_gen, orthogonal=True),
)

common_freqs = tdi_data_lisa.data_info.frequency_samples_array
wf_xhm = IMRPhenomXHM(common_freqs, f_ref, parameter_conversion=bilby.gw.conversion.generate_component_masses)
wf_xhm.update_waveform(params)
lisa.inject_signal(
    wf_xhm.waveform_container,
    params["ecliptic_longitude"],
    params["ecliptic_latitude"],
    params["polarization"],
    params["coalescence_time"],
)
taiji.inject_signal(
    wf_xhm.waveform_container,
    params["ecliptic_longitude"],
    params["ecliptic_latitude"],
    params["polarization"],
    params["coalescence_time"],
)
tianqin.inject_signal(
    wf_xhm.waveform_container,
    params["ecliptic_longitude"],
    params["ecliptic_latitude"],
    params["polarization"],
    params["coalescence_time"],
)

########################################################################################
# sampling settings
likelihood = LikelihoodBilbyInterface(
    waveform=wf_xhm,
    detector=(lisa, taiji, tianqin,),
    channels=('A', 'E', 'T')
    )

priors = {}
priors['chi_1'] = bilby.core.prior.Uniform(name='chi_1', minimum=-0.99, maximum=0.99)
priors['chi_2'] = bilby.core.prior.Uniform(name='chi_2', minimum=-0.99, maximum=0.99)
priors['ecliptic_longitude'] = bilby.core.prior.Uniform(name='ecliptic_longitude', minimum=0, maximum=2 * lal.PI, boundary='periodic')
priors['ecliptic_latitude'] = bilby.core.prior.Cosine(name='ecliptic_latitude')
priors['inclination'] = bilby.core.prior.Sine(name='inclination')
priors['polarization'] = bilby.core.prior.Uniform(name='polarization', minimum=0, maximum=lal.PI, boundary='periodic')
priors['reference_phase'] = bilby.core.prior.Uniform(name='reference_phase', minimum=0, maximum=2 * lal.PI, boundary='periodic')
priors['coalescence_time'] = bilby.core.prior.Uniform(name='coalescence_time', minimum=tc-100, maximum=tc+100)
priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=5e5, maximum=2e6)
priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.05, maximum=0.99)
priors['luminosity_distance'] = bilby.core.prior.LogUniform(name='luminosity_distance', minimum=1e4, maximum=5e6)
priors = bilby.core.prior.PriorDict(dictionary=priors)

label = "inj_MBHB_network_HM"
outdir = f"outdir_{label}"
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label=label,
    sampler='pymultinest',
    npoints=2048,
    )
