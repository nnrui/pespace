import os
# The jax backend is unused on this taichi path; keep jax off the GPU
os.environ['JAX_PLATFORMS']='cpu'
# Select the GPU for this MPI rank when the scheduler provides these env variables
local_rank_id = int(os.environ.get('MPI_LOCALRANKID', 0))
gpus_list = [g.strip() for g in os.environ.get('GPU_DEVICE_ORDINAL', '').split(',') if g.strip()]
if gpus_list:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_list[local_rank_id % len(gpus_list)]

import bilby
import lal
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
# ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

from pespace.detector.antenna import InterferometerAntenna, FDResponseModelMarset2018
from pespace.detector.tdi import TDIChannelData, FDMichelsonConstantEqualArm
from pespace.detector.orbit import available_orbit_models
from pespace.inference.interface_bilby import LikelihoodBilbyInterfaceTaichi
from tiwave.waveforms import IMRPhenomXAS

########################################################################################
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
params = bilby.gw.conversion.generate_mass_parameters(params)

tdi_data = TDIChannelData()
tdi_data.set_fd_data_from_zero(
    channels=tdi_chan, 
    duration=duration, 
    delta_time=dt,
    start_time=t_start,
    minimum_frequency=f_min,
    maximum_frequency=f_max,
    )
tdi_data.set_fd_noise_power_density_from_model("LISA_SciRDv1", tdi_generation=tdi_gen)
noise = tdi_data.get_fd_noise_realization()
tdi_data.add_into_fd_data(noise)

orbit_model = available_orbit_models['LISA_analytic']
response_model = FDResponseModelMarset2018()
tdi_combination = FDMichelsonConstantEqualArm(generation=tdi_gen, orthogonal=True)
lisa = InterferometerAntenna(
    name="lisa",
    tdi_data=tdi_data,
    orbit_model=orbit_model,
    response_model=response_model,
    tdi_combination=tdi_combination,
)

wf_xas = IMRPhenomXAS(tdi_data.frequency_samples, f_ref, parameter_conversion=bilby.gw.conversion.generate_component_masses)
wf_xas.update_waveform(params)
lisa.inject_signal(
    wf_xas.waveform_container,
    params["ecliptic_longitude"],
    params["ecliptic_latitude"],
    params["polarization"],
    params["coalescence_time"],
)

########################################################################################
# set sampling
likelihood = LikelihoodBilbyInterfaceTaichi(
    waveform=wf_xas,
    detector=lisa,
    channels=('A', 'E', 'T')
    )

priors = {}
priors["chi_1"] = bilby.core.prior.Uniform(name="chi_1", minimum=-0.99, maximum=0.99)
priors["chi_2"] = bilby.core.prior.Uniform(name="chi_2", minimum=-0.99, maximum=0.99)
priors["ecliptic_longitude"] = bilby.core.prior.Uniform(name="ecliptic_longitude", minimum=0, maximum=2 * lal.PI, boundary="periodic")
priors["ecliptic_latitude"] = bilby.core.prior.Cosine(name="ecliptic_latitude")
priors["inclination"] = bilby.core.prior.Sine(name="inclination")
priors["polarization"] = bilby.core.prior.Uniform(name="polarization", minimum=0, maximum=lal.PI, boundary="periodic")
priors["reference_phase"] = bilby.core.prior.Uniform(name="reference_phase", minimum=0, maximum=2 * lal.PI, boundary="periodic")
priors["coalescence_time"] = bilby.core.prior.Uniform(name="coalescence_time", minimum=tc-20, maximum=tc+20)
priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(name="chirp_mass", minimum=params['chirp_mass']-5e3, maximum=params['chirp_mass']+5e3)
priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(name="mass_ratio", minimum=0.2, maximum=0.9)
priors["luminosity_distance"] = bilby.core.prior.Uniform(name="luminosity_distance", minimum=4e4, maximum=8e4)
priors = bilby.core.prior.PriorDict(dictionary=priors)

label = "inj_MBHB_LISA"
outdir = f"outdir_{label}"
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label=label,
    sampler='pymultinest',
    npoints=2048,
    )
