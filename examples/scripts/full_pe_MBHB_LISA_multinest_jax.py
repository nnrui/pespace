import os
_local_rank = int(os.environ.get("MPI_LOCALRANKID"))
_gpus = ["0", "1"]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[_local_rank % len(_gpus)]
print(f"[rank {_local_rank}] using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

from collections import namedtuple

import bilby
import lal
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

from pespace.detector.antenna import InterferometerAntenna, FDResponseModelMarset2018
from pespace.detector.orbit import KaplerianHeliocentric
from pespace.detector.tdi import FDMichelsonConstantEqualArm, TDIChannelData
from pespace.inference.interface_bilby import LikelihoodBilbyInterfaceJax
from pespace.waveform.interface_ripple import IMRPhenomXAS


###############################################################################
# set injection
tdi_gen = "2.0"
tdi_chan = ("A", "E", "T")

dt = 10
f_min = 1e-4
f_max = 0.5 * (1 / dt)
f_ref = f_min
t_start = 0.0

num_tsamples = 2 ** np.ceil(np.log2(7 * lal.DAYJUL_SI / dt))
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

tdi_data = TDIChannelData(backend="jax")
tdi_data.set_fd_data_from_zero(
    channels=tdi_chan,
    duration=duration,
    delta_time=dt,
    start_time=t_start,
    minimum_frequency=f_min,
    maximum_frequency=f_max,
)
tdi_data.set_fd_noise_power_density_from_model("LISA_SciRDv1", tdi_generation=tdi_gen)
noise = tdi_data.get_fd_noise_realization(seed=2026, output_type="numpy")
tdi_data.add_into_fd_data(noise)

orbit_model = KaplerianHeliocentric(2.5e9, 0.0, -np.pi / 9, backend="jax")
response_model = FDResponseModelMarset2018(backend="jax")
tdi_combination = FDMichelsonConstantEqualArm(generation=tdi_gen, orthogonal=True, backend="jax")
lisa = InterferometerAntenna(
    name="lisa",
    tdi_data=tdi_data,
    orbit_model=orbit_model,
    response_model=response_model,
    tdi_combination=tdi_combination,
    backend="jax",
)

wf_xas = IMRPhenomXAS(tdi_data.frequency_samples, f_ref)

# injection: the ripple waveform needs chirp_mass/symmetric_mass_ratio/...
params_inj = bilby.gw.conversion.generate_mass_parameters(params)
WaveformParams = namedtuple("WaveformParams", wf_xas.required_params)
wf_params_inj = WaveformParams(**{name: params_inj[name] for name in wf_xas.required_params})

lisa.inject_signal(
    wf_xas.get_waveform(wf_params_inj),
    params["ecliptic_longitude"],
    params["ecliptic_latitude"],
    params["polarization"],
    params["coalescence_time"],
)

###############################################################################
# set sampling
def q_to_eta(params):
    q = params["mass_ratio"]
    eta = q / (1.0 + q) ** 2
    params["symmetric_mass_ratio"] = eta
    return params

likelihood = LikelihoodBilbyInterfaceJax(
    waveform=wf_xas,
    detector=lisa,
    channels=("A", "E", "T"),
    parameter_conversion=q_to_eta,
)

priors = {}
# priors["chi_1"] = bilby.core.prior.Uniform(name="chi_1", minimum=-0.99, maximum=0.99)
# priors["chi_2"] = bilby.core.prior.Uniform(name="chi_2", minimum=-0.99, maximum=0.99)
# priors["ecliptic_longitude"] = bilby.core.prior.Uniform(name="ecliptic_longitude", minimum=0, maximum=2 * lal.PI, boundary="periodic")
# priors["ecliptic_latitude"] = bilby.core.prior.Cosine(name="ecliptic_latitude")
# priors["inclination"] = bilby.core.prior.Sine(name="inclination")
# priors["polarization"] = bilby.core.prior.Uniform(name="polarization", minimum=0, maximum=lal.PI, boundary="periodic")
# priors["reference_phase"] = bilby.core.prior.Uniform(name="reference_phase", minimum=0, maximum=2 * lal.PI, boundary="periodic")
# priors["coalescence_time"] = bilby.core.prior.Uniform(name="coalescence_time", minimum=tc - 100, maximum=tc + 100)
priors["chi_1"] = params_inj["chi_1"]
priors["chi_2"] = params_inj["chi_2"]
priors["ecliptic_longitude"] = params_inj["ecliptic_longitude"]
priors["ecliptic_latitude"] = params_inj["ecliptic_latitude"]
priors["inclination"] = params_inj["inclination"]
priors["polarization"] = params_inj["polarization"]
priors["reference_phase"] = params_inj["reference_phase"]
priors["coalescence_time"] = params_inj["coalescence_time"]

priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(name="chirp_mass", minimum=5e5, maximum=2e6)
priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(name="mass_ratio", minimum=0.05, maximum=0.99)
priors["luminosity_distance"] = bilby.core.prior.LogUniform(name="luminosity_distance", minimum=1e4, maximum=5e6)
priors = bilby.core.prior.PriorDict(dictionary=priors)

label = "full_pe_MBHB_LISA_multinest_jax"
outdir = f"outdir_{label}"
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label=label,
    sampler="pymultinest",
    npoints=2048,
)
