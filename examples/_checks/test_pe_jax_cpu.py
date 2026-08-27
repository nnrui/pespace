from collections import namedtuple
import os

num_physical_cores = 16
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_physical_cores}"
os.environ[
    "XLA_FLAGS"
] += " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
print("num cores:", len(jax.devices("cpu")))

import lal
import numpy as np
import bilby

from pespace.detector.antenna import InterferometerAntenna, FDResponseModelMarset2018
from pespace.detector.tdi import TDIChannelData, FDMichelsonConstantEqualArm
from pespace.detector.orbit import KaplerianHeliocentric
from pespace.inference.interface_jax import WhittleLikelihoodJax
from pespace.inference.sampler import BlackjaxSMCwithNUTS
from pespace.waveform.interface_ripple import IMRPhenomXAS

########################################################################################
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
tdi_combination = FDMichelsonConstantEqualArm(
    generation=tdi_gen, orthogonal=True, backend="jax"
)
lisa = InterferometerAntenna(
    name="lisa",
    tdi_data=tdi_data,
    orbit_model=orbit_model,
    response_model=response_model,
    tdi_combination=tdi_combination,
    backend="jax",
)

wf_xas = IMRPhenomXAS(tdi_data.frequency_samples, f_ref)
WaveformParams = namedtuple("WaveformParams", wf_xas.required_params)
wf_params_inj = WaveformParams(
    **{name: params[name] for name in wf_xas.required_params}
)
lisa.inject_signal(
    wf_xas.get_waveform(wf_params_inj),
    params["ecliptic_longitude"],
    params["ecliptic_latitude"],
    params["polarization"],
    params["coalescence_time"],
)

########################################################################################
# set sampling
likelihood = WhittleLikelihoodJax(
    waveform=wf_xas, detector=lisa, channels=("A", "E", "T")
)

priors = {
    "chi_1": {"fixed": params["chi_1"]},
    "chi_2": {"fixed": params["chi_2"]},
    "ecliptic_longitude": {"fixed": params["ecliptic_longitude"]},
    "ecliptic_latitude": {"fixed": params["ecliptic_latitude"]},
    "inclination": {"fixed": params["inclination"]},
    "polarization": {"fixed": params["polarization"]},
    "reference_phase": {"fixed": params["reference_phase"]},
    "coalescence_time": {"fixed": params["coalescence_time"]},
    "luminosity_distance": {"fixed": params["luminosity_distance"]},
    "chirp_mass": {"distribution": "Uniform", "low": 5e5, "high": 2e6},
    "mass_ratio": {"distribution": "Uniform", "low": 0.05, "high": 0.99},
    "symmetric_mass_ratio": {
        "transform": lambda mass_ratio: mass_ratio / (1.0 + mass_ratio) ** 2,
        "from": ("mass_ratio",),
    },
}

sampler = BlackjaxSMCwithNUTS(likelihood, priors, rkey_seed=2026, label="test_pe_jax_cpu")
results = sampler.run()
