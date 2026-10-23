import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from collections import namedtuple

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

import numpy as np
import lal
import bilby

from pespace.detector.antenna import (
    FDResponseModelMarset2018,
    InterferometerAntenna,
)
from pespace.detector.orbit import KaplerianHeliocentric
from pespace.detector.tdi import (
    FDMichelsonConstantEqualArm,
    TDIChannelData,
)
from pespace.inference.likelihood_jax import WhittleLikelihoodJax
from pespace.inference.sampler.flowmc import FlowMCSampler
from pespace.inference.sampler.base import Cosine, Sine
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
likelihood = WhittleLikelihoodJax(
    waveform=wf_xas,
    detector=lisa,
    channels=("A", "E", "T"),
)

prior_dict = {
    "chi_1": {"distribution": "Uniform", "low": -0.99, "high": 0.99},
    "chi_2": {"distribution": "Uniform", "low": -0.99, "high": 0.99},
    "ecliptic_longitude": {"distribution": "Uniform", "low": 0.0, "high": 2.0 * np.pi},
    "ecliptic_latitude": {"distribution": Cosine},
    "inclination": {"distribution": Sine},
    "polarization": {"distribution": "Uniform", "low": 0.0, "high": np.pi},
    "reference_phase": {"distribution": "Uniform", "low": 0.0, "high": 2.0 * np.pi},
    "coalescence_time": {"distribution": "Uniform", "low": tc-10.0, "high": tc+10.0},
    "chirp_mass": {"distribution": "Uniform", "low": params_inj['chirp_mass']-5e3, "high": params_inj['chirp_mass']+5e3},
    "mass_ratio": {"distribution": "Uniform", "low": 0.2, "high": 0.9},
    "luminosity_distance": {"distribution": "Uniform", "low": 4e4, "high": 8e4},

    "symmetric_mass_ratio": {
        "transform": lambda mass_ratio: mass_ratio / (1.0 + mass_ratio) ** 2,
        "from": ("mass_ratio",),
    },
}

    
bundle_config = dict(
    n_local_steps=100,
    n_global_steps=1000,
    n_training_loops=20,
    n_production_loops=10,
    n_epochs=20,
    rq_spline_hidden_units=(128, 128),
    rq_spline_n_bins=16,
    rq_spline_n_layers=10,
    n_NFproposal_batch_size=1,
    n_max_examples=50000,
    global_thinning=100,
    early_stopping=True,
    early_stopping_tolerance=0.1,
    verbose=True,
    chain_batch_size=64,
    mala_step_size=np.array(
        [1e-2, 1e-2, 1e-3, 2e-3, 1e-1,
         6e-1, 6e-1, 8e-2, 4e-2, 5e-3, 6e-2]),
    adapt_step_size=True, 
    adapt_step_size_per_dim=False,
    history_window=500,
    n_temperatures=8, 
    max_temperature=20,
)

label="full_pe_MBHB_LISA_flowmc"
n_chains = 64

sampler = FlowMCSampler(
    likelihood,
    prior_dict,
    label=label,
    outdir=f"outdir_{label}",
    rkey_seed=2026,
    resource_strategy_bundles="RQSpline_MALA_PT",
    bundle_config=bundle_config,
    n_chains=n_chains,
    likelihood_batch_size=64,
    checkpoint_interval=600.0,

)
sampler.run()

