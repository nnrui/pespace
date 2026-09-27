import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import bilby
import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
fp = "f64"
import jax.numpy as jnp
assert len(jax.devices()) ==1 
arch = jax.devices()[0].platform

# gen_IMRPhenomXAS_hphc is used on purpose (no tf array), to be fair with the
# tiwave scripts, which are run with include_tf=False.
from pespace.detector.tdi import TDIChannelData
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc


DAYJUL_SI = 86400.0
tdi_chan = ("A", "E", "T")
dt = 10
f_min = 1e-4
f_max = 0.5 * (1 / dt)
f_ref = f_min

rng = np.random.default_rng(seed=2026)
test_num = 10
sample_num = 2 * test_num
params_samples = {
    "chi_1": rng.uniform(-0.99, 0.99, sample_num),
    "chi_2": rng.uniform(-0.99, 0.99, sample_num),
    "ecliptic_longitude": rng.uniform(0.0, 2 * np.pi, sample_num),
    "ecliptic_latitude": rng.uniform(-np.pi / 2, np.pi / 2, sample_num),
    "inclination": rng.uniform(0.0, np.pi, sample_num),
    "polarization": rng.uniform(0.0, np.pi, sample_num),
    "reference_phase": rng.uniform(0.0, 2 * np.pi, sample_num),
    "coalescence_time": rng.uniform(-100, 100, sample_num),
    "chirp_mass": rng.uniform(5e5, 2e6, sample_num),
    "mass_ratio": rng.uniform(0.05, 0.99, sample_num),
    "luminosity_distance": np.exp(rng.uniform(np.log(1e4), np.log(1e6), sample_num)),
}
params_samples = pd.DataFrame(params_samples)
params_samples = bilby.gw.conversion.generate_mass_parameters(params_samples)

params_array = np.column_stack(
    [
        params_samples["chirp_mass"],
        params_samples["symmetric_mass_ratio"],
        params_samples["chi_1"],
        params_samples["chi_2"],
        params_samples["luminosity_distance"],
        np.zeros(sample_num),
        params_samples["reference_phase"],
        params_samples["inclination"],
    ]
)
params_warmup = jnp.asarray(params_array[:test_num])
params_timed = jnp.asarray(params_array[test_num:])


# jit'ed ripple IMRPhenomXAS h+/hx for one system (no tf array)
waveform_hphc = jax.jit(
    lambda freqs, params: gen_IMRPhenomXAS_hphc(freqs, params, f_ref)
)


def timing_wf_calculation(duration: float) -> tuple[int, float]:
    tdi_data = TDIChannelData(backend="jax")
    tdi_data.set_fd_data_from_zero(
        channels=tdi_chan,
        duration=duration,
        delta_time=dt,
        start_time=0.0,
        minimum_frequency=f_min,
        maximum_frequency=f_max,
    )
    freq_length = tdi_data.data_info.frequency_series_length
    frequencies = tdi_data.frequency_samples

    waveform_compiled = waveform_hphc.lower(frequencies, params_warmup[0]).compile()

    pols_warmup = []
    for params in params_warmup:
        pols_warmup.append(waveform_compiled(frequencies, params))
    jax.block_until_ready(pols_warmup)

    st = time.perf_counter()
    pols_timed = []
    for params in params_timed:
        pols_timed.append(waveform_compiled(frequencies, params))
    jax.block_until_ready(pols_timed) # sync once for the whole queue
    ed = time.perf_counter()
    time_per_wf = (ed - st) / test_num

    return freq_length, time_per_wf


perf = []
for duration in np.round(np.exp(np.linspace(0, np.log(2*365), 24))) * DAYJUL_SI:
    perf.append(timing_wf_calculation(duration))
perf_np = np.array(perf)

np.savetxt(f"perf_jax_wf_{arch}_{fp}.dat", perf_np)
