import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import bilby
import numpy as np
import pandas as pd
import taichi as ti
ti.init(arch=ti.cuda, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

from pespace.detector.tdi import TDIChannelData
from tiwave.waveforms import IMRPhenomXAS


DAYJUL_SI = 86400.0
tdi_gen = "2.0"
tdi_chan = ("A", "E", "T")
dt = 10
f_min = 1e-4
f_max = 0.5 * (1 / dt)
f_ref = f_min

params_inject = dict(
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
    coalescence_time=0.0,
)
params_inject = bilby.gw.conversion.generate_mass_parameters(params_inject)

rng = np.random.default_rng(seed=2026)
test_num = 10
params_samples = {
    "chi_1": rng.uniform(-0.99, 0.99, test_num),
    "chi_2": rng.uniform(-0.99, 0.99, test_num),
    "ecliptic_longitude": rng.uniform(0.0, 2 * np.pi, test_num),
    "ecliptic_latitude": rng.uniform(-np.pi / 2, np.pi / 2, test_num),
    "inclination": rng.uniform(0.0, np.pi, test_num),
    "polarization": rng.uniform(0.0, np.pi, test_num),
    "reference_phase": rng.uniform(0.0, 2 * np.pi, test_num),
    "coalescence_time": rng.uniform(-100, 100, test_num),
    "chirp_mass": rng.uniform(5e5, 2e6, test_num),
    "mass_ratio": rng.uniform(0.05, 0.99, test_num),
    "luminosity_distance": np.exp(rng.uniform(np.log(1e4), np.log(1e6), test_num)),
}
params_samples = pd.DataFrame(params_samples)
params_samples = bilby.gw.conversion.generate_mass_parameters(params_samples)


def timing_wf_calculation(duration):
    tdi_data = TDIChannelData()
    tdi_data.set_fd_data_from_zero(
        channels=tdi_chan,
        duration=duration,
        delta_time=dt,
        start_time=0.0,
        minimum_frequency=f_min,
        maximum_frequency=f_max,
    )
    freq_length = tdi_data.data_info.frequency_series_length

    wf_xas = IMRPhenomXAS(tdi_data.frequency_samples, f_ref, include_tf=False)
    wf_xas.update_waveform(params_inject) # call ahead of timing to compile and cache
    ti.sync() # flush the warm-up launch (CUDA kernels are launched asynchronously)

    st = time.perf_counter()
    for i in range(test_num):
        wf_xas.update_waveform(params_samples.iloc[i])
    ti.sync()
    ed = time.perf_counter()
    time_per_wf = (ed - st) / test_num

    return freq_length, time_per_wf


perf = []
for duration in np.round(np.exp(np.linspace(0, np.log(2*365), 24))) * DAYJUL_SI:
    perf.append(timing_wf_calculation(duration))
perf_np = np.array(perf)

arch = ti.lang.impl.current_cfg().arch.name
fp = ti.lang.impl.current_cfg().default_fp.to_string()
np.savetxt(f"perf_ti_wf_{arch}_{fp}.dat", perf_np)
