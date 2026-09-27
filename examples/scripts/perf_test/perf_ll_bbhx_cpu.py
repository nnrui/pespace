from cuda.pathfinder import load_nvidia_dynamic_lib
load_nvidia_dynamic_lib("cublas")
load_nvidia_dynamic_lib("cusparse")

import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

backend = "cpu"
import numpy as xp
arch = "cpu"

import bilby
import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from bbhx.likelihood import Likelihood
from bbhx.utils.constants import PC_SI, YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
from pespace.detector.tdi import TDIChannelData


DAYJUL_SI = 86400.0
tdi_gen = "2.0"
chan = 'AET'
dt = 10.0
f_min = 1e-4
f_max = 0.5 * (1 / dt)
f_ref = 0.0 

# using PhenomD, single (2,2) harmonic
ampphase_kwards = dict(
    run_phenomd=True,
)
modes = [(2, 2)]

orbits = EqualArmlengthOrbits(force_backend=backend)
orbits.configure(linear_interp_setup=True)
response_kwards = dict(
    orbits=orbits,
    TDItag=chan,
    tdi2=True,
)

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
params_samples = bilby.gw.conversion.generate_mass_parameters(params_samples)

params_warmup = {
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
params_warmup = bilby.gw.conversion.generate_mass_parameters(params_warmup)


class DirectLikelihood(Likelihood):
    """
    bbhx ``Likelihood`` restricted to ``direct=True`` (full resolution, no interpolation).
    """

    def get_ll(
        self,
        params,
        return_extracted_snr=False,
        phase_marginalize=False,
        **waveform_kwargs
    ):

        # store info
        self.phase_marginalize = phase_marginalize
        self.return_extracted_snr = return_extracted_snr

        # setup kwargs properly
        waveform_kwargs["freqs"] = self.data_freqs
        waveform_kwargs["fill"] = False
        # ---------------------------------------------------------------------------------
        # change 1
        # waveform_kwargs["direct"] = False  (sparse grid + cubic-spline interp)
        waveform_kwargs["direct"] = True  # full-resolution template, no interpolation
        waveform_kwargs["squeeze"] = False  # stock default -> keep (num_bin_all, 3, N)
        waveform_kwargs["compress"] = True  # stock default -> combine harmonics
        # ---------------------------------------------------------------------------------

        # get information from waveform generators
        # ---------------------------------------------------------------------------------
        # change 2
        # templateChannels, inds_start, ind_lengths = self.waveform_gen(
        #     *params, **waveform_kwargs
        # )
        templateChannels = self.waveform_gen(*params, **waveform_kwargs)
        templateChannels = list(templateChannels)  # (num_bin_all, 3, N) -> per-binary (3, N)
        inds_start = np.zeros(self.waveform_gen.num_bin_all, dtype=np.int32)
        ind_lengths = np.full(
            self.waveform_gen.num_bin_all, self.data_stream_length, dtype=np.int32
        )
        # ---------------------------------------------------------------------------------

        # get flattened templates to remove channel dimension
        templateChannels = [tc.flatten() for tc in templateChannels]

        # adjust for cupy vs numpy
        try:
            templateChannels_ptrs = np.asarray(
                [tc.data.ptr for tc in templateChannels], dtype=np.int64
            )
        except AttributeError:
            templateChannels_ptrs = np.asarray(
                [tc.__array_interface__["data"][0] for tc in templateChannels],
                dtype=np.int64,
            )

        # initialize inner product info
        self.d_h = np.zeros(self.waveform_gen.num_bin_all, dtype=self.xp.complex128)
        self.h_h = np.zeros(self.waveform_gen.num_bin_all, dtype=self.xp.complex128)

        self.like_gen(
            self.d_h,
            self.h_h,
            self.data_channels,
            self.noise_factors,
            templateChannels_ptrs,
            inds_start,
            ind_lengths,
            self.data_stream_length,
            self.waveform_gen.num_bin_all,
        )

        # phase marginalize in d_h term
        d_h_temp = self.d_h if not self.phase_marginalize else self.xp.abs(self.d_h)
        out = -1 / 2 * (self.d_d + self.h_h - 2 * d_h_temp).real
        # get out of cupy if needed
        try:
            out = out.get()

        except AttributeError:
            pass

        if self.return_extracted_snr:
            return np.array([out, d_h_temp.real / np.sqrt(self.h_h.real)]).T
        else:
            return out


wave_gen = BBHWaveformFD(
    amp_phase_kwargs=ampphase_kwards,
    response_kwargs=response_kwards,
    force_backend=backend,
)


def timing_ll_calculation(duration):
    tdi_data = TDIChannelData()
    tdi_data.set_fd_data_from_zero(
            channels=tuple(chan),
            duration=duration,
            delta_time=dt,
            start_time=0.0,
            minimum_frequency=f_min,
            maximum_frequency=f_max,
        )
    tdi_data.set_fd_noise_power_density_from_model("LISA_SciRDv1", tdi_generation=tdi_gen)
    freq_length = tdi_data.data_info.frequency_series_length
    data_freqs = xp.asarray(tdi_data.data_info.frequency_samples_array)
    noise = tdi_data.get_fd_noise_realization(seed=2026, output_type="numpy")
    psd_dict = tdi_data.fd_noise_power_density.to_numpy()  # {channel: (N,)}
    psd = xp.asarray(np.stack([psd_dict[channel] for channel in chan]))

    t_ref = 0.8 * duration
    inject_vector = np.asarray(
        [
            params_inject["mass_1"],
            params_inject["mass_2"],
            params_inject["chi_1"],
            params_inject["chi_2"],
            params_inject["luminosity_distance"] * 1e6 * PC_SI,
            params_inject["reference_phase"],
            f_ref,
            params_inject["inclination"],
            params_inject["ecliptic_longitude"],
            params_inject["ecliptic_latitude"],
            params_inject["polarization"],
            t_ref + params_inject["coalescence_time"],
        ]
    )
    waveform_kwargs: dict = dict(
        modes=modes,
        t_obs_start=0.0,
        t_obs_end=duration / YRSID_SI,
    )

    data_channels = wave_gen(
        *inject_vector,
        freqs=data_freqs,
        direct=True,  # data is generated without interpolation
        **waveform_kwargs,
    )[0]  # direct=True + compress=True returns a dense (num_bin_all, 3, N) array
    data_channels = data_channels + xp.asarray(
        np.stack([getattr(noise, channel) for channel in chan])
    )

    likelihood = DirectLikelihood(
        wave_gen, data_freqs, data_channels, psd, force_backend=backend
    )

    # (12, test_num): [m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref]
    params_batched = np.vstack(
        [
            params_samples["mass_1"],
            params_samples["mass_2"],
            params_samples["chi_1"],
            params_samples["chi_2"],
            params_samples["luminosity_distance"] * 1e6 * PC_SI,
            params_samples["reference_phase"],
            np.full(test_num, f_ref),
            params_samples["inclination"],
            params_samples["ecliptic_longitude"],
            params_samples["ecliptic_latitude"],
            params_samples["polarization"],
            t_ref + params_samples["coalescence_time"],
        ]
    )
    params_warmup_batched = np.vstack(
        [
            params_warmup["mass_1"],
            params_warmup["mass_2"],
            params_warmup["chi_1"],
            params_warmup["chi_2"],
            params_warmup["luminosity_distance"] * 1e6 * PC_SI,
            params_warmup["reference_phase"],
            np.full(test_num, f_ref),
            params_warmup["inclination"],
            params_warmup["ecliptic_longitude"],
            params_warmup["ecliptic_latitude"],
            params_warmup["polarization"],
            t_ref + params_warmup["coalescence_time"],
        ]
    )

    st = time.perf_counter()
    ll_inj = likelihood.get_ll(inject_vector, **waveform_kwargs)
    print("log_likelihood at injection: ", ll_inj)
    print("time (first call, includes warm-up): ", time.perf_counter() - st)

    params_loop = params_batched.T
    ll_loop = []
    st = time.perf_counter()
    for i in range(test_num):
        ll_loop.append(likelihood.get_ll(params_loop[i], **waveform_kwargs))
    time_per_ll = (time.perf_counter() - st) / test_num
    print("time for clean call (average)", time_per_ll)

    # warm-up for the batched code path (first-call setup), excluded from timing
    likelihood.get_ll(params_warmup_batched, **waveform_kwargs)

    st = time.perf_counter()
    ll_batched = likelihood.get_ll(params_batched, **waveform_kwargs)
    time_per_ll_batched = (time.perf_counter() - st) / test_num
    print("time for batched call (average)", time_per_ll_batched)

    assert np.allclose(np.ravel(ll_loop), ll_batched, rtol=1e-12, atol=0.0)

    return freq_length, time_per_ll, time_per_ll_batched


perf_ll = []
for duration in np.round(np.exp(np.linspace(0, np.log(2*365), 24))) * DAYJUL_SI:
    perf_ll.append(timing_ll_calculation(duration))
perf_ll_np = np.array(perf_ll)

output = f"perf_bbhx_ll_{arch}.dat"
np.savetxt(output, perf_ll_np)
