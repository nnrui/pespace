import sys
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/pespace")
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/tiwave")
# sys.path.append('/home/hydrogen/workspace/Space_GW/pespace')
# sys.path.append('/home/hydrogen/workspace/Space_GW/tiwave')
import os
from pathlib import Path

import numpy as np
import taichi as ti
import bilby
import h5py

from pespace.constants import DAY_SI, PI
from pespace.detectors import TDIChannelsData, SpaceborneInterferometer
from pespace.likelihood import FrequencyDomainLikelihood
from pespace.orbits import KaplerianHeliocentric
from tiwave.waveforms import IMRPhenomD

local_rank_id = int(os.environ['MPI_LOCALRANKID'])
gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False)

label = "LDC1-1_v1_MBHB"
outdir = f"output/outdir_{label}"
Path(outdir).mkdir(exist_ok=True)


########################################################################################
# Read in the data
prefix = "/home/changfenggroup/nrui/works/codes/gw_space/LDC_data"
# prefix = "/home/hydrogen/workspace/Space_GW/LDC/LDC_data"
dataname = "LDC1-1_MBHB_v1_1_TD"
file_path = f"{prefix}/{dataname}.hdf5"
with h5py.File(file_path, "r") as f:
    source_parameters = {}
    for parameter, value in f["H5LISA/GWSources/MBHB-0"].items():
        source_parameters[parameter] = value[()]
    TDI_data = f["H5LISA/PreProcess/TDIdata"][()]
time_array, X_array, Y_array, Z_array = TDI_data.T
A_array = (Z_array - X_array)/np.sqrt(2)
E_array = (X_array - 2*Y_array + Z_array)/np.sqrt(6)
T_array = (X_array + Y_array + Z_array)/np.sqrt(3)

cadence = source_parameters["Cadence"]
duration = source_parameters["CoalescenceTime"] - time_array[0] + DAY_SI
end_idx = int((duration + time_array[0]) // cadence)
input_TDI_data_array = np.array([A_array[:end_idx], E_array[:end_idx]])

# Set TDI data and detector
TDI_chans = ("A", "E")
TDI_gen = "1.5"
LDC_mbhb = TDIChannelsData(label="LDC1-1_MBHB", minimum_frequency=1e-4, maximum_frequency=0.1)
LDC_mbhb.set_time_domain_data_from_input_array(channels=TDI_chans, 
                                               generation=TDI_gen, 
                                               duration=duration, 
                                               cadence=cadence,
                                               TDI_data_array=input_TDI_data_array,
                                               start_time=time_array[0])
LDC_mbhb.Fourier_transform_time_domain_data_to_frequency_domain()
LDC_mbhb.set_frequency_domain_noise_power_density_from_model("LISA_SciRDv1")
det = SpaceborneInterferometer(name='LISA_LDC1-1_MBHB', TDI_data=LDC_mbhb, orbit=KaplerianHeliocentric(2.5e9, 0.0, 0.0))
det.initialize_response_container_in_frequency_domain()

# Set waveform
params_tiwave= dict(
    mass_1 = source_parameters["Mass1"],
    mass_2 = source_parameters["Mass2"],
    chi_1 = source_parameters["Spin1"],
    chi_2 = source_parameters["Spin2"],
    luminosity_distance = source_parameters["Distance"]*1000,  # note: the LDC using the unit of Gpc
    inclination = source_parameters["InitialPolarAngleL"],
    reference_phase = source_parameters["PhaseAtCoalescence"], # remember setting the reference_frequency to coalescence frequency
    coalescence_time = source_parameters["CoalescenceTime"],
    ecliptic_latitude = source_parameters["EclipticLatitude"],
    ecliptic_longitude = source_parameters["EclipticLongitude"],
    polarization = source_parameters["InitialAzimuthalAngleL"],
)
params_tiwave = bilby.gw.conversion.generate_mass_parameters(params_tiwave)
wf = IMRPhenomD(det.TDI_data.frequency_samples)
wf.update_waveform(params_tiwave)
f_peak = wf.amplitude_coefficients[None].f_peak/wf.source_parameters[None].M_sec
wf = IMRPhenomD(det.TDI_data.frequency_samples, reference_frequency=f_peak)

# Set prior
priors = {}
priors['chi_1'] = bilby.core.prior.Uniform(name='chi_1', minimum=-0.99, maximum=0.99)
priors['chi_2'] = bilby.core.prior.Uniform(name='chi_2', minimum=-0.99, maximum=0.99)
priors['ecliptic_longitude'] = bilby.core.prior.Uniform(name='ecliptic_longitude', minimum=0, maximum=2 * PI, boundary='periodic')
priors['ecliptic_latitude'] = bilby.core.prior.Cosine(name='ecliptic_latitude')
priors['inclination'] = bilby.core.prior.Sine(name='inclination')
priors['polarization'] = bilby.core.prior.Uniform(name='polarization', minimum=0, maximum=PI, boundary='periodic')
priors['reference_phase'] = bilby.core.prior.Uniform(name='reference_phase', minimum=0, maximum=2 * PI, boundary='periodic')
priors['coalescence_time'] = bilby.core.prior.Uniform(name='coalescence_time', minimum=params_tiwave['coalescence_time']-0.1*DAY_SI, maximum=params_tiwave['coalescence_time']+0.1*DAY_SI)

priors['total_mass'] = bilby.core.prior.LogUniform(name='total_mass', minimum=5e4, maximum=5e7)
priors['mass_ratio'] = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.05, maximum=0.99)
priors['luminosity_distance'] = bilby.core.prior.LogUniform(name='luminosity_distance', minimum=1e3, maximum=1e6)

# Run sampler and plot results
result = bilby.run_sampler(
    likelihood=FrequencyDomainLikelihood(wf, det),
    priors=priors,
    outdir=outdir,
    label=label,
    sampler='pymultinest',
    npoints=1024,
    # evidence_tolerance=0.1
    )
# result = bilby.core.result.read_in_result(f"{outdir}/{label}_result.json")
params_tiwave.pop("mass_1")
params_tiwave.pop("mass_2")
params_tiwave.pop("chirp_mass")
params_tiwave.pop("symmetric_mass_ratio")
result.plot_corner(parameters=params_tiwave, quantiles=[0.05, 0.95])
