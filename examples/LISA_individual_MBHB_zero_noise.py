import sys
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/pespace")
sys.path.append("/home/changfenggroup/nrui/works/codes/gw_space/tiwave")
import os

import taichi as ti
import bilby
from pespace.constants import DAY_SI
from pespace.detectors import TDIChannelsData, SpaceborneInterferometer
from pespace.likelihood import FrequencyDomainLikelihood
from tiwave.waveforms import IMRPhenomD

local_rank_id = int(os.environ['MPI_LOCALRANKID'])
gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
ti.init(arch=ti.cuda, default_fp=ti.f64, cpu_max_num_threads=1)


duration = 30*DAY_SI  # 1 year observation
cadence = 10
TDI_chans = ("A", "E")
TDI_gen = "1.5"
parameters = dict(
    total_mass=3089053.9,
    mass_ratio=0.11,
    chi_1=0.3986046314480332,
    chi_2=0.5465882372512956,
    luminosity_distance=6000.42017466175677,
    inclination=0.30782038099413395,
    reference_phase=0.0,
    coalescence_time=0.0,
    ecliptic_latitude=-0.5256036732051035,
    ecliptic_longitude=1.1637,
    polarization=1.30782038099413395,
)

lisa_mbhb = TDIChannelsData(label="inj_MBHB_lisa", minimum_frequency=1e-5, maximum_frequency=0.1)
lisa_mbhb.set_frequency_domain_data_with_zero_value(channels=TDI_chans, generation=TDI_gen, duration=duration, cadence=cadence)
lisa_mbhb.set_frequency_domain_noise_power_density_from_model("LISA_SciRDv1")

lisa = SpaceborneInterferometer(name='LISA', TDI_data=lisa_mbhb, orbit="LISA_analytic")
lisa.initialize_response_container_in_frequency_domain()
wf = IMRPhenomD(lisa.TDI_data.frequency_samples)
wf.update_waveform(parameters)
lisa.inject_frequency_domain_signal(wf.waveform_container, parameters['ecliptic_longitude'], parameters['ecliptic_latitude'], parameters['polarization'])

priors = {}
priors['chi_1'] = parameters["chi_1"]
priors['chi_2'] = parameters["chi_2"]
priors['ecliptic_longitude'] = parameters["ecliptic_longitude"]
priors['ecliptic_latitude'] = parameters["ecliptic_latitude"]
priors['inclination'] = parameters["inclination"]
priors['polarization'] = parameters["polarization"]
priors['reference_phase'] = parameters["reference_phase"]
priors['coalescence_time'] = parameters["coalescence_time"]
priors['total_mass'] = bilby.core.prior.LogUniform(name='total_mass', minimum=2.5e6, maximum=3.5e6)
priors['mass_ratio'] = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.05, maximum=0.5)
priors['luminosity_distance'] = bilby.gw.prior.Uniform(name='luminosity_distance', minimum=4000, maximum=8000)

label = "LISA_MBHB_injection_zero_noise"
outdir = f"outdir_{label}"
result = bilby.run_sampler(
    likelihood=FrequencyDomainLikelihood(wf, lisa),
    priors=priors,
    outdir=outdir,
    label=label,
    sampler='pymultinest',
    npoints=2048)

# PLOT RESULT
plot_parameter = ['total_mass','mass_ratio','luminosity_distance']
result.plot_corner(parameters=plot_parameter, 
                   truths=parameters,
                   quantiles=[0.05, 0.95])
