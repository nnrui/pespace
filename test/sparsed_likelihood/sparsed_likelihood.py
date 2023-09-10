
import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

import numpy as np
import bilby 

# from peSpace.detectors import LISALike
# from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
# from peSpace.likelihood import SparseLikelihood

# det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
#                maximum_frequency=1, TDI_channels=('A', 'E'), 
#                TDI_generation='2.0')

# parameters = dict(total_mass=4e6,
#                   mass_ratio=1/3,
#                   luminosity_distance=36594.3,
#                   chi_1 = 0.2,
#                   chi_2 = 0.4,
#                   coalescence_time=0.0,
#                   ecliptic_longitude = 3.335,
#                   ecliptic_latitude = 1.468,
#                   polarization = 2.237,
#                   inclination = 1.047,
#                   coalescence_phase = 0.0,)
# parameters = bilby.gw.conversion.generate_all_bbh_parameters(parameters)

# det.generate_FD_noise_realization_from_psd(seed=0)
# det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
# det.save_data()

# # det.set_strains_FD_from_file('LISA_detector_data_None.hdf5')
# likelihood_sparse = SparseLikelihood(sparse_ratio=1000, waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
#                                      detector=det)

# priors = {}
# priors['total_mass'] = bilby.core.prior.Uniform(name='M', minimum=1e6, maximum=1e7)
# priors['mass_ratio'] = bilby.core.prior.Uniform(name='q', minimum=1.0, maximum=10.0)
# priors['chi_1'] = bilby.core.prior.Uniform(name='chi1', minimum=-1.0, maximum=1.0)
# priors['chi_2'] = bilby.core.prior.Uniform(name='chi2', minimum=-1.0, maximum=1.0)
# priors['luminosity_distance'] = bilby.core.prior.Uniform(name='dist', minimum=10e3, maximum=100e3)
# priors['coalescence_time'] = bilby.core.prior.Uniform(name='Deltat', minimum=-600, maximum=600)
# priors['coalescence_phase'] = bilby.core.prior.Uniform(name='phi', minimum=-np.pi, maximum=np.pi, boundary='periodic')
# priors['ecliptic_longitude'] = bilby.core.prior.Uniform(name='lambda', minimum=-np.pi, maximum=np.pi, boundary='periodic')
# priors['ecliptic_latitude'] = bilby.core.prior.Cosine(name='beta')
# priors['polarization'] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
# priors['inclination'] = bilby.core.prior.Sine(name='inc')


# # priors['total_mass'] = parameters['total_mass']
# # priors['mass_ratio'] = parameters['mass_ratio']
# priors['chi_1'] = parameters['chi_1']
# priors['chi_2'] = parameters['chi_2']
# # priors['luminosity_distance'] = parameters['luminosity_distance']
# priors['coalescence_time'] = parameters['coalescence_time']
# priors['coalescence_phase'] = parameters['coalescence_phase']
# # priors['ecliptic_longitude'] =parameters['ecliptic_longitude']
# # priors['ecliptic_latitude'] = parameters['ecliptic_latitude']
# priors['polarization'] = parameters['polarization']
# priors['inclination'] = parameters['inclination']


# # RUN SAMPLER
# outdir = 'outdir_pespace_sparsed_likelihood'
# label = 'sparsed_likelihood_M_q_dL_lam_beta'
# result = bilby.run_sampler(
#     likelihood=likelihood_sparse,
#     priors=priors,
#     conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
#     outdir=outdir,
#     label=label,
#     sampler='dynesty',
#     npoints=2048,
#     # walks=200,
#     # nact=10,
#     # maxmcmc=15000,
#     queue_size=72,
#     dlogz=0.1)


# PLOT RESULT
result = bilby.result.read_in_result('/home/hydrogen/workspace/Space_GW/peSpace/test/sparsed_likelihood/outdir_pespace_sparsed_likelihood/sparsed_likelihood_M_q_dL_lam_beta_result.json')
result.plot_corner(parameters=dict(total_mass=4e6,
                                   mass_ratio=1/3,
                                   luminosity_distance=36594.3,
                                   ecliptic_longitude = 3.335,
                                   ecliptic_latitude = 1.468,
                                   coalescence_phase = 0.0,), 
                   quantiles=[0.05, 0.95])
