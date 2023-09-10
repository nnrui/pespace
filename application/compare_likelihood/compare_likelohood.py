import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')
import time
from multiprocessing import Pool

import numpy as np
from bilby.gw.conversion import total_mass_and_mass_ratio_to_component_masses
  

from peSpace.detectors import LISALike
from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from peSpace.likelihood import FullLikelihood, SparseLikelihood




det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=1, TDI_channels=('A', 'E', 'T'), 
               TDI_generation='2.0')

parameters = dict(mass_1 = 1.5e6,
                  mass_2 = 5e5,
                  luminosity_distance=36594.3,
                  chi_1 = 0.2,
                  chi_2 = 0.4,
                  coalescence_time=0.0,
                  ecliptic_longitude = 3.335,
                  ecliptic_latitude = 1.468,
                  polarization = 2.237,
                  inclination = 3.14/3.0,
                  coalescence_phase = 0.0,)

# det.generate_FD_noise_realization_from_psd(seed=0)
# det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
# det.save_data()

# det.set_strains_FD_from_file('LISA_detector_data_None.hdf5')
# likelihood_sparse = SparseLikelihood(sparse_ratio=1, waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
#                             detector=det)
# likelihood_full = FullLikelihood(waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
#                             detector=det)
# likelihood_sparse.parameters.update(parameters)
# likelihood_full.parameters.update(parameters)
# print(likelihood_sparse.log_likelihood())
# print(likelihood_full.log_likelihood())

det.set_strains_FD_from_file('LISA_detector_data_None.hdf5')
likelihood_sparse = SparseLikelihood(sparse_ratio=100, waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
                            detector=det)
likelihood_full = FullLikelihood(waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
                            detector=det)
likelihood_sparse.parameters.update(parameters)
likelihood_full.parameters.update(parameters)

def diff_likelihood(mass_para):
    mass_1, mass_2 = mass_para
    mass_dict = {'mass_1':mass_1, 'mass_2':mass_2}
    likelihood_sparse.parameters.update(mass_dict)
    likelihood_full.parameters.update(mass_dict)
    logl_sparse = likelihood_sparse.log_likelihood()
    logl_full = likelihood_full.log_likelihood()
    relative_diff = np.abs(0.5 * (logl_sparse - logl_full) / (logl_sparse + logl_full))
    
    return relative_diff

num = 7200
rng = np.random.default_rng(seed=0)
log10_total_mass =rng.uniform(5, 8, num)
q = rng.uniform(0.2, 1.0, num)
mass_1, mass_2 = total_mass_and_mass_ratio_to_component_masses(q, 10**log10_total_mass)
mass_para = np.vstack([mass_1, mass_2]).T
st = time.perf_counter()
with Pool(processes=72) as p:
    diff_logl = p.map(diff_likelihood, mass_para)
ed = time.perf_counter()
print('time consuming', ed-st)

np.savetxt('log10_total_mass.txt', log10_total_mass)
np.savetxt('q.txt', q)
np.savetxt('diff_logl.txt', diff_logl)
# print(diff_logl)