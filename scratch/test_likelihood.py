import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')
import time

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

det.generate_FD_noise_realization_from_psd(seed=0)
det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
det.save_data()

det.set_strains_FD_from_file('LISA_detector_data_None.hdf5')
likelihood = SparseLikelihood(sparse_ratio=100, waveform_func=IMRPhenomD_h22_Amplitude_Phase_tf,
                            detector=det)

likelihood.parameters.update(parameters)
st = time.perf_counter()
log_l = likelihood.log_likelihood()
ed = time.perf_counter()
print(log_l)
print('time consuming: ', ed-st)

