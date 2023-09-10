import sys
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

from peSpace.detectors import LISALike
from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from peSpace.utilities import inner_product, polarization_tensor_SSB, time_in_band_leading_order

# det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
#                maximum_frequency=0.1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
#                TDI_generation='1.5')

# parameters = dict(mass_1 = 1.5e6,
#                   mass_2 = 5e5,
#                   luminosity_distance=360000,
#                   chi_1 = 0.2,
#                   chi_2 = 0.4,
#                   coalescence_time=0.0,
#                   ecliptic_longitude = 1.2,
#                   ecliptic_latitude = 0.8,
#                   polarization = 0.4,
#                   inclination = 0.2,
#                   coalescence_phase = 0.0,)
# waveform = IMRPhenomD_h22_Amplitude_Phase_tf(det.frequency_array, parameters)

# det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
# print(inner_product(det.signals['E'], det.signals['E'], det.psd_array['E'], 1./det.duration))


# polarization_tensor_SSB(1.2, 1., 1., 'z')


print(time_in_band_leading_order(1.5e6, 5e5, 1e-4))