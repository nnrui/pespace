import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

import h5py

from peSpace.detectors import LISALike
from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf



det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=1, TDI_channels=('X', 'Y', 'Z', 'A', 'E', 'T'), 
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
waveform = IMRPhenomD_h22_Amplitude_Phase_tf(det.frequency_array, parameters)

# single_links = det.generate_singlelink_responses(waveform, parameters)
# print(single_links)

# TDI_dict = det.TDI_responses(waveform, parameters)
# print(TDI_dict)

# det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
# print(det.signals)
# print(det.strains_FD)
# for chan in ('X', 'Y', 'Z', 'A', 'E', 'T'):
#     resdual = det.strains_FD[chan] - det.signals[chan]
#     print(resdual)

# det.generate_FD_noise_realization_from_psd()
# print(det.noise)
# print(det.strains_FD)
# for chan in ('X', 'Y', 'Z', 'A', 'E', 'T'):
#     resdual = det.strains_FD[chan] - det.noise[chan]
#     print(resdual)

# det.generate_FD_noise_realization_from_psd()
# det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
# det.plot_FD_data_amplitude()

# import bilby
# from matplotlib import pyplot as plt
# import numpy as np

# psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=det.frequency_array, psd_array=det.psd_array['A'])
# noise_realization, frequencies = psd.get_noise_realisation(sampling_frequency=0.1, duration=864000)

# fig, ax = plt.subplots()
# ax.loglog(det.frequency_array, np.abs(det.noise['A']), label='self noise realization')
# ax.loglog(frequencies, np.abs(noise_realization), label='bilby noise realization')
# ax.loglog(det.frequency_array, np.sqrt(det.psd_array['A']), label='asd')
# ax.set_xlim(1e-5,0.1)
# ax.legend()
# fig.savefig('noise_bilby_pespace.png')

# print(det.optimal_snr())

# det.generate_FD_noise_realization_from_psd()
# det.inject_signal_FD(parameters, IMRPhenomD_h22_Amplitude_Phase_tf)
# det.save_data()

data = det.set_strains_FD_from_file('/home/hydrogen/workspace/Space_GW/peSpace/scratch/LISA_detector_data.hdf5')
print(type(data['strains_TD']))