import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')
import time
import copy

import numpy as np
from matplotlib import pyplot as plt

from peSpace.detectors import LISALike
from peSpace.waveform import IMRPhenomD_h22_Amplitude_Phase_tf
from peSpace.likelihood import FullLikelihood, SparseLikelihood

det = LISALike(name='LISA', duration=864000, cadance=10, minimum_frequency=1e-4, 
               maximum_frequency=1, TDI_channels=('A', 'E', 'T'), 
               TDI_generation='2.0')

parameters_1 = dict(mass_1 = 1.5e6,
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
det.inject_signal_FD(parameters_1, IMRPhenomD_h22_Amplitude_Phase_tf)
print(det.optimal_snr())

strain = det.signals['A']
strain_amp = np.abs(det.signals['A'])
strain_phase = np.unwrap(np.angle(det.signals['A']), discont=np.pi, period=2*np.pi)

plt.figure()
plt.semilogx(det.frequency_array, strain.real, label='real')
plt.semilogx(det.frequency_array, strain.imag, label='imag')
plt.legend()
plt.savefig('full_response_real_imag.png')

plt.figure()
plt.loglog(det.frequency_array, strain_amp)
plt.savefig('full_response_amp.png')

plt.figure()
plt.semilogx(det.frequency_array, strain_phase)
plt.savefig('full_response_phase.png')

