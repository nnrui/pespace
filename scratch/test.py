import numpy as np
from matplotlib import pyplot as plt
import time

# # test polarization tensor
# ################################################################################
# import sys
# sys.path.append('/home/hydrogen/workspace/LISA/LDC_software/lib/python3.6/site-packages/')
# import pyFDresponse as FD_Resp
# from prototype_func import get_polarization_tensor_SSB

# HSplus = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]])
# HScross = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])

# lam_array = np.random.uniform(low=0, high=2*np.pi, size=1000)
# beta_array = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=1000)
# psi_array = np.random.uniform(low=0, high=np.pi, size=1000)

# points = np.vstack([lam_array, beta_array, psi_array]).T
# print('generate random points of (lam, beta, psi) to compare the polarization tensor given by MLDC and code here', points.shape)

# Hplus_diff = []
# Hcross_diff = []
# for lam, beta, psi in points:
#     O1 = FD_Resp.funcO1(lam, beta, psi)
#     invO1 = FD_Resp.funcinverseO1(lam, beta, psi)
#     Hplus_MLDC = np.dot(O1, np.dot(HSplus, invO1))
#     Hcross_MLDC = np.dot(O1, np.dot(HScross, invO1))

#     Hplus = get_polarization_tensor_SSB(lam, beta, psi, 'plus')
#     Hcross = get_polarization_tensor_SSB(lam, beta, psi, 'cross')
#     # print(Hplus)
#     # print(Hplus_MLDC)

#     Hplus_diff.append(Hplus-Hplus_MLDC)
#     Hcross_diff.append(Hcross-Hcross_MLDC)

# Hplus_diff = np.asarray(Hplus_diff)
# Hcross_diff = np.asarray(Hcross_diff)

# print('summation of difference of Hplus', np.sum(Hplus_diff.flatten()))
# print('summation of difference of Hcross', np.sum(Hcross_diff.flatten()))








# test single link response
# ###################################################################################
import lal
import lalsimulation as lalsim


tc = 0.0
phi_c = 0.0
mass_1 = 1.5e6
mass_2 = 5e5
luminosity_distance = 36000
# mass_1 = 100
# mass_2 = 90
# luminosity_distance = 500

chi_1 = 0.0
chi_2 = 0.0
theta_jn = 0.0
waveform_dictionary = lal.CreateDict()

minimum_frequency = 1e-5
maximum_frequency = 1
duration = 2592000
cadance = 5
f_low = np.maximum(minimum_frequency, 1.0/duration)
f_high = np.minimum(maximum_frequency, 1.0/(2*cadance))
# frequency_array = np.arange(f_low, f_high, 1.0/duration)
# print(len(frequency_array))
frequency_array = np.linspace(f_low, f_high, 1000)





st = time.perf_counter()
amp_h22, phase_h22, tf_h22 = lalsim.SimIMRPhenomDFrequencySequenceh22AmpPhasetf(frequency_array, mass_1, mass_2, chi_1, chi_2, luminosity_distance, tc, phi_c, waveform_dictionary)
print('generate waveform by lalsim using time: ', time.perf_counter()-st)

np.savetxt('frequency_array.txt', frequency_array)
np.savetxt('amp.txt', amp_h22.data)
np.savetxt('phase.txt', phase_h22.data)
np.savetxt('tf.txt', tf_h22.data)

frequency_array = np.loadtxt('frequency_array.txt')
amp = np.loadtxt('amp.txt')
phase = np.loadtxt('phase.txt')
tf = np.loadtxt('tf.txt')

waveform = dict(frequency_array = frequency_array,
                amp = amp,
                phase = phase,
                tf = tf)
parameters = dict(EclipticLongitude = 0.2,
                  EclipticLatitude = 0.8,
                  PolarizationAngle = 0.0,
                  Inclination = 0.0,
                  PhaseAtCoalescence = phi_c)

import prototype_func
from prototype_func import generate_singlelink_responses, TDI_combination, PSD_1, PSD_2, constellation_center_p0_MLDC1
# print(constellation_center_p0_MLDC1(tf_h22.data))
# print(constellation_center_p0_MLDC1(tf_h22.data).shape)
st = time.perf_counter()
singlelink_response = generate_singlelink_responses(waveform=waveform, parameters=parameters)
TDI_XYZ_1 = TDI_combination(frequency_array, singlelink_response, 'XYZ1.5')
print('generate_singlelink_responses using time: ', time.perf_counter()-st)
TDI_XYZ_2 = TDI_combination(frequency_array, singlelink_response, 'XYZ2.0')


plt.figure()
plt.loglog(frequency_array, np.abs(TDI_XYZ_1['X']), color='tab:blue', linestyle='dashed', label='GW_X_TDI1.0')
plt.loglog(frequency_array, np.abs(TDI_XYZ_2['X']), color='tab:blue', linestyle='solid', label='GW_X_TDI2.0')
plt.loglog(frequency_array, np.sqrt(PSD_1(frequency_array)), color='tab:orange', linestyle='dashed',label='sqrt(noise_X_TDI1.0)')
plt.loglog(frequency_array, np.sqrt(PSD_2(frequency_array)), color='tab:orange', linestyle='solid',label='sqrt(noise_X_TDI2.0)')
# plt.ylim(-1e-19, 1e-19)
# plt.xlim(1e-4, 1e-1)
plt.legend()
plt.savefig('TDI12_X.png')
# singlelink_response_MLDC = 



###################################################################################
# from utilities import cutoff_frequency_SMBH
# import lalsimulation as lalsim
# mass_1 = 1.5e8
# mass_2 = 0.5e8
# print('f_cut ', cutoff_frequency_SMBH(mass_1 + mass_2))
# f_peak = lalsim.IMRPhenomDGetPeakFreq(mass_1, mass_2, 0, 0)
# print('peak frequency ', f_peak)

# ###################################################################################
# from utilities import time_in_band_leading_order, estimate_imr_duration
# from bilby.gw.utils import calculate_time_to_merger
# import lal
# import lalsimulation as lalsim
# mass_1 =  30
# mass_2 = 29
# f_start = 20
# print('time_in_band_leading_order ', time_in_band_leading_order(mass_1, mass_2, start_frequency=f_start, safety_factor=1))
# print('estimate_imr_duration ', estimate_imr_duration(mass_1, mass_2, 0.0, 0.0, f_start, 1.0))
# print('bilby.gw.calculate_time_to_merger ', calculate_time_to_merger(f_start, mass_1, mass_2, chi=0, safety=1.0))
# print('lalsim ', lalsim.SimInspiralTaylorF2ReducedSpinChirpTime( f_start, mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI, 0, -1))

# # from pycbc.pnutils import _get_imr_duration
# # print(_get_imr_duration(mass_1, mass_2, 0.0, 0.0, 1e-4, approximant="TaylorF2")/1.1)
# # time_in_band_leading_order  5045023.886655574
# # estimate_imr_duration  -7431981.956713552
# # calculate_time_to_merger,  5111214.818438125
