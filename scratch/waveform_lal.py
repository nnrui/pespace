import numpy as np
from matplotlib import pyplot as plt
import time


import lal
import lalsimulation as lalsim

from peSpace.utilities import cutoff_frequency_PhenomD


f_ref = 0.0
# in bbhx, f_ref is fixed to f_peak
# f_ref= 1e-3
phi_ref = 0.0 
t_ref = 0.0
# mass_1 = 30
# mass_2 = 29
# luminosity_distance = 500
tc = 0.0
phi_c = 0.0

mass_1 = 1.5e6
mass_2 = 5e5
luminosity_distance = 360000
# zero spin make the difference samll,
# for IMRPhenomD when phase_peak, and tf_peak are not exactly zero
# chi_1 = 0.2
# chi_2 = 0.4
chi_1 = 0.0
chi_2 = 0.0
theta_jn = 0.0
phase = 0.0
waveform_dictionary = lal.CreateDict()
# t_ref = 0.0

mass_1_SI = mass_1*lal.MSUN_SI
mass_2_SI = mass_2*lal.MSUN_SI
luminosity_distance_SI = luminosity_distance*1e6*lal.PC_SI

# minimum_frequency = 20
# maximum_frequency = 1024
# duration = 4
# cadance = 1/2048
minimum_frequency = 1e-4
maximum_frequency = 0.1
duration = 5045024
cadance = 10
f_cut = cutoff_frequency_PhenomD(mass_1, mass_2)
sampling_frequency = 2*f_cut
f_low = np.maximum(minimum_frequency, 1.0/duration)
f_high = np.minimum(maximum_frequency, f_cut)
frequency_array = np.arange(f_low, f_high, 1.0/duration)
print('cutoff frequency ', f_cut)
print('sampling_frequency ', sampling_frequency)
print('cadance ', 1/sampling_frequency)
print(f_low, f_high)
print(len(frequency_array))




f_peak = lalsim.IMRPhenomDGetPeakFreq(mass_1, mass_2, chi_1, chi_2)
print('peak frequency', f_peak)




st = time.perf_counter()
amp_h22, phase_h22, tf_h22 = lalsim.SimIMRPhenomDFrequencySequenceh22AmpPhasetf(frequency_array, mass_1, mass_2, chi_1, chi_2, luminosity_distance, tc, phi_c, waveform_dictionary)
print('lalsim waveform func call use time : ', time.perf_counter()-st)
# hp_lal = lalsim.SimIMRPhenomDFrequencySequence(frequency_array, phase, f_ref, mass_1_SI, mass_2_SI, chi_1, chi_2, luminosity_distance_SI, waveform_dictionary, lalsim.NoNRT_V)
# print(hp.__dir__())
# print(hp.data.__dir__())
# print(hp.sampleUnits)
# print(amp.data, ph, tf)

tf_h22_phaseDeriv = (phase_h22.data[1:]-phase_h22.data[:-1])/(frequency_array[1]-frequency_array[0])
# print(tf/2/np.pi)

print(lalsim.SimIMRPhenomDChirpTime(mass_1_SI, mass_2_SI, chi_1, chi_2, minimum_frequency))
print(tf_h22.data[0])
print(tf_h22.data[-1])
print(tf_h22.data[-1]-tf_h22.data[0])















# from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
# phenomhm = PhenomHMAmpPhase(use_gpu=False, run_phenomd=True)
# st = time.perf_counter()
# phenomhm(
#     mass_1,
#     mass_2,
#     chi_1,
#     chi_2,
#     luminosity_distance_SI,
#     phi_ref,
#     f_ref,
#     t_ref,
#     length=len(frequency_array),
#     freqs=frequency_array,
#     modes=[(2,2)]
# )
# print('bbhx: ', time.perf_counter()-st)
# # get important quantities
# freqs_bbhx = phenomhm.freqs_shaped  # shape (num_bin_all, length)
# amps_bbhx = phenomhm.amp  # shape (num_bin_all, num_modes, length)
# phase_bbhx = -phenomhm.phase - 2*np.pi*frequency_array*tc  # shape (num_bin_all, num_modes, length)
# tf_bbhx = phenomhm.tf  # shape (num_bin_all, num_modes, length)







plt.figure()
plt.loglog(frequency_array, amp_h22.data, label='lal')
# plt.loglog(frequency_array, amps_bbhx[0][0], label='bbhx')
plt.axvline(f_peak, color='tab:red')
plt.legend()
plt.savefig('amp.png')

plt.figure()
plt.semilogx(frequency_array, phase_h22.data, label='lal')
# plt.semilogx(frequency_array, phase_bbhx[0][0], label='bbhx')
plt.axhline(0.0, color='tab:red')
plt.axvline(f_peak, color='tab:red')
plt.ylim(-10,1)
plt.xlim(5e-4, 1e-1)
plt.legend()
plt.savefig('phase.png')

plt.figure()
plt.loglog(frequency_array, np.abs(phase_h22.data), label='lal')
# plt.loglog(frequency_array, np.abs(phase_bbhx[0][0]), label='bbhx')
plt.axvline(f_peak, color='tab:red')
plt.legend()
plt.savefig('phase_log.png')

plt.figure()
plt.semilogx(frequency_array, tf_h22.data, label='lal')
# plt.semilogx(frequency_array, tf_bbhx[0][0], label='bbhx')
plt.semilogx(frequency_array[:-1], -tf_h22_phaseDeriv, label='lal_phaseDeriv')
plt.axhline(0.0, color='tab:red')
plt.axvline(f_peak, color='tab:red')
# plt.ylim(-2000,500)
# plt.xlim(5e-4, 1e-1)
plt.legend()
plt.savefig('tf.png')

plt.figure()
plt.loglog(frequency_array, np.abs(tf_h22.data), label='lal')
# plt.loglog(frequency_array, np.abs(tf_bbhx[0][0]), label='bbhx')
plt.loglog(frequency_array[:-1], np.abs(tf_h22_phaseDeriv), label='lal_phaseDeriv')
plt.axvline(f_peak, color='tab:red')
plt.legend()
plt.savefig('tf_log.png')


# ##################################################################################################
# # Fourier Transfromation
# from  bilby.core import utils

# td_strain_lal = utils.infft(amp_h22.data*np.exp(1.0j*phase_h22.data), 1/cadance)
# td_strain_bbhx = utils.infft(amp_h22.data*np.exp(1.0j*phase_bbhx[0][0]), 1/cadance)
# print(len(td_strain_lal))
# print(len(amp_h22.data))

# plt.figure()
# plt.plot(range(len(td_strain_lal)), td_strain_lal, label='lal')
# plt.plot(range(len(td_strain_lal)), td_strain_bbhx, label='bbhx')
# plt.legend()
# plt.savefig('time_domain.png')




























# # use phase/tf information from last waveform run
# from bbhx.response.fastfdresponse import LISATDIResponse

# phi_ref = 0.0
# inc = np.pi/3
# beta = -1.37/np.pi
# lam = 2.27/np.pi
# psi = np.pi/7


# response = LISATDIResponse()
# response(frequency_array[:-1],
#         inc,
#         lam,
#         beta,
#         psi,
#         phi_ref,
#         len(frequency_array)-1,
#         phase=phase_h22.data[:-1],
#         tf=-tf_h22/2/np.pi,
#         modes=[(2,2)])
# # print(response.__dir__())
# # print(response.transferL1[0][0])
# T_a = response.transferL1[0][0]
# plt.figure()
# plt.semilogx(frequency_array[:-1], T_a.imag)
# plt.savefig('transfer_func_A.png')



# # plot parts of the response
# for i in range(1, 4):

#     # (2, 2) mode
#     index = response.modes.index((2,2))
#     # response.transferL1
#     transfer = getattr(response, f"transferL{i}")
#     plt.semilogx(freqs, transfer.real[0, index], label=f"L{i} Real", ls="solid", color=f"C{i-1}")
#     plt.semilogx(freqs, transfer.imag[0, index], label=f"L{i} Imag", ls="dashed", color=f"C{i-1}")
# plt.legend()
# plt.xlabel("Frequency (Hz)")
# plt.ylabel(r"$\mathcal{T}(f, t(f))$")