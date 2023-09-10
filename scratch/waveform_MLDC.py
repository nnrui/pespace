import sys 
sys.path.append('/home/hydrogen/workspace/LISA/LDC_software/lib/python3.6/site-packages')
import pyIMRPhenomD
# print(pyIMRPhenomD.__dir__())
# print(help(pyIMRPhenomD))

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline



minimum_frequency = 1e-4
maximum_frequency = 1e-1
duration = 2592000
cadance = 10
f_low = np.maximum(minimum_frequency, 1.0/duration)
f_high = np.minimum(maximum_frequency, 1.0/(2*cadance))
frequency_array = np.arange(f_low, f_high, 1.0/duration)

phi_ref = 0.0
fRef = 0.0
m1_Msun = 1.5e6
m2_Msun = 5e5
chi1 = 0.2
chi2 = 0.4
dist_Mpc = 36000
m1_SI = m1_Msun*pyIMRPhenomD.MSUN_SI
m2_SI = m2_Msun*pyIMRPhenomD.MSUN_SI
dist_SI = dist_Mpc*1e6*pyIMRPhenomD.PC_SI

wf_MLDC = pyIMRPhenomD.IMRPhenomDh22AmpPhase(frequency_array, phi_ref, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
freq_MLDC, amp_MLDC, phase_MLDC = wf_MLDC.GetWaveform()
tfspline = spline(freq_MLDC, 1./(2.*np.pi) * phase_MLDC).derivative()
tf = tfspline(frequency_array)
print(len(frequency_array))
plt.figure()
plt.loglog(freq_MLDC, amp_MLDC)
plt.savefig('amp_MLDC.png')
plt.figure()
plt.loglog(freq_MLDC, np.abs(phase_MLDC))
plt.savefig('phase_log_MLDC.png')
plt.figure()
plt.loglog(freq_MLDC, phase_MLDC)
plt.savefig('phase_MLDC.png')
plt.figure()
plt.semilogx(freq_MLDC, tf)
plt.savefig('tf_MLDC.png')
plt.figure()
plt.loglog(freq_MLDC, np.abs(tf))
plt.savefig('tf_log_MLDC.png')

