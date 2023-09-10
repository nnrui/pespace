import time

import numpy as np
import lal
import lalsimulation as lalsim

from matplotlib import pyplot as plt


def f_dot(Mc, f):
    return 96/5 * np.pi**(8/3) * (lal.MTSUN_SI*Mc)**(5/3) * f**(11/3) 


# f_array = np.linspace(1e-4, 1e-1, 1000)
# plt.figure()
# plt.loglog(f_array, 3e5*f_dot(1e5, f_array))
# plt.savefig('dynamic_frequency_space.png')


# f_CUT = 0.2
# print(f_CUT/(lal.MTSUN_SI*1e5))


df_min = 1/(24*3600*30)
df_max = 0.4/100

def generate_dynamic_frequency_array_heterodyne(frequencies, total_mass):
    delta_f = frequencies[1] - frequencies[0]
    idx = [False] * len(frequencies)
    i = 0
    while i<len(frequencies):
        idx[i] = True
        f = frequencies[i]
        df = 1e5*f_dot(total_mass, f)
        if df<df_min:
            df = delta_f
        elif df>df_max:
            df = df_max
        i += int(df//delta_f)


    return idx


frequencies = np.arange(1e-4, 0.1, 1/(24*3600*30))
total_mass = 1e6

f = frequencies[generate_dynamic_frequency_array_heterodyne(frequencies, total_mass)]
print(len(f))
print(f)


rng = np.random.default_rng()

a = rng.uniform(-1, 1, int(1e8))
p = rng.uniform(0, 2*np.pi, int(1e8))
a = a*np.exp(1j*p)

st = time.perf_counter()
abs1 = np.conj(a)*a
ed = time.perf_counter()
print(abs1[0:10].real, 'using conj, time consuming: ', ed-st)

st = time.perf_counter()
abs2 = np.abs(a)**2
ed = time.perf_counter()
print(abs2[0:10], 'using abs, time consuming: ', ed-st)