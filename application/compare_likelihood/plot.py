import numpy as np
from matplotlib import pyplot as plt

log10_total_mass = np.loadtxt('/home/hydrogen/workspace/Space_GW/peSpace/application/compare_likelihood/log10_total_mass.txt')
q = np.loadtxt('/home/hydrogen/workspace/Space_GW/peSpace/application/compare_likelihood/q.txt')
diff_logl = np.loadtxt('/home/hydrogen/workspace/Space_GW/peSpace/application/compare_likelihood/diff_logl.txt')

log10_diff_logl = np.log(diff_logl)
plt.figure()
plt.hist(log10_diff_logl, density=True, bins=50)
plt.savefig('diff_logl.png')


plt.figure()
plt.scatter(log10_total_mass, q, s=1, c=log10_diff_logl)
plt.xlabel('log10_total_mass')
plt.ylabel('q')
cbar = plt.colorbar()
cbar.set_label(r'$\ln \Delta {L}$')
plt.savefig('diff_logl_scatter.png')