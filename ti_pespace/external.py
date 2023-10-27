# Wrapping ti function for external call.
# Functions are called with CPU and serially.
# Automated parallelization is turned off.

import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

