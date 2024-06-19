import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/peSpace')

import taichi as ti
from ti_peSpace.utilities import *

ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.kernel
def test():
    print('sinc(x):')
    print(sinc(0.0))
    print(sinc(0.5))
    
    print('polarization_tensor_SSB(lam, beta, psi):')
    pol = polarization_tensor_SSB(1.2, 2.3, 2.5)
    print(pol.plus)
    print(pol.cross)

    print('GW_propagation_unit_vector_k(lam, beta):')
    k = GW_propagation_unit_vector_k(1.2, 2.3)
    print(k)
    print(k.norm())

test()
