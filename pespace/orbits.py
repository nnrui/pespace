import taichi as ti
import taichi.math as tm
import numpy as np

from .constants import *
from .utils import vec3

sqrt3 = np.sqrt(3)
Omega0 = 2 * PI / YEAR_SI                        # Omega0: orbital angular velocity of the constellation
e = ARM_LENGTH_LISA_SI / (2 * AU_SI * sqrt3)     # e: orbital eccentricity
a_sec = AU_SEC

# orbit vectors of detector constellation in SSB
OrbitVectorStruct = ti.types.struct(n1  = vec3,      # unit vectors of link23, in SSB coordinate
                                    n2  = vec3,      # unit vectors of link31, in SSB coordinate
                                    n3  = vec3,      # unit vectors of link12, in SSB coordinate
                                    p1_det = vec3,      # vector of the node 1 relative to the center of costellation, p1 = p0 + p1_det, in SSB coordinate, in unit of sec
                                    p2_det = vec3,      # vector of the node 2 relative to the center of costellation, p2 = p0 + p2_det, in SSB coordinate, in unit of sec
                                    p3_det = vec3,      # vector of the node 3 relative to the center of costellation, p3 = p0 + p3_det, in SSB coordinate, in unit of sec
                                    p0  = vec3,      # vector of the center of the costellation, in SSB coordinate, in unit of sec
                                    )

@ti.func
def LISA_analytic(t: ti.f64) -> OrbitVectorStruct:
    '''
    from Eq. 48, 49, 50 in LISA Data Challenge Manual
    All returned length is in the unit of second
    '''
    alpha = Omega0 * t
    c = tm.cos(alpha)
    s = tm.sin(alpha)

    return OrbitVectorStruct(n1  = vec3([-1./2. * c * s, 1./2. * (1. + c*c), sqrt3/2. * s]),
                             n2  = 1/4 * vec3([c*s - sqrt3*(1 + s*s), sqrt3*c*s - (1 + c*c), -sqrt3*s - 3*c]),
                             n3  = 1/4 * vec3([c*s + sqrt3*(1 + s*s), -sqrt3*c*s - (1 + c*c), -sqrt3*s + 3*c]),
                             p1_det = vec3([-(1 + s*s), c*s, -sqrt3*c]) * a_sec * e,
                             p2_det = vec3([1 + s*s + sqrt3*c*s, -(c*s + sqrt3*(1+c*c)), -(3.*s - sqrt3*c)]) * a_sec * e / 2,
                             p3_det = vec3([1 + s*s - sqrt3*c*s, (-c*s + sqrt3*(1+c*c)),  (3.*s + sqrt3*c)]) * a_sec * e / 2,
                             p0  = vec3([c, s, 0.0]) * a_sec
                             )


available_orbit_models = {'LISA_analytic': LISA_analytic,
                            # to be implemented
                          'Taiji_analytic': None,
                          'Taiji_numerical': None,
                          'Tianqin_analytic': None}