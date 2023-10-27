import numpy as np
from numpy import sin, cos, sqrt

from .constants import *


Omega0 = 2 * PI / YEAR_SI                    # Omega0: orbital angular velocity of the constellation
e = ARM_LENGTH_LISA_SI/(2*AU_SI*sqrt(3))     # e: orbital eccentricity
a_sec = AU_SEC


def _LISA_analytic(t):
    '''
    note that all returned length is in the unit of second
    '''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)

    n1  = np.array([[-1./2*c*s, 1./2*(1 + c**2), sqrt(3)/2*s]])
    n2  = 1/4 * np.array([[c*s - sqrt(3)*(1 + s**2), sqrt(3)*c*s - (1 + c**2), -sqrt(3)*s - 3*c]])
    n3  = 1/4 * np.array([[c*s + sqrt(3)*(1 + s**2), -sqrt(3)*c*s - (1 + c**2), -sqrt(3)*s + 3*c]])
    p1L = np.array([[-(1+s**2), c*s, -sqrt(3)*c]])*a_sec*e
    p2L = np.array([[1/2*(sqrt(3)*c*s + (1+s**2)), 1/2*(-c*s - sqrt(3)*(1+c**2)), -sqrt(3)/2*(sqrt(3)*s - c)]])*a_sec*e
    p3L = np.array([[1/2*(-sqrt(3)*c*s + (1+s**2)), 1/2*(-c*s + sqrt(3)*(1+c**2)), -sqrt(3)/2*(-sqrt(3)*s - c)]])*a_sec*e
    p0  = np.array([[a_sec*c, a_sec*s, 0.0*t]]) 

    return {'n1'   : n1   ,
            'n2'   : n2   ,
            'n3'   : n3   ,
            'n1_T' : n1.T ,
            'n2_T' : n2.T ,
            'n3_T' : n3.T ,
            'p1L'  : p1L  ,
            'p2L'  : p2L  ,
            'p3L'  : p3L  ,
            'p0'   : p0   ,
            }

orbit_models = {'LISA_analytic': _LISA_analytic,
                # to be implemented
                'Taiji_analytic': None,
                'Taiji_numerical': None,
                'Tianqin_analytic': None}

def get_constellation_vectors_from_orbits(time, orbit='LISA_analytic'):
    func = orbit_models[orbit]
    return func(time)