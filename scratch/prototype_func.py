import numpy as np
from numpy import sin, cos, sqrt
# Note the normalization factor of pi used in the definition. 
# Use sinc(x / np.pi) to obtain the unnormalized sinc function.
def sinc(x):
    return np.sinc(x/np.pi)

import lal





#### Armlength
lisaL = 2.5e9 # LISA's arm meters
lisaLT = lisaL/lal.C_SI # LISA's armn in sec

#### Noise levels
### Optical Metrology System noise
## Decomposition
Sloc = (1.7e-12)**2    # m^2/Hz
Ssci = (8.9e-12)**2    # m^2/Hz
Soth = (2.e-12)**2     # m^2/Hz
## Global
Soms_d_dict = {'Proposal':(10.e-12)**2, 'SciRDv1': (15.e-12)**2, 'MRDv1': (10.e-12)**2}  # m^2/Hz

### Acceleration
Sa_a_dict = {'Proposal':(3.e-15)**2, 'SciRDv1': (3.e-15)**2, 'MRDv1': (2.4e-15)**2}  # m^2/sec^4/Hz


lisaD = 0.3  # TODO check it
lisaP = 2.0  # TODO check it




def func():
    '''
    description
    
    Parameters
    ==========


    Returns:
    ========

    '''

    return None




def get_polarization_tensor_SSB(lam, beta, psi, mode):
    '''
    return the polarization tensor in SSB, symbols follow the convention in LDC Manual: LISA-LCST-SGS-MAN-001

    Parameters
    ==========
    EclipticLongitude: lambda, 
    EclipticLatitude: beta, note that beta is (-pi/2, pi/2)
    PolarizationAngle: psi, 
    mode: one of 'plus', 'cross', 'x', 'y', 'breathing', 'longitudinal'

    Returns:
    ========
    array: 3*3 array
    '''
    # todo the constant should compute only once to reduce compution burden.
    p = np.array([[sin(lam)*cos(psi) - sin(beta)*cos(lam)*sin(psi), 
                    -(sin(beta)*sin(lam)*sin(psi)) - cos(lam)*cos(psi), 
                    cos(beta)*sin(psi)]])
    q = np.array([[-cos(lam)*cos(psi)*sin(beta)-sin(lam)*sin(psi), 
                    -cos(psi)*sin(beta)*sin(lam)+cos(lam)*sin(psi),
                    cos(beta)*cos(psi)]])
    if mode == 'plus':
        polarization_tensor = p.T@p - q.T@q
    elif mode == 'cross':
        polarization_tensor = p.T@q + q.T@p
    elif mode == 'x':
        pass
    elif mode == 'y':
        pass
    elif mode == 'breathing':
        pass
    elif mode == 'longitudinal':
        pass

    return polarization_tensor

# def generate_geometric_



def link_unit_vector():
    '''
    unit-vector along the link
    
    Parameters
    ==========


    Returns:
    ========

    '''

    return None


# ref the InterferometerGeometry in bilby
# TODO reduce the repeating evaulate cos, sin, ...
# orbit function of LISA in SSB, copied from MLDC1, t passed in is in second
# using the notation following the LDC Manual: LISA-LCST-SGS-MAN-001
YEAR_SI = lal.YRJUL_SI
AU_SI = lal.AU_SI
C_SI = lal.C_SI
arm_length_LISA =2.5e9                    # arm length in metter


Omega0 = 2*np.pi/YEAR_SI                  # Omega0: orbital angular velocity of the constellation
e = arm_length_LISA/(2*AU_SI*sqrt(3))     # e: orbital eccentricity
a = AU_SI                                 # a: 1 AU
# l = 0.0                                   # inital phase of rotation around the center of the constellation
# kappa = 0.0                                   # inital phase of orbital motion around the sun

def constellation_center_p0_MLDC1(t):
    '''Cartesian SSB components of the center of the constellation'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return np.array([a*c, a*s, 0.0*t]) 
    # return a*c
def node1_p1L_MLDC(t):
    '''Cartesian SSB components of the spacecraft positions, measured from constellation center'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return np.array([-(1+s**2), c*s, -sqrt(3)*c])*a*e

def node2_p2L_MLDC(t):
    '''Cartesian SSB components of the spacecraft positions, measured from constellation center'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return np.array([1/2*(sqrt(3)*c*s + (1+s**2)), 1/2*(-c*s - sqrt(3)*(1+c**2)), -sqrt(3)/2*(sqrt(3)*s - c)])*a*e

def node3_p3L_MLDC(t):
    '''Cartesian SSB components of the spacecraft positions, measured from constellation center'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return np.array([1/2*(-sqrt(3)*c*s + (1+s**2)), 1/2*(-c*s + sqrt(3)*(1+c**2)), -sqrt(3)/2*(-sqrt(3)*s - c)])*a*e

# Cartesian SSB components of unit vectors
def link1_unit_vector_n1_MLDC(t):
    '''unit vector of link 1 in SSB'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return np.array([-1./2*c*s, 1./2*(1 + c**2), sqrt(3)/2*s])

def link2_unit_vector_n2_MLDC(t):
    '''unit vector of link 2 in SSB'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return 1./4 * np.array([c*s - sqrt(3)*(1 + s**2), sqrt(3)*c*s - (1 + c**2), -sqrt(3)*s - 3*c])

def link3_unit_vector_n3_MLDC(t):
    '''unit vector of link 1 in SSB'''
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    return 1./4 * np.array([c*s + sqrt(3)*(1 + s**2), -sqrt(3)*c*s - (1 + c**2), -sqrt(3)*s + 3*c])



trajdict_MLDC = {
    'p0': constellation_center_p0_MLDC1,
    'p1L': node1_p1L_MLDC,
    'p2L': node2_p2L_MLDC,
    'p3L': node3_p3L_MLDC,
    'n1': link1_unit_vector_n1_MLDC,
    'n2': link2_unit_vector_n2_MLDC,
    'n3': link3_unit_vector_n3_MLDC,
    }

def GW_propagation_unit_vector_k(lam, beta):                    # note that beta is (-pi/2, pi/2)
    return np.array([-cos(beta)*cos(lam), -cos(beta)*sin(lam), -sin(beta)])




def contraction_between_link_vector_polarization_tensor(tf, lam, beta, psi, mode):
    '''
    evaluate the n_i \e^{+, \times}_ij n_j, note that the \e^{+, \times}_ij is constant
    depend on (lambda, beta, psi), but n_i, n_j are varying with time.

    Parameters
    ==========

    mode: one of 'plus', 'cross', 'x', 'y', 'breathing', 'longitudinal'

    Returns:
    ==========
    array
    '''
    if mode in ["plus", "cross", "x", "y", "breathing", "longitudinal"]:
        polarization_tensor = get_polarization_tensor_SSB(lam, beta, psi, mode)
    else:
        raise Exception('mode passed in should be one of (\'plus\', \'cross\', \'x\', \'y\', \
                        \'breathing\', \'longitudinal\'), but you give {}'.format(mode) )
    

    return None

def transfer_function():


    pass


def delay_from_heliocenter():
    pass 



def generate_singlelink_responses(waveform, parameters):


    frequency_array = waveform['frequency_array']
    amp = waveform['amp']
    phase = waveform['phase']
    tf = waveform['tf']
    n = len(frequency_array)
    # print(type(phase))
    # print(tf.shape)
    h22 = amp*np.exp(1j*phase) # NOTE whether the returned phase should include the minus
    # TODO fishing this in lalsim, why don't let the lalsim directly return h_cross and h_plus

    lam = parameters['EclipticLongitude']
    beta = parameters['EclipticLatitude']

    psi = parameters['PolarizationAngle']
    inc = parameters['Inclination']
    phi0 = parameters['PhaseAtCoalescence']

    n1 = link1_unit_vector_n1_MLDC(tf).T
    n2 = link2_unit_vector_n2_MLDC(tf).T
    n3 = link3_unit_vector_n3_MLDC(tf).T
    p1L = node1_p1L_MLDC(tf).T
    p2L = node2_p2L_MLDC(tf).T
    p3L = node3_p3L_MLDC(tf).T
    p0 = constellation_center_p0_MLDC1(tf).T

    k = GW_propagation_unit_vector_k(lam, beta)

    # print('n1 shape: ', n1.shape)
    # print('n2 shape: ', n2.shape)
    # print('n3 shape: ', n3.shape)
    # print('k shape: ', k.shape)
    # print('p1L shape: ', p1L.shape)
    # print('p2L shape: ', p2L.shape)
    # print('p3L shape: ', p3L.shape)
    # print('p0 shape: ', p0.shape)
    # print(np.atleast_2d(n1[0]))
    # print(np.atleast_2d(n1[0]).T)




    Y22 = lal.SpinWeightedSphericalHarmonic(inc, phi0, -2, 2, 2)
    # SpinWeightedSphericalHarmonic(REAL8 theta, REAL8 phi, int s, int l, int m) -> COMPLEX16
    Y2m2star = np.conjugate(lal.SpinWeightedSphericalHarmonic(inc, phi0, -2, 2, -2))
    hplus = (Y22*h22 + Y2m2star*np.conjugate(h22))/2
    hcross = (Y22*h22 - Y2m2star*np.conjugate(h22))*1j/2
    pol_tensor_plus = get_polarization_tensor_SSB(lam, beta, psi, 'plus')
    pol_tensor_cross = get_polarization_tensor_SSB(lam, beta, psi, 'cross')
    
 
    n1Hn1 = np.zeros(n, dtype='complex')
    n2Hn2 = np.zeros(n, dtype='complex')
    n3Hn3 = np.zeros(n, dtype='complex')
    kn1 = np.zeros(n, dtype='float')
    kn2 = np.zeros(n, dtype='float')
    kn3 = np.zeros(n, dtype='float')
    kp1Lp2L = np.zeros(n, dtype='float')
    kp2Lp3L = np.zeros(n, dtype='float')
    kp3Lp1L = np.zeros(n, dtype='float')
    kp0 = np.zeros(n, dtype='float')

    for i in range(n):
        n1_i = np.atleast_2d(n1[i])
        n1_i_T = n1_i.T
        n2_i = np.atleast_2d(n2[i])
        n2_i_T = n2_i.T
        n3_i = np.atleast_2d(n3[i])
        n3_i_T = n3_i.T
        n1Hn1[i] = n1_i@(pol_tensor_plus*hplus[i] + pol_tensor_cross*hcross[i])@n1_i_T
        n2Hn2[i] = n2_i@(pol_tensor_plus*hplus[i] + pol_tensor_cross*hcross[i])@n2_i_T
        n3Hn3[i] = n3_i@(pol_tensor_plus*hplus[i] + pol_tensor_cross*hcross[i])@n3_i_T
    # print(n1Hn1)
    # print(n2Hn2)
    # print(n3Hn3)
    # print(n1Hn1.shape)
    # print(n2Hn2.shape)
    # print(n3Hn3.shape)
        kn1[i] = k@n1_i_T
        kn2[i] = k@n2_i_T
        kn3[i] = k@n3_i_T
    # print(kn1)
    # print(kn1.shape)
        kp1Lp2L[i] = np.dot(k, (p1L[i] + p2L[i]))
        kp2Lp3L[i] = np.dot(k, (p2L[i] + p3L[i]))
        kp3Lp1L[i] = np.dot(k, (p3L[i] + p1L[i]))
    # print(kp1Lp2L.shape)
    # print(kp1Lp2L)
        kp0[i] = np.dot(k, p0[i])


    factor_sinc = np.pi*frequency_array*arm_length_LISA/C_SI
    factorsinc12 = sinc(factor_sinc * (1.-kn3))
    factorsinc21 = sinc(factor_sinc * (1.+kn3))
    factorsinc23 = sinc(factor_sinc * (1.-kn1))
    factorsinc32 = sinc(factor_sinc * (1.+kn1))
    factorsinc31 = sinc(factor_sinc * (1.-kn2))
    factorsinc13 = sinc(factor_sinc * (1.+kn2))
    

    factor_exp = -1j*np.pi*frequency_array/C_SI
    factorcexp12 = np.exp(factor_exp*(arm_length_LISA+kp1Lp2L))
    factorcexp23 = np.exp(factor_exp*(arm_length_LISA+kp2Lp3L))
    factorcexp31 = np.exp(factor_exp*(arm_length_LISA+kp3Lp1L))

    # # TODO use approprate unit to reduce repeat dividing c
    prefactor = -1j*np.pi*frequency_array*arm_length_LISA/C_SI
    factorcexp0 = np.exp(-1j*2*np.pi*frequency_array*kp0/C_SI)
    commonfac = prefactor * factorcexp0



    link12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12
    link21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12
    link23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23
    link32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23
    link31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31
    link13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31



    return {'link12': link12, 
            'link21': link21,
            'link23': link23,
            'link32': link32,
            'link31': link31,
            'link13': link13,}








def TDI_combination(frequency_array, singlelink_response, TDI='XYZ1.5'):
    '''
    Parameters
    ==========
    singlelink_response: dict, contains response of different links ();
    TDI: one of XYZ_1.5, AET_1.5, XYZ_2.0, AET_2.0, 
    '''
    TDI_dict  = {}
    if TDI=='XYZ1.5':
        x = np.pi*frequency_array*arm_length_LISA/C_SI
        z = np.exp(-2*1j*x)
        prefactor = (1 - z**2)
        Xraw = singlelink_response['link21'] + z*singlelink_response['link12'] - singlelink_response['link31'] - z*singlelink_response['link13']
        Yraw = singlelink_response['link32'] + z*singlelink_response['link23'] - singlelink_response['link12'] - z*singlelink_response['link21']
        Zraw = singlelink_response['link13'] + z*singlelink_response['link31'] - singlelink_response['link23'] - z*singlelink_response['link32']
        TDI_dict['X'] = prefactor * Xraw
        TDI_dict['Y'] = prefactor * Yraw
        TDI_dict['Z'] = prefactor * Zraw
        return TDI_dict    
    elif TDI == 'AET1.5':
        pass
    elif TDI == 'XYZ2.0':
        x = np.pi*frequency_array*arm_length_LISA/C_SI
        z = np.exp(-2*1j*x)
        # prefactor = (1 - z**2)
        prefactor = (1 - z**2 - z**4 + z**6)
        Xraw = singlelink_response['link21'] + z*singlelink_response['link12'] - singlelink_response['link31'] - z*singlelink_response['link13']
        Yraw = singlelink_response['link32'] + z*singlelink_response['link23'] - singlelink_response['link12'] - z*singlelink_response['link21']
        Zraw = singlelink_response['link13'] + z*singlelink_response['link31'] - singlelink_response['link23'] - z*singlelink_response['link32']
        TDI_dict['X'] = prefactor * Xraw
        TDI_dict['Y'] = prefactor * Yraw
        TDI_dict['Z'] = prefactor * Zraw
        return TDI_dict


@np.vectorize
def PSD_1(f):
    Spm, Sop = lisanoises(f, 'SciRDv1')
    S = 16*sin(2*np.pi*f*arm_length_LISA/C_SI)**2 * (Sop + (3+cos(2*2*np.pi*f*arm_length_LISA/C_SI))*Spm )
    return S

@np.vectorize
def PSD_2(f):
    Spm, Sop = lisanoises(f, 'SciRDv1')
    S = 64*sin(2*np.pi*f*arm_length_LISA/C_SI)**2 * sin(4*np.pi*f*arm_length_LISA/C_SI)**2 * (Sop + (3+cos(2*2*np.pi*f*arm_length_LISA/C_SI))*Spm)
    return S







def lisanoises(f, model="SciRDv1",unit='relativeFrequency'):
    """
    Return the analytic approximation of the two components of LISA noise,
    i.e. the acceleration and the
    @param f is the frequency array
    @param model is the noise model:
        * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
        * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
        * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
    @param unit is the unit of the output: 'relativeFrequency' or 'displacement'
    """
    if  model == 'mldc':
        Spm = 2.5e-48 * (1.0 + (f/1.0e-4)**-2) * f**(-2)
        defaultL = 16.6782
        Sop = 1.8e-37 * (lisaLT/defaultL)**2 * f**2

    elif model == 'newdrs':  # lisalight, to be used with lisaL = 1Gm, lisaP = 2
        Spm = 6.00314e-48 * f**(-2)                                 # 4.6e-15 m/s^2/sqrt(Hz)
        defaultL = 16.6782
        defaultD = 0.4
        defaultP = 1.0
        Sops = 6.15e-38 * (lisaLT/defaultL)**2 * (defaultD/lisaD)**4 * (defaultP/lisaP)      # 11.83 pm/sqrt(Hz)
        Sopo = 2.81e-38                                                                                                         # 8 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2

    elif model == 'LCESAcall':
        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_dict['Proposal'] *(1.0 +(0.4e-3/frq)**2+(frq/9.3e-3)**4)
        ## In displacement
        Sa_d = Sa_a*(2.*np.pi*frq)**(-4.)
        ## In relative frequency unit
        Sa_nu = Sa_d*(2.0*np.pi*frq/C_SI)**2
        Spm =  Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_dict['Proposal'] * (1. + (2.e-3/f)**4)
        ## In relative frequency unit
        Soms_nu = Soms_d*(2.0*np.pi*frq/C_SI)**2
        Sop =  Soms_nu


    elif model == 'Proposal':
        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_dict['Proposal'] *(1.0 +(0.4e-3/frq)**2)*(1.0+(frq/8e-3)**4)
        ## In displacement
        Sa_d = Sa_a*(2.*np.pi*frq)**(-4.)
        ## In relative frequency unit
        Sa_nu = Sa_d*(2.0*np.pi*frq/C_SI)**2
        Spm =  Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_dict['Proposal'] * (1. + (2.e-3/f)**4)
        ## In relative frequency unit
        Soms_nu = Soms_d*(2.0*np.pi*frq/C_SI)**2
        Sop =  Soms_nu

    elif model=='SciRDv1' or model=='MRDv1':
        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_dict[model] *(1.0 +(0.4e-3/frq)**2)*(1.0+(frq/8e-3)**4)
        ## In displacement
        Sa_d = Sa_a*(2.*np.pi*frq)**(-4.)
        ## In relative frequency unit
        Sa_nu = Sa_d*(2.0*np.pi*frq/C_SI)**2
        Spm =  Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_dict[model] * (1. + (2.e-3/f)**4)
        ## In relative frequency unit
        Soms_nu = Soms_d*(2.0*np.pi*frq/C_SI)**2
        Sop =  Soms_nu

    else:
        raise NotImplementedError(model)

    if unit=='displacement':
        return Sa_d, Soms_d
    elif unit=='relativeFrequency':
        return Spm, Sop
    else:
        raise NotImplementedError(unit)



