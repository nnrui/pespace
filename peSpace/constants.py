import lal 


Mf_CUT_PhenomD = 0.2   # copied from lalsimultion

C_SI = lal.C_SI
MTSUN_SI = lal.MTSUN_SI
MSUN_SI = lal.MSUN_SI
YEAR_SI = lal.YRJUL_SI
AU_SI = lal.AU_SI
AU_sec = AU_SI/C_SI

arm_length_LISA_SI = 2.5e9  # meters
arm_length_LISA_sec = arm_length_LISA_SI/C_SI # seconds

PI = lal.PI

DEBUG = True