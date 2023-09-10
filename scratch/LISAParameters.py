"""
This script defines the parameters of the LISA instrument
"""

"""
    Copyright (C) 2017 Stas Babak, Antoine Petiteau for the LDC team

    This file is part of LISA Data Challenge.

    LISA Data Challenge is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

##################################################
#                                                #
#                LISA Parameters                 #
#                  version 1.0                   #
#                                                #
#         S. Babak, A. Petiteau, ...             #
#      for the LISA Data Challenge team          #
#                                                #
##################################################

import numpy as np
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
