# This file is part of the Astrometry.net suite.
# Copyright 2008 Dustin Lang.
#
# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.
#
# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA		 02110-1301 USA

import ctypes
from ctypes import *
import ctypes.util
import sys
import os.path

_hp = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '_healpix.so'))


# returns (ra, dec) in degrees.
def healpix_to_radecdeg(hp, nside=1., dx=0.5, dy=0.5):
    ra = ctypes.c_double(0)
    dec = ctypes.c_double(0)
    _hp.healpix_to_radecdeg(hp, nside,
							c_double(dx),
							c_double(dy),
                            ctypes.pointer(ra),
                            ctypes.pointer(dec))
    return (ra.value, dec.value)




