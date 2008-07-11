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

from astrometry.util.healpix2 import *
from pylab import *

if __name__ == '__main__':
    nside = 2
    for hp in range(12 * nside**2):
        (ra,dec) = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
        text(ra, dec, '%i' % hp)

        nsteps = 20
        stepvals = [x/float(nsteps) for x in range(nsteps+1)]
        rstepvals = [1-x for x in stepvals]
        ones = [1.0]*(nsteps+1)
        zeros = [0.0]*(nsteps+1)

        x = stepvals + ones + rstepvals + zeros
        y = zeros + stepvals + ones + rstepvals

        edges = [healpix_to_radecdeg(hp, nside, xx, yy) for (xx,yy) in zip(x,y)]
        ra = [ra for (ra,dec) in edges]
        dec = [dec for (ra,dec) in edges]
        plot(ra, dec, 'k-')

    xlabel('RA (deg)')
    ylabel('Dec (deg)')
    title('Healpixes (nside=%i)' % i)
    savefig('healpix.png')
    
