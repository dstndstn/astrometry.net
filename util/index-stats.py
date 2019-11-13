#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

from astrometry.util.index import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import match
from astrometry.util.starutil_numpy import *
from astrometry.util.fits import *

from optparse import *

from pylab import *
from numpy import *
import os

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--prefix', dest='prefix', help='Prefix for output plot names')
    parser.add_option('-r', '--range', dest='range', type='float', help='Set search radius range (in arcsec) of stars-1 plot, default 15')
    parser.set_defaults(prefix='', range=15.)
    opt,args = parser.parse_args()

    if 'plots' in args: # DEBUG!
        cat = fits_table('cat2.fits')
        xyz = radectoxyz(cat.ra, cat.dec)
        R = 15.
        inds,dists = match(xyz, xyz, deg2rad(R/3600.))
        notself = (inds[:,0] != inds[:,1])
        clf()
        hist(rad2deg(dists[notself]) * 3600., 200)
        title('ImSim reference catalog')
        xlabel('Distance between pairs of sources (arcsec)')
        ylabel('Counts')
        xlim(0, R)
        savefig('cat-stars-1.png')

        cat = fits_table('stars3.fits')
        xyz = radectoxyz(cat.ra, cat.dec)
        R = 15.
        inds,dists = match(xyz, xyz, deg2rad(R/3600.))
        notself = (inds[:,0] != inds[:,1])
        clf()
        hist(rad2deg(dists[notself]) * 3600., 200)
        title('ImSim reference catalog -- stars only')
        xlabel('Distance between pairs of sources (arcsec)')
        ylabel('Counts')
        xlim(0, R)
        savefig('cat-stars-2.png')

        I1 = inds[notself,0]
        I2 = inds[notself,1]
        clf()
        RA = cat.ra + (cat.ra > 180)*-360
        dra = RA[I1]-RA[I2]
        ddec = cat.dec[I1]-cat.dec[I2]
        #plot(dra, ddec, 'r.')
        (H,xe,ye) = histogram2d(dra, ddec, bins=(200,200))
        H=H.T
        imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
               interpolation='nearest', origin='lower', cmap=antigray)

        xlabel('dRA (deg)')
        ylabel('dDec (deg)')
        axis('equal')
        savefig('cat-stars-4.png')

        cat = fits_table('gals2.fits')
        xyz = radectoxyz(cat.ra, cat.dec)
        R = 15.
        inds,dists = match(xyz, xyz, deg2rad(R/3600.))
        notself = (inds[:,0] != inds[:,1])
        clf()
        hist(rad2deg(dists[notself]) * 3600., 200)
        title('ImSim reference catalog -- galaxies only')
        xlabel('Distance between pairs of sources (arcsec)')
        ylabel('Counts')
        xlim(0, R)
        savefig('cat-stars-3.png')
        sys.exit(0)

    for indfn in args:
        print('Reading index', indfn)
        null = None
        I = index_load(indfn, 0, null)
        print('Loaded.')
        NS = index_nstars(I)
        NQ = index_nquads(I)
        print('Index has', NS, 'stars and', NQ, 'quads')
        DQ = index_get_quad_dim(I)
        print('Index has "quads" with %i stars' % (DQ))
        DC = index_get_quad_dim(I)
        print('Index has %i-dimensional codes' % (DC))

        iname = os.path.basename(I.indexname).replace('.fits', '')
        # stars
        print('Getting stars...')
        stars = index_get_stars(I)
        print(stars.shape)

        ra,dec = xyztoradec(stars)
        ra += (ra > 180)*-360

        # FIXME --!
        #ra *= cos(deg2rad(ra))
        rng = [[-10,10],[-10,10]]
        clf()
        (H,xe,ye) = histogram2d(ra, dec, bins=(100,100), range=rng)
        H=H.T
        binarea = (xe[1]-xe[0])*(ye[1]-ye[0])
        print('Bin area:', binarea, 'deg^2')
        binarea *= 3600.
        print(binarea, 'arcmin^2')
        imshow(H/binarea, extent=(min(xe), max(xe), min(ye), max(ye)),
               aspect='auto',
               interpolation='nearest', origin='lower', cmap=antigray)
        colorbar()
        axis('equal')
        xlabel('RA (deg)')
        ylabel('Dec (deg)')
        title('Reference source density in %s' % iname)
        savefig(opt.prefix + 'stars-2.png')

        R = opt.range
        print('Finding pairs within', R, 'arcsec')
        inds,dists = match(stars, stars, deg2rad(R/3600.))
        print('inds', inds.shape, 'dists', dists.shape)

        notself = (inds[:,0] != inds[:,1])
        clf()
        hist(rad2deg(dists[notself]) * 3600., 200)
        xlabel('Star pair distances (arcsec)')
        ylabel('Counts')
        xlim(0, R)
        savefig(opt.prefix + 'stars-1.png')


        # codes
        print('Getting codes...')
        codes = index_get_codes(I)
        print('shape', codes.shape)

        # code slices
        cx = codes[:,0]
        cy = codes[:,1]
        dx = codes[:,2]
        dy = codes[:,3]
        clf()
        (H,xe,ye) = histogram2d(cx, cy, bins=(100,100))
        H=H.T
        imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
               interpolation='nearest', origin='lower', cmap=antigray)
        axis('equal')
        xlabel('cx')
        ylabel('cy')
        savefig(opt.prefix + 'codes-1.png')

        clf()
        (H,xe,ye) = histogram2d(append(cx, dx), append(cy, dy), bins=(100,100))
        H=H.T
        imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
               interpolation='nearest', origin='lower', cmap=antigray)
        axis('equal')
        xlabel('cx, dx')
        ylabel('cy, dy')
        savefig(opt.prefix + 'codes-2.png')

        clf()
        (H,xe,ye) = histogram2d(cx, dx, bins=(100,100))
        H=H.T
        imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
               interpolation='nearest', origin='lower', cmap=antigray)
        axis('equal')
        xlabel('cx')
        ylabel('dx')
        savefig(opt.prefix + 'codes-3.png')

        clf()
        xx = append(cx, dx)
        yy = append(cy, dy)
        (H,xe,ye) = histogram2d(append(xx, 1.0-xx), append(yy, 1.0-yy), bins=(100,100))
        H=H.T
        imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
               interpolation='nearest', origin='lower', cmap=antigray)
        axis('equal')
        xlabel('cx, dx')
        ylabel('cy, dy')
        title('duplicated for A-B swap')
        savefig(opt.prefix + 'codes-4.png')

        for pnum,arrs in [(5, [cx,cy,dx,dy]),
                          (6, [append(cx, 1-cx), append(cy, 1-cy), append(dx, 1-dx), append(dy, 1-dy)])]:
            clf()
            anames = ['cx', 'cy', 'dx', 'dy']
            sp = 0
            for i,a1 in enumerate(arrs):
                for j,a2 in enumerate(arrs):
                    sp += 1
                    if j > i:
                        continue
                    subplot(4,4, sp)
                    if i == j:
                        hist(a1, 100, range=[-0.3, 1.3])
                    else:
                        (H,xe,ye) = histogram2d(a2, a1, bins=(100,100), range=[[-0.3,1.3],[-0.3,1.3]])
                        H=H.T
                        imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
                               interpolation='nearest', origin='lower', cmap=antigray)
                        axis([-0.3,1.3,-0.3,1.3])
                        axis('scaled')
                    xticks([],[])
                    yticks([],[])
                    if i == 3:
                        xlabel(anames[j])
                    if j == 0:
                        ylabel(anames[i])
            savefig(opt.prefix + 'codes-%i.png' % pnum)


        index_free(I)
        
