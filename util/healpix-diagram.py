# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

from astrometry.util.util import *
from astrometry.util.starutil_numpy import *
from numpy import *
from pylab import *

def cosdeg(y):
    return cos(deg2rad(y))

def plot_healpix_on_radec(nside=2, cosdec=False, fn=None, textkws=None):

    def ycoord(y):
        if cosdec:
            return sin(deg2rad(y))
        else:
            return y

    clf()
    
    # RA,Dec grid lines.
    for x in range(-50,410,10):
        axvline(x=x, lw=0.5, ls='-', color='0.8')
    for y in range(-90,100,10):
        axhline(y=ycoord(y), lw=0.5, ls='-', color='0.8')

    if textkws is None:
        textkws = {}

    if not 'horizontalalignment' in textkws:
        textkws['horizontalalignment'] = 'center'
    if not 'verticalalignment' in textkws:
        textkws['verticalalignment'] = 'center'

    for hp in range(12 * nside**2):
        (racen,deccen) = healpix_to_radec(hp, nside, 0.5, 0.5)
        text(racen, ycoord(deccen), '%i' % hp, **textkws)

        nsteps = 20
        stepvals = [x/float(nsteps) for x in range(nsteps+1)]
        rstepvals = [1-x for x in stepvals]
        ones = [1.0]*(nsteps+1)
        zeros = [0.0]*(nsteps+1)

        x = stepvals + ones + rstepvals + zeros
        y = zeros + stepvals + ones + rstepvals

        edges = [healpix_to_radec(hp, nside, xx, yy) for (xx,yy) in zip(x,y)]
        ras = [ra for (ra,dec) in edges]
        decs = [dec for (ra,dec) in edges]

        if max(ras) > 270 and min(ras) < 90:
            # wrap-around!
            ras1 = []
            ras2 = []
            for ra in ras:
                if ra < 180:
                    ras1.append(ra)
                    ras2.append(ra+360)
                else:
                    ras1.append(ra-360)
                    ras2.append(ra)
            plot(ras1, ycoord(decs), 'k--')
            plot(ras2, ycoord(decs), 'k-')
            if racen > 180:
                text(racen-360, ycoord(deccen), '%i' % hp, **textkws)
            else:
                text(racen+360, ycoord(deccen), '%i' % hp, **textkws)

        else:
            plot(ras, ycoord(decs), 'k-')

    xlabel('RA (deg)')
    ylabel('Dec (deg)')
    ylim(ycoord(-90), ycoord(90))
    xlim(-40,400)
    yt = arange(-90, 100, 10)
    yticks(ycoord(yt), ['%g'%y for y in yt])
    title('Healpixes (nside=%i)' % nside)
    if fn is None:
        fn = 'healpix-%i.png' % nside
    savefig(fn)

def plot_radec_on_healpix(nside=1, bighp=1):
    clf()
    xlabel('Healpix x')
    ylabel('Healpix y')

    (ralo,rahi,declo,dechi) = healpix_radec_bounds(bighp, 1)

    ragrid = 5
    decgrid = 5
    rastep = 0.5
    decstep = 0.5

    ralo = floor(ralo/ragrid) *ragrid
    rahi = (1+floor(rahi/ragrid)) *ragrid
    declo = floor(declo/decgrid) *decgrid
    dechi = (1+floor(dechi/decgrid)) *decgrid

    for ra in arange(ralo,rahi,ragrid):
        xy=[]
        lastok = True
        for dec in arange(declo, dechi, decstep):
            (bhp, hpx, hpy) = radectohealpixf(ra, dec, nside)
            #print 'ra,dec', ra,dec, 'bighp', bhp, 'x,y', hpx,hpy
            if bhp != bighp:
                if lastok and len(xy):
                    plot([x for x,y in xy], [y for x,y in xy], 'r-')
                    xy = []
                lastok = False
                continue
            lastok = True
            xy.append((hpx,hpy))
        if len(xy):
            plot([x for x,y in xy], [y for x,y in xy], 'r-')

    for dec in arange(declo, dechi, decgrid):
        xy=[]
        lastok = True
        for ra in arange(ralo,rahi,rastep):
            (bhp, hpx, hpy) = radectohealpixf(ra, dec, nside)
            if bhp != bighp:
                if lastok and len(xy):
                    plot([x for x,y in xy], [y for x,y in xy], 'r-')
                    xy = []
                lastok = False
                continue
            lastok = True
            xy.append((hpx,hpy))
        if len(xy):
            plot([x for x,y in xy], [y for x,y in xy], 'r-')

    axhline(y=0, lw=0.5, ls='-', color='0.8')
    axhline(y=nside, lw=0.5, ls='-', color='0.8')
    axvline(x=0, lw=0.5, ls='-', color='0.8')
    axvline(x=nside, lw=0.5, ls='-', color='0.8')
    axis('scaled')
    axis([-0.1, nside+0.1, -0.1, nside+0.1])
    title('RA,Dec lines in healpix space, for big healpix %i' % bighp)
    savefig('hp-radec-%i.png' % bighp)

def plot_smallcircles_on_healpix(nside=1, bighp=1):
    clf()
    xlabel('Healpix x')
    ylabel('Healpix y')

    #xstep = ystep = 0.05
    #radius = 1.3

    xstep = ystep = 0.1
    radius = 2.5

    ncircle = 60

    runit = sqrt(deg2distsq(radius))

    figure(figsize=(6,6))
    for x in arange(0, 1.001, xstep):
        for y in arange(0, 1.001, ystep):
            xyz = array(healpix_to_xyz(bighp, nside, x, y))
            r,d = xyztoradec(xyz)
            dra,ddec = derivatives_at_radec(r,d)
            dra = dra[0,:]
            dra /= vector_norm(dra)
            ddec = ddec[0,:]
            ddec /= vector_norm(ddec)
            assert(abs(dot(xyz, dra))  < 1e-15)
            assert(abs(dot(xyz, ddec)) < 1e-15)
            assert(abs(dot(dra, ddec)) < 1e-15)
            #print dot(xyz,dra)
            #print dot(xyz,ddec)
            #print dot(dra,ddec)
            theta = linspace(0, 2*pi, ncircle)
            xyzs = array([xyz + runit*sin(t)*dra + runit*cos(t)*ddec
                          for t in theta])
            xyzs /= vector_norm(xyzs)
            #(bighp, hpx, hpy) = xyz_to_healpixf(p[0],p[1],p[2], nside)
            HPs = array([xyztohealpixf(p[0],p[1],p[2], nside) for p in xyzs])
            bighps = HPs[:,0]
            hpx = HPs[:,1]
            hpy = HPs[:,2]
            I = (bighps == bighp)
            if sum(I):
                plot(hpx[I], hpy[I], 'r-')

    axhline(y=0, lw=0.5, ls='-', color='0.8')
    axhline(y=nside, lw=0.5, ls='-', color='0.8')
    axvline(x=0, lw=0.5, ls='-', color='0.8')
    axvline(x=nside, lw=0.5, ls='-', color='0.8')
    axis('scaled')
    margin = 0.05
    axis([-margin, nside+margin, -margin, nside+margin])
    title('Small circles in healpix space, for big healpix %i' % bighp)
    xlabel('Healpix x')
    ylabel('Healpix y')
    savefig('hp-circles-%i.png' % bighp)

if __name__ == '__main__':

    for bighp in [1,5,9]:
        plot_smallcircles_on_healpix(nside=1, bighp=bighp)
        plot_radec_on_healpix(nside=1, bighp=bighp)

    plot_healpix_on_radec(nside=1, fn='healpix-1a.png')
    plot_healpix_on_radec(nside=3, fn='healpix-3a.png', textkws={'fontsize':8})

    plot_healpix_on_radec(nside=1, cosdec=True)
    plot_healpix_on_radec(nside=3, cosdec=True, textkws={'fontsize':8})


