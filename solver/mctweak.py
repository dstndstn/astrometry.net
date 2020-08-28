# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import emcee
import triangle

from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.fits import *
from astrometry.solver.solver import *


class McTweak(object):
    def __init__(self, wcs, xy, rd):
        self.refra  = rd.ra
        self.refdec = rd.dec
        self.testxy = np.vstack((xy.x, xy.y)).T
        nt = len(xy)
        sig2 = 1.
        self.testsig2 = np.zeros(nt) + sig2

        self.W = wcs.get_width()
        self.H = wcs.get_height()
        self.distractors = 0.25
        ## Accept: set to ~inf?
        self.logodds_bail = -1e100
        self.logodds_accept = 1e12

        self.wcs = wcs
        
    def __call__(self, args):
        # plug args into wcs

        # make a local copy...
        wcs = Sip(self.wcs)
        set_sip_args(wcs, args)

        # sip.radec2pixelxy uses the *inverse* SIP polynomials... compute 'em
        sip_compute_inverse_polynomials(wcs, 20, 20, 1, self.W, 1, self.H)
        
        ok,x,y = wcs.radec2pixelxy(self.refra, self.refdec)
        refxy = np.vstack((x,y)).T
        
        logodds = verify_star_lists_np(refxy, self.testxy, self.testsig2,
                                       self.W * self.H, self.distractors,
                                       self.logodds_bail, self.logodds_accept)
        return logodds



def set_sip_args(wcs, args):
    args = list(reversed(args))
    r = args.pop()
    d = args.pop()
    wcs.set_crval((r,d))
    CD = (args.pop(), args.pop(), args.pop(), args.pop())
    wcs.set_cd(CD)

    order = wcs.a_order
    for p in range(0, order+1):
        for q in range(0, order+1-p):
            if p+q <= 1:
                continue
            assert(p + q <= order)
            wcs.set_a_term(p, q, args.pop())

    order = wcs.b_order
    for p in range(0, order+1):
        for q in range(0, order+1-p):
            if p+q <= 1:
                continue
            assert(p + q <= order)
            wcs.set_b_term(p, q, args.pop())

    assert(len(args) == 0)

def get_sip_args(wcs):
    W,H = wcs.get_width(), wcs.get_height()
    S = max(W, H)
    args = []
    sigs = []
    r,d = wcs.get_crval()
    pixscale = wcs.pixel_scale()
    args.extend([r,d])
    sigs.extend([pixscale/3600.]*2)

    cd1,cd2,cd3,cd4 = wcs.get_cd()
    args.extend([cd1,cd2,cd3,cd4])
    sigs.extend([max(x/1000., pixscale/3600./S) for x in [cd1,cd2,cd3,cd4]])
    
    order = wcs.a_order
    for p in range(0, order+1):
        for q in range(0, order+1-p):
            if p+q <= 1:
                continue
            assert(p + q <= order)
            args.append(wcs.get_a_term(p, q))
            sigs.append(S**-(p+q))
    order = wcs.b_order
    for p in range(0, order+1):
        for q in range(0, order+1-p):
            if p+q <= 1:
                continue
            assert(p + q <= order)
            args.append(wcs.get_b_term(p, q))
            sigs.append(S**-(p+q))
    return args, sigs


def mctweak(wcs, xy, rd):
    obj = McTweak(wcs, xy, rd)

    # Initial args
    args,sigs = get_sip_args(wcs)

    print('Args:', args)
    print('Sigs:', sigs)
    print('Number of arguments:', len(args))
    print('Logodds:', obj(args))

    ndim, nwalkers = len(args), 100
    p0 = emcee.utils.sample_ball(args, sigs, size=nwalkers)
    print('p0', p0.shape)

    ps = PlotSequence('mctweak')

    W,H = wcs.get_width(), wcs.get_height()
    mywcs = Sip(wcs)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, obj)
    lnp0, rstate = None, None
    pp = []
    for step in range(10000):
        print('Step', step)
        p0,lnp0,rstate = sampler.run_mcmc(p0, 1, lnprob0=lnp0, rstate0=rstate)
        print('Best logprob:', np.max(lnp0))
        i = np.argmax(lnp0)
        print('Best args:', p0[i,:])

        pp.extend(sampler.flatchain)
        sampler.reset()
        
        if step % 100 != 0:
            continue

        
        plt.clf()
        plt.plot(obj.testxy[:,0], obj.testxy[:,1], 'r.')
        for args in p0[np.random.permutation(nwalkers)[:10],:]:
            set_sip_args(mywcs, args)
            sip_compute_inverse_polynomials(mywcs, 20, 20, 1, W, 1, H)
            ok,x,y = mywcs.radec2pixelxy(obj.refra, obj.refdec)
            plt.plot(x, y, 'bo', mec='b', mfc='none', alpha=0.25)

            ex = 10.
            ngridx = ngridy = 10
            stepx = stepy = 100
            xgrid = np.linspace(0, W, ngridx)
            ygrid = np.linspace(0, H, ngridy)
            X = np.linspace(0, W, int(np.ceil(W/stepx)))
            Y = np.linspace(0, H, int(np.ceil(H/stepy)))
            for x in xgrid:
                DX,DY = [],[]
                xx,yy = [],[]
                for y in Y:
                    dx,dy = mywcs.get_distortion(x, y)
                    xx.append(x)
                    yy.append(y)
                    DX.append(dx)
                    DY.append(dy)
                DX = np.array(DX)
                DY = np.array(DY)
                xx = np.array(xx)
                yy = np.array(yy)
                EX = DX + ex * (DX - xx)
                EY = DY + ex * (DY - yy)
                #plot(xx, yy, 'k-', alpha=0.5)
                plt.plot(EX, EY, 'b-', alpha=0.1)

            for y in ygrid:
                DX,DY = [],[]
                xx,yy = [],[]
                for x in X:
                    dx,dy = mywcs.get_distortion(x, y)
                    DX.append(dx)
                    DY.append(dy)
                    xx.append(x)
                    yy.append(y)
                DX = np.array(DX)
                DY = np.array(DY)
                xx = np.array(xx)
                yy = np.array(yy)
                EX = DX + ex * (DX - xx)
                EY = DY + ex * (DY - yy)
                #plot(xx, yy, 'k-', alpha=0.5)
                plt.plot(EX, EY, 'b-', alpha=0.1)
                
        for x in xgrid:
            plt.plot(x+np.zeros_like(Y), Y, 'k-', alpha=0.5)
        for y in ygrid:
            plt.plot(X, y+np.zeros_like(X), 'k-', alpha=0.5)
                
        plt.axis([1, W, 1, H])
        plt.axis('scaled')
        ps.savefig()

        pp = np.vstack(pp)
        print('pp', pp.shape)
        
        # plt.clf()
        # triangle.corner(pp, plot_contours=False)
        # ps.savefig()

        pp = []

wcs = Tan('bok-01.wcs', 0)
sip = Sip(wcs)

# sip.a_order = 3
# sip.b_order = 3
# sip.ap_order = 4
# sip.bp_order = 4
sip.a_order = 2
sip.b_order = 2
sip.ap_order = 3
sip.bp_order = 3

xy = fits_table('bok-01.axy')
rd = fits_table('bok-01.rdls')

mctweak(sip, xy, rd)

