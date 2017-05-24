# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import sys
import os
from astrometry.sdss.dr9 import *
from astrometry.sdss import *
from astrometry.util.plotutils import *

from scipy.ndimage.filters import *

def test_vs_idl():
    run, camcol, field = 1752, 3, 163
    band ='r'
    bandnum = band_index(band)

    datadir = os.path.join(os.path.dirname(__file__), 'testdata')

    sdss = DR9(basedir=datadir)
    sdss.retrieve('psField', run, camcol, field)
    psfield = sdss.readPsField(run, camcol, field)

    ps = PlotSequence('klpsf')

    # These psf*.fits files were produced on the NYU system via:
    #
    #  psfield = mrdfits('psField-001752-3-0163.fit', 3)
    #  psfimage = sdss_psf_recon(psfield, 1000., 0.)
    #  mwrfits,psfimage,'psf1k0.fits',/CREATE
    #
    for x,y,fn in [(0.,0.,'psf00.fits'),
                   (0., 1000., 'psf01k.fits'),
                   (1000., 0., 'psf1k0.fits'),
                   (0., 2000., 'psf02k.fits'),
                   (2000., 0., 'psf2k0.fits'),
                   (600.,500.,'psf.fits')]:

        psf0 = pyfits.open(os.path.join(datadir, fn))[0].data

        # The IDL code adds 0.5 to the pixel coords
        psf = psfield.getPsfAtPoints(bandnum, x + 0.5, y + 0.5)

        psf = psf.astype(np.float32)

        def show(im):
            plt.imshow(np.log10(np.maximum(im, 1e-4)), interpolation='nearest', origin='lower')
        
        plt.clf()
        show(psf0)
        plt.gray()
        plt.colorbar()
        plt.title('IDL: %.1f, %.1f' % (x,y))
        ps.savefig()

        plt.clf()
        show(psf)
        plt.colorbar()
        plt.title('Me: %.1f, %.1f' % (x,y))
        ps.savefig()

        plt.clf()
        plt.imshow(psf - psf0, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.title('Diff: %.1f, %.1f' % (x,y))
        ps.savefig()

        diff = psf - psf0

        print('Diff:', diff.min(), diff.max())
        rms = np.sqrt(np.mean(diff**2))
        print('RMS:', rms)
        assert(np.all(np.abs(diff) < 5e-8))
        assert(rms < 2e-8)

def test_vs_stars():
    ps = PlotSequence('klpsf')

    #run, camcol, field, band = 4948, 6, 249, 'r'
    run, camcol, field, band = 94, 6, 11, 'r'
    test_vs_stars_on(run, camcol, field, band, ps)

    return
    run, camcol, field = 1752, 3, 163
    band ='r'
    test_vs_stars_on(run, camcol, field, band, ps)
    
    #756-z6-700
    run, camcol, field = 756, 6, 700
    band ='z'
    test_vs_stars_on(run, camcol, field, band, ps)

def test_vs_stars_on(run, camcol, field, band, ps):

    bandnum = band_index(band)
    datadir = os.path.join(os.path.dirname(__file__), 'testdata')
    sdss = DR9(basedir=datadir)

    sdss.retrieve('psField', run, camcol, field)
    psfield = sdss.readPsField(run, camcol, field)

    sdss.retrieve('frame', run, camcol, field, band)
    frame = sdss.readFrame(run, camcol, field, band)
    img = frame.image
    H,W = img.shape

    fn = sdss.retrieve('photoObj', run, camcol, field)
    T = fits_table(fn)
    T.mag  = T.get('psfmag')[:,bandnum]
    T.flux = T.get('psfflux')[:,bandnum]
    # !!
    T.x = T.colc[:,bandnum] - 0.5
    T.y = T.rowc[:,bandnum] - 0.5
    T.cut(T.prob_psf[:,bandnum] == 1)
    T.cut(T.nchild == 0)
    T.cut(T.parent == -1)
    T.cut(T.flux > 1.)
    print(len(T), 'after flux cut')
    #T.cut(T.flux > 10.)
    #print len(T), 'after flux cut'
    #T.cut(T.flux > 20.)
    #print len(T), 'after flux cut'
    T.cut(np.argsort(-T.flux)[:25])
    # margin
    m = 30
    T.cut((T.x >= m) * (T.x < (W-m)) * (T.y >= m) * (T.y < (H-m)))
    #T.cut(np.argsort(T.mag))
    T.cut(np.argsort(-np.abs(T.x - T.y)))
    
    print(len(T), 'PSF stars')
        
    #R,C = 5,5
    #plt.clf()

    eigenpsfs  = psfield.getEigenPsfs(bandnum)
    eigenpolys = psfield.getEigenPolynomials(bandnum)
    RR,CC = 2,2
    
    xx,yy = np.meshgrid(np.linspace(0, W, 12), np.linspace(0, H, 8))

    ima = dict(interpolation='nearest', origin='lower')
    
    plt.clf()
    mx = None
    for i,(psf,poly) in enumerate(zip(eigenpsfs, eigenpolys)):
        print()
        print('Eigen-PSF', i)
        XO,YO,C = poly
        kk = np.zeros_like(xx)
        for xo,yo,c in zip(XO,YO,C):
            dk = (xx ** xo) * (yy ** yo) * c
            #print 'xo,yo,c', xo,yo,c, '-->', dk
            kk += dk
        print('Max k:', kk.max(), 'min', kk.min())
        print('PSF range:', psf.min(), psf.max())
        print('Max effect:', max(np.abs(kk.min()), kk.max()) * max(np.abs(psf.min()), psf.max()))
        
        plt.subplot(RR,CC, i+1)
        plt.imshow(psf * kk.max(), **ima)
        if mx is None:
            mx = (psf * kk.max()).max()
        else:
            plt.clim(-mx * 0.05, mx * 0.05)
        plt.colorbar()
    ps.savefig()

    psfs = psfield.getPsfAtPoints(bandnum, xx,yy)
    #print 'PSFs:', psfs
    ph,pw = psfs[0].shape
    psfs = np.array(psfs).reshape(xx.shape + (ph,pw))
    print('PSFs shape:', psfs.shape)
    psfs = psfs[:,:,15:36,15:36]
    ny,nx,ph,pw = psfs.shape
    psfmos = np.concatenate([psfs[i,:,:,:] for i in range(ny)], axis=1)
    print('psfmos', psfmos.shape)
    psfmos = np.concatenate([psfmos[i,:,:] for i in range(nx)], axis=1)
    print('psfmos', psfmos.shape)
    plt.clf()
    plt.imshow(np.log10(np.maximum(psfmos + 1e-3, 1e-3)), **ima)
    ps.savefig()
    
    diffs = None
    rmses = None
    
    for i in range(len(T)):

        xx = T.x[i]
        yy = T.y[i]
        
        ix = int(np.round(xx))
        iy = int(np.round(yy))
        dx = xx - ix
        dy = yy - iy

        S = 25
        stamp = img[iy-S:iy+S+1, ix-S:ix+S+1]

        L = 6
        Lx = lanczos_filter(L, np.arange(-L, L+1) - dx)
        Ly = lanczos_filter(L, np.arange(-L, L+1) - dy)
        sx = correlate1d(stamp, Lx, axis=1, mode='constant')
        shim = correlate1d(sx,  Ly, axis=0, mode='constant')
        shim /= (Lx.sum() * Ly.sum())
        
        psf = psfield.getPsfAtPoints(bandnum, xx, yy)
        mod = psf / psf.sum() * T.flux[i]

        psf2 = psfield.getPsfAtPoints(bandnum, yy, xx)
        mod2 = psf2 / psf2.sum() * T.flux[i]

        psf3 = psfield.getPsfAtPoints(bandnum, xx, yy).T
        mod3 = psf3 / psf3.sum() * T.flux[i]

        mods = [mod, mod2, mod3]

        if diffs is None:
            diffs = [np.zeros_like(m) for m in mods]
            rmses = [np.zeros_like(m) for m in mods]
            
        for m,rms,diff in zip(mods, rmses, diffs):
            diff += (m - shim)
            rms +=  (m - shim)**2

        if i > 10:
            continue
        
        def show(im):
            plt.imshow(np.log10(np.maximum(1e-3, im + 1e-3)), vmin=-3, vmax=0, **ima)
            plt.hot()
            plt.colorbar()
        
        R,C = 3,4
        plt.clf()
        plt.subplot(R,C,1)
        show(shim)
        plt.subplot(R,C,2)
        show(mods[0])
        plt.subplot(R,C,2+C)
        plt.imshow(mods[0] - shim, vmin=-0.05, vmax=0.05, **ima)
        plt.hot()
        plt.colorbar()

        plt.subplot(R,C,3)
        show(mods[1])

        plt.subplot(R,C,3+C)
        plt.imshow(mods[1] - shim, vmin=-0.05, vmax=0.05, **ima)
        plt.hot()
        plt.colorbar()
        
        plt.subplot(R,C,4)
        show(mods[2])
        
        plt.subplot(R,C,4+C)
        plt.imshow(mods[2] - shim, vmin=-0.05, vmax=0.05, **ima)
        plt.hot()
        plt.colorbar()

        plt.subplot(R,C,3+2*C)
        plt.imshow(mods[1] - mods[0], vmin=-0.05, vmax=0.05, **ima)
        plt.hot()
        plt.colorbar()

        plt.subplot(R,C,4+2*C)
        plt.imshow(mods[2] - mods[0], vmin=-0.05, vmax=0.05, **ima)
        plt.hot()
        plt.colorbar()

        plt.suptitle('%.0f, %.0f' % (xx,yy))
        ps.savefig()

    for diff in diffs:
        diff /= len(T)

    rmses = [np.sqrt(rms / len(T)) for rms in rmses]

    print('rms median', np.median(rmses[0]), 'mean', np.mean(rmses[0]))

    r0,r1 = [np.percentile(rmses[0], p) for p in [10,90]]
    
    R,C = 2,len(diffs)
    plt.clf()

    for i,(d,r) in enumerate(zip(diffs, rmses)):
        plt.subplot(R,C, 1+i)
        plt.imshow(d, vmin=-0.05, vmax=0.05, **ima)
        plt.colorbar()
        plt.hot()

        plt.subplot(R,C, 1+i+C)
        plt.imshow(r, vmin=r0, vmax=r1, **ima)
        plt.colorbar()
        plt.hot()
    
    ps.savefig()

            
        
if __name__ == '__main__':
    #test_vs_idl()
    test_vs_stars()
    
