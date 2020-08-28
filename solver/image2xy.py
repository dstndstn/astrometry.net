# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function

"""
NAME:
      image2xy

PURPOSE:
      Extract sources from a FITS file

INPUTS:
      Takes a single FITS file as input

OPTIONAL INPUTS:

KEYWORD PARAMETERS:

OUTPUTS:

OPTIONAL OUTPUTS:

COMMENTS:

MODIFICATION HISTORY:
       K. Mierle, 2007-Jan - Initial version based on image2xy.c
       Hogg, 2007-May - simplexy options change
"""

# You need ctypes and a recent (1.0) numpy for this to work. I've included
# pyfits so you don't have to. 

try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
import sys
import scipy
import os
import time
from simplexy import simplexy

# Default settings
dpsf = 1.0
plim = 8.0
dlim = dpsf
saddle = 5.0
maxper = 1000
maxnpeaks = 10000
maxsize = 1000
halfbox= 100

def source_extract(image_data, srcext=None):


    x,y,flux,sigma = simplexy(image_data, dpsf=dpsf, plim=plim,
                              dlim=dlim, saddle=saddle, maxper=maxper,
                              maxnpeaks=maxnpeaks, maxsize=maxsize,
                              halfbox=halfbox)

    print('simplexy: shapes',x.shape, y.shape)

    cx = pyfits.Column(name='X', format='E', array=x, unit='pix')
    cy = pyfits.Column(name='Y', format='E', array=y, unit='pix')
    cflux = pyfits.Column(name='FLUX', format='E', array=flux)

    tbhdu = pyfits.new_table([cx, cy, cflux])
    h = tbhdu.header
    h.add_comment('Parameters used in source extraction')
    h.update('dpsf',     dpsf, 'Gaussian psf width')
    h.update('plim',     plim, 'Significance to keep')
    h.update('dlim',     dlim, 'Closest two peaks can be')
    h.update('saddle',   saddle, 'Saddle in difference (in sig)')
    h.update('maxper',   maxper, 'Max num of peaks per object')
    h.update('maxpeaks', maxnpeaks, 'Max num of peaks total')
    h.update('maxsize',  maxsize, 'Max size of extended objects')
    h.update('halfbox',  halfbox, 'Half-size of sliding sky window')
    if srcext != None:
        h.update('srcext', i, 'Extension number in src image')
    h.update('estsigma', sigma, 'Estimated source image variance')
    h.add_comment('The X and Y points are specified assuming 1,1 is ')
    h.add_comment('the center of the leftmost bottom pixel of the ')
    h.add_comment('image in accordance with the FITS standard.')
    h.add_comment('Extracted by image2xy.py')
    h.add_comment('on %s %s' % (time.ctime(), time.tzname[0]))
    cards = tbhdu.header.ascardlist()
    cards['TTYPE1'].comment = 'X coordinate'
    cards['TTYPE2'].comment = 'Y coordinate'
    cards['TTYPE3'].comment = 'Flux of source'

    return tbhdu

def extract(fitsfile):
    outfile = pyfits.HDUList()

    # Make empty HDU; no image
    outfile.append(pyfits.PrimaryHDU()) 

    for i, hdu in enumerate(fitsfile):
        if (i == 0 and hdu.data != None) or isinstance(hdu, pyfits.ImageHDU):
            print(hdu.data.shape+(i,))
            if i == 0:
                print('Image: Primary HDU (number 0) %sx%s' % hdu.data.shape)
            else:
                print('Image: Extension HDU (number %s) %sx%s' % tuple((i,)+hdu.data.shape))

            tbhdu = source_extract(image_data)

            outfile.append(tbhdu)

    return x,y,flux,sigma, outfile

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: image2xy.py image.fits")
    else:
        infile = sys.argv[1]
        fitsfile = pyfits.open(infile)
        x,y,flux,sigma, outfile = extract(fitsfile)
        newfile = infile.replace('.fits','.xy.fits')
        try:
            outfile.writeto(newfile)
        except IOError:
            # File probably exists
            print('File %s appears to already exist; deleting!' % newfile)
            import os
            os.unlink(newfile)
            outfile.writeto(newfile)
