# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from pylab import *
from numpy import *
from math import *

# I can never remember where all the bands are...

rawdata = [
    #              min  max  mid fwhm   label-offset
    ('galex',   'fuv', 135, 175, 0, 0,  0),
    ('galex',   'nuv', 175, 280, 0, 0,  0),
    ('usnob', 'B [OJ]', 350, 540, 0, 0,      0),
    ('usnob', 'R [EF]', 590, 690, 0, 0,      1),
    ('usnob', 'N', 715, 900, 0, 0,      0),
    # Wikipedia:UBV... says: 364 nm for U, 442 nm for B, 540 nm for V.
    ('Johnson', 'U', 0,0, 364, 200, 0),
    ('Johnson', 'B', 375,500, 0,0,1),
    ('Johnson', 'V', 500,600, 0,0,0),
    ('Johnson', 'R', 550,725, 0,0,1),
    ('Johnson', 'I', 725,875, 0,0,0),
    ('sdss',  'u',   0,   0, 354,  57,  0),
    ('sdss',  'g',   0,   0, 477, 137,  1),
    ('sdss',  'r',   0,   0, 623, 137,  0),
    ('sdss',  'i',   0,   0, 763, 153,  1),
    ('sdss',  'z',   0,   0, 913,  95,  0),

    # I got these from eyeballing the synphot throughputs
    ('hst', '275', 250, 300, 0, 0,    0),
    ('hst', '336', 310, 370, 0, 0,    1),
    ('hst', '475', 400, 550, 0, 0,    0),
    ('hst', '814', 700, 950, 0, 0,    0),
    ('hst', '110', 900,1400, 0, 0,    1),
    ('hst', '160',1400,1700, 0, 0,    0),
    
    ('ukidss','Z', 830,  925, 0,   0,   0),
    ('ukidss','Y', 970, 1070, 0,   0,   1),
    ('ukidss','J',1170, 1330, 0,   0,   0),
    ('ukidss','H',1490, 1780, 0,   0,   1),
    ('ukidss','K',2030, 2370, 0,   0,   0),
    ('2mass', 'J',   0,   0, 1240, 0,   0),
    ('2mass', 'H',   0,   0, 1600, 0,   0),
    ('2mass', 'Ks',  0,   0, 2160, 0,   0),
    ('spitzer', '1', 0,   0, 3600, 0,   0),
    ('spitzer', '2', 0,   0, 4500, 0,   0),
    ('spitzer', '3', 0,   0, 5800, 0,   0),
    ('spitzer', '4', 0,   0, 8000, 0,   0),
    ]

surveys = array([survey for (survey, band, a,b,c,d,e) in rawdata])
los = array([lo or mid - fwhm/2. for (survey, band, lo, hi, mid, fwhm, e) in rawdata])
his = array([hi or mid + fwhm/2. for (survey, band, lo, hi, mid, fwhm, e) in rawdata])

survs = set(surveys)
surveymids = {}
surveylos = {}
for s in survs:
    surveymids[s] = sqrt(los[surveys == s].min() * his[surveys == s].max())
    surveylos[s] = los[surveys == s].min()

ymargin = 0.1
xmargin = 1.1

clf()
lastsurvey = ''
h = 0
for (survey, band, mn, mx, mid, fwhm, off) in rawdata:
    lo = mn or mid - fwhm/2.
    hi = mx or mid + fwhm/2.

    if survey != lastsurvey:
        h += 1
        text(surveylos[survey] / xmargin, h, survey,
             horizontalalignment='right',
             verticalalignment='center')
        lastsurvey = survey
    semilogx([lo, hi], [h, h], 'ko-')
    text(sqrt(lo*hi), h+ymargin, band,
         horizontalalignment='center',
         verticalalignment='bottom',
         fontsize='10')

    if mn:
        txt = '%g-%g' % (mn, mx)
    else:
        if fwhm:
            txt = '%g(%g)' % (mid, fwhm)
        else:
            txt = '%g' % (mid)
    text(sqrt(lo*hi), h-1.5*(ymargin*(1+off)), txt,
         horizontalalignment='center',
         verticalalignment='top',
         fontsize='7')

a=axis()
axis([70, 1.1e4, 0, h+1])

y0 = 0.1
y1 = 0.25
mx = 1.0e4
mn = 75
for (x, y, txt) in [(1e-11, y0, 'Gamma'),
                    (1e-9,  y1, 'X-ray'),
                    (400e-9,   y0, 'purple'),
                    (500e-9,   y1, 'green'),
                    (700e-9,   y0, 'red'),
                    (10e-6,   y0, 'infrared'),
                    ]:
    ha = 'left'
    x /= 1e-9
    if x >= mx:
        x = mx
        ha = 'right'
    if x < mn:
        x = mn
    text(x, y, txt,
         verticalalignment='bottom',
         horizontalalignment=ha,
         fontsize='8')

xticks([100, 1000, 10000], ['100 nm', '1 um', '10 um'])
yticks([],[])
xlabel('Wavelength')
title('Bands')
savefig('bands.png')
    

# USNOB
# http://www.iop.org/EJ/article/1538-3881/125/2/984/202452.tb1.html
# (B)
# O: 350-500 nm
# J: 385-540 nm
#  : 395-540
# (R)
# E: 620-670 nm
# F: 610-690 nm
#  : 630-690
#  : 590-690
# (N)
# N: 730-900 nm
#  : 715-900

# 2MASS
# http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_1.new.html
# J: 1.24 um
# H: 1.66 um
# Ks: 2.16 um

# SDSS
# http://www.astro.princeton.edu/PBOOK/camera/camera.htm
# 5 channels
# u: 354 +- 57/2 nm
# g: blue-green    477 +- 137/2 nm
# r: red           623 +- 137/2 nm
# i: far-red       763 +- 153/2 nm
# z: near-infrared 913 +- 95/2 nm

# Spitzer - IRAC
# http://www.spitzer.caltech.edu/technology/irac.shtml
# 4 channels
# 5.12 x 5.12 arcmin
# 3.6, 4.5, 5.8, and 8 microns
# 256 x 256 pixels

# GALEX
# http://www.galex.caltech.edu/QUICKFACTS/quickfacts.html
# fuv: 135 to 175 nm
# nuv: 175 to 280 nm

# HST
# many!
# http://www.stsci.edu/hst/acs/documents/handbooks/cycle17/c05_imaging2.html#356971

