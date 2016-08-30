# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from math import pi,cos,sin,radians,degrees,asin,atan2,sqrt,acos,floor
from functools import reduce

def hms2ra(h, m, s):
    return 15. * (h + (m + s/60.)/60.)

def dms2dec(sign, d, m, s):
    return sign * (d + (m + s/60.)/60.)

def tokenize_hms(s):
    s = s.strip()
    # split on whitespace
    tokens = s.split()
    # also split on colons
    tokens = reduce(list.__add__, [t.split(':') for t in tokens])
    assert(len(tokens) < 4)
    assert(len(tokens) > 0)
    h = len(tokens) >= 1 and float(tokens[0]) or 0
    m = len(tokens) >= 2 and float(tokens[1]) or 0
    s = len(tokens) >= 3 and float(tokens[2]) or 0
    return (h,m,s)

def hmsstring2hms(st):
    (h,m,s) = tokenize_hms(st)
    return (h,m,s)

def hmsstring2ra(st):
    return hms2ra(*hmsstring2hms(st))

# returns (sign, deg, minutes, second)
def dmsstring2dms(s):
    sign = 1.0
    if s[0] == '-':
        sign = -1.0
        s = s[1:]
    elif s[0] == '+':
        s = s[1:]
    (d,m,s) = tokenize_hms(s)
    return (sign, d, m, s)

def dmsstring2dec(s):
    return dms2dec(*dmsstring2dms(s))

# RA in degrees
def ra2hms(ra):
    h = ra * 24. / 360.
    hh = int(floor(h))
    m = (h - hh) * 60.
    mm = int(floor(m))
    s = (m - mm) * 60.
    return (hh, mm, s)

# Dec in degrees
# returns (sign, degrees, minutes, seconds)
def dec2dms(dec):
    sgn = (dec > 0) and 1 or -1
    d = dec * sgn
    dd = int(floor(d))
    m = (d - dd) * 60.
    mm = int(floor(m))
    s = (m - mm) * 60.
    return (sgn, d, m, s)

# RA in degrees
def ra2hmsstring(ra, separator=' '):
    (h,m,s) = ra2hms(ra)
    ss = int(floor(s))
    ds = int(round((s - ss) * 1000.0))
    return separator.join(['%0.2i' % h, '%0.2i' % m, '%0.2i.%0.3i' % (ss,ds)])

# Dec in degrees
def dec2dmsstring(dec, separator=' '):
    (sign,d,m,s) = dec2dms(dec)
    # whole number of seconds
    ss = int(floor(s))
    # milliseconds
    ds = int(round((s - ss) * 1000.0))
    # fix rounding errors...
    if ds >= 1000:
        ds -= 1000
        ss += 1
    return separator.join(['%+0.2i' % int(d*sign), '%0.2i' % m, '%0.2i.%0.3i' % (ss,ds)])

# RA, Dec in degrees
def radectoxyz(ra, dec):
    rarad = radians(ra)
    decrad = radians(dec)
    cosd = cos(decrad)
    return (cosd * cos(rarad), cosd * sin(rarad), sin(decrad))

# RA, Dec in degrees
def xyztoradec(x,y,z):
    return (degrees(xy2ra(x, y)), degrees(z2dec(z)))

def xyzarrtoradec(xyz):
    return (degrees(xy2ra(xyz[0], xyz[1])), degrees(z2dec(xyz[2])))

def rad2deg(r):    return 180.0*r/pi
def deg2rad(d):    return d*pi/180.0
def rad2arcmin(r): return 10800.0*r/pi
def arcmin2rad(a): return a*pi/10800.0
def rad2arcsec(r): return 648000.0*r/pi
def arcsec2rad(a): return a*pi/648000.0
def arcsec2deg(a): return rad2deg(arcsec2rad(a))
def radec2x(r,d):  return cos(d)*cos(r) # r,d in radians
def radec2y(r,d):  return cos(d)*sin(r) # r,d in radians
def radec2z(r,d):  return sin(d)        # r,d in radians
def z2dec(z):      return asin(z)     # result in radians
def xy2ra(x,y):
    "Convert x,y to ra in radians"
    r = atan2(y,x)
    r += 2*pi*(r<0.)
    return r

def degrees_between(ra1, dec1, ra2, dec2):
    return arcsec2deg(arcsec_between(ra1, dec1, ra2, dec2))

# RA,Decs in degrees.
def arcsec_between(ra1, dec1, ra2, dec2):
    xyz1 = radectoxyz(ra1, dec1)
    xyz2 = radectoxyz(ra2, dec2)
    d2 = sum([(a-b)**2 for (a,b) in zip(xyz1, xyz2)])
    return distsq2arcsec(d2)

def rad2distsq(rad):
    return 2. * (1. - cos(rad))

def arcsec2distsq(arcsec):
    return rad2distsq(arcsec2rad(arcsec))

def arcsec2dist(arcsec):
    return sqrt(arcsec2distsq(arcsec))

def dist2arcsec(dist):
    return distsq2arcsec(dist**2)

def distsq2arcsec(dist2):
    return rad2arcsec(distsq2rad(dist2))

def distsq2rad(dist2):
    return acos(1. - dist2 / 2.)
