# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

from math import *  # Needed if we're going to exclude pylab :)

PIl = 3.1415926535897932384626433832795029
def rad2deg(r):    return 180.0*r/PIl
def deg2rad(d):    return d*PIl/180.0
def rad2arcmin(r): return 10800.0*r/PIl
def arcmin2rad(a): return a*PIl/10800.0
def rad2arcsec(r): return 648000.0*r/PIl
def arcsec2rad(a): return a*PIl/648000.0
def radec2x(r,d):  return cos(d)*cos(r) # r,d in radians
def radec2y(r,d):  return cos(d)*sin(r) # r,d in radians
def radec2z(r,d):  return sin(d)        # r,d in radians
def z2dec(z):      return arcsin(z)     # result in radians
def xy2ra(x,y):
    "Convert x,y to ra in radians"
    r = arctan2(y,x)
    r += 2*PIl*(r>=0.00)
    return r

# wrapper to give us a tuple.
def radec2xyz(r,d): return (radec2x(r,d), radec2y(r,d), radec2z(r,d))

# Takes min/max RA & DEC (which a number of the catalogs provide) and
# compute the xyz coordinates of the box on the surface of the sphere.
def radecbox2xyz(min_r, max_r, min_d, max_d):
    return [radec2xyz(min_r, min_d), radec2xyz(min_r, max_d), \
        radec2xyz(max_r, max_d), radec2xyz(max_r, min_d)]



#define radscale2xyzscale(r) (sqrt(2.0-2.0*cos(r/2.0)))
def star_coords(s,r):
    # eta is a vector perpendicular to r
    etax = -r[1]
    etay = +r[0]
    etaz = 0.0
    eta_norm = sqrt(etax * etax + etay * etay)
    etax /= eta_norm
    etay /= eta_norm

    # xi =  r cross eta
    xix = -r[2] * etay
    xiy =  r[2] * etax
    xiz =  r[0] * etay - r[1] * etax
    sdotr = dot(s,r)

    return (
            s[0] * xix / sdotr +
            s[1] * xiy / sdotr +
            s[2] * xiz / sdotr,
            s[0] * etax / sdotr +
            s[1] * etay / sdotr
           )

def radectohealpix(ra, dec):
    x = radec2x(ra, dec)
    y = radec2y(ra, dec)
    z = radec2z(ra, dec)
    return xyztohealpix(x, y, z)

def xyztohealpix(x, y, z) :
    "Return the healpix catalog number associated with point (x,y,z)"

    # the polar pixel center is at (z,phi/pi) = (2/3, 1/4)
    # the left pixel center is at (z,phi/pi) = (0, 0)
    # the right pixel center is at (z,phi/pi) = (0, 1/2)
    twothirds = 2.0 / 3.0

    phi = arctan2(y, x)
    if phi < 0.0:
        phi += 2.0 * pi

    phioverpi = phi / pi

    if z >= twothirds or z <= -twothirds:
        # definitely in the polar regions.
        # the north polar healpixes are 0,1,2,3
        # the south polar healpixes are 8,9,10,11
        if z >= twothirds:
            offset = 0
        else:
            offset = 8

        pix = int(phioverpi * 2.0)

        return offset + pix

    # could be polar or equatorial.
    offset = int(phioverpi * 2.0)
    phimod = phioverpi - offset * 0.5

    z1 =  twothirds - (8.0 / 3.0) * phimod
    z2 = -twothirds + (8.0 / 3.0) * phimod

    if z >= z1 and z >= z2:
        # north polar
        return offset
    if z <= z1 and z <= z2:
        # south polar
        return offset + 8
    if phimod < 0.25:
        # left equatorial
        return offset + 4
    # right equatorial
    return ((offset+1)%4) + 4


# Brought over from starutil.c & plotcat.c
def project_equal_area(point):
    x, y, z = point
    xp = x * sqrt(1. / (1. + z))
    yp  = y * sqrt(1. / (1. + z))
    xp = 0.5 * (1.0 + xp)
    yp = 0.5 * (1.0 + yp)
    return (xp, yp)

def project_hammer_aitoff_x(point):
    x, y, z = point
    theta = atan(float(x) / (z + 0.000001))
    r = sqrt(x * x + z * z)
    if z < 0:
        if x < 0:
            theta = theta - pi 
        else:
            theta = pi + theta
    theta /= 2.0
    zp = r * cos(theta)
    xp = r * sin(theta)
    assert zp >= -0.01
    return project_equal_area((xp, y, zp))


def getxy(projectedpoint, N):
    px, py = projectedpoint
    px = 0.5 + (px - 0.5) * 0.99
    py = 0.5 + (py - 0.5) * 0.99
    X = round(px * 2*N)
    Y = round(py * N)
    return (X, Y)

