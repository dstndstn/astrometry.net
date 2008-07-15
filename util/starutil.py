from math import pi,cos,sin,radians,asin,atan2,sqrt,acos

# RA, Dec in degrees
def radectoxyz(ra, dec):
    rarad = radians(ra)
    decrad = radians(dec)
    cosd = cos(decrad)
    return (cosd * cos(rarad), cosd * sin(rarad), sin(decrad))

def rad2deg(r):    return 180.0*r/pi
def deg2rad(d):    return d*pi/180.0
def rad2arcmin(r): return 10800.0*r/pi
def arcmin2rad(a): return a*pi/10800.0
def rad2arcsec(r): return 648000.0*r/pi
def arcsec2rad(a): return a*pi/648000.0
def radec2x(r,d):  return cos(d)*cos(r) # r,d in radians
def radec2y(r,d):  return cos(d)*sin(r) # r,d in radians
def radec2z(r,d):  return sin(d)        # r,d in radians
def z2dec(z):      return asin(z)     # result in radians
def xy2ra(x,y):
    "Convert x,y to ra in radians"
    r = atan2(y,x)
    r += 2*pi*(r>=0.00)
    return r

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
