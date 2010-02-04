from numpy import *
import datetime

# scalars (racenter, deccenter) in deg
# scalar radius in deg
# arrays (ra,dec) in deg
# returns array of booleans
def points_within_radius(racenter, deccenter, radius, ra, dec):
	return radecdotproducts(racenter, deccenter, ra, dec) >= cos(deg2rad(radius))

def points_within_radius_range(racenter, deccenter, radiuslo, radiushi, ra, dec):
	d = radecdotproducts(racenter, deccenter, ra, dec)
	return (d <= cos(deg2rad(radiuslo))) * (d >= cos(deg2rad(radiushi)))

# scalars (racenter, deccenter) in deg
# arrays (ra,dec) in deg
# returns array of cosines
def radecdotproducts(racenter, deccenter, ra, dec):
	xyzc = radectoxyz(racenter, deccenter).T
	xyz = radectoxyz(ra, dec)
	return dot(xyz, xyzc)[:,0]

# RA, Dec in degrees: scalars or 1-d arrays.
# returns xyz of shape (N,3)
def radectoxyz(ra_deg, dec_deg):
    ra  = deg2rad(ra_deg)
    dec = deg2rad(dec_deg)
    cosd = cos(dec)
    xyz = vstack((cosd * cos(ra),
				  cosd * sin(ra),
				  sin(dec))).T
    assert(xyz.shape[1] == 3)
    return xyz

# RA,Dec in degrees
# returns (dxyz_dra, dxyz_ddec)
def derivatives_at_radec(ra_deg, dec_deg):
    ra  = deg2rad(ra_deg)
    dec = deg2rad(dec_deg)
    cosd = cos(dec)
    nsd = -sin(dec)
    return (180./pi * vstack((cosd * -sin(ra),
							  cosd * cos(ra),
							  sin(dec))).T,
			180./pi * vstack((nsd * cos(ra),
							  nsd * sin(ra),
							  cosd)).T)

def xyztoradec(xyz):
	'''
	Converts positions on the unit sphere to RA,Dec in degrees.

	'xyz' must be a numpy array, either of shape (3,) or (N,3)

	Returns a tuple (RA,Dec).

	If 'xyz' is a scalar, RA,Dec are scalars.

	If 'xyz' is shape (N,3), RA,Dec are shape (N,).

	>>> xyztoradec(array([1,0,0]))
	(0.0, 0.0)

	>>> xyztoradec(array([ [1,0,0], [0,1,0], [0,0,1]]))
	(array([  0.,  90.,   0.]), array([  0.,   0.,  90.]))

	>>> xyztoradec(array([0,1,0]))
	(90.0, 0.0)

	>>> xyztoradec(array([0,0,1]))
	(0.0, 90.0)

	'''
	if len(xyz.shape) == 1:
		# HACK!
		rs,ds = xyztoradec(xyz[newaxis,:])
		return (rs[0], ds[0])
	(nil,three) = xyz.shape
	assert(three == 3)
	ra = arctan2(xyz[:,1], xyz[:,0])
	ra += 2*pi * (ra < 0)
	dec = arcsin(xyz[:,2] / norm(xyz)[:,0])
	return (rad2deg(ra), rad2deg(dec))


#####################

# RA,Decs in degrees.
def arcsec_between(ra1, dec1, ra2, dec2):
	'''
	Computes the angle between two (arrays of) RA,Decs.

	>>> from numpy import round
	>>> print round(arcsec_between(0, 0, 1, 0), 6)
	3600.0

	>>> print round(arcsec_between(array([0, 1]), array([0, 0]), 1, 0), 6)
	[ 3600.     0.]

	>>> print round(arcsec_between(1, 0, array([0, 1]), array([0, 0])), 6)
	[ 3600.     0.]

	>>> print round(arcsec_between(array([0, 1]), array([0, 0]), array([0, 1]), array([0, 0])), 6)
	[[    0.  3600.]
	 [ 3600.     0.]]

	'''
	xyz1 = radectoxyz(ra1, dec1)
	xyz2 = radectoxyz(ra2, dec2)
	# (n,3) (m,3)

	s0 = xyz1.shape[0]
	s1 = xyz2.shape[0]
	d2 = zeros((s0,s1))
	for s in range(s0):
		d2[s,:] = sum((xyz1[s,:] - xyz2)**2, axis=1)
	if s0 == 1 and s1 == 1:
		d2 = d2[0,0]
	elif s0 == 1:
		d2 = d2[0,:]
	elif s1 == 1:
		d2 = d2[:,0]
	return distsq2arcsec(d2)

def degrees_between(ra1, dec1, ra2, dec2):
	return arcsec2deg(arcsec_between(ra1, dec1, ra2, dec2))


def distsq2rad(dist2):
    return arccos(1. - dist2 / 2.)
def distsq2arcsec(dist2):
    return rad2arcsec(distsq2rad(dist2))

def rad2deg(r):
    return 180.0*r/pi
def rad2arcsec(r):
    return 648000.0*r/pi

def arcsec2rad(a):
    return a*pi/648000.0
def arcsec2deg(a):
    return rad2deg(arcsec2rad(a))



# x can be an array of shape (N,D)
# returns an array of shape (N,1)
def norm(x):
	if len(x.shape) == 2:
		return sqrt(sum(x**2, axis=1))[:,newaxis]
	else:
		return sqrt(sum(x**2))
		

# proper motion (dl, db, dra, or ddec) in mas/yr
# dist in kpc
# returns velocity in km/s
def pmdisttovelocity(pm, dist):
	# (pm in deg/yr) * (dist in kpc) to (velocity in km/s)
	pmfactor = 1/3.6e6 * pi/180. * 0.977813952e9
	return pm * dist * pmfactor


# ra, dec in degrees
# pmra = d(RA*cos(Dec))/dt, pmdec = dDec/dt, in deg/yr or mas/yr
# returns (l,b, pml,pmb) in degrees and [the same units as pmra,pmdec]
#    pml is d(l*cos(b))/dt
def pm_radectolb(ra, dec, pmra, pmdec):
	(l1, b1) = radectolb(ra, dec)
	# the Jo Bovy method:
	(a,d) = galactic_pole
	alphangp = deg2rad(a)
	deltangp = deg2rad(d)
	delta = deg2rad(dec)
	alpha = deg2rad(ra)
	b = deg2rad(b1)

	cosphi = ((sin(deltangp) - sin(delta)*sin(b)) /
			  (cos(delta)*cos(b)))
	sinphi = ((sin(alpha - alphangp) * cos(deltangp)) /
			  cos(b))

	dlcosb =  cosphi * pmra + sinphi * pmdec
	db     = -sinphi * pmra + cosphi * pmdec
	return (l1, b1, dlcosb, db)

# ra, dec in degrees
# returns (l,b) in degrees
def radectolb(ra, dec):
	(xhat, yhat, zhat) = galactic_unit_vectors()
	xyz = radectoxyz(ra, dec)
	xg = dot(xyz, xhat)
	yg = dot(xyz, yhat)
	zg = dot(xyz, zhat)
	# danger, will robinson, danger!
	# abuse the xyztoradec routine to convert xyz in the galactic
	# unit sphere to (l,b) in galactic coords.
	(l,b) = xyztoradec(hstack((xg, yg, zg)))
	# galactic system is left-handed so "l" comes out backward.
	l = 360. - l
	return (l,b)

# ra,dec in degrees
# dist in kpc
# pmra is d(ra * cos(dec))/dt  in mas/yr
# pmdec is in mas/yr
# returns (pmra, pmdec) in the same units
def remove_solar_motion(ra, dec, dist, pmra, pmdec):
	(xhat, yhat, zhat) = galactic_unit_vectors()
	# (we only need yhat)
	# V_sun in kpc / yr
	vsun = 240. * 1.02268944e-9 * yhat.T
	# unit vectors on celestial sphere
	unitxyz = radectoxyz(ra, dec)
	# heliocentric positions in kpc
	xyz = dist[:,newaxis] * unitxyz
	# numerical difference time span in yr
	dyr = 1.
	# transverse displacements on celestial unit sphere
	unitxyz2 = radectoxyz(ra  + pmra/cos(deg2rad(dec)) /3.6e6 * dyr,
						  dec + pmdec/3.6e6 * dyr)
	# heliocentric transverse displacement of the observed star in kpc
	dxyz = (unitxyz2 - unitxyz) * dist[:,newaxis]
	# galactocentric displacement in kpc
	dxyz -= vsun * dyr
	# new 3-space position in kpc
	xyz3 = xyz + dxyz
	# back to the lab, deg
	(ra3,dec3) = xyztoradec(xyz3)
	# adjusted angular displacement, deg
	dra = ra3 - ra
	# tedious RA wrapping
	dra += 360. * (dra < -180)
	dra -= 360. * (dra >  180)
	# convert back to proper motions
	return ((dra * cos(deg2rad(dec3)) / dyr) * 3.6e6,
			((dec3 - dec) / dyr) * 3.6e6)

# the north galactic pole, (RA,Dec), in degrees, from Bovy.
galactic_pole = (192.85948, 27.12825)
# vs Wikipedia's (192.859508, 27.128336)

# returns (xhat, yhat, zhat), unit vectors in the RA,Dec unit sphere
# of the galactic coordinates.
def galactic_unit_vectors():
	# direction to GC 
	xhat = radectoxyz(266.405100, -28.936175).T
	# direction to Galactic Pole
	zhat = radectoxyz(*galactic_pole).T
	# coordinate system is left-handed
	yhat = cross(xhat.T, zhat.T).T
	#print 'yhat', yhat
	#print 'norms:', norm(xhat.T), norm(yhat.T), norm(zhat.T)
	# recompute xhat to ensure it is normal: the directions here
	# are inconsistent at the 1e-6 level.
	xhat = cross(zhat.T, yhat.T).T
	#print 'xhat', xhat
	#print 'xhat2', xhat2
	print 'dot products:',sum(xhat*yhat),sum(xhat*zhat),sum(yhat*zhat)
	return (xhat, yhat, zhat)


def mjdtodate(mjd):
	jd = mjdtojd(mjd)
	return jdtodate(jd)

def jdtodate(jd):
	unixtime = (jd - 2440587.5) * 86400. # in seconds
	return datetime.datetime.utcfromtimestamp(unixtime)

def mjdtojd(mjd):
	return mjd + 2400000.5

def timedeltatodays(dt):
	return dt.days + (dt.seconds + dt.microseconds/1e6)/86400.

def datetomjd(d):
	d0 = datetime.datetime(1858, 11, 17, 0, 0, 0)
	dt = d - d0
	# dt is a timedelta object.
	return timedeltatodays(dt)

# UTC for 2000 January 1.5
J2000 = datetime.datetime(2000,1,1,12,0,0,0,tzinfo=None)

def ecliptic_basis(eclipticangle = 23.43928):
	Equinox= array([1,0,0])
	CelestialPole = array([0,0,1])
	YPole = cross(CelestialPole, Equinox)
	EclipticAngle= deg2rad(eclipticangle)
	EclipticPole= (CelestialPole * cos(EclipticAngle) - YPole * sin(EclipticAngle))
	Ydir = cross(EclipticPole, Equinox)
	return (Equinox, Ydir, EclipticPole)










def hms2ra(h, m, s):
	return 15. * (h + (m + s/60.)/60.)

def tokenize_hms(s):
	s = s.strip()
	tokens = s.split()
	tokens = reduce(list.__add__, [t.split(':') for t in tokens])
	h = len(tokens) >= 1 and float(tokens[0]) or 0
	m = len(tokens) >= 2 and float(tokens[1]) or 0
	s = len(tokens) >= 3 and float(tokens[2]) or 0
	return (h,m,s)

def hmsstring2ra(st):
	(h,m,s) = tokenize_hms(st)
	return hms2ra(h, m, s)

def dms2dec(sign, d, m, s):
	return sign * (d + (m + s/60.)/60.)

def dmsstring2dec(s):
	sign = (s[0] == '-') and -1.0 or 1.0
	if s[0] == '-' or s[0] == '+':
		s = s[1:]
	(d,m,s) = tokenize_hms(s)
	return dms2dec(sign, d, m, s)

# RA in degrees
def ra2hms(ra):
	h = ra * 24. / 360.
	hh = int(floor(h))
	m = (h - hh) * 60.
	mm = int(floor(m))
	s = (m - mm) * 60.
	return (hh, mm, s)

# Dec in degrees
def dec2dms(dec):
	sgn = (dec > 0) and 1. or -1.
	d = dec * sgn
	dd = int(floor(d))
	m = (d - dd) * 60.
	mm = int(floor(m))
	s = (m - mm) * 60.
	if s >= 60.:
		m += 1.
		s -= 60.
	return (sgn*d, m, s)

# RA in degrees
def ra2hmsstring(ra, separator=' '):
	(h,m,s) = ra2hms(ra)
	ss = int(floor(s))
	ds = int(round((s - ss) * 1000.0))
	return separator.join(['%0.2i' % h, '%0.2i' % m, '%0.2i.%0.3i' % (ss,ds)])

# Dec in degrees
def dec2dmsstring(dec, separator=' '):
	(d,m,s) = dec2dms(dec)
	ss = int(floor(s))
	ds = int(round((s - ss) * 1000.0))
	return ' '.join(['%+0.2i' % d, '%0.2i' % m, '%0.2i.%0.3i' % (ss,ds)])

def xyzarrtoradec(xyz):
	return (degrees(xy2ra(xyz[0], xyz[1])), degrees(z2dec(xyz[2])))

def deg2rad(d):    return d*pi/180.0
def deg2arcmin(d): return d * 60.
def deg2arcsec(d): return d * 3600.
def rad2arcmin(r): return 10800.0*r/pi
def arcmin2rad(a): return a*pi/10800.0
def arcmin2deg(a): return a/60.
def arcmin2rad(a): return deg2rad(arcmin2deg(a))
def radec2x(r,d):  return cos(d)*cos(r) # r,d in radians
def radec2y(r,d):  return cos(d)*sin(r) # r,d in radians
def radec2z(r,d):  return sin(d)        # r,d in radians
def z2dec(z):      return asin(z)     # result in radians
def xy2ra(x,y):
    "Convert x,y to ra in radians"
    r = atan2(y,x)
    r += 2*pi*(r<0.)
    return r


def rad2distsq(rad):
    return 2. * (1. - cos(rad))

def arcsec2distsq(arcsec):
    return rad2distsq(arcsec2rad(arcsec))

def arcsec2dist(arcsec):
    return sqrt(arcsec2distsq(arcsec))

def arcmin2distsq(arcmin):
    return rad2distsq(arcmin2rad(arcmin))

def arcmin2dist(arcmin):
    return sqrt(arcmin2distsq(arcmin))

def dist2arcsec(dist):
    return distsq2arcsec(dist**2)





if __name__ == '__main__':
	import doctest
	doctest.testmod()
