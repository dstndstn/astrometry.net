from numpy import sin, atleast_1d, zeros, logical_and
from math import pi

def lanczos_filter(order, x):
	x = atleast_1d(x)
	nz = logical_and(x != 0., logical_and(x < order, x > -order))
	filt = zeros(len(x), float)
	#filt[nz] = order * sin(pi * x[nz]) * sin(pi * x[nz] / order) / ((pi * x[nz])**2)
	pinz = pi * x[nz]
	filt[nz] = order * sin(pinz) * sin(pinz / order) / (pinz**2)
	filt[x == 0] = 1.
	#filt[x >  order] = 0.
	#filt[x < -order] = 0.
	return filt

# Given a range of integer coordinates that you want to, eg, cut out
# of an image, [xlo, xhi], and bounds for the image [xmin, xmax],
# returns the range of coordinates that are in-bounds, and the
# corresponding region within the desired cutout.
def get_overlapping_region(xlo, xhi, xmin, xmax):
	if xlo > xmax or xhi < xmin or xlo > xhi or xmin > xmax:
		return ([], [])

	assert(xlo <= xhi)
	assert(xmin <= xmax)
	
	xloclamp = max(xlo, xmin)
	Xlo = xloclamp - xlo

	xhiclamp = min(xhi, xmax)
	Xhi = Xlo + (xhiclamp - xloclamp)

	#print 'xlo, xloclamp, xhiclamp, xhi', xlo, xloclamp, xhiclamp, xhi
	assert(xloclamp >= xlo)
	assert(xloclamp >= xmin)
	assert(xloclamp <= xmax)
	assert(xhiclamp <= xhi)
	assert(xhiclamp >= xmin)
	assert(xhiclamp <= xmax)
	#print 'Xlo, Xhi, (xmax-xmin)', Xlo, Xhi, xmax-xmin
	assert(Xlo >= 0)
	assert(Xhi >= 0)
	assert(Xlo <= (xhi-xlo))
	assert(Xhi <= (xhi-xlo))

	return (slice(xloclamp, xhiclamp+1), slice(Xlo, Xhi+1))
