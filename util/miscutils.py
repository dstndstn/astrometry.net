# Given a range of integer coordinates that you want to, eg, cut out
# of an image, [xlo, xhi], and bounds for the image [xmin, xmax],
# returns the range of coordinates that are in-bounds, and the
# corresponding region within the desired cutout.
def get_overlapping_region(xlo, xhi, xmin, xmax):
	if xlo > xmax or xhi < xmin:
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
