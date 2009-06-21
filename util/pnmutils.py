from numpy import *

# Writes a numpy array as PGM to the given file handle.  The numpy
# array will be converted to integer values.  If maxval=255, 8-bit
# ints; if maxval>255, 16-bit ints.
def write_pgm(x, f, maxval=255):
	(h,w) = x.shape
	if maxval >= 65536:
		raise 'write_pgm: maxval must be < 65536'

	f.write('P5 %i %i %i\n' % (w, h, maxval))
	if maxval <= 255:
		f.write(getbuffer(x.astype(uint8)))
	else:
		f.write(getbuffer(x.astype(uint16)))
