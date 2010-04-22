import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('image', interpolation='nearest', cmap='gray', origin='lower')
rc('image', resample=False)
from pylab import *
import pyfits

I0 = pyfits.open('small.fits')[0].data
I1 = pyfits.open('unhp.fits')[0].data
clf()
imshow(I0, vmin=1000, vmax=1200)
colorbar()
title('Original image')
savefig('I0.png')
H = pyfits.open('hp.fits')[0].data
clf()
imshow(H, vmin=1000, vmax=1200)
colorbar()
title('Healpix image')
savefig('H.png')

clf()
imshow(I1, vmin=1000, vmax=1200)
colorbar()
title('Resampled image')
savefig('I1.png')
clf()
imshow(I1-I0, vmin=-10, vmax=10)
colorbar()
gray()
savefig('diff.png')

for i in [0, 99] + range(1, 99):
	clf()
	fn = 'step-%02i.fits' % i
	print 'Reading', fn
	In = pyfits.open(fn)[0].data
	imshow(In, vmin=1000, vmax=1200)
	colorbar()
	title('Resampled image, step %i' % i)
	savefig('Istep-%02i.png' % i)

	clf()
	imshow(In-I0, vmin=-10, vmax=10)
	colorbar()
	rms = sqrt(mean((In-I0).ravel()**2))
	med = median(abs((In-I0).ravel()))
	title('Resampled image error, step %i.  RMS=%.2f, Median=%.2f' % (i, rms, med))
	savefig('Ierrstep-%02i.png' % i)

