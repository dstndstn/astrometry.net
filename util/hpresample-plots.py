import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('image', interpolation='nearest', cmap='gray', origin='lower')
rc('image', resample=False)
from pylab import *
import pyfits
import sys

dosteps = 'steps' in sys.argv[1:]

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

bins = linspace(900, 1100, 101)

clf()
subplot(2,1,1)
hist(I0.ravel(), bins=bins)
I0r = I0.ravel()
i = logical_and(I0r >= 900, I0r <= 1100)
mn = mean(I0r[i])
st = std(I0r[i])
plot(bins, len(I0r[i]) / (sqrt(2.*pi)*st) * (bins[1]-bins[0]) * exp(-(bins - mn)**2 / (2.*st**2)), 'r-')
title('Original image: N(%.1f, %.1f)' % (mn,st))

subplot(2,1,2)
hist(I1.ravel(), bins=bins)
I1r = I1.ravel()
i = logical_and(I1r >= 900, I1r <= 1100)
mn = mean(I1r[i])
st = std(I1r[i])
plot(bins, len(I1r[i]) / (sqrt(2.*pi)*st) * (bins[1]-bins[0]) * exp(-(bins - mn)**2 / (2.*st**2)), 'r-')
title('Resampled image: N(%.1f, %.1f)' % (mn,st))
subplots_adjust(hspace=0.25)
savefig('hists.png')

clf()
subplot(111)
title('Resampled image: pixel error distribution')
hist((I1-I0).ravel(), bins=linspace(-25, 25, 51))
xlabel('Resampled pixel - Original pixel')
savefig('histdiffs.png')

clf()
#s0 = I0.copy()
#s0.sort()
#s1 = I1.copy()
#s1.sort()
#plot(s0, s1, 'r.', alpha=0.05)
plot(I0.ravel(), I1.ravel(), 'r.', alpha=0.05)
xlabel('Original pixel value')
ylabel('Resampled pixel value')
axis([900,1100,900,1100])
savefig('sorted.png')

sys.exit(0)

ii = [0, 99] + range(1, 99)
RMS = zeros_like(ii).astype(float)
MED = zeros_like(RMS)

for i in ii:
	fn = 'step-%02i.fits' % i
	print 'Reading', fn
	In = pyfits.open(fn)[0].data

	D = (In - I0)[50:-50,50:-50]
	#rms = sqrt(mean((In-I0).ravel()**2))
	#med = median(abs((In-I0).ravel()))
	rms = sqrt(mean(D.ravel()**2))
	med = median(abs(D.ravel()))
	RMS[i] = rms
	MED[i] = med

	if dosteps:
		clf()
		imshow(In, vmin=1000, vmax=1200)
		colorbar()
		title('Resampled image, step %i' % i)
		savefig('Istep-%02i.png' % i)

		clf()
		imshow(In-I0, vmin=-10, vmax=10)
		colorbar()
		title('Resampled image error, step %i.  Central RMS=%.2f, Median=%.2f' % (i, rms, med))
		savefig('Ierrstep-%02i.png' % i)

clf()
plot(RMS, 'r.-')
xlabel('Iteration #')
ylabel('RMS error of central region')
title('Inverse-resampling: RMS error')
ylim(ymin=0)
savefig('rms.png')

clf()
plot(MED, 'r.-')
xlabel('Iteration #')
ylabel('Median error of central region')
title('Inverse-resampling: median error')
ylim(ymin=0)
savefig('median.png')

clf()
p1 = plot(MED, 'r.-')
p2 = plot(RMS, 'b.-')
xlabel('Iteration #')
ylabel('Error of central region')
legend((p1,p2), ('Median','RMS'))
title('Inverse-resampling error')
ylim(ymin=0)
savefig('errors.png')



clf()
semilogy(RMS, 'r.-')
xlabel('Iteration #')
ylabel('RMS error of central region')
title('Inverse-resampling: RMS error')
savefig('logrms.png')

clf()
semilogy(MED, 'r.-')
xlabel('Iteration #')
ylabel('Median error of central region')
title('Inverse-resampling: median error')
savefig('logmedian.png')

clf()
p1 = semilogy(MED, 'r.-')
p2 = semilogy(RMS, 'b.-')
xlabel('Iteration #')
ylabel('Error of central region')
legend((p1,p2), ('Median','RMS'))
title('Inverse-resampling error')
savefig('logerrors.png')
