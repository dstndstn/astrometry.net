import matplotlib
matplotlib.use('Agg')
import os.path
from pylab import *
from numpy import *
# test_tweak 2>tt.py
from tt import *

if __name__ == '__main__':
	#print 'me:', __file__
	#tt = os.path.join(os.path.dirname(__file__), 'test_tweak')
	

	clf()
	truedxy = xy_2 - origxy_2
	obsdxy  = noisyxy_2 - origxy_2
	xy = xy_2

	X1 = linspace(-100, 2100, 100)
	Y1 = gridy
	X1,Y1 = meshgrid(X1,Y1)
	X1 = X1.T
	Y1 = Y1.T
	X2 = gridx
	Y2 = linspace(-100, 2100, 100)
	X2,Y2 = meshgrid(X2,Y2)
	truesipx_x = zeros_like(X1)
	truesipy_x = zeros_like(X1)
	truesipx_y = zeros_like(Y2)
	truesipy_y = zeros_like(Y2)
	for xo,yo,c in truesip_a_2:
		truesipx_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
	for xo,yo,c in truesip_b_2:
		truesipy_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
	for xo,yo,c in truesip_a_2:
		truesipx_x += c * (X1 - x0)**xo * (Y1 - y0)**yo
	for xo,yo,c in truesip_b_2:
		truesipy_x += c * (X1 - x0)**xo * (Y1 - y0)**yo
	sipx_x = zeros_like(X1)
	sipy_x = zeros_like(X1)
	sipx_y = zeros_like(Y2)
	sipy_y = zeros_like(Y2)
	for xo,yo,c in sip_a_2:
		sipx_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
	for xo,yo,c in sip_b_2:
		sipy_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
	for xo,yo,c in sip_a_2:
		sipx_x += c * (X1 - x0)**xo * (Y1 - y0)**yo
	for xo,yo,c in sip_b_2:
		sipy_x += c * (X1 - x0)**xo * (Y1 - y0)**yo

	x = xy[:,0]
	y = xy[:,1]
	truedx = truedxy[:,0]
	truedy = truedxy[:,1]
	obsdx = obsdxy[:,0]
	obsdy = obsdxy[:,1]

	subplot(2,2,1)
	plot(x, truedx, 'bs', mec='b', mfc='None')
	plot(x, obsdx, 'r.')
	plot(X1, -truesipx_x, 'b-', alpha=0.2)
	plot(X1, -sipx_x, 'r-', alpha=0.2)
	xlabel('x')
	ylabel('dx')
	xlim(-100, 2100)

	subplot(2,2,2)
	plot(x, truedy, 'bs', mec='b', mfc='None')
	plot(x, obsdy, 'r.')
	plot(X1, -truesipy_x, 'b-', alpha=0.2)
	plot(X1, -sipy_x, 'r-', alpha=0.2)
	xlabel('x')
	ylabel('dy')
	xlim(-100, 2100)


	subplot(2,2,3)
	plot(y, truedx, 'bs', mec='b', mfc='None')
	plot(y, obsdx, 'r.')
	plot(Y2, -truesipx_y, 'b-', alpha=0.2)
	plot(Y2, -sipx_y, 'r-', alpha=0.2)
	xlabel('y')
	ylabel('dx')
	xlim(-100, 2100)

	subplot(2,2,4)
	plot(y, truedy, 'bs', mec='b', mfc='None')
	plot(y, obsdy, 'r.')
	plot(Y2, -truesipy_y, 'b-', alpha=0.2)
	plot(Y2, -sipy_y, 'r-', alpha=0.2)
	xlabel('y')
	ylabel('dy')
	xlim(-100, 2100)

	savefig('tt2.png')
	
