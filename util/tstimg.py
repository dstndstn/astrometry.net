import pyfits
from numpy import *

if __name__ == '__main__':
	
	W,H = 10,10
	sigma = 1.
	X,Y = meshgrid(range(W), range(H))
	img = 50 + 200 * exp(-0.5 * ((X - W/2)**2 + (Y - H/2)**2)/(sigma**2))
	pyfits.writeto('tstimg.fits', img, clobber=True)
	

