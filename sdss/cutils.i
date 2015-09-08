/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */
%module cutils

%{
#include <math.h>

static double deg2rad(double x) {
	return x * (M_PI / 180.0);
}
static double rad2deg(double x) {
	return x * (180.0 / M_PI);
}
 %}
%inline %{

#if CONFUSE_EMACS
 }
#endif

PyObject* munu_to_prime(double mu, double nu, double color) {
	return NULL;
	// return (xp,yp)
	/*
	def munu_to_prime(self, mu, nu, color=0):
		'''
		mu = a + b * rowm + c * colm
		nu = d + e * rowm + f * colm

		So

		[rowm; colm] = [b,c; e,f]^-1 * [mu-a; nu-d]

		[b,c; e,f]^1 = [B,C; E,F] in the code below, so

		[rowm; colm] = [B,C; E,F] * [mu-a; nu-d]

		'''
		a, b, c, d, e, f = self._get_abcdef()
		#print 'mu,nu', mu, nu, 'a,d', a,d
		determinant = b * f - c * e
		#print 'det', determinant
		B =  f / determinant
		C = -c / determinant
		E = -e / determinant
		F =  b / determinant
		#print 'B', B, 'mu-a', mu-a, 'C', C, 'nu-d', nu-d
		#print 'E', E, 'mu-a', mu-a, 'F', F, 'nu-d', nu-d
		mua = mu - a
		# in field 6955, g3, 809 we see a~413
		#if mua < -180.:
		#	mua += 360.
		mua = 360. * (mua < -180.)
		yprime = B * mua + C * (nu - d)
		xprime = E * mua + F * (nu - d)
		return xprime,yprime
	 */
}

PyObject* prime_to_pixel(double xp, double yp) {
	return NULL;
	/*
	def prime_to_pixel(self, xprime, yprime,  color=0):
		color0 = self._get_ricut()
		g0, g1, g2, g3 = self._get_drow()
		h0, h1, h2, h3 = self._get_dcol()
		px, py, qx, qy = self._get_cscc()

		# #$(%*&^(%$%*& bad documentation.
		(px,py) = (py,px)
		(qx,qy) = (qy,qx)

		qx = qx * np.ones_like(xprime)
		qy = qy * np.ones_like(yprime)
		#print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
		xprime -= np.where(color < color0, px * color, qx)
		yprime -= np.where(color < color0, py * color, qy)

		# Now invert:
		#   yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
		#   xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
		x = xprime - h0
		# dumb-ass Newton's method
		dx = 1.
		# FIXME -- should just update the ones that aren't zero
		# FIXME -- should put in some failsafe...
		while max(np.abs(np.atleast_1d(dx))) > 1e-10:
			xp    = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
			dxpdx = 1 +      h1     + h2 * 2*x +  h3 * 3*x**2
			dx = (xprime - xp) / dxpdx
			#print 'Max Newton dx', max(abs(dx))
			x += dx
		y = yprime - (g0 + g1 * x + g2 * x**2 + g3 * x**3)
		return (x, y)
	 */
}

/*
 Convention: ra,dec in degrees
 node,incl in radians
 resulting mu,nu in degrees.
 */
PyObject* radec_to_munu(double ra, double dec, double node, double incl) {
	ra  = deg2rad(ra);
	dec = deg2rad(dec);
	/*
	 double mu = node + atan2(sin(ra - node) * cos(dec) * cos(incl)
	 + sin(dec) * sin(incl),
	 cos(ra - node) * cos(dec));
	 double nu = asin(-sin(ra - node) * cos(dec) * sin(incl)
	 + sin(dec) * cos(incl));
	 */
	// This version is faster
	const double sinramnode = sin(ra - node);
	const double cosdec = cos(dec);
	const double sindec = sin(dec);
	const double sinincl = sin(incl);
	const double cosincl = cos(incl);
	double mu = node + atan2(sinramnode * cosdec * cosincl + sindec * sinincl,
							 cos(ra - node) * cosdec);
	double nu = asin(-sinramnode * cosdec * sinincl + sindec * cosincl);

	mu = rad2deg(mu);
	nu = rad2deg(nu);
	if (mu < 0)
		mu += 360.0;
	if (mu > 360)
		mu -= 360.0;
	return PyTuple_Pack(2, PyFloat_FromDouble(mu), PyFloat_FromDouble(nu));
}

static double getlistval(PyObject* cachedvals, int i) {
	PyObject* po = PyList_GET_ITEM(cachedvals, i);
	assert(PyFloat_Check(po));
	return PyFloat_AsDouble(po);
}

/*
 Convention: ra,dec in degrees
 node,incl in radians

 ASSUMES color < riCut
 ASSUMES color = 0
 */
PyObject* radec_to_pixel(double ra, double dec, PyObject* cachedvals) {
	//double node, double incl,
	//double a, double b, double c,
	//double d, double e, double f,
	//double px, double py) {

	assert(PyList_Check(cachedvals));
	assert(PyList_Size(cachedvals) == 25);

	double node = getlistval(cachedvals, 0);
	double incl = getlistval(cachedvals, 1);
	double a    = getlistval(cachedvals, 2);
	double d    = getlistval(cachedvals, 5);
	double B    = getlistval(cachedvals, 8);
	double C    = getlistval(cachedvals, 9);
	double E    = getlistval(cachedvals, 10);
	double F    = getlistval(cachedvals, 11);
	//double px   = getlistval(cachedvals, 12);
	//double py   = getlistval(cachedvals, 13);
	double g0   = getlistval(cachedvals, 16);
	double g1   = getlistval(cachedvals, 17);
	double g2   = getlistval(cachedvals, 18);
	double g3   = getlistval(cachedvals, 19);
	double h0   = getlistval(cachedvals, 20);
	double h1   = getlistval(cachedvals, 21);
	double h2   = getlistval(cachedvals, 22);
	double h3   = getlistval(cachedvals, 23);

	ra  = deg2rad(ra);
	dec = deg2rad(dec);

	// from previous  (radec_to_munu)
	const double sinramnode = sin(ra - node);
	const double cosdec = cos(dec);
	const double sindec = sin(dec);
	const double sinincl = sin(incl);
	const double cosincl = cos(incl);
	double mu = node + atan2(sinramnode * cosdec * cosincl + sindec * sinincl,
							 cos(ra - node) * cosdec);
	double nu = asin(-sinramnode * cosdec * sinincl + sindec * cosincl);

	mu = rad2deg(mu);
	nu = rad2deg(nu);
	if (mu > 360.0)
		mu -= 360.0;
	if (mu < 0)
		mu += 360.0;

	//printf("c: mu,nu %g,%g\n", mu, nu);

	// munu_to_pixel
	//   munu_to_prime,
	double mua = mu - a;
	if (mua < -180)
		mua += 360;
	double xp = E * mua + F * (nu - d);
	double yp = B * mua + C * (nu - d);

	//printf("c: xp,yp %g,%g\n", xp,yp);
	//   prime_to_pixel
	// color terms would go here...

	// Now invert:
	//   yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
	//   xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
	double x = xp - h0;
	double dx = 1.;
	double dxpdx;
	double xpi;
	do {
		//xp = x + h0 + h1 * x + h2 * x*x + h3 * x*x*x;
		xpi = x + h0 + x * (h1 + x * (h2 + x * h3));
		dxpdx = 1. + h1 + x * (h2 * 2. + h3 * 3. * x);
		dx = (xp - xpi) / dxpdx;
		x += dx;
	} while (dx > 1e-10);
	double y = yp - (g0 + x * (g1 + x * (g2 + x * g3)));
	return PyTuple_Pack(2, PyFloat_FromDouble(x), PyFloat_FromDouble(y));
}



%}

