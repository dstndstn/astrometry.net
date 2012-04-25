%module cutils

%{
#include <math.h>

static double deg2rad(double x) {
	return x * (M_PI / 180.);
}
static double rad2deg(double x) {
	return x * (180. / M_PI);
}
 %}
%inline %{

#if CONFUSE_EMACS
 }
#endif

//void 
PyObject*
radec_to_munu(double ra, double dec, double node, double incl) {
	//, double* pmu, double* pnu) {
	ra = deg2rad(ra);
	dec = deg2rad(dec);
	/*
	 double mu = node + atan2(sin(ra - node) * cos(dec) * cos(incl) + sin(dec) * sin(incl),
	 cos(ra - node) * cos(dec));
	 double nu = asin(-sin(ra - node) * cos(dec) * sin(incl) + sin(dec) * cos(incl));
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
	//if (pmu) *pmu = mu;
	//if (pnu) *pnu = nu;
	//return Py_BuildValue("(dd)", mu, nu);
	return PyTuple_Pack(2, PyFloat_FromDouble(mu), PyFloat_FromDouble(nu));
}
%}

