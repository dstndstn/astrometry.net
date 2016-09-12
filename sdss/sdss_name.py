from starutil_numpy import ra2hms, dec2dms

# RA,Dec in degrees
def sdss_name(ra, dec):
    '''
    Returns the SDSS name, JHHMMSS.ss+DDMMSS.s

    >>> print sdss_name(15, 1)
    J010000.00+010000.0

    >>> print sdss_name(0, 0)
    J000000.00+000000.0

    >>> print sdss_name(0, 1./3600)
    J000000.00+000001.0

    >>> print sdss_name(0, 1./36000)
    J000000.00+000000.1

    # Truncation
    >>> print sdss_name(0, 1./36001)
    J000000.00+000000.0

    >>> print sdss_name(15/60., 1)
    J000100.00+010000.0

    >>> print sdss_name(15/3600., 1)
    J000001.00+010000.0

    >>> print sdss_name(15/360000., 1)
    J000000.01+010000.0

    >>> print sdss_name(15/360001., 1)
    J000000.00+010000.0

    >>> print sdss_name(375., -0.5)
    J010000.00-003000.0

    >>> print sdss_name(375., 0.5)
    J010000.00+003000.0

    >>> print sdss_name(360., 90.)
    J000000.00+900000.0
    '''
    (rh,rm,rs) = ra2hms(ra)
    (sgn,dd,dm,ds) = dec2dms(dec)
    # According to http://www.sdss.org/dr3/coverage/IAU.html
    # the coordinates are truncated, not rounded.
    rcs = int(rs * 100.)
    dds = int(ds * 10.)

    return 'J%02i%02i%02i.%02i%s%02i%02i%02i.%01i' % (
        rh, rm, rcs / 100, rcs % 100,
        '+' if sgn >= 0 else '-',
        dd, dm, dds / 10, dds % 10)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
