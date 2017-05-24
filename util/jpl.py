# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import re
import datetime
from astrometry.util.starutil_numpy import *
import numpy as np

## FIXME -- requires at least one digit before the decimal place.
floatre = r'[+-]?[\d]+(.[\d]*)?([eE][+-]?[\d]*)?'

def floatvar(x):
    return '(?P<%s>%s)' % (x, floatre)

# For parsing orbital elements type output.
elemrexstr = (r'^' + floatvar('jd')
              + '.*' + ' EC= ' + floatvar('e')
              + '.*' + ' IN= ' + floatvar('i')
              + '.*' + ' OM= ' + floatvar('Omega')
              + ' W = ' + floatvar('pomega')
              + '.*' + 'MA= ' + floatvar('M')
              + '.*' + 'A = ' + floatvar('a')
              )
elemrex = re.compile(elemrexstr, re.MULTILINE | re.DOTALL)

# For finding "System GM" -- only in orbital elements type
sysgmrexstr = '^System GM *: ' + floatvar('gm') + r' '#AU\^3/d\^2$'
sysgmrex = re.compile(sysgmrexstr)

# For parsing X,V type output
xvrexstr = ('^' + floatvar('jd') + r' = (?P<ad>A\.D\. .*?)'
            + r'^\s+' + floatvar('x0')
            + ' +' + floatvar('x1')
            + ' +' + floatvar('x2')
            + '.*?'
            + r'^\s+' + floatvar('v0')
            + ' +' + floatvar('v1')
            + ' +' + floatvar('v2')
            )
xvrex = re.compile(xvrexstr, re.MULTILINE | re.DOTALL)

# For parsing "observer" type output (RA,Dec)
# With QUANTITIES=1; angle format=DEG
radecrexstr = (
    '^ ' + '(?P<datetime>[\d]{4}-[\w]{3}-[\d]{2} [\d]{2}:[\d]{2}) .*?'
               + floatvar('ra') + ' *?' + floatvar('dec'))
radecrex = re.compile(radecrexstr, re.MULTILINE | re.DOTALL)

'''
For output like:

#Date       UT      R.A. (J2000) Decl.    Delta     r     El.    Ph.   m1     Sky Motion
#            h m s                                                            "/min    P.A.
# geocentric
2007 04 01 000000   23.1983  -02.463     2.953   2.070   22.9  10.8  17.1    1.45    057.9
'''
#radecrexstr2 = (
#    '^' + '(?P<datetime>[\d]{4} [\d]{2} [\d]{2} [\d]{2}:[\d]{2}) .*?'
#               + floatvar('ra') + ' *?' + floatvar('dec'))
#radecrex = re.compile(radecrexstr, re.MULTILINE | re.DOTALL)



# Returns a list of lists of elements, plus a list of the JDs.
#   ([jd1, jd2, ...], [   [a1, e1, i1, Omega1, pomega1, M1, GM1], ... ])
# Where  i, Omega, pomega, M   are in radians
def parse_orbital_elements(s, needSystemGM=True):
    if needSystemGM:
        m = sysgmrex.search(s)
        if not m:
            print('Did not find "System GM" entry')
            return None
        gm = float(m.group('gm'))
    else:
        gm = 1.
    allE = []
    alljd = []
    for m in elemrex.finditer(s):
        d = m.groupdict()
        E = [np.deg2rad(x) if rad else x
             for (x,rad) in zip([float(d[x]) for x in
                                 ['a',  'e',   'i',  'Omega', 'pomega','M' ]],
                                [False, False, True, True,    True,    True])]
        E.append(gm)
        allE.append(E)
        alljd.append(float(d['jd']))
    return alljd,allE

# Returns (x, v, jd), each as numpy arrays.
#     x in AU
#     v in AU/day
def parse_phase_space(s):
    all_x = []
    all_v = []
    all_jd = []
    for m in xvrex.finditer(s):
        d = m.groupdict()
        x = np.array([float(d[k]) for k in ['x0','x1','x2']])
        v = np.array([float(d[k]) for k in ['v0','v1','v2']])
        all_x.append(x)
        all_v.append(v)
        all_jd.append(float(d['jd']))
    return (np.array(all_x), np.array(all_v), np.array(all_jd))

# Returns (ra,dec,jd), each as numpy arrays.
#   RA,Dec in J2000 deg
def parse_radec(s):
    all_ra = []
    all_dec = []
    all_jd = []
    for m in radecrex.finditer(s):
        d = m.groupdict()
        all_ra.append(float(d['ra']))
        all_dec.append(float(d['dec']))
        t = datetime.datetime.strptime(d['datetime'], '%Y-%b-%d %H:%M')
        # 2000-Jan-01 12:00
        all_jd.append(datetojd(t))
    return (np.array(all_ra), np.array(all_dec), np.array(all_jd))
