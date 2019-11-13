#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import

from astrometry.util.fits import *
from astrometry.util.starutil_numpy import *
from astrometry.util.find_data_file import *
from os.path import basename,dirname
import numpy as np

def get_photoobj_filename(photoobjdir, rr, run, camcol, field):
    fn = os.path.join(photoobjdir, rr, '%i'%run, '%i'%camcol,
                      'photoObj-%06i-%i-%04i.fits' % (run, camcol, field))
    return fn


def read_photoobjs_in_wcs(wcs, margin, cols=None,
                          cutToPrimary=True,
                          wfn=None,
                          sdss=None):
    '''
    Read photoObjs that are inside the given 'wcs', plus 'margin' in degrees.
    '''
    import logging
    log = logging.getLogger('read_photoobjs_in_wcs')

    ra,dec = wcs.radec_center()
    rad = wcs.radius()
    rad += np.hypot(14., 10.) / 2 / 60.
    # a little extra margin
    rad += margin

    if sdss is None:
        from astrometry.sdss import DR9
        sdss = DR9()
    if wfn is None:
        wfn = sdss.filenames.get('window_flist', None)
        if wfn is None:
            wfn = 'window_flist.fits'
    if not os.path.exists(wfn):
        print('File does not exist:', wfn, '; downloading...')
        wfn = sdss.retrieve('window_flist', None, None, None, rerun='xxx')
        print('Retrieved', wfn)
    else:
        print('Using', wfn)

    print('Searching for run,camcol,fields with radius', rad, 'deg')
    RCF = radec_to_sdss_rcf(ra, dec, radius=rad*60., tablefn=wfn)
    log.debug('Found %i fields possibly in range' % len(RCF))

    pixmargin = margin * 3600. / wcs.pixel_scale()
    W,H = wcs.get_width(), wcs.get_height()
    
    TT = []
    for run,camcol,field,r,d in RCF:
        log.debug('RCF %i/%i/%i' % (run, camcol, field))
        rr = sdss.get_rerun(run, field=field)
        if rr in [None, '157']:
            log.debug('Rerun 157')
            continue

        fn = sdss.retrieve('photoObj', run, camcol, field, rerun=rr)
        #fn = get_photoobj_filename(rr, run, camcol, field)

        T = fits_table(fn, columns=cols)
        if T is None:
            log.debug('read 0 from %s' % fn)
            continue
        log.debug('read %i from %s' % (len(T), fn))

        # while we're reading it, record its length for later...
        #get_photoobj_length(rr, run, camcol, field)

        ok,x,y = wcs.radec2pixelxy(T.ra, T.dec)
        x -= 1
        y -= 1
        T.cut((x > -pixmargin) * (x < (W + pixmargin)) *
              (y > -pixmargin) * (y < (H + pixmargin)))
        if cutToPrimary:
            T.cut((T.resolve_status & 256) > 0)
            log.debug('cut to %i within target area and PRIMARY.' % len(T))
        else:
            log.debug('cut to %i within target area.' % len(T))
        if len(T) == 0:
            continue
        TT.append(T)
    if not len(TT):
        return None
    T = merge_tables(TT)
    return T


class RaDecToRcf(object):
    def __init__(self, tablefn=None):
        self.kd = None
        self.sdssxyz = None
        if tablefn is None:
            tablefn = find_data_file('dr7fields.fits')
        self.tab = fits_table(tablefn)
        if self.tab is None:
            raise Exception('Failed to read table of SDSS fields from file: "' + str(tablefn) + '"')

    def __call__(self, ra, dec, spherematch=True, radius=0, contains=False):
        T = self.tab
        # HACK - magic 13x9 +1 arcmin.
        if radius == 0:
            radius = sqrt(14.**2 + 10.**2)/2.
        d2 = arcmin2distsq(radius)
        if self.sdssxyz is None:
            self.sdssxyz = radectoxyz(T.ra, T.dec)
        if not spherematch:
            rcfs = []
            for r,d in broadcast(ra,dec):
                xyz = radectoxyz(r,d)
                dist2s = sum((xyz - self.sdssxyz)**2, axis=1)
                I = flatnonzero(dist2s < d2)
                rcfs.append(zip(T[I].run, T[I].camcol,
                                T[I].field, T[I].ra, T[I].dec))
        else:
            from astrometry.libkd import spherematch
            if self.kd is None:
                self.kd = spherematch.tree_build(self.sdssxyz)
            rds = array([x for x in broadcast(ra,dec)])
            xyz = radectoxyz(rds[:,0], rds[:,1]).astype(double)
            kd2 = spherematch.tree_build(xyz)
            notself = False
            inds,D = spherematch.trees_match(self.kd, kd2, np.sqrt(d2),
                                             notself=notself, permuted=True)
            if len(inds) == 0:
                return []
            I = np.argsort(D[:,0])
            inds = inds[I]
            rcfs = [[] for i in range(len(rds))]
            cols = T.columns()
            gotem = False
            if contains:
                if ('ramin' in cols and 'ramax' in cols and
                    'decmin' in cols and 'decmax' in cols):
                    gotem = True
                    for j,i in inds:
                        (r,d) = rds[i]
                        if (r >= T.ramin[j] and r <= T.ramax[j]
                            and d >= T.decmin[j] and d <= T.decmax[j]):
                            rcfs[i].append((T.run[j], T.camcol[j],
                                            T.field[j], T.ra[j], T.dec[j]))
                        #print '%i fields contain the first query RA,Dec' % len(rcfs[0])
                else:
                    print('you requested fields *containing* the query RA,Dec,')
                    print('but the fields list file \"%s\" doesn\'t contain RAMIN,RAMAX,DECMIN, and DECMAX columns' % tablefn)
            if not gotem:
                for j,i in inds:
                    rcfs[i].append((T.run[j], T.camcol[j], T.field[j],
                                    T.ra[j], T.dec[j]))

        if isscalar(ra) and isscalar(dec):
            return rcfs[0]
        return rcfs
        


# RA,Dec are either scalars or iterables.
# Radius is in *arcmin*.  sigh.
# If scalars, returns a list of (run, camcol, field, ra, dec) tuples, one for each matching field.
# If iterable, returns a list containing one list per query (ra,dec) of the same tuple.
def radec_to_sdss_rcf(ra, dec, spherematch=True, radius=0, tablefn=None, contains=False):
    RD = RaDecToRcf(tablefn=tablefn)
    return RD(ra, dec, spherematch=spherematch, radius=radius, contains=contains)



def OLD_radec_to_sdss_rcf(ra, dec, spherematch=True, radius=0, tablefn=None, contains=False):
    # This file is generated by merging the files "dr7_e.fits", "dr7_g.fits", and "dr7_a.fits",
    # whose construction is described in http://trac.astrometry.net/browser/trunk/projects/sdss-tests/README
    # (and in comments below that I didn't notice before writing this)
    if tablefn is None:
        tablefn = find_data_file('dr7fields.fits')
    sdss = table_fields(tablefn)
    if sdss is None:
        print('Failed to read table of SDSS fields from file', tablefn)
        raise Exception('Failed to read table of SDSS fields from file: "' + str(tablefn) + '"')
    sdssxyz = radectoxyz(sdss.ra, sdss.dec)
    ## HACK - magic 13x9 arcmin.
    if radius == 0:
        radius = sqrt(13.**2 + 9.**2)/2.
    radius2 = arcmin2distsq(radius)
    if not spherematch:
        rcfs = []
        for r,d in broadcast(ra,dec):
            xyz = radectoxyz(r,d)
            dist2s = sum((xyz - sdssxyz)**2, axis=1)
            I = flatnonzero(dist2s < radius2)
            if False:
                print('I:', I)
                print('fields:', sdss[I].run, sdss[I].field, sdss[I].camcol)
                print('RA', sdss[I].ra)
                print('Dec', sdss[I].dec)
            rcfs.append(zip(sdss[I].run, sdss[I].camcol, sdss[I].field, sdss[I].ra, sdss[I].dec))
    else:
        from astrometry.libkd import spherematch
        rds = array([x for x in broadcast(ra,dec)])
        xyz = radectoxyz(rds[:,0], rds[:,1]).astype(double)
        (inds,dists) = spherematch.match(xyz, sdssxyz, sqrt(radius2))
        #print 'found %i matches' % len(inds)
        if len(inds) == 0:
            return []
        #print 'inds:', inds.shape
        I = np.argsort(dists[:,0])
        #print 'dists:', dists.shape
        inds = inds[I,:]
        rcfs = [[] for i in range(len(rds))]
        cols = sdss.columns()
        gotem = False
        if contains:
            if 'ramin' in cols and 'ramax' in cols and 'decmin' in cols and 'decmax' in cols:
                gotem = True
                for i,j in inds:
                    (r,d) = rds[i]
                    if r >= sdss.ramin[j] and r <= sdss.ramax[j] and d >= sdss.decmin[j] and d <= sdss.decmax[j]:
                        rcfs[i].append((sdss.run[j], sdss.camcol[j], sdss.field[j], sdss.ra[j], sdss.dec[j]))
                print('%i fields contain the first query RA,Dec' % len(rcfs[0]))
            else:
                print('you requested fields *containing* the query RA,Dec,')
                print('but the fields list file \"%s\" doesn\'t contain RAMIN,RAMAX,DECMIN, and DECMAX columns' % tablefn)
        if not gotem:
            for i,j in inds:
                rcfs[i].append((sdss.run[j], sdss.camcol[j], sdss.field[j], sdss.ra[j], sdss.dec[j]))


    if isscalar(ra) and isscalar(dec):
        return rcfs[0]
    return rcfs

# The field list was created starting with dstn's list of fields in DR7:
#  fitscopy dr7_e.fits"[col RUN;FIELD;CAMCOL;RA=(RAMIN+RAMAX)/2;DEC=(DECMIN+DECMAX)/2]" e.fits
#  fitscopy dr7_g.fits"[col RUN;FIELD;CAMCOL;RA=(RAMIN+RAMAX)/2;DEC=(DECMIN+DECMAX)/2]" g.fits
#  fitscopy dr7_a.fits"[col RUN;FIELD;CAMCOL;RA=(RAMIN+RAMAX)/2;DEC=(DECMIN+DECMAX)/2]" a.fits
#  tabmerge g.fits e.fits
#  tabmerge g.fits+1 e.fits+1
#  tabmerge a.fits+1 e.fits+1
#  mv e.fits dr7fields.fits
#  rm g.fits a.fits

'''
cd ~/sdss-tests
casjobs.py $SDSS_CAS_USER $SDSS_CAS_PASS querywait @dr7_ngood.sql
casjobs.py $SDSS_CAS_USER $SDSS_CAS_PASS querywait @dr7_ngood2.sql
casjobs.py $SDSS_CAS_USER $SDSS_CAS_PASS outputdownloaddelete mydb.goodfields2 /tmp/dr7.fits
fitscopy /tmp/dr7.fits"[col RA=(ramin+ramax)/2;DEC=(decmin+decmax)/2;run;field;camcol;ngood;ramin;ramax;decmin;decmax]" dr7fields.fits

casjobs.py $SDSS_CAS_USER $SDSS_CAS_PASS querywait @s82_ngood.sql
# Stripe82 has no RunQA table.
casjobs.py $SDSS_CAS_USER $SDSS_CAS_PASS querywait @s82_ngood2.sql
casjobs.py $SDSS_CAS_USER $SDSS_CAS_PASS outputdownloaddelete mydb.s82goodfields2 s82.fits
fitscopy s82.fits"[col RA=(ramin+ramax)/2;DEC=(decmin+decmax)/2;run;field;camcol;ngood;ramin;ramax;decmin;decmax]" s82fields.fits

'''

def main():
    import sys
    from optparse import OptionParser
    
    parser = OptionParser(usage='%prog [options] <ra> <dec>')
    parser.add_option('-f', dest='fields', help='FITS table of fields to use; default is astrometry/data/dr7fields.fits')
    parser.add_option('-c', dest='contains', action='store_true', help='Print only fields that *contain* the given point; requires RAMIN,RAMAX,DECMIN,DECMAX fields.')
    parser.add_option('-b', '--bands', dest='bands', help='Retrieve fpCs of the given bands; default "ugriz"')
    parser.add_option('-t', dest='filetypes', help='Retrieve this file type (fpC, fpM, psField, tsField, tsObj, etc)', action='append', default=['fpC'])
    parser.add_option('-r', dest='radius', type=float, default=15., help='Search radius (arcmin)')
    parser.set_defaults(fields=None, contains=False, bands='ugriz')

    (opt, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        print()
        print('Got extra arguments:', args)
        sys.exit(-1)

    # parse RA,Dec.
    try:
        ra = float(args[0])
    except ValueError:
        ra = hmsstring2ra(args[0])
    try:
        dec = float(args[1])
    except ValueError:
        dec = dmsstring2dec(args[1])
    
    tablefn = None
    if opt.fields is not None:
        if os.path.exists(opt.fields):
            tablefn = opt.fields
        else:
            tablefn = find_data_file(opt.fields)
        if tablefn is None:
            print('Failed to find list of fields:', opt.fields)
            sys.exit(-1)
    
    # arcmin
    radius = opt.radius
    rcfs = radec_to_sdss_rcf(ra,dec,radius=radius, tablefn=tablefn, contains=opt.contains)
    print('ra,dec', ra,dec)
    print('rcfs:', rcfs)
    print()
    for (r,c,f,ra1,dec1) in rcfs:
        print('%i %i %i (dist: %g arcmin)' % (r,c,f, deg2arcmin(degrees_between(ra,dec,ra1,dec1))))

    print()
    for (r,c,f,ra1,dec1) in rcfs:
        print('http://cas.sdss.org/dr7/en/get/frameByRCFZ.asp?R=%i&C=%i&F=%i&Z=0&submit1=Get+Image' % (r,c,f))

    print()
    for (r,c,f,ra1,dec1) in rcfs:
        print('wget "http://cas.sdss.org/dr7/en/get/frameByRCFZ.asp?R=%i&C=%i&F=%i&Z=0&submit1=Get+Image" -O sdss-%04i-%i-%04i.jpg' % (r,c,f,r,c,f))

    from .sdss_das import sdss_das_get
    for (r,c,f,ra1,dec1) in rcfs:
        for t in opt.filetypes:
            for b in opt.bands:
                R = sdss_das_get(t, None, r, c, f, b)
                if R is False:
                    continue
                if t == 'fpC':
                    fpc = sdss_filename('fpC', r, c, f, b)
                    os.system('gunzip -cd %s.gz > %s' % (fpc,fpc))
                    wcs = Tan(filename=fpc)
                    x,y = wcs.radec2pixelxy(ra, dec)
                    x,y = int(x),int(y)
                    os.system('imcopy %s"[%i:%i,%i:%i]" !/tmp/cut-%s' % (fpc, max(0, x-100), x+100, max(0, y-100), y+100, fpc))
                    os.system('an-fitstopnm -i /tmp/cut-%s -N 1150 -X 1400 | pnmtopng > cut-%s.png' % (fpc, fpc))
                    print('R,C,F', r,c,f)
                    print('x,y', x,y)

if __name__ == '__main__':
    main()

