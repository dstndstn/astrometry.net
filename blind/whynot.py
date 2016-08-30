#! /usr/bin/env python
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function

from astrometry.util.index import *
from astrometry.util.sip import *
from astrometry.util.starutil_numpy import *
from astrometry.util.fits import *
from astrometry.libkd.spherematch import match

from numpy import *

from optparse import OptionParser
import sys

def allcombinations(fronts, backs):
    #print 'fronts', fronts, 'backs', backs
    if len(fronts) == 0:
        return fronts
    if len(backs) == 0:
        return fronts
    newf = []
    for f in fronts:
        for b in backs[0]:
            newf.append(f + [b])
    if len(newf) == 0:
        return []
    return allcombinations(newf, backs[1:])
    #return allcombinations([f + [backs[0]] for f in fronts],
    #                       backs[1:])

def whynot(field, index, qidx, wcs):
    # find index stars in the field.
    (ra,dec) = wcs.get_radec_center()
    rad = arcsec2deg(wcs.get_pixel_scale() * hypot(wcs.get_width(), wcs.get_height()))
    
    (xyz, radec, starinds) = index_search_stars(index, ra, dec, rad)
    print('xyz', xyz.shape)
    print('radec', radec.shape)
    print('starinds', starinds.shape)

    istars = tabledata()
    istars.xyz = xyz
    istars.radec = radec
    istars.starind = starinds

    W,H = wcs.get_width(), wcs.get_height()
    pix = array([wcs.radec2pixelxy(r, d) for (r,d) in radec])
    print('pix', pix.shape)
    # within image bounds
    I = (pix[:,0] > 0) * (pix[:,0] < W) * (pix[:,1] > 0) * (pix[:,1] < H)

    # find nearby pairs of stars...
    nsigma = 3
    pixeljitter = 1
    sigma = hypot(index.index_jitter / wcs.get_pixel_scale(), pixeljitter)
    rad = nsigma * sigma

    fieldxy = vstack((field.x, field.y)).T
    print('field', fieldxy.shape)

    (cinds, cdists) = match(pix, fieldxy, rad)
    print('matches:', cinds.shape)
    #print cdists.shape

    corrs = tabledata()
    corrs.dist = cdists[:,0]
    corrs.star = starinds[cinds[:,0]]
    corrs.field = cinds[:,1]

    allquads = []

    # All quads built from stars in the field...
    for starind in starinds[I]:
        #print qidx, starind
        quads = qidxfile_get_quad_list(qidxfile_addr(qidx), starind)
        #print 'quads:', quads.shape
        allquads.append(quads)

    allquads = unique(hstack(allquads))
    print('%i unique quads touch stars in the field' % len(allquads))

    '''
    "quads" object: all quads that touch index stars in this field.

    .quad : (int) quad index
    .stars : (DQ x int) star indices
    .starsinfield: (DQ x bool) stars in field
    
    '''
    quads = tabledata()
    quads.quad = allquads

    qstars = quadfile_get_stars_for_quads(quadfile_addr(index), quads.quad)
    print('stars in quads:', qstars.shape)
    #print qstars

    quads.stars = qstars

    quads.starsinfield = array([[s in starinds[I] for s in q] for q in qstars])
    #print quads.starsinfield

    allin = quads.starsinfield.min(axis=1)
    #print quads.allin

    quads.allin = allin

    quadsin = quads[allin]
    print('%i quads use stars that are in the field' % (len(quadsin)))
    print('stars:', starinds)

    c_s2f = {}
    for c in corrs: #i in range(len(corrs)):
        #print 'corr', c
        if not c.star in c_s2f:
            c_s2f[c.star] = []
        c_s2f[c.star].append(c.field)
        #c_s2f[corrs.star[i]]
    #print c_s2f
    
    # [[[213], [35], [265], [63]], [[186]], [[]], [[11], [19]], ...]
    # For each quad, for each star in the quad, the list of field stars corresponding.
    fq = []
    for q in quadsin.stars:
        fq1 = []
        for s in q:
            if s in c_s2f:
                fq1.append(c_s2f[s])
            else:
                fq1.append([])
        fq.append(fq1)
    #print fq

    # For each quad, the list of sets of field stars corresponding.
    # [[[213, 35, 265, 63]], [], [],
    fq2 = []
    for q in fq:
        #print 'q:', q
        #ac = allcombinations([q[0]], q[1:])
        #ac = allcombinations([[]], q)
        #print '--> ac:', ac
        fq2.append(allcombinations([[]], q))
    #print fq2

    quadsin.fq2 = fq2

    hasf = array([len(x) > 0 for x in quadsin.fq2])
    #print 'hasf:', hasf

    okquads = quadsin[hasf]

    #forder = argsort([max(x) for x in okquads.fq2])
    #okquads = okquads[forder]

    print('ok quads:', len(okquads))
    print('quad nums:', okquads.quad)
    print('stars:', okquads.stars)
    print('field stars:', okquads.fq2)

    #qmatches = table_data()
    #fqs = []
    #qs = []
    #for okq,fq in zip(okquads, okquads.fq2):
    #    fqs = fqs + fq
    #    qs.append(okq)
    #qmatches.fieldquad = fqs

    qmatches = []
    for okq,fqs in zip(okquads, okquads.fq2):
        for fq in fqs:
            #print 'field stars', fq
            #print 'quad:', okq.quad
            qmatches.append((fq, okq))

    forder = argsort([max(fq) for fq,okq in qmatches])
    #print 'forder', forder

    print()
    print('Quads in the order they will be found')
    for i in forder:
        fq,okq = qmatches[i]
        print()
        print('object', max(fq))
        print('field stars', fq)
        print('quad:', okq.quad)




    # Index stars, by sweep (or "r" mag)...
    # --which quads are they part of

    #skdt = starkd_addr(index)
    istars.sweep = array([startree_get_sweep(index.starkd, si) for si in istars.starind])

    mystars = istars[I]
    order = argsort(mystars.sweep)

    print()
    print()
    print('Index stars:', end=' ')
    for ms in mystars[order]:
        print()
        print('Star', ms.starind, 'sweep', ms.sweep, 'radec', ms.radec)
        print('Field star(s) nearby:', len(c_s2f.get(ms.starind, [])))
        for c in corrs[corrs.star == ms.starind]:
            print('  field star', c.field, 'dist', c.dist, 'pixels')
        myquads = quads[array([(ms.starind in q.stars) for q in quads])]
        print('In %i quads' % len(myquads))
        for q in myquads:
            print('  quad %i: %i stars in the field' % (q.quad, sum(q.starsinfield)))
            if q.allin:
                print('    stars:', q.stars)
                print('    correspondences for stars:', [c_s2f.get(s, None) for s in q.stars])


if __name__ == '__main__':
    parser = OptionParser(usage='%prog <field> <index> <qidx> <wcs>')

    opt, args = parser.parse_args()

    if len(args) != 4:
        parser.print_help()
        sys.exit(-1)
        
    fieldfn = args[0]
    indexfn = args[1]
    qidxfn = args[2]
    wcsfn = args[3]

    log_init(3)

    field = fits_table(fieldfn)
    index = index_load(indexfn, 0, None)
    qidx = qidxfile_open(qidxfn)
    wcs = Sip(filename=wcsfn)
    
    whynot(field, index, qidx, wcs)
    
