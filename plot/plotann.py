#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import sys
import os
from optparse import OptionParser

# from util/addpath.py
if __name__ == '__main__':
    mydir = sys.path[0]
    andir = os.path.dirname(mydir)
    rootdir = os.path.dirname(andir)
    sys.path.insert(1, rootdir)

from astrometry.plot.plotstuff import *
from astrometry.util.fits import *

def match_kdtree_catalog(wcs, catfn):
    from astrometry.libkd.spherematch import tree_open, tree_close, tree_build_radec, tree_free, trees_match, tree_permute
    from astrometry.libkd import spherematch_c
    from astrometry.util.starutil_numpy import deg2dist, xyztoradec
    import numpy as np
    import sys

    rc,dc = wcs.get_center()
    rr = wcs.get_radius()
    kd = tree_open(catfn)
    kd2 = tree_build_radec(np.array([rc]), np.array([dc]))
    r = deg2dist(rr)
    I,J,nil = trees_match(kd, kd2, r, permuted=False)
    del kd2
    xyz = kd.get_data(I.astype(np.uint32))
    I = kd.permute(I)
    del kd
    tra,tdec = xyztoradec(xyz)
    return tra, tdec, I
    

def get_annotations(wcs, opt):
    # Objects to annotate:
    annobjs = []
    
    if opt.uzccat:
        # FIXME -- is this fast enough, or do we need to cut these
        # targets first?
        T = fits_table(opt.uzccat)
        for i in range(len(T)):
            if not wcs.is_inside(T.ra[i], T.dec[i]):
                continue
            annobjs.append((T.ra[i], T.dec[i], 'uzc', ['UZC %s' % T.zname[i]]))

    if opt.hipcat:
        # FIXME -- is this fast enough, or do we need to cut these
        # targets first?
        T = fits_table(opt.hipcat)
        for i in range(len(T)):
            if not wcs.is_inside(T.ra[i], T.dec[i]):
                continue
            if opt.hiplabel:
                txt = ['HIP %i (%.1f)' % (T.hip[i], T.vmag[i])]
            else:
                txt = ['']
            annobjs.append((T.ra[i], T.dec[i], 'HIP', txt))
            
    if opt.abellcat:
        T = fits_table(opt.abellcat)
        for i in range(len(T)):
            if not wcs.is_inside(T.ra[i], T.dec[i]):
                continue
            annobjs.append((T.ra[i], T.dec[i], 'abell', ['Abell %i' % T.aco[i]]))

    if opt.t2cat:
        #print 'Matching Tycho2...'
        tra,tdec,I2 = match_kdtree_catalog(wcs, opt.t2cat)
        T = fits_table(opt.t2cat, hdu=6)
        for r,d,t1,t2,t3 in zip(tra,tdec, T.tyc1[I2], T.tyc2[I2], T.tyc3[I2]):
            if not wcs.is_inside(r, d):
                continue
            annobjs.append((r, d, 'tycho2', ['Tycho-2 %i-%i-%i' % (t1,t2,t3)]))

    return annobjs


def get_annotations_for_wcs(wcs, opt):
    anns = get_annotations(wcs, opt)
    circs = []
    if opt.ngcnames:
        T = fits_table(opt.ngcnames)
        T.cut(np.array([len(s) for s in T.name]))
        T.isngc = np.array([not s.startswith('I') for s in T.name])
        T.num = np.array([int(s.replace('I','').strip()) for s in T.name])
        namemap = {}
        for X,nm in zip(zip(T.isngc, T.num), T.object):
            if not X in namemap:
                namemap[X] = []
            namemap[X].append(nm)
    else:
        namemap = None

    for nm,isngc,cat in [('NGC', True, opt.ngccat), ('IC', False, opt.iccat)]:
        if not cat:
            continue
        T = fits_table(cat)
        num = T.get(nm.lower() + 'num')
        for i in range(len(T)):
            # FIXME -- include NGC object radius...?
            if not wcs.is_inside(float(T.ra[i]), float(T.dec[i])):
                continue
            names = ['%s %i' % (nm, num[i])]
            if namemap:
                more = namemap.get((isngc, num[i]), None)
                if more:
                    names += more #' / '.join(more)
            if T.radius[i]:
                circs.append((T.ra[i], T.dec[i], nm.lower(), names, T.radius[i]))
            else:
                anns.append((T.ra[i], T.dec[i], nm.lower(), names))

    if opt.hdcat:
        ra,dec,I = match_kdtree_catalog(wcs, opt.hdcat)
        for r,d,i in zip(ra,dec,I):
            if not wcs.is_inside(r, d):
                continue
            # good ol' HD catalog and its sensible numbering scheme
            anns.append((r, d, 'hd', ['HD %i' % (i+1)]))

    if opt.brightcat:
        T = fits_table(opt.brightcat)
        if opt.nbright:
            T.cut(np.argsort(T.vmag))
        nb = 0
        for r,d,n1,n2,vmag in zip(T.ra, T.dec, T.name1, T.name2, T.vmag):
            if not wcs.is_inside(r, d):
                continue
            # print('Bright-star catalog:', n1, n2, type(n1))
            names = []
            n1 = n1.strip()
            if len(n1):
                if '\\u' in n1:
                    try:
                        # py2 only
                        n1 = unicode(n1)
                    except:
                        pass
                    import json
                    x = json.dumps(n1)
                    x = x.replace('\\\\','\\')
                    n1 = json.loads(x)
                names.append(n1)
            n2 = n2.strip()
            if len(n2):
                names.append(n2)
            if len(names) == 0:
                # skip unnamed stars
                continue
            anns.append((r, d, 'bright', names, vmag))
            nb += 1
            if opt.nbright and nb >= opt.nbright:
                break

    jobjs = []
    for ann in anns:
        r,d,typ,names = ann[:4]
        ok,x,y = wcs.radec2pixelxy(float(r),float(d))
        dd = dict(type=typ, names=names, pixelx=float(x), pixely=float(y),
                  radius=0.)
        if len(ann) == 5:
            mag = ann[4]
            dd.update(vmag=float(mag))
        jobjs.append(dd)
    for r,d,typ,names,rad in circs:
        ok,x,y = wcs.radec2pixelxy(float(r),float(d))
        pixscale = wcs.pixel_scale()
        pixrad = (rad * 3600.) / pixscale
        jobjs.append(dict(type=typ, names=names, pixelx=float(x), pixely=float(y),
                          radius=float(pixrad)))

    return jobjs

class OptDuck(object):
    pass

def get_empty_opts():
    opt = OptDuck()
    opt.ngc = False
    opt.bright = False
    opt.brightcat = None
    opt.nbright = 0
    opt.hdcat = None
    opt.uzccat = None
    opt.t2cat = None
    opt.abellcat = None
    opt.ngccat = None
    opt.ngcnames = None
    opt.iccat = None
    opt.hipcat = None
    opt.hiplabel = False
    return opt

if __name__ == '__main__':
    parser = OptionParser('usage: %prog <wcs.fits file> <image file> <output.{jpg,png,pdf} file>\n' +
                          '    OR     %prog <wcs.fits>   for JSON output')
    parser.add_option('--scale', dest='scale', type=float,
                      help='Scale plot by this factor')
    parser.add_option('--no-ngc', dest='ngc', action='store_false', default=True)
    parser.add_option('--no-bright', dest='bright', action='store_false', default=True)
    parser.add_option('--no-const', dest='const', action='store_false', default=True, help='Do not plot constellations')
    parser.add_option('--hdcat', dest='hdcat',
                      help='Path to Henry Draper catalog hd.fits')
    parser.add_option('--uzccat', dest='uzccat',
                      help='Path to Updated Zwicky Catalog uzc2000.fits')
    parser.add_option('--tycho2cat', dest='t2cat',
                      help='Path to Tycho-2 KD-tree file')
    parser.add_option('--abellcat', dest='abellcat',
                      help='Path to Abell catalog abell-all.fits')

    parser.add_option('--hipcat', dest='hipcat',
                      help='Path to Hipparcos catalog hip.fits')
    parser.add_option('--hiplabel', action='store_true',
                      help='Label Hipparcos stars')
    
    parser.add_option('--ngccat', dest='ngccat',
                      help='Path to NGC catalog openngc-ngc.fits -- ONLY USED FOR JSON OUTPUT!')
    parser.add_option('--ngcnames', dest='ngcnames',
                      help='Path to openngc-names.fits for aliases')
    parser.add_option('--iccat', dest='iccat',
                      help='Path to IC catalog openngc-ic.fits -- ONLY USED FOR JSON OUTPUT!')

    parser.add_option('--ngcfrac', dest='ngcfrac', type=float, default=0.,
                      help='Minimum fraction of image size to plot NGC/IC objs; default %default')

    parser.add_option('--brightcat', dest='brightcat',
                      help='Path to bright-star catalog -- ONLY USED FOR JSON OUTPUT!')

    parser.add_option('--nbright', dest='nbright', type=int, default=0,
                      help='Max number of bright stars')

    parser.add_option('--target', '-t', dest='target', action='append',
                      default=[],
                      help='Add named target (eg "M 31", "NGC 1499")')

    parser.add_option('--target-rd', '-T', dest='targetrd', nargs=3, action='append',
                      default=[],
                      help='Add a custom target with RA,Dec (eg, "-T \'My Star\' 34.33 0.87"')

    parser.add_option('--no-grid', dest='grid', action='store_false',
                      default=True, help='Turn off grid lines')
    parser.add_option('--grid-size', dest='gridsize', type=float,
                      default=0.1,
                      help='Grid spacing (in degrees), default %default')
    parser.add_option('--grid-color', dest='gridcolr',
                      default='0.2:0.2:0.2',
                      help='Grid color, default %default')
    parser.add_option('--grid-label', dest='gridlab', type=float,
                      default=0.2,
                      help='Grid label spacing (in degrees), default %default')

    parser.add_option('--tcolor', dest='textcolor', default='green',
                      help='Text color')
    parser.add_option('--tsize', dest='textsize', default=18, type=float,
                      help='Text font size')
    parser.add_option('--halign', dest='halign', default='C',
                      help='Text horizontal alignment')
    parser.add_option('--valign', dest='valign', default='B',
                      help='Text vertical alignment')
    parser.add_option('--tox', dest='tox', default=0, type=float,
                      help='Text offset x')
    parser.add_option('--toy', dest='toy', default=-10, type=float,
                      help='Text offset y')
    parser.add_option('--lw', dest='lw', default=2, type=float,
                      help='Annotations line width')
    parser.add_option('--ms', dest='ms', default=0., type=float,
                      help='Marker size')
    parser.add_option('--rd', dest='rd', action='append', default=[],
                      help='Plot RA,Dec markers')
    parser.add_option('--xy', dest='xy', action='append', default=[],
                      help='Plot x,y markers')
    parser.add_option('--quad', dest='quad', action='append', default=[],
                      help='Plot quad from given match file')
    parser.add_option('--pastel', action='store_true',
                      help='Pastel colors for constellations and bright stars?')
    parser.add_option('-v', '--verbose',
        action='store_true', dest='verbose', help='be chatty')

    opt,args = parser.parse_args()
    dojson = False
    if len(args) == 3:
        imgfn = args[1]
        outfn = args[2]
    else:
        if len(args) == 1:
            dojson = True
        else:
            parser.print_help()
            sys.exit(-1)

    import logging
    logformat = '%(message)s'
    from astrometry.util.util import log_init #, LOG_VERB, LOG_MSG
    loglvl = 2
    if opt.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logformat)
        loglvl += 1
    else:
        logging.basicConfig(level=logging.INFO, format=logformat)
    log_init(loglvl)

    wcsfn = args[0]

    if dojson:
        from astrometry.util.util import anwcs
        wcs = anwcs(wcsfn,0)
        jobjs = get_annotations_for_wcs(wcs, opt)
        import json
        j = json.dumps(jobjs)
        print(j)
        sys.exit(0)

    fmt = PLOTSTUFF_FORMAT_JPG
    s = outfn.split('.')
    if len(s):
        s = s[-1].lower()
        if s in Plotstuff.format_map:
            fmt = s
    plot = Plotstuff(outformat=fmt, wcsfn=wcsfn)
    #plot.wcs_file = wcsfn
    #plot.outformat = fmt
    #plotstuff_set_size_wcs(plot.pargs)

    if opt.ms:
        plot.markersize = opt.ms
    plot.fontsize = opt.textsize

    plot.outfn = outfn
    img = plot.image
    img.set_file(imgfn)

    if opt.scale:
        plot.scale_wcs(opt.scale)
        plot.set_size_from_wcs()
        #W,H = img.get_size()

    plot.plot('image')

    if opt.grid:
        plot.color = opt.gridcolr
        plot.plot_grid(opt.gridsize, opt.gridsize,
                       opt.gridlab, opt.gridlab)

    ann = plot.annotations
    ann.NGC = opt.ngc
    ann.constellations = opt.const
    ann.constellation_labels = opt.const
    ann.constellation_labels_long = opt.const
    ann.bright = opt.bright
    ann.ngc_fraction = opt.ngcfrac
    if opt.pastel:
        ann.constellation_pastel = True
        ann.bright_pastel = True
        #print(ann.constellation_pastel)
    if opt.hdcat:
        ann.HD = True
        ann.HD_labels = True
        ann.hd_catalog = opt.hdcat

    anns = get_annotations(plot.wcs, opt)
    for r,d,typ,names in anns:
        name = ' / '.join(names)
        ann.add_target(r, d, name)

    plot.color = opt.textcolor
    plot.fontsize = opt.textsize
    plot.lw = opt.lw
    plot.valign = opt.valign
    plot.halign = opt.halign
    plot.label_offset_x = opt.tox;
    plot.label_offset_y = opt.toy;

    if len(opt.target):
        for t in opt.target:
            if plot_annotations_add_named_target(ann, t):
                raise RuntimeError('Unknown target', t)
    # if you want to plot normal vs named targets differently:
    # plot.plot('annotations')
    # ann.clear_targets()
    # ann.NGC = False
    # ann.constellations = False
    # ann.bright = False
    # ann.HD = False
    # plot.color = 'red'
    if len(opt.targetrd):
        for name,ra,dec in opt.targetrd:
            try:
                ra = float(ra)
            except:
                print('Failed to parse RA string as float:', ra)
                raise
            try:
                dec = float(dec)
            except:
                print('Failed to parse Dec string:', dec)
                raise
            ann.add_target(ra, dec, name)
    plot.plot('annotations')

    if False:
        # Could plot bright stars like this, rather than via setting "ann.bright = True"
        n = bright_stars_n()
        plot.markersize = 5
        plot.color = 'cyan'
        plot.marker = 'circle'
        plot.apply_settings()
        for i in range(n):
            s = bright_stars_get(i)
            if len(s.name) == 0 and len(s.common_name) == 0:
                continue
            ok,x,y = plotstuff_radec2xy(plot, s.ra, s.dec)
            if not ok:
                continue
            if x < 0 or y < 0 or x > plot.W or y > plot.H:
                continue
            #print('Star', s.name, '(%s)' % s.common_name, 'at %.1f, %.1f' % (x,y))
            txt = s.common_name
            if len(txt) == 0:
                continue
                #txt = s.name
            # Grab first common name
            txt = txt.split('/')[0]
            plot.marker_xy(x, y)
            plot.text_xy(x, y, txt)
        plot.stroke()

    for rdfn in opt.rd:
        rd = plot.radec
        rd.fn = rdfn
        plot.plot('radec')

    for fn in opt.xy:
        xy = plot.xy
        xy.fn = fn
        plot.plot('xy')
        
    for mfn in opt.quad:
        match = fits_table(mfn)
        for m in match:
            qp = m.quadpix
            xy = [(qp[0], qp[1])]
            #plot.move_to_xy(qp[0], qp[1])
            for d in range(1, m.dimquads):
                #plot.line_to_xy(qp[2 * d], qp[2 * d + 1])
                xy.append((qp[2 * d], qp[2 * d + 1]))
            #plot.stroke()
            plot.polygon(xy)
            plot.close_path()
            plot.stroke()
        
    plot.write()
