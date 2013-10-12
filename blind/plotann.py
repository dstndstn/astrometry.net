#! /usr/bin/env python
import sys
import os
from optparse import OptionParser

# from util/addpath.py
if __name__ == '__main__':
    mydir = sys.path[0]
    andir = os.path.dirname(mydir)
    rootdir = os.path.dirname(andir)
    sys.path.insert(1, rootdir)

from astrometry.blind.plotstuff import *
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
    I,J,d = trees_match(kd, kd2, r, permuted=False)
    # HACK
    #I2,J,d = trees_match(kd, kd2, r)
    xyz = spherematch_c.kdtree_get_positions(kd, I)
    #print 'I', I
    I2 = tree_permute(kd, I)
    #print 'I2', I2
    tree_free(kd2)
    tree_close(kd)
    tra,tdec = xyztoradec(xyz)
    return tra, tdec, I2
    

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
    # NGC/IC 2000
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
        #print 'Matching HD...'
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
            print 'Bright-star catalog:', n1, n2
            names = []
            n1 = n1.strip()
            if len(n1):
                if '\\u' in n1:
                    n1 = unicode(n1)
                    import simplejson
                    x = simplejson.dumps(n1)
                    x = x.replace('\\\\','\\')
                    n1 = simplejson.loads(x)
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
        dd = dict(type=typ, names=names, pixelx=x, pixely=y,
                  radius=0.)
        if len(ann) == 5:
            mag = ann[4]
            dd.update(vmag=mag)
        jobjs.append(dd)
    for r,d,typ,names,rad in circs:
        ok,x,y = wcs.radec2pixelxy(float(r),float(d))
        pixscale = wcs.pixel_scale()
        pixrad = (rad * 3600.) / pixscale
        jobjs.append(dict(type=typ, names=names, pixelx=x, pixely=y,
                          radius=pixrad))

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

    parser.add_option('--ngccat', dest='ngccat',
                      help='Path to NGC2000 catalog ngc2000.fits -- ONLY USED FOR JSON OUTPUT!')
    parser.add_option('--ngcnames', dest='ngcnames',
                      help='Path to ngc2000names.fits for aliases')
    parser.add_option('--iccat', dest='iccat',
                      help='Path to IC2000 catalog ic2000.fits -- ONLY USED FOR JSON OUTPUT!')

    parser.add_option('--ngcfrac', dest='ngcfrac', type=float, default=0.,
                      help='Minimum fraction of image size to plot NGC/IC objs')

    parser.add_option('--brightcat', dest='brightcat',
                      help='Path to bright-star catalog -- ONLY USED FOR JSON OUTPUT!')

    parser.add_option('--nbright', dest='nbright', type=int, default=0,
                      help='Max number of bright stars')

    parser.add_option('--target', '-t', dest='target', action='append',
                      default=[],
                      help='Add named target (eg "M 31", "NGC 1499")')
    parser.add_option('--no-grid', dest='grid', action='store_false',
                      default=True, help='Turn off grid lines')
    parser.add_option('--grid-size', dest='gridsize', type=float,
                      default=0.1,
                      help='Grid spacing (in degrees), default %default')
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
    parser.add_option('--toy', dest='toy', default=0, type=float,
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

    wcsfn = args[0]

    if dojson:
        from astrometry.util.util import anwcs
        wcs = anwcs(wcsfn,0)
        jobjs = get_annotations_for_wcs(wcs, opt)
        import simplejson
        json = simplejson.dumps(jobjs)
        print json
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

    plot.outfn = outfn
    img = plot.image
    img.set_file(imgfn)

    if opt.scale:
        plot.scale_wcs(opt.scale)
        plot.set_size_from_wcs()
        #W,H = img.get_size()

    plot.plot('image')

    if opt.grid:
        plot.color = 'gray'
        plot.plot_grid(opt.gridsize, opt.gridsize,
                       opt.gridlab, opt.gridlab)

    ann = plot.annotations
    ann.NGC = opt.ngc
    ann.constellations = opt.const
    ann.constellation_labels = opt.const
    ann.constellation_labels_long = opt.const
    ann.bright = opt.bright
    ann.ngc_fraction = opt.ngcfrac
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

    plot.plot('annotations')

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
