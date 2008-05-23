import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
os.environ['PATH'] = '/bin:/usr/bin:/home/gmaps/test/quads:/home/gmaps/test/tilecache/render'

import logging
import os.path
import shutil
import sys

from django.db import models

from astrometry.net.portal.models import Job, Submission, DiskFile, Calibration, Tag
from astrometry.net.portal.log import log
from astrometry.net.portal.convert import convert, is_tarball, get_objs_in_field, FileConversionError
from astrometry.util.run_command import run_command

# http://www.techcrunch.com/get-youtube-movie/
# will get any youtube video as a local ".flv" download.
# mplayer -vo png foo.flv
# mencoder -o sky.avi -ovc lavc -lavcopts vcodec=mpeg4:keyint=50:autoaspect mf://sky-0*.png -mf fps=10:type=png
# mencoder -o ann.avi -ovc lavc -lavcopts vcodec=mpeg4:keyint=50:autoaspect mf://ann-*.png -mf fps=10:type=png

if __name__ == '__main__':

    # FIREBall
    # subid = 'test-200804-28074176'
    # outdir = '/tmp/movie/B'

    # 
    subid = 'test-200804-68274868'
    outdir = '/tmp/movie/C'
    
    blankskyfn = os.path.join(outdir, 'sky-blank.png')

    render_sky = False
    render_ann = True

    ann_args = {
        #'grid': 10., # grid spacing in arcmin
        'grid': 600., # grid spacing in arcmin
        }

    ann_out_prefix = 'ann-'
    sky_out_prefix = 'sky-'

    if render_sky:
        (ramin,ramax,decmin,decmax) = (0, 360, -85, 85)
        (w, h) = (300, 300)
        layers = [ 'tycho', 'grid' ]
        gain = -1
        cmd = ('tilerender'
               + ' -x %f -X %f -y %f -Y %f' % (ramin, ramax, decmin, decmax)
               + ' -w %i -h %i' % (w, h)
               + ''.join((' -l ' + l) for l in layers)
               + ' -s' # arcsinh
               + ' -g %g' % gain
               + ' > %s' % blankskyfn
               )
        (rtn, out, err) = run_command(cmd)
        if rtn:
            print 'Failed to tilerender the blank sky.'
            print 'out', out
            print 'err', err
            sys.exit(-1);


    subs = Submission.objects.all().filter(subid=subid)
    if len(subs) != 1:
        print 'Got %i submissions, not 1.' % len(subs)
        sys.exit(-1)
    sub = subs[0]
    jobs = sub.jobs.all().order_by('fileorigname')
    
    print 'Got %i jobs.' % len(jobs)

    ns = 0
    nu = 0
    for i, job in enumerate(jobs):
        if job.solved():
            ns += 1
            if render_ann:
                print 'Job %i: creating annotation.' % (i+1)
                annfn = convert(job, job.diskfile, 'annotation-big', ann_args)
            if render_sky:
                print 'Job %i: creating on-the-sky.' % (i+1)
                skyfn = convert(job, job.diskfile, 'onsky-dot')
        else:
            nu += 1
            if render_ann:
                print 'Job %i: getting original.' % (i+1)
                annfn = convert(job, job.diskfile, 'fullsizepng')
            if render_sky:
                skyfn = None

        if render_ann:
            newannfn = os.path.join(outdir, ann_out_prefix + job.fileorigname)
            shutil.move(annfn, newannfn)

        if render_sky:
            newskyfn = os.path.join(outdir, sky_out_prefix + job.fileorigname)
            if skyfn:
                shutil.move(skyfn, newskyfn)
            else:
                os.symlink(blankskyfn, newskyfn)

    print '%i solved and %i unsolved.' % (ns, nu)
    
