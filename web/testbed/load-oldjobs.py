import os
import os.path
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'an.settings'
sys.path.extend(['/home/gmaps/test/tilecache',
                 '/home/gmaps/test/an-common',
                 '/home/gmaps/test/',
                 '/home/gmaps/django/lib/python2.4/site-packages'])

from an.util.run_command import run_command
from an.testbed.models import OldJob, TestbedJob


def load_jobdir(jobdir):
    print 'Job', jobdir
    jobdatafile = os.path.join(jobdir, 'jobdata.db')
    if not os.path.exists(jobdatafile):
        print 'no jobdata file.'
        return
    path = os.path.abspath(jobdir)
    print 'path:', path
    (path,id) = os.path.split(path)
    (path,epoch) = os.path.split(path)
    (path,site) = os.path.split(path)
    jobid = '%s-%s-%s' % (site, epoch, id)
    print 'jobid=', jobid
    cmd = 'sqlite %s "select * from jobdata"' % jobdatafile
    (rtn, out, err) = run_command(cmd)
    if rtn:
        print 'sqlite failed: %i' % rtn
        return
    jobdata = {}
    outlines = out.split('\n')
    for ln in outlines:
        terms = ln.split('|')
        if len(terms) != 2:
            print 'line: "%s"' % ln
        else:
            jobdata[terms[0]] = terms[1]
    #for k,v in jobdata.items():
    #    print '  ',k,'=',v

    imgfn = None
    if 'imagefilename' in jobdata:
        imgfn = jobdata['imagefilename']
    W = 0
    if 'imageW' in jobdata:
        W = jobdata['imageW']
    H = 0
    if 'imageH' in jobdata:
        H = jobdata['imageH']
        
    oj = OldJob(
        jobid = jobid,
        jobdir = os.path.abspath(jobdir),
        email = jobdata['email'],
        imagefile = imgfn,
        solved = ('cd11' in jobdata),
        imagew = W,
        imageh = H,
        )
    oj.save()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage:  load.py <job-dir> ...'
        sys.exit(-1)

    for jobdir in sys.argv[1:]:
        load_jobdir(jobdir)

