import os
import os.path
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'an.settings'
sys.path.extend(['/home/gmaps/test/tilecache',
                 '/home/gmaps/test/an-common',
                 '/home/gmaps/test/',
                 '/home/gmaps/django/lib/python2.4/site-packages'])
os.environ['LD_LIBRARY_PATH'] = '/home/gmaps/test/an-common'
os.environ['PATH'] = '/bin:/usr/bin:/home/gmaps/test/quads'

import sip

from django.contrib.auth.models import User

from an.util.run_command import run_command
from an.testbed.models import OldJob, TestbedJob
from an.portal.wcs import TanWCS
from an.portal.job import AstroField



if __name__ == '__main__':

    # BIG HACK! - look through LD_LIBRARY_PATH if this is still needed...
    if not sip.libraryloaded():
        sip.loadlibrary('/home/gmaps/test/an-common/_sip.so')

    ojs = OldJob.objects.all()
    print 'Oldjobs: %i' % len(ojs)
    ojs = ojs.filter(solved=True)
    print 'Solved: %i' % len(ojs)

    # DEBUG
    ojs = ojs[:100]

    # HACK
    us = User.objects.all().filter(username='testbed@astrometry.net')
    if len(us) != 1:
        print 'Failed to find user.'
        sys.exit(-1)
    user = us[0]
    
    for oj in ojs:
        if not oj.jobdir or not oj.imagefile:
            #len(oj.jobdir) == 0 or len(oj.imagefile) == 0:
            continue
        
        field = AstroField(user = user,
                           xcol = 'X',
                           ycol = 'Y',
                           imagew = oj.imagew,
                           imageh = oj.imageh,
                           )
        field.save()
        fn = field.filename()
        os.symlink(os.path.join(oj.jobdir, oj.imagefile), fn)

        wcs = TanWCS(file=os.path.join(oj.jobdir, 'wcs.fits'))
        wcs.save()

        tj = TestbedJob(field = field,
                        wcs = wcs,
                        #origid = oj.jobid,
                        )
        tj.save()
        print 'added testbed job', tj.id


    
