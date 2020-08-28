import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
import settings

import django
django.setup()

from astrometry.net.models import *

import logging
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)


from collections import Counter

SL = SkyLocation.objects.all()
print(SL.count(), 'sky locations')
c = Counter([(s.nside, s.healpix) for s in SL])

for (ns,hp),n in c.most_common():
    if n == 1:
        break
    print('Nside,Healpix', ns, hp, '-->', n)

    S = SkyLocation.objects.filter(nside=ns, healpix=hp)
    print('Found', S.count())
    s0 = S[0]
    for s in S[1:]:
        # SkyLocations.calibrations --> Calibration.sky_location
        print('fixing dup: cals', s.calibrations)
        if s.calibrations is None:
            continue
        for cal in s.calibrations.all():
            cal.sky_location = s0
            cal.save()
        s.delete()
