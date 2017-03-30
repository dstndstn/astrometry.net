from __future__ import print_function
import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)

from astrometry.net.models import *
from log import *
from django.contrib.auth.models import User

import django
django.setup()


users = User.objects.filter(email__contains='godard')
print(users.count(), 'Users match')
bestuser = None
nmax = 0
for u in users:
    nsub = u.submissions.count()
    print('  User', u.id, 'has', nsub, 'Submissions')
    if nsub > nmax:
        bestuser = u
        nmax = nsub

from social.apps.django_app.default.models import UserSocialAuth

socs = {}
for u in users:
    soc = UserSocialAuth.objects.filter(user=u)
    print('  User', u.id, 'has', soc.count(), 'social auths', [s.id for s in soc])
    for s in soc:
        socs[s.id] = s

if len(socs) == 1:
    # Make the single social auth -> the User with the most submissions.
    soc = socs.values()[0]
    olduser = soc.user
    print("Social auth's OLD user:", olduser.id, olduser)
    print('Updating to', bestuser.id, bestuser)
    #soc.user = bestuser
    #soc.save()
