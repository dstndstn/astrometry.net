from __future__ import print_function
import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)

import django
django.setup()

from astrometry.net.models import *
from log import *
from django.contrib.auth.models import User

#users = User.objects.filter(email__contains='godard')
#users = User.objects.filter(id=36789)
#users = User.objects.filter(id__in=[29376, 43427])
users = User.objects.filter(id__in=[29427,44180])
#users = User.objects.filter(profile__apikey='mevrrxqunbicseno')

#u = bestuser
#nsub = u.submissions.count()
#print('  User', u.id, 'has', nsub, 'Submissions')

print(users.count(), 'Users match')
if False:
    bestuser = None
    nmax = 0
    for u in users:
        nsub = u.submissions.count()
        print('  User', u.id, 'has', nsub, 'Submissions')
        if nsub > nmax:
            bestuser = u
            nmax = nsub
        print('  API key', u.profile.apikey)
for u in users:
    nsub = u.submissions.count()
    print('  User', u.id, 'has', nsub, 'Submissions')

#bestuser = User.objects.get(id=43427)
bestuser = users[0]
print('Updating to user:', bestuser, 'id', bestuser.id)

#users = list(users) + [bestuser]

#sys.exit(0)
        
#from social.apps.django_app.default.models import UserSocialAuth
from social_django.models import UserSocialAuth

socs = {}
for u in users:
#for u in [bestuser]:
    soc = UserSocialAuth.objects.filter(user=u)
    print('  User', u.id, 'has', soc.count(), 'social auths', [s.id for s in soc], [s.provider for s in soc])
    for s in soc:
        socs[s.id] = s
        #s.user = u
        s.user = bestuser
        s.save()

#bestuser = users[0]
sys.exit(0)

if len(socs) == 1:
    # Make the single social auth -> the User with the most submissions.
    ss = list(socs.values())
    soc = ss[0]
    olduser = soc.user
    print("Social auth's OLD user:", olduser.id, olduser)
    print('Updating to', bestuser.id, bestuser)
    soc.user = bestuser
    soc.save()
