from __future__ import print_function
import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)

import settings
import django
django.setup()
from astrometry.net.models import *
from log import *
from django.contrib.auth.models import User


old_users = User.objects.filter(email='christoph@nieswand.org')
#old_users = User.objects.filter(id=15193)
#old_users = User.objects.filter(profile__apikey='')
print('Found old_user(s):', old_users)

for u in old_users:
    print('  ', u.id, u, u.email)
    print(u.get_full_name(), u.profile, u.first_name, u.get_username(), u.username)
    print('date joined', u.date_joined)
    #u.profile.create_api_key()
    #u.profile.save()
    #print('New API key:', u.profile.apikey)
    #print(dir(u))

    # print('Generating new API key')
    # u.profile.create_api_key()
    # u.profile.save()
    # print(u.get_full_name(), u.profile, u.first_name, u.get_username(), u.username)

sys.exit(0)

#new_users = User.objects.filter(email__icontains='')
#new_user = User.objects.get(first_name='Haven')
#new_users = User.objects.filter(id=19432)
new_users = User.objects.filter(email__icontains='myneni')
print('New users:', new_users)

for u in new_users:
    print('  ', u)
    print('  ', u.id, u, u.email)
    print(u.get_full_name(), u.profile, u.first_name, u.get_username(), u.username)

new_user = new_users[0]

print('New user:', new_user)

#sys.exit(0)

update = True
#update = False

for old_user in old_users:
    for clz,field in [(CommentReceiver, 'owner'),
                      (FlaggedUserImage, 'user'),
                      (TaggedUserImage, 'tagger'),
                      (UserImage, 'user'),
                      (Submission, 'user'),
                      (Album, 'user')]:
        objs = clz.objects.filter(**{field: old_user})
        print('Found', objs.count(), 'instances of', clz, 'owned by old user', old_user)

        if update:
            for obj in objs:
                setattr(obj, field, new_user)
                obj.save()
