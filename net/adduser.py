import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import sys

from django.contrib.auth.models import User
from django.core.validators import email_re

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: %s <email> <password>' % sys.argv[0]
        sys.exit(-1)
    email = sys.argv[1]
    password = sys.argv[2]
    if email_re.match(email) is None:
        print 'Invalid email address: ', email
        print 'Usage: %s <email> <password>'
        sys.exit(-1)

    print 'Adding user...'
    user = User.objects.create_user(email, email, password)
    user.save()
    print 'Added user.'

