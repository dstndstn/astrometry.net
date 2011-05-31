import logging

from astrometry.net1 import settings

logfile = settings.PORTAL_LOGFILE
logging.basicConfig(level=logging.DEBUG,
                    #format='%(asctime)s %(levelname)s %(message)s',
                    format='%(pathname)s %(name)s %(message)s',
                    filename=logfile,
                    )

# disable django's logging every SQL call
l = logging.getLogger('django.db.backends')
l.setLevel(logging.INFO)


def log(*msg):
    logging.debug(' '.join([str(m).decode('latin_1', 'backslashreplace') for m in msg]))
    #logging.debug(' '.join([str(m).encode('latin_1', 'backslashreplace') for m in msg]))
    #logging.debug(' '.join([str(m).encode('latin_1') for m in msg]))
    #logging.debug(' '.join(map(str, msg)))
