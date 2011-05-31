import logging

from astrometry.net import settings

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    filename=settings.SERVER_LOGFILE,
                    )

def log(*msg):
    logging.debug(' '.join([str(m).decode('utf8', 'backslashreplace') for m in msg]))

