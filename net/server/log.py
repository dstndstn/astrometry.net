import logging

from astrometry.net import settings

logfile = settings.SERVER_LOGFILE
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    filename=logfile,
                    )

def log(*msg):
    logging.debug(' '.join([str(m).decode('utf8', 'backslashreplace') for m in msg]))

