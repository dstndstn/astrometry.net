import logging

from astrometry.server import settings

logfile = settings.LOGFILE
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    filename=logfile,
                    )

def log(*msg):
    logging.debug(' '.join([str(m).decode('utf8', 'backslashreplace') for m in msg]))

