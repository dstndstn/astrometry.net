import logging

from an import gmaps_config

logfile = gmaps_config.vo_logfile
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    filename=logfile,
                    )

def log(*msg):
    logging.debug(' '.join(map(str, msg)))
