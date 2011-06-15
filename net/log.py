import logging
logger = logging.getLogger(__name__)
#debug = logger.debug
#loginfo = logger.info
#logmsg = logger.info

def loginfo(*args):
    logger.info([' '.join(str(a) for a in args)])
logmsg = loginfo

def debug(*args):
    logger.debug([' '.join(str(a) for a in args)])
logdebug = debug
