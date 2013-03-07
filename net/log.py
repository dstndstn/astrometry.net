import logging
logger = logging.getLogger(__name__)
#debug = logger.debug
#loginfo = logger.info
#logmsg = logger.info

def _getstr(args):
    try:
        return ' '.join(str(a) for a in args)
    except:
        return ' '.join(unicode(a) for a in args)

def loginfo(*args):
    ss = _getstr(args)
    logger.info(ss)

logmsg = loginfo

def debug(*args):
    ss = _getstr(args)
    logger.debug(ss)

logdebug = debug
