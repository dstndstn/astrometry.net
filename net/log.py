import logging
logger = logging.getLogger(__name__)

def _getstr(args):
    try:
        return ' '.join(str(a) for a in args)
    except:
        return ' '.join(unicode(a) for a in args)

def loginfo(*args):
    ss = _getstr(args)
    logger.info(ss)

def logwarn(*args):
    ss = _getstr(args)
    logger.warning(ss)

logmsg = loginfo

def debug(*args):
    ss = _getstr(args)
    logger.debug(ss)

logdebug = debug
