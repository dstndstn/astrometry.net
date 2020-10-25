import os
import shutil

from astrometry.net.log import logmsg

class TempfileMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization.

    def __call__(self, request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.
        request.tempfiles = []
        request.tempdirs = []

        response = self.get_response(request)

        # Code to be executed for each request/response after
        # the view is called.
        for dirnm in request.tempdirs:
            if os.path.exists(dirnm):
                try:
                    shutil.rmtree(dirnm)
                except OSError as e:
                    logmsg('Failed to delete temp dir', dirnm, ':', e)
        for fn in request.tempfiles:
            if os.path.exists(fn):
                try:
                    os.remove(fn)
                except OSError as e:
                    logmsg('Failed to delete temp file', fn, ':', e)

        return response

    
