import os
import shutil

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
            shutil.rmtree(dirnm)
        for fn in request.tempfiles:
            os.remove(fn)

        return response

    
