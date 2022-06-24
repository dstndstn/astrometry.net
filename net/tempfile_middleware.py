import os
import shutil

from astrometry.net.log import logmsg

class TempfileMiddleware:
    # One-time configuration and initialization (one time for the whole app)
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Called for each request
        request.tempfiles = []
        request.tempdirs = []

        response = self.get_response(request)

        if response.streaming:
            def wrap_streaming_content(content, request):
                for chunk in content:
                    yield chunk
                #print('Sent all content')
                delete_tempfiles_for_request(request)
            #print('Streaming response')
            response.streaming_content = wrap_streaming_content(response.streaming_content,
                                                                request)
            return response
        else:
            delete_tempfiles_for_request(request)
            return response

def delete_tempfiles_for_request(request):
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
