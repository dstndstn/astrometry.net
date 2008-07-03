import ctypes
import ctypes.util
import sys

from SocketServer import ThreadingTCPServer, BaseRequestHandler

_backend = None
_libname = ctypes.util.find_library('libbackend.so')
if _libname:
    _backend = ctypes.CDLL(_libname)
else:
    import os.path
    p = os.path.join(os.path.dirname(__file__), 'libbackend.so')
    _backend = ctypes.CDLL(p)





class BackendHandler(BaseRequestHandler):
    def handle(self):
        print 'Got request from ', self.client_address
        # self.request: a socket
        # self.client_address: ('123.4.5.6', 4567)
        # self.server
        f = self.request.makefile('rw')
        #self.request.send("Hello.\n");
        axy = f.readline()
        print 'Axy is', axy
        f.write('Hello\n')

        backend = self.server.backend

        jobfn = '/tmp/job.axy'
        job = backend.backend_read_job_file(be, jobfn)
        if not job:
            print 'Failed to read job.'
            return
        backend.backend_run_job(be, job)
        backend.job_free(job)

if __name__ == '__main__':

    import os.path
    p = os.path.join(os.path.dirname(__file__), '../etc/backend-test.cfg')

    _backend.log_init(3)
    be = _backend.backend_new()
    configfn = p
    if _backend.backend_parse_config_file(be, configfn):
        print 'Failed to initialize backend.'
        sys.exit(-1)

    server_address = ('127.0.0.1', 9999)
    request_handler_class = BackendHandler
    ss = ThreadingTCPServer(server_address, request_handler_class)
    ss.backend = _backend
    print
    print 'Waiting for network connections...'
    print
    ss.serve_forever()

