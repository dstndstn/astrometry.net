import ctypes
import ctypes.util
import sys
import os
import os.path
import thread

from optparse import OptionParser
from SocketServer import ThreadingTCPServer, BaseRequestHandler

_backend = None
_libname = ctypes.util.find_library('libbackend.so')
if _libname:
    _backend = ctypes.CDLL(_libname)
else:
    p = os.path.join(os.path.dirname(__file__), 'libbackend.so')
    _backend = ctypes.CDLL(p)


class BackendHandler(BaseRequestHandler):
    def handle(self):
        print 'Got request from ', self.client_address
        # self.request: a socket
        # self.client_address: ('123.4.5.6', 4567)
        # self.server

        f = self.request.makefile('rw')
        f.write('Hello\n')
        f.flush()
        backend = self.server.backend
        backend.log_to_fd(f.fileno())

        while True:
            cmdline = f.readline().strip()
            print 'Command is', cmdline
            args = cmdline.split(' ')
            cmd = args[0]
            args = args[1:]

            if cmd == 'job':
                jobpath = args[0]
                cancelfile = args[1]
                jobdir = os.path.dirname(jobpath)

                job = backend.backend_read_job_file(be, jobpath)
                if not job:
                    print 'Failed to read job.'
                    return
                backend.job_set_base_dir(job, jobdir)
                backend.job_set_cancel_file(job, cancelfile)
                backend.backend_run_job(be, job)
                backend.job_free(job)
                break

            elif cmd == 'info':
                print 'pwd is', os.getcwd()
                print 'pid is', os.getpid()
                print 'thread id is', thread.get_ident()
        

        # probably not necessary:
        backend.log_to_fd(0)

        f.close()



if __name__ == '__main__':
    port = 9999
    configfn = '../etc/backend.cfg'

    usage = 'backend_daemon.py [args]'
    parser = OptionParser(usage)
    parser.add_option("-p", "--port", dest="port",
                      help="port to listen on", default=port)
    parser.add_option("-c", "--config", dest="configfn",
                      help="config filename", default=configfn)
    (options, args) = parser.parse_args()
    if len(args):
        parser.error("incorrect number of arguments")
        sys.exit(-1)

    port = options.port
    configfn = options.configfn

    server_address = ('127.0.0.1', port)
    request_handler_class = BackendHandler
    ss = ThreadingTCPServer(server_address, request_handler_class)
    ss.backend = _backend
    print
    print 'Waiting for network connections on', ss.server_address
    print

    _backend.log_init(3)
    _backend.log_set_thread_specific()

    be = _backend.backend_new()
    if _backend.backend_parse_config_file(be, configfn):
        print 'Failed to initialize backend.'
        sys.exit(-1)

    # ??
    ss.daemon_threads = True
    print
    print 'Waiting for network connections on', ss.server_address
    print

    ss.serve_forever()

    print
    print "serve_forever() finished."
    print

