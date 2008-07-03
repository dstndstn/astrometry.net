from SocketServer import ThreadingTCPServer, BaseRequestHandler

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

if __name__ == '__main__':
    server_address = ('127.0.0.1', 9999)
    request_handler_class = BackendHandler
    ss = ThreadingTCPServer(server_address, request_handler_class)
    ss.serve_forever()

