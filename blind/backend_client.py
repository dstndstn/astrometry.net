import sys
import socket
import time

if __name__ == '__main__':
    jobfile = sys.argv[1]
    cancelfile = sys.argv[2]

    daemon = ('127.0.0.1', 9999)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(daemon)
    s.sendall('job %s %s\n' % (jobfile, cancelfile))
    
    f = s.makefile('r')
    # Pipe from socket to stderr.
    while True:
        try:
            line = f.readline()
            if line == '':
                break
            sys.stderr.write(line)
        except:
            break
    f.close()
    s.close()

