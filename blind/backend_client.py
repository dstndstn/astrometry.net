import sys
import socket
import time

#import backend_daemon

if __name__ == '__main__':
    jobfile = sys.argv[1]
    cancelfile = sys.argv[2]

    #daemon = ('127.0.0.1', 9999)
    daemon = ('127.0.0.1', 10000)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(daemon)

    #s.send('cancel %s' % cancelfile)
    s.sendall('job %s %s\n' % (jobfile, cancelfile))
    
    f = s.makefile('r')
    while True:
        #time.sleep(1)
        try:
            line = f.readline()
            #print 'got', line
            if line == '':
                break
            sys.stderr.write(line)
        except:
            break
    f.close()
    s.close()
