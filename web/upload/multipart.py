#! /usr/bin/env python

import re
import sys
import os

def log(x):
    print >> sys.stderr, x

#
# StreamParser - parse data a chunk at a time.
# |
# |--StateMachine - delegate parsing to one of several states.
#    |              StateMachine uses StateMachineState
#    |
#    |--MessageParser - parses a message with a header and a body.
#       |               MessageParser uses HeaderState
#       |
#       |--Multipart - parses multipart/form-data POSTs by alternating
#                      between "part header" and "part body" states.
#                      Multipart uses PartHeaderState
#                      Multipart uses PartBodyState
#
#
#
# StateMachineState - parses according to the current state.
# |
# |--HeaderState - parses lines of text of the form "header: value",
# |  |             terminated by CRLF, until a blank line is found.
# |      |
# |      |--PartHeaderState - parses the header of one part of a multi-part
# |                           message.
# |
# |--PartBodyState - parses the body of one part of a multi-part message.
#

#
# I then tacked on the ability to write some parts to files and keep others
# in memory:
#
# Multipart
# |
# |--FileMultipart
#
#
# PartBodyState
# |
# |--FileBodyState - streams the body of one part of a multi-part message to
#                    a file.

class StreamParser(object):
    def __init__(self, fin=None):
        self.fin = fin
        # unparsed data
        self.data = ''
        self.blocksize = 4096
        self.error = False
        self.errorstring = None
        self.needmore = False
        self.bytes_parsed = 0

    # Returns False if EOF is reached or an error occurs
    def readmore(self):
        newdata = self.fin.read(self.blocksize)
        log('read %d' % len(newdata))
        if len(newdata) == 0:
            return False
        return self.moreinput(newdata)

    # Returns False if a parsing error has occurred
    def moreinput(self, newdata):
        self.data += newdata
        self.needmore = False
        while not (self.error or self.needmore or (len(self.data) == 0)):
            #log('processing...')
            #log('data len is %d, data is "%s"' % (len(self.data), self.data))
            self.process()
        return not self.error

    # Sets self.error if a parsing error occurs;
    # sets self.needmore if more data is needed to proceed.
    def process(self):
        pass

    # Drops 'nchars' bytes off the front of the data buffer.
    def discard_data(self, nchars):
        self.data = self.data[nchars:]
        self.bytes_parsed += nchars


class StateMachine(StreamParser):
    def __init__(self, fin=None):
        super(StateMachine, self).__init__(fin)
        self.state = None
        self.state_transition('start')

    def initial_state(self):
        return None

    def set_state(self, newstate):
        self.state = newstate

    def process(self):
        if not self.state:
            log('Error: no state')
            self.error = True
            self.errorstring = 'Server error (no state)'
            return
        self.state.process()

    # Called by a StateMachineState to indicate that a transition to another
    # state should take place.
    #   'nextstate': string describing the next state.
    def state_transition(self, nextstate):
        if nextstate == 'start':
            self.set_state(self.initial_state())
            return
        log('no such state: "%s"' % nextstate)


class StateMachineState(object):
    def __init__(self, machine=None):
        self.machine = machine

    def process(self):
        pass


class MessageParser(StateMachine):
    def __init__(self, fin=None):
        # the state (a HeaderState object) that will parse the message header
        self.header_state = HeaderState(self)
        # dictionary of headers from the header parser.
        self.headers = None
        # the state (a StateMachineState object) that will parse the message body
        self.body_state = None
        # how many bytes had been parsed when the body was reached.
        self.body_offset = 0
        super(MessageParser, self).__init__(fin)

    def initial_state(self):
        return self.header_state

    def state_transition(self, trans):
        if trans == 'header-done':
            log('header finished.')
            self.headers = self.header_state.headers
            for k,v in self.headers.items():
                log('  ' + str(k) + " = " + str(v))

            self.state_transition('body')
            self.body_offset = self.bytes_parsed
            return
        elif trans == 'body':
            self.body = self.get_body_state()
            if not self.body:
                log('No body state found.')
                self.error = True
                self.errorstring = 'Server error (no body state)'
            self.set_state(self.body)
            return
        super(MessageParser, self).state_transition(trans)

        # Returns a StateMessageState to parse the body of the message.
        def get_body_state(self):
            return None

CRLF = '\r\n'
CR = '\r'
LWSP = ' \t'
ALPHANUM = r'A-Za-z0-9_\-'

class Multipart(MessageParser):
    def __init__(self, fin=None):
        super(Multipart, self).__init__(fin)
        # the parts of this multi-part message.
        # each part is a dictionary.
        self.parts = []
        self.currentpart = None
        self.boundary = None


    # returns a dictionary that maps normal form elements to
    # their data values.
    def get_form_values(self):
        res = {}
        for p in self.parts:
            if not 'field' in p or not 'data' in p:
                continue
            field = p['field']
            data = p['data']
            if field and data:
                res[field] = data
        return res

    def get_uploaded_files(self):
        res = []
        for p in self.parts:
            if not ('field' in p and
                    'filename' in p and
                    'local-filename' in p and
                    'datalen' in p):
                continue
            f = {}
            f['field'] = p['field']
            f['user-filename'] = p['filename']
            f['filename'] = p['local-filename']
            f['length'] = p['datalen']
            res.append(f)

        #if len(res) == 0:
        #    return None
        return res

    def state_transition(self, trans):
        log('state_transition to ' + trans)

        if trans == 'part-header-start':
            self.currentpart = {}
            self.set_state(self.get_part_header_state())
            return

        elif trans == 'part-header-done':
            headers = self.state.headers
            self.currentpart['headers'] = headers

            key = 'Content-Disposition'
            if key in headers:
                cd = headers[key]
                cdre = re.compile(r'^form-data; name="(?P<name>[' + ALPHANUM + r']+)"' +
                                  r'(?P<filename_given>; filename="(?P<filename>[' + ALPHANUM + r'\.' + r']*)")?$')
                match = cdre.match(cd)
                if match:
                    field = match.group('name')
                    filename = match.group('filename')
                    given = match.group('filename_given')
                    if field:
                        self.currentpart['field'] = field
                    if given:
                        if filename:
                            self.currentpart['filename'] = filename
                        else:
                            self.currentpart['filename'] = ''
                else:
                    log('Failed to parse Content-Disposition: ' + cd)
            key = 'Content-Type'
            if key in headers:
                ct = headers[key]
                self.currentpart['content-type'] = ct

            self.state_transition('part-body-start')
            return

        elif trans == 'part-body-start':
            self.set_state(self.get_part_body_state(self.boundary))
            return

        elif trans == 'part-body-done':
            if not self.currentpart:
                log('Preamble ended.')
            else:
                datalen = self.state.get_data_length()
                data = self.state.get_data()
                log('Body ended.  Got %d bytes of data.' % datalen)
                log('Data is: ***%s***' % data)
                if data:
                    self.currentpart['data'] = data
                else:
                    self.currentpart['datalen'] = datalen
                self.add_part(self.currentpart)

            self.currentpart = {}
            return

        elif trans == 'start-next-part':
            self.state_transition('part-header-start')
            return

        elif trans == 'last-part-body-done':
            self.state_transition('end')
            return

        elif trans == 'end':
            return

        super(Multipart, self).state_transition(trans)

    def add_part(self, part):
        self.parts.append(part)

    # Overrides MessageParser.get_body_state(); starts us off in the
    # "preamble" state, which acts basically the same as a body that should
    # be empty.
    def get_body_state(self):
        if not 'Content-Type' in self.headers:
            log('No Content-Type.')
            self.error = True
            self.errorstring = 'No Content-Type header'
            return None
        ct = self.headers['Content-Type']
        if not ct.startswith('multipart/form-data'):
            log('Content-type is "%s", not multipart/form-data.' % ct)
            self.error = True
            self.errorstring = 'Content-Type is not multipart/form-data'
            return None
        ctre = re.compile(r'^multipart/form-data; boundary=(?P<boundary>[' + ALPHANUM + r']+)$')
        res = ctre.match(ct)
        if not res:
            log('no boundary found')
            raise ValueError, 'boundary not found in Content-Type string.'
        self.boundary = res.group('boundary')
        log('boundary: "%s"' % self.boundary)
        return self.get_part_body_state(self.boundary)

    # Returns a StateMachineState that will parse the next "part header".
    def get_part_header_state(self):
        return PartHeaderState(self)

    # Returns a StateMachineState that will parse the next "part body".
    #   "boundary" - the boundary of this multi-part message.
    def get_part_body_state(self, boundary):
        return PartBodyState(self, boundary)


class HeaderState(StateMachineState):
    def __init__(self, machine=None):
        super(HeaderState, self).__init__(machine)
        self.headers = {}

    # processes lines of text (terminated by CRLF) until a blank line is
    # found.  Calls process_line() on each line.  Calls
    # StreamParser.discard_data() after processing each line (except the last
    # blank line).  Calls header_done() when the blank line has been found.
    def process(self):
        stream = self.machine
        if len(stream.data) < 2:
            stream.needmore = True
            return
        ind = stream.data.find(CRLF)
        if (ind == -1):
            stream.needmore = True
            return
        if (ind == 0):
            #stream.discard_data(2)
            self.header_done()
            return
        self.process_line(ind)
        stream.discard_data(ind+2)

    def header_done(self):
        self.machine.state_transition('header-done')

    def process_line(self, lineend):
        stream = self.machine
        line = stream.data[:lineend]
        linere = re.compile(r'^' +
                            '(?P<name>[' + ALPHANUM + r']+)' +
                            r':' +
                            r'[' + LWSP + r']+' +
                            r'(?P<value>[' + LWSP + r'\.\(\)\?,+*:;/="' + ALPHANUM + r']+)' +
                            '$')
        match = linere.match(line)
        if not match:
            log('HeaderState: no match for line "%s"' % line)
            # Be forgiving...
            #stream.error = True
            return
        #log('matched: "%s"' % stream.data[match.start(0):match.end(0)])
        self.process_header(match.group('name'), match.group('value'))

    def process_header(self, name, value):
        self.headers[name] = value


class PartHeaderState(HeaderState):
    def header_done(self):
        log('part header finished.')
        for k,v in self.headers.items():
            log('  ' + str(k) + " = " + str(v))
        # chomp the CRLF.
        self.machine.discard_data(2)
        self.machine.state_transition('part-header-done')

    def reset_headers(self):
        self.headers = {}

class PartBodyState(StateMachineState):
    def __init__(self, machine, boundary):
        super(PartBodyState, self).__init__(machine)
        self.boundary = boundary
        self.data = None
        self.reset_data()
    def process(self):
        stream = self.machine
        bdy = self.boundary
        if (len(stream.data) < len(bdy) + 8):
            stream.needmore = True
            return
        bdyre = re.compile(CRLF + '--' + bdy + r'(?P<dashdash>--)?' + CRLF)
        match = bdyre.search(stream.data)
        if match:
            # got it!
            start = match.start(0)
            relen = match.end(0) - start
            if (start > 0):
                # chomp the non-matching data off the front first.
                self.handle_data(stream.data[:start])
                stream.discard_data(start)
            stream.discard_data(relen)
            self.part_body_done()
            if match.group('dashdash'):
                # it's the end of the last part.
                self.last_part_body_done()
            else:
                self.start_next_part()
            return

        # boundary not found.  chomp the data up to the last CR, if it
        # exists, or the whole string otherwise.
        ind = stream.data.rfind(CR)
        if (ind == -1):
            # no CR found; chomp the whole string.
            ind = len(stream.data)
        if (ind == 0):
            # string starts with CR, is long enough to be the boundary, but
            # isn't... chomp the whole thing.
            ind = len(stream.data)
        self.handle_data(stream.data[:ind])
        stream.discard_data(ind)

    def get_data_length(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def part_body_done(self):
        self.machine.state_transition('part-body-done')

    def last_part_body_done(self):
        self.machine.state_transition('last-part-body-done')

    def start_next_part(self):
        self.machine.state_transition('start-next-part')

    def handle_data(self, data):
        #log('PartBodyState: handling data ***%s***' % data)
        self.data += data

    def reset_data(self):
        self.data = ''


class FileMultipart(Multipart):
    def __init__(self, fin=None):
        super(FileMultipart, self).__init__(fin)
        self.writefields = {}

    # Overrides Multipart.get_part_body_state().
    #
    # Returns a StateMachineState to parse the body of one part of a
    # multi-part message.  If this part's header contains a
    # Content-Disposition header which has the "name" and "filename"
    # parameters, and the "name" is a key in the "writefields"
    # dictionary, then a FileBodyState() is returned.
    def get_part_body_state(self, boundary):
        #log('FileMultipart: get_part_body_state')
        superme = super(FileMultipart, self)
        if not self.currentpart:
            #log('no current part.')
            return superme.get_part_body_state(boundary)

        key = 'field'
        fnkey = 'filename'
        if not ((key in self.currentpart) and \
                (fnkey in self.currentpart)):
            log('SaveToFile: no field and filename keys.')
            return superme.get_part_body_state(boundary)

        fn = self.currentpart[fnkey]
        field = self.currentpart[key]
        filename = self.get_filename(field, fn, self.currentpart)
        if not filename:
            #log('filename is not in writefields.')
            return superme.get_part_body_state(boundary)
        self.currentpart['local-filename'] = filename
        log('writing this part to file %s' % filename)
        return FileBodyState(self, boundary, filename)

    # Computes the filename to write the data of this "part body" to.
    def get_filename(self, field, filename, currentpart):
        if not field in self.writefields:
            return None
        return self.writefields[field]

    def set_bytes_written(self, nb):
        self.currentpart['bytes-written'] = nb

class FileBodyState(PartBodyState):
    filename = None
    fid = None
    datalen = 0

    def __init__(self, machine, boundary, filename):
        self.filename = filename
        self.fid = open(filename, 'wb')
        super(FileBodyState, self).__init__(machine, boundary)
        self.data = None

    def handle_data(self, data):
        self.datalen += len(data)
        self.fid.write(data)
        self.machine.set_bytes_written(self.datalen)

    def get_data_length(self):
        return self.datalen

    def part_body_done(self):
        self.fid.close()
        self.fid = None
        key = 'UPLOAD_FILE_MODE'
        if key in os.environ:
            mode = int(os.environ['UPLOAD_FILE_MODE'], 0)
            os.chmod(self.filename, mode)
        super(FileBodyState, self).part_body_done()



def multipartmain(args):
    #mp = Multipart(sys.stdin)
    if len(sys.argv):
        sys.argv = sys.argv[1:]
        fin = open(sys.argv[0], 'rb')
    else:
        fin = sys.stdin
    mp = FileMultipart(fin)
    mp.writefields['file'] = '/tmp/contents-file'
    #mp.blocksize = 1
    while mp.readmore():
        pass
    if mp.error:
        print 'Parser failed.'
    else:
        print 'Parser succeeded'
        print 'Message headers:'
        for k,v in mp.headers.items():
            print '  ', k, '=', v
        for p in mp.parts:
            print '      Part:'
            for k,v in p.items():
                print '        ', k, '=', v

if __name__ == '__main__':
    multipartmain(sys.argv)

