#!/usr/bin/env python

# This file was imported into the Astrometry.net suite and *very* heavily
# modified.      The original copyright is below.  Our modifications are
# Copyright 2007 Dustin Lang and licensed under the terms of the GPL
# version 2.


# Copyright (c) 2006, Lindsey Simon <lsimon@finetooth.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#         Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#         Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time
import random
import re
import sys
import cgi
import cgitb
import multipart

#from an.upload.models import UploadedFile

def log(x):
    print >> sys.stderr, x

class Upload(multipart.FileMultipart):
    def __init__(self, fin=None):
        if not fin:
            fin = sys.stdin
        super(Upload, self).__init__(fin)
        self.basedir = None
        self.id_field = None
        # an UploadedFile object
        self.upload = None
        self.uid = None
        self.filename = None
        self.bytes_written = 0
        self.starttime = int(time.time())

    def set_basedir(self, bdir):
        from an.upload.models import UploadedFile
        log('Upload: setting base dir to %s' % bdir)
        self.basedir = bdir
        UploadedFile.set_default_base_dir(self.basedir)
        if self.upload:
            self.upload.set_base_dir(self.basedir)

    def set_id_field(self, field):
        self.id_field = field

    # Sets all filesize columns to the current number of bytes
    # received and written.
    def set_filesize(self):
        if not self.upload:
            return
        self.upload.predictedsize = self.bytes_written
        self.upload.filesize = self.bytes_written
        self.upload.byteswritten = self.bytes_written

    def add_part(self, part):
        super(Upload, self).add_part(part)

        # Here we're looking for the "upload_id" part.
        if 'field' in part and 'data' in part and \
           part['field'] == self.id_field:
            self.id_part(part)

        if 'field' in part and 'filename' in part:
            self.file_part(part)

        #for k in part:
        #    log('part: ' + str(k))


    # called from add_part when the part containing the file is
    # encountered.
    def file_part(self, part):
        from an.upload.models import UploadedFile
        if not self.upload:
            return
        if 'filename' in part:
            log('Filename on the user\'s machine: "%s"' % part['filename'])
            self.upload.userfilename = part['filename']
            self.update_progress()

    # called from add_part when the part containing the upload_id
    # field is encountered.
    def id_part(self, part):
        from an.upload.models import UploadedFile

        uid = part['data']
        if not UploadedFile.isValidId(uid):
            log('Invalid id "%s"' % uid)
            self.error = True
            self.errorstring = 'Invalid upload ID'
            return

        log('Setting ID to %s' % uid)
        self.uid = uid
        
        # We will have read the Content-Length header by now...
        key = 'Content-Length'
        if not key in self.headers:
            log('Content-Length unknown.')
            self.error = True
            self.errorstring = 'Content-Length unknown'
            return
        contentlength = int(self.headers[key])

        predictedsize = contentlength \
                        - (self.bytes_parsed - self.body_offset) \
                        - (len(self.boundary) + 6)
        # magic number 6: the ending boundary token is CRLF "--" BOUNDARY CRLF
        log('Predicted file size: %d' % predictedsize)
        log('ID is %s' % self.uid)

        now = int(time.time())
        self.upload = UploadedFile(uploadid=self.uid, ### HACK - this doesn't seem to work??
                                   starttime=now,
                                   nowtime=now,
                                   predictedsize=predictedsize,
                                   )
        self.upload.uploadid = self.uid
        
        self.filename = self.upload.get_filename()
        if os.path.exists(self.filename):
            log('File for upload ID "%s" already exists.' % id)
            self.error = True
            self.errorstring = 'Upload ID already exists'
            return
        #log('Upload ID "%s" already exists.' % id)
        #self.error = True
        #self.errorstring = 'Upload ID already exists'
        self.update_progress()

    def update_progress(self):
        if not self.upload:
            return
        if self.bytes_written > self.upload.predictedsize:
            self.upload.predictedsize = self.bytes_written
        now = int(time.time())
        self.upload.nowtime = now
        self.upload.byteswritten = self.bytes_written
        self.upload.errorstring = self.errorstring
        self.upload.save()

    # Computes the filename to write the data of this "part body" to.
    def get_filename(self, field, filename, currentpart):
        if not self.uid:
            log('Upload ID is not known.')
            return None
        if not self.filename:
            log('Filename is not known.')
            return None
        return self.filename

    def set_bytes_written(self, nb):
        super(Upload, self).set_bytes_written(nb)
        self.bytes_written = nb



# This method gets called when run from Apache with the
# "SetHandler python-program" directive.
def handler(req):
    from mod_python import apache

    # Need this to compensate for mod_python hiding the environment...
    os.environ.update(req.subprocess_env)

    from an.upload.models import UploadedFile

    req.content_type = 'text/html'
    req.write('<html><head></head><body>')
    req.write('<p>Upload in progress...\n')
    postfix = '</p></body></html>'
    try:
        upload_base_dir = os.environ['UPLOAD_DIR']
    except KeyError:
        upload_base_dir = '/tmp'
    try: 
        upload_id_field = os.environ['UPLOAD_ID_STRING']
    except KeyError:
        upload_id_field = 'upload_id'

    up = Upload(req)
    up.set_basedir(upload_base_dir)
    up.set_id_field(upload_id_field)

    # Re-format the input headers to pass to the parser...
    hdr = ''
    for k,v in req.headers_in.items():
        hdr += str(k) + ': ' + str(v) + '\r\n'
    hdr += '\r\n'
    #log('Input headers:\n' + hdr)

    ok = up.moreinput(hdr)
    if not ok:
        log('Header parsing failed.')
        req.write('upload failed (reading headers)\n' + postfix)
        #up.update_progress()
        return apache.OK
    
    while up.readmore():
        # For testing progress meter with fast connections :)
        #time.sleep(1)
        up.update_progress()

    #fcopy = open('/tmp/in', 'wb')
    #fcopy.write(hdr)
    #while True:
    #   log('reading...')
    #   data = req.read(1024)
    #   fcopy.write(data)
    #   fcopy.flush()
    #   if len(data) == 0:
    #       log('no data')
    #       break
    #   ok = up.moreinput(data)
    #   if not ok:
    #       log('parsing failed.')
    #       break
    #   log('writing progress...')
    #   up.update_progress()
    #fcopy.close()

    #while True:
    #   log('reading...')
    #   ok = up.readmore()
    #   if not ok:
    #       break
    #   log('writing progress...')
    #   up.update_progress()

    if up.error:
        log('Parser failed.')
        req.write('upload failed (reading body)\n' + postfix)
        up.update_progress()
        return apache.OK

    log('Upload succeeded')
    up.set_filesize()
    up.update_progress()
    #log('Message headers:')
    #if up.headers:
    #    for k,v in up.headers.items():
    #        log('       ' + str(k) +  '=' + str(v))
    for p in up.parts:
        log('  Part:')
        for k,v in p.items():
            if k == 'data':
                if ('field' in p) and (p['field'] == upload_id_field):
                    log('    ' + str(k) +  '=' + str(v))
                continue
            log('    ' + str(k) +  '=' + str(v))

    #for p in up.parts:
    formvals = up.get_form_values()
    log('form values:')
    for k,v in formvals.items():
        log('  ' + str(k) +  '=' + str(v))

    filevals = up.get_uploaded_files()
    log('uploaded files:')
    for f in filevals:
        for k,v in f.items():
            log('  ' + str(k) +  '=' + str(v))
        log('')

    if len(filevals) == 0:
        req.write('no such file.\n' + postfix)
        up.error = True
        up.errorstring = 'No such file.'
        up.update_progress()
        return apache.OK

    thefile = filevals[0]
    if thefile['length'] == 0:
        req.write('empty file.\n' + postfix)
        up.error = True
        up.errorstring = 'Empty file.'
        up.update_progress()
        return apache.OK

    req.write('done.</p>\n')
    req.write('<div style="visibility:hidden;" id="upload-succeeded" />')
    req.write('</body></html>')
    return apache.OK


#########
# main  #
#########

if __name__ == '__main__':
    #cgitb.enable()

    # send headers so that the browser doesn't freak out
    print "Content-Type: text/html"
    print

    try:
        upload_base_dir = os.environ['UPLOAD_DIR']
    except KeyError:
        upload_base_dir = '/tmp'
    try: 
        upload_id_field = os.environ['UPLOAD_ID_STRING']
    except KeyError:
        upload_id_field = 'upload_id'

    up = Upload()
    #up.blocksize = 1
    up.set_basedir(upload_base_dir)
    up.set_id_field(upload_id_field)

    while up.readmore():
        up.update_progress()

    #while True:
    #print 'reading...'
    #ok = up.readmore()
    #if not ok:
    #    break
    #print 'writing progress...'
    #sys.stdout.flush()
    #up.update_progress()
    if up.error:
        log('Parser failed.')
        up.abort()
    log('Parser succeeded')
    log('Message headers:')
    if up.headers:
        for k,v in up.headers.items():
            log('       ' + str(k) +  '=' + str(v))
    for p in up.parts:
        log('    Part:')
        for k,v in p.items():
            log('       ' + str(k) +  '=' + str(v))

    print 'Upload complete.'
