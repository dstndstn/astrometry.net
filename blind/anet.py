#!/bin/env python

"""
   anet -- command line interface to astrometry.net website. 
           Finds WCS solutions to your FITS images.
   
   Written by J. Bloom (jbloom@astro.berkeley.edu)
   June 2008 Copyright (c) Josh Bloom
   
   This program is free software licensed with the GNU Public License Version 3.
   For a full copy of the license please go here http://www.gnu.org/licenses/licenses.html#GPL
   
   SETUP:
      * should work with python2.4 or higher (only tested on 2.5)
      * you need pyfits.py (> 1.3) in your PYTHONPATH (available from STSCI). This will in turn require you
        to have numpy (or numarray) installed. If you dont get any import errors when you
          #user> python 
             >>> import pyfits
        then you are ok.
      * edit the variables astrometry_username and astrometry_email as appropriate
      * you might want to edit job_dict, since I assume a platescale around 0.1 - 2.5"/pix to help with speed up.
        you could also speed things along if you know the parity of your images
         
   USAGE: python anet.py {fitsfile.fits | *.fits}

   TODO:  add an arg parser to allow user to tweak control over the URL calls
      
"""

import os, urllib, sys, datetime, copy
import urllib2, cookielib
import threading
import mimetools, mimetypes
import os, stat
from cStringIO import StringIO
import pyfits 
import time

__author__  = "J. S. Bloom"
__version__ = "1.0"

# Put your name and the email you registered with astrometry.net here.
astrometry_username = "---Your Name Here---"
astrometry_email    = "---Your Email Here---"                  


## some helper classes first.
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

class MultipartPostHandler(urllib2.BaseHandler):
    
    handler_order = urllib2.HTTPHandler.handler_order - 10 # needs to run first
    # Controls how sequences are uncoded. If true, elements may be given multiple values by
    #  assigning a sequence.
    doseq = 1

    def http_request(self, request):
        data = request.get_data()
        if data is not None and type(data) != str:
            v_files = []
            v_vars = []
            try:
                 for(key, value) in data.items():
                     if type(value) == file:
                         v_files.append((key, value))
                     else:
                         v_vars.append((key, value))
            except TypeError:
                systype, value, traceback = sys.exc_info()
                raise TypeError, "not a valid non-string sequence or mapping object", traceback

            if len(v_files) == 0:
                data = urllib.urlencode(v_vars, self.doseq)
            else:
                boundary, data = self.multipart_encode(v_vars, v_files)

                contenttype = 'multipart/form-data; boundary=%s' % boundary
                if(request.has_header('Content-Type')
                   and request.get_header('Content-Type').find('multipart/form-data') != 0):
                    print "Replacing %s with %s" % (request.get_header('content-type'), 'multipart/form-data')
                request.add_unredirected_header('Content-Type', contenttype)

            request.add_data(data)
        return request

    def multipart_encode(vars, files, boundary = None, buf = None):
        if boundary is None:
            boundary = mimetools.choose_boundary()
        if buf is None:
            buf = StringIO()
        for(key, value) in vars:
            buf.write('--%s\r\n' % boundary)
            buf.write('Content-Disposition: form-data; name="%s"' % key)
            buf.write('\r\n\r\n' + str(value) + '\r\n')
        for(key, fd) in files:
            file_size = os.fstat(fd.fileno())[stat.ST_SIZE]
            filename = fd.name.split('/')[-1]
            contenttype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            buf.write('--%s\r\n' % boundary)
            buf.write('Content-Disposition: form-data; name="%s"; filename="%s"\r\n' % (key, filename))
            buf.write('Content-Type: %s\r\n' % contenttype)
            # buffer += 'Content-Length: %s\r\n' % file_size
            fd.seek(0)
            buf.write('\r\n' + fd.read() + '\r\n')
        buf.write('--' + boundary + '--\r\n\r\n')
        buf = buf.getvalue()
        return boundary, buf
    multipart_encode = Callable(multipart_encode)

    https_request = http_request

opener = urllib2.build_opener(MultipartPostHandler)

results = []

class ImageContainer:
	astrometry_dot_net_url = "http://live.astrometry.net/"
	
	def __init__(self,call_string,name,verbose=True):
		self.reqid       = None
		self.name  = name
		self.call_string = call_string
		self.status      = "unknown"
		self.stat        = "unknown"
		self.newhead = None
		self.verbose = verbose

		self._make_request()
		self._get_req_id()
		self._get_job_status()
		self._get_new_wcs()
		self._replace_new_wcs()
		
		results.append((self.name,self.status,self.stat,self.reqid))
		
	def _make_request(self):
		
		self.time_started= datetime.datetime.now()
		self.status      = "submitted"
		if self.verbose:
			print "** Submitting WCS request for image = %s" % self.name
			print datetime.datetime.now()
		self.req  = opener.open(self.astrometry_dot_net_url + "index.php", self.call_string)
		if self.verbose:
			print "   (Finished uploading image = %s)" % self.name
			print datetime.datetime.now()
		self.status      = "returned"
	
	def _get_req_id(self):
		if self.status != "returned":
			self.status = "failed"
		tmp = self.req.read().splitlines()
		gotit = False
		for i in range(len(tmp)):
			if tmp[i].find("<title>") != -1:
				gotit = True
				break
		if gotit:
			tmp = tmp[i+1].split("Job ")
			self.reqid = tmp[1].split()[0]
			self.status = "got req id"
		return
	
	def _get_job_status(self,timeout=200.0):
		if self.status != "got req id":
			print "bad job status"
			return
			
		got_status = False
		start = datetime.datetime.now()
		call = self.astrometry_dot_net_url + "status.php?" + urllib.urlencode({"job": self.reqid})
		timeout = datetime.timedelta(seconds=timeout)
		if self.verbose:
			print "   If you'd like to check the status of %s, go to: \n    %s" % (self.name,call)
		while not got_status and datetime.datetime.now() - start < timeout:
			f = urllib.urlopen(call)
			tmp = f.readlines()
			for i in range(len(tmp)):
				if tmp[i].find("<tr><td>Status:</td><td>") != -1:
					self.stat = tmp[i+1].split("</td>")[0]
					if self.stat in ["Failed", "Solved"]:
						got_status = True
						break
			time.sleep(1)
		# print "   Status of file %s (req id = %s).... %s" % (self.name,self.reqid,self.stat)

	def _get_new_wcs(self):
		#print "here1 (%s)" % self.stat
		
		if self.stat != "Solved":
			return
		call = self.astrometry_dot_net_url + "status.php?" + urllib.urlencode({"job": self.reqid, "get": "wcs.fits"})
		self.newhead = "wcs-" + self.reqid + ".fits"
		urllib.urlretrieve(call,self.newhead)
	
	def _replace_new_wcs(self,delnew=True):

		if self.newhead is None or self.stat != "Solved":
			return
		
		if self.name.find(".fits") == -1:
			## not a fits image
			self.status = "wcs=%s" % self.newhead
			return
		wascompressed = False
		if self.name.endswith(".gz"):
			os.system("gunzip " + self.name)
			wascompressed = True
			self.name = self.name.split(".gz")[0]
		tmp = pyfits.open(self.name,"update")
		tmp1 = pyfits.open(self.newhead,"readonly")
		tmp2 = tmp1[0].header
		del tmp2["SIMPLE"]
		del tmp2["BITPIX"]
		del tmp2["NAXIS"]
		
		## copy the header over
		tmp1.close()
		for c in tmp2.ascardlist():
			tmp[0].header.update(c.key,c.value,c.comment)

		tmp.verify("silentfix")
		tmp.close(output_verify='warn')
		if delnew:
			os.remove(self.newhead)
		if wascompressed:
			os.system("gzip " + self.name)
			self.name += ".gz"

		if self.verbose:	
			print "Finished WCS request for image %s (%s)" % (self.name,self.stat)
		
		
class AstrometrySolver:
	
	job_dict = {"uname": astrometry_username,"email": astrometry_email, "fsunit" :"arcsecperpix",\
		"fstype-ul": 1, "fsu": 1.1, "fsl": 0.9, "xysrc": "img", "parity": 2, "index": "10arcmin", "tweak": 1,\
		"tweak_order": 2, "imgfile": "","submit": "Submit"}

	def __init__(self,verbose=True):
		self.verbose= verbose
		self.threads = []
		
	def _make_request(self,imgfile=None,pixel_size_range = [0.2,1.1], tweak_astrometry=True):
		
		if imgfile is None or not os.path.isfile(imgfile):
			if self.verbose:
				print "! imgfile is bad"
			return
		tmp =copy.copy(self.job_dict)
		tmp.update({"imgfile": open(imgfile,"rb"), "fsl": pixel_size_range[0], "fsu": pixel_size_range[1], "tweak": int(tweak_astrometry)})
		if imgfile.find(".fits") == -1:
			tmp.update({"index": "auto"})
		self.threads.append(threading.Timer(0.0,ImageContainer,args=[tmp,imgfile],kwargs={'verbose': self.verbose}))
		self.threads[-1].start()

		#print opener.open(self.astrometry_dot_net_url, tmp).read()
		
		#params = urllib.urlencode(tmp)
		#print self.astrometry_dot_net_url + "?" + params
		#f = url
	
	def get_wcs(self,imlist=None,howmany_at_a_time=5,pixel_size_range = [0.2,1.1]):
		
		print "Verbose is set to %s" % repr(self.verbose)
		if imlist is None:
			return
		
		if type(imlist) == type("a"):
			self._make_request(imlist,pixel_size_range = pixel_size_range)
			self.threads[-1].join()
			
		if type(imlist) == type([]):
			nsets = len(imlist)/howmany_at_a_time + 1
			for i in range(nsets):
				# print (i, (i*howmany_at_a_time),((1 + i)*howmany_at_a_time))
				for im in imlist[(i*howmany_at_a_time):((i+1)*howmany_at_a_time)]:
					if im.find(".fits") == -1:
						## probably cannot trust the image scale to be small
						self._make_request(im,pixel_size_range=[0.1,500],tweak_astrometry=False)
					else:
						self._make_request(im)
				## wait until the last guy finishes before firing off more
				self.threads[-1].join()
		
		self.threads[-1].join()
	
	def __str__(self):
		a = "RESULTS OF THE SUBMITTED JOBS\n"
		a += "%-45s\t%-10s\t%-10s\t%-15s\n" % ("name","status","stat","reqid")
		a += "*"*100 + "\n"
		for r in results:
			a += "%-45s\t'%-10s'\t%-10s\t%-15s\n" % (os.path.basename(r[0]),r[1],r[2],r[3])
		return a
		#results.append((self.name,self.status,self.stat,self.reqid))
		
if __name__ == "__main__":
	
	if len(sys.argv) <= 1:
		print __doc__
	else:
		a = AstrometrySolver()
		a.get_wcs(imlist=sys.argv[1:],pixel_size_range = [0.2,2])
		print a
	
		
