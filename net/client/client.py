#!/usr/bin/env python3

# This file is part of the Astrometry.net suite.
# Copyright 2009 Dustin Lang
# Licensed under a 3-clause BSD style license - see LICENSE
# https://github.com/dstndstn/astrometry.net

from __future__ import print_function
import os
import sys
import time
import base64
import shutil

try:
    # py3
    from urllib.parse import urlencode, quote
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
except ImportError:
    # py2
    from urllib import urlencode, quote
    from urllib2 import urlopen, Request, HTTPError

#from exceptions import Exception
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application  import MIMEApplication

from email.encoders import encode_noop

import json
def json2python(data):
    try:
        return json.loads(data)
    except:
        pass
    return None
python2json = json.dumps

class MalformedResponse(Exception):
    pass
class RequestError(Exception):
    pass

class Client(object):
    default_url = 'https://nova.astrometry.net/api/'

    def __init__(self,
                 apiurl = default_url):
        self.session = None
        self.apiurl = apiurl

    def get_url(self, service):
        return self.apiurl + service

    def send_request(self, service, args={}, file_args=None):
        '''
        service: string
        args: dict
        '''
        if self.session is not None:
            args.update({ 'session' : self.session })
        print('Python:', args)
        json = python2json(args)
        print('Sending json:', json)
        url = self.get_url(service)
        print('Sending to URL:', url)

        # If we're sending a file, format a multipart/form-data
        if file_args is not None:
            import random
            boundary_key = ''.join([random.choice('0123456789') for i in range(19)])
            boundary = '===============%s==' % boundary_key
            headers = {'Content-Type':
                       'multipart/form-data; boundary="%s"' % boundary}
            data_pre = (
                '--' + boundary + '\n' +
                'Content-Type: text/plain\r\n' +
                'MIME-Version: 1.0\r\n' +
                'Content-disposition: form-data; name="request-json"\r\n' +
                '\r\n' +
                json + '\n' +
                '--' + boundary + '\n' +
                'Content-Type: application/octet-stream\r\n' +
                'MIME-Version: 1.0\r\n' +
                'Content-disposition: form-data; name="file"; filename="%s"' % file_args[0] +
                '\r\n' + '\r\n')
            data_post = (
                '\n' + '--' + boundary + '--\n')
            data = data_pre.encode() + file_args[1] + data_post.encode()

        else:
            # Else send x-www-form-encoded
            data = {'request-json': json}
            print('Sending form data:', data)
            data = urlencode(data)
            data = data.encode('utf-8')
            print('Sending data:', data)
            headers = {}

        request = Request(url=url, headers=headers, data=data)

        try:
            f = urlopen(request)
            print('Got reply HTTP status code:', f.status)
            txt = f.read()
            print('Got json:', txt)
            result = json2python(txt)
            print('Got result:', result)
            stat = result.get('status')
            print('Got status:', stat)
            if stat == 'error':
                errstr = result.get('errormessage', '(none)')
                raise RequestError('server error message: ' + errstr)
            return result
        except HTTPError as e:
            print('HTTPError', e)
            txt = e.read()
            open('err.html', 'wb').write(txt)
            print('Wrote error text to err.html')

    def login(self, apikey):
        args = { 'apikey' : apikey }
        result = self.send_request('login', args)
        sess = result.get('session')
        print('Got session:', sess)
        if not sess:
            raise RequestError('no session in result')
        self.session = sess

    def _get_upload_args(self, **kwargs):
        args = {}
        for key,default,typ in [('allow_commercial_use', 'd', str),
                                ('allow_modifications', 'd', str),
                                ('publicly_visible', 'y', str),
                                ('scale_units', None, str),
                                ('scale_type', None, str),
                                ('scale_lower', None, float),
                                ('scale_upper', None, float),
                                ('scale_est', None, float),
                                ('scale_err', None, float),
                                ('center_ra', None, float),
                                ('center_dec', None, float),
                                ('parity',None,int),
                                ('radius', None, float),
                                ('downsample_factor', None, int),
                                ('positional_error', None, float),
                                ('tweak_order', None, int),
                                ('crpix_center', None, bool),
                                ('invert', None, bool),
                                ('use_sextractor', None, bool),
                                ('image_width', None, int),
                                ('image_height', None, int),
                                ('x', None, list),
                                ('y', None, list),
                                ('album', None, str),
                                ]:
            if key in kwargs:
                val = kwargs.pop(key)
                val = typ(val)
                args.update({key: val})
            elif default is not None:
                args.update({key: default})
        print('Upload args:', args)
        return args

    def url_upload(self, url, **kwargs):
        args = dict(url=url)
        args.update(self._get_upload_args(**kwargs))
        result = self.send_request('url_upload', args)
        return result

    def upload(self, fn=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        if fn is not None:
            try:
                f = open(fn, 'rb')
                file_args = (fn, f.read())
            except IOError:
                print('File %s does not exist' % fn)
                raise
        return self.send_request('upload', args, file_args)

    def submission_images(self, subid):
        result = self.send_request('submission_images', {'subid':subid})
        return result.get('image_ids')

    def overlay_plot(self, service, outfn, wcsfn, wcsext=0):
        from astrometry.util import util as anutil
        wcs = anutil.Tan(wcsfn, wcsext)
        params = dict(crval1 = wcs.crval[0], crval2 = wcs.crval[1],
                      crpix1 = wcs.crpix[0], crpix2 = wcs.crpix[1],
                      cd11 = wcs.cd[0], cd12 = wcs.cd[1],
                      cd21 = wcs.cd[2], cd22 = wcs.cd[3],
                      imagew = wcs.imagew, imageh = wcs.imageh)
        result = self.send_request(service, {'wcs':params})
        print('Result status:', result['status'])
        plotdata = result['plot']
        plotdata = base64.b64decode(plotdata)
        open(outfn, 'wb').write(plotdata)
        print('Wrote', outfn)

    def sdss_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('sdss_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def galex_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('galex_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def myjobs(self):
        result = self.send_request('myjobs/')
        return result['jobs']

    def job_status(self, job_id, justdict=False):
        result = self.send_request('jobs/%s' % job_id)
        if justdict:
            return result
        stat = result.get('status')
        if stat == 'success':
            result = self.send_request('jobs/%s/calibration' % job_id)
            print('Calibration:', result)
            result = self.send_request('jobs/%s/tags' % job_id)
            print('Tags:', result)
            result = self.send_request('jobs/%s/machine_tags' % job_id)
            print('Machine Tags:', result)
            result = self.send_request('jobs/%s/objects_in_field' % job_id)
            print('Objects in field:', result)
            result = self.send_request('jobs/%s/annotations' % job_id)
            print('Annotations:', result)
            result = self.send_request('jobs/%s/info' % job_id)
            print('Calibration:', result)

        return stat

    def annotate_data(self,job_id):
        """
        :param job_id: id of job
        :return: return data for annotations
        """
        result = self.send_request('jobs/%s/annotations' % job_id)
        return result

    def sub_status(self, sub_id, justdict=False):
        result = self.send_request('submissions/%s' % sub_id)
        if justdict:
            return result
        return result.get('status')

    def jobs_by_tag(self, tag, exact):
        exact_option = 'exact=yes' if exact else ''
        result = self.send_request(
            'jobs_by_tag?query=%s&%s' % (quote(tag.strip()), exact_option),
            {},
        )
        return result

class ClientRunnerOptions(object):
    def __init__(self, **entries):
        self.server = Client.default_url
        self.public = 'y'
        self.allow_mod = 'd'
        self.allow_commercial = 'd'

        self.__dict__.update(entries)

    def __getattr__(self, name):
        try:
            return object.__getattr__(self, name)
        except:
            return None

def run_client(opt):
    args = {}
    args['apiurl'] = opt.server
    c = Client(**args)
    c.login(opt.apikey)

    if opt.upload or opt.upload_url or opt.upload_xy:
        if opt.wcs or opt.kmz or opt.newfits or opt.corr or opt.annotate:
            opt.wait = True

        kwargs = dict(
            allow_commercial_use=opt.allow_commercial,
            allow_modifications=opt.allow_mod,
            publicly_visible=opt.public)
        if opt.scale_lower and opt.scale_upper:
            kwargs.update(scale_lower=opt.scale_lower,
                          scale_upper=opt.scale_upper,
                          scale_type='ul')
        elif opt.scale_est and opt.scale_err:
            kwargs.update(scale_est=opt.scale_est,
                          scale_err=opt.scale_err,
                          scale_type='ev')
        elif opt.scale_lower or opt.scale_upper:
            kwargs.update(scale_type='ul')
            if opt.scale_lower:
                kwargs.update(scale_lower=opt.scale_lower)
            if opt.scale_upper:
                kwargs.update(scale_upper=opt.scale_upper)

        for key in ['scale_units', 'center_ra', 'center_dec', 'radius',
                    'downsample_factor', 'positional_error', 'tweak_order', 'crpix_center',
                    'album', 'invert', 'use_sextractor']:
            if getattr(opt, key) is not None:
                kwargs[key] = getattr(opt, key)
        if opt.parity is not None:
            kwargs.update(parity=int(opt.parity))

        if opt.upload:
            upres = c.upload(opt.upload, **kwargs)
        if opt.upload_xy:
            from astrometry.util.fits import fits_table
            T = fits_table(opt.upload_xy)
            kwargs.update(x=[float(x) for x in T.x], y=[float(y) for y in T.y])
            upres = c.upload(**kwargs)
        if opt.upload_url:
            upres = c.url_upload(opt.upload_url, **kwargs)

        stat = upres['status']
        if stat != 'success':
            print('Upload failed: status', stat)
            print(upres)
            sys.exit(-1)

        opt.sub_id = upres['subid']

    if opt.wait:
        if opt.solved_id is None:
            if opt.sub_id is None:
                print("Can't --wait without a submission id or job id!")
                sys.exit(-1)

            while True:
                stat = c.sub_status(opt.sub_id, justdict=True)
                print('Got status:', stat)
                jobs = stat.get('jobs', [])
                if len(jobs):
                    for j in jobs:
                        if j is not None:
                            break
                    if j is not None:
                        print('Selecting job id', j)
                        opt.solved_id = j
                        break
                time.sleep(5)

        while True:
            stat = c.job_status(opt.solved_id, justdict=True)
            print('Got job status:', stat)
            if stat.get('status','') in ['success']:
                success = (stat['status'] == 'success')
                break
            elif stat.get('status','') in ['failure']:
                print("Image solving failed")
                sys.exit(-1)
            time.sleep(5)

    if opt.solved_id:
        # we have a jobId for retrieving results
        retrieveurls = []
        if opt.wcs:
            # We don't need the API for this, just construct URL
            url = opt.server.replace('/api/', '/wcs_file/%i' % opt.solved_id)
            retrieveurls.append((url, opt.wcs))
        if opt.kmz:
            url = opt.server.replace('/api/', '/kml_file/%i/' % opt.solved_id)
            retrieveurls.append((url, opt.kmz))
        if opt.newfits:
            url = opt.server.replace('/api/', '/new_fits_file/%i/' % opt.solved_id)
            retrieveurls.append((url, opt.newfits))
        if opt.corr:
            url = opt.server.replace('/api/', '/corr_file/%i' % opt.solved_id)
            retrieveurls.append((url, opt.corr))

        for url,fn in retrieveurls:
            print('Retrieving file from', url, 'to', fn)
            with urlopen(url) as r:
                with open(fn, 'wb') as w:
                    shutil.copyfileobj(r, w)
            print('Wrote to', fn)

        if opt.annotate:
            result = c.annotate_data(opt.solved_id)
            with open(opt.annotate,'w') as f:
                f.write(python2json(result))

    if opt.wait:
        # behaviour as in old implementation
        opt.sub_id = None

    if opt.sdss_wcs:
        (wcsfn, outfn) = opt.sdss_wcs
        c.sdss_plot(outfn, wcsfn)
    if opt.galex_wcs:
        (wcsfn, outfn) = opt.galex_wcs
        c.galex_plot(outfn, wcsfn)

    if opt.sub_id:
        print(c.sub_status(opt.sub_id))
    if opt.job_id:
        print(c.job_status(opt.job_id))

    if opt.jobs_by_tag:
        tag = opt.jobs_by_tag
        print(c.jobs_by_tag(tag, None))
    if opt.jobs_by_exact_tag:
        tag = opt.jobs_by_exact_tag
        print(c.jobs_by_tag(tag, 'yes'))

    if opt.myjobs:
        jobs = c.myjobs()
        print(jobs)

def get_args():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('--server', dest='server', default=Client.default_url,
                      help='Set server base URL (eg, %default)')
    parser.add_option('--apikey', '-k', dest='apikey',
                      help='API key for Astrometry.net web service; if not given will check AN_API_KEY environment variable')
    parser.add_option('--upload', '-u', dest='upload', help='Upload a file')
    parser.add_option('--upload-xy', dest='upload_xy', help='Upload a FITS x,y table as JSON')
    parser.add_option('--wait', '-w', dest='wait', action='store_true', help='After submitting, monitor job status')
    parser.add_option('--wcs', dest='wcs', help='Download resulting wcs.fits file, saving to given filename; implies --wait if --urlupload or --upload')
    parser.add_option('--newfits', dest='newfits', help='Download resulting new-image.fits file, saving to given filename; implies --wait if --urlupload or --upload')
    parser.add_option('--corr', dest='corr', help='Download resulting corr.fits file, saving to given filename; implies --wait if --urlupload or --upload')
    parser.add_option('--kmz', dest='kmz', help='Download resulting kmz file, saving to given filename; implies --wait if --urlupload or --upload')
    parser.add_option('--annotate','-a',dest='annotate',help='store information about annotations in give file, JSON format; implies --wait if --urlupload or --upload')
    parser.add_option('--urlupload', '-U', dest='upload_url', help='Upload a file at specified url')
    parser.add_option('--scale-units', dest='scale_units',
                      choices=('arcsecperpix', 'arcminwidth', 'degwidth', 'focalmm'), help='Units for scale estimate ("arcsecperpix", "arcminwidth", "degwidth", or "focalmm")')
    #parser.add_option('--scale-type', dest='scale_type',
    #                  choices=('ul', 'ev'), help='Scale bounds: lower/upper or estimate/error')
    parser.add_option('--scale-lower', dest='scale_lower', type=float, help='Scale lower-bound')
    parser.add_option('--scale-upper', dest='scale_upper', type=float, help='Scale upper-bound')
    parser.add_option('--scale-est', dest='scale_est', type=float, help='Scale estimate')
    parser.add_option('--scale-err', dest='scale_err', type=float, help='Scale estimate error (in PERCENT), eg "10" if you estimate can be off by 10%')
    parser.add_option('--ra', dest='center_ra', type=float, help='RA center')
    parser.add_option('--dec', dest='center_dec', type=float, help='Dec center')
    parser.add_option('--radius', dest='radius', type=float, help='Search radius around RA,Dec center')
    parser.add_option('--downsample', dest='downsample_factor', type=int, help='Downsample image by this factor')
    parser.add_option('--positional_error', dest='positional_error', type=float, help='How many pixels a star may be from where it should be.')
    parser.add_option('--parity', dest='parity', choices=('0','1'), help='Parity (flip) of image')
    parser.add_option('--tweak-order', dest='tweak_order', type=int, help='SIP distortion order (default: 2)')
    parser.add_option('--crpix-center', dest='crpix_center', action='store_true', default=None, help='Set reference point to center of image?')
    parser.add_option('--invert', action='store_true', default=None, help='Invert image before detecting sources -- for white-sky, black-stars images')
    parser.add_option('--use-source-extractor', dest='use_sextractor', action='store_true', default=None, help='Use SourceExtractor for source detection?')
    parser.add_option('--image-width', type=int, help='Set image width for x,y lists')
    parser.add_option('--image-height', type=int, help='Set image height for x,y lists')
    parser.add_option('--album', type=str, help='Add image to album with given title string')
    parser.add_option('--sdss', dest='sdss_wcs', nargs=2, help='Plot SDSS image for the given WCS file; write plot to given PNG filename')
    parser.add_option('--galex', dest='galex_wcs', nargs=2, help='Plot GALEX image for the given WCS file; write plot to given PNG filename')
    parser.add_option('--jobid', '-i', dest='solved_id', type=int,help='retrieve result for jobId instead of submitting new image')
    parser.add_option('--substatus', '-s', dest='sub_id', help='Get status of a submission')
    parser.add_option('--jobstatus', '-j', dest='job_id', help='Get status of a job')
    parser.add_option('--jobs', '-J', dest='myjobs', action='store_true', help='Get all my jobs')
    parser.add_option('--jobsbyexacttag', '-T', dest='jobs_by_exact_tag', help='Get a list of jobs associated with a given tag--exact match')
    parser.add_option('--jobsbytag', '-t', dest='jobs_by_tag', help='Get a list of jobs associated with a given tag')
    parser.add_option( '--private', '-p',
        dest='public',
        action='store_const',
        const='n',
        default='y',
        help='Hide this submission from other users')
    parser.add_option('--allow_mod_sa','-m',
        dest='allow_mod',
        action='store_const',
        const='sa',
        default='d',
        help='Select license to allow derivative works of submission, but only if shared under same conditions of original license')
    parser.add_option('--no_mod','-M',
        dest='allow_mod',
        action='store_const',
        const='n',
        default='d',
        help='Select license to disallow derivative works of submission')
    parser.add_option('--no_commercial','-c',
        dest='allow_commercial',
        action='store_const',
        const='n',
        default='d',
        help='Select license to disallow commercial use of submission')

    opt,args = parser.parse_args()

    if opt.apikey is None:
        # try the environment
        opt.apikey = os.environ.get('AN_API_KEY', None)
    if opt.apikey is None:
        parser.print_help()
        print()
        print('You must either specify --apikey or set AN_API_KEY')
        sys.exit(-1)

    return opt

if __name__ == '__main__':
    print("Running with args %s"%sys.argv)

    opt = get_args()

    run_client(opt)
