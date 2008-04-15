import logging
import math
import os
import re

from django import newforms as forms
from django.newforms.util import ErrorList
from django.db import models
from django.http import HttpResponse, HttpResponseRedirect
from django.newforms import widgets, ValidationError, form_for_model
from django.template import Context, RequestContext, loader

#from an.portal.models import Job, AstroField, UserPreferences

from an import gmaps_config
from an import settings
from an.vo.log import log
from an.vo.votable import *
from an.util.file import file_size

from an.vo.models import Image as voImage

#intersects = [ 'COVERS', 'ENCLOSED', 'CENTER', 'OVERLAPS' ]

class PointedTable(VOTable):
    def __init__(self):
        super(PointedTable, self).__init__()

        flds = [
            # MUST
            VOField('image_title', 'char', '*', 'VOX:Image_Title'),
            VOField('image_format', 'char', '*', 'VOX:Image_Format'),
            VOField('image_url', 'char', '*', 'VOX:Image_AccessReference'),
            VOField('RA_center', 'double', None, 'POS_EQ_RA_MAIN'),
            VOField('DEC_center', 'double', None, 'POS_EQ_DEC_MAIN'),
            VOField('naxis', 'int', '*', 'VOX:Image_Naxis'),
            VOField('scale', 'double', '*', 'VOX:Image_Scale'),

            # SHOULD
            VOField('instrument', 'char', '*', 'INST_ID'),
            VOField('jdate', 'double', None, 'VOX:Image_MJDateObs'),
            VOField('crpix', 'double', '*', 'VOX:WCS_CoordRefPixel'),
            VOField('crval', 'double', '*', 'VOX:WCS_CoordRefValue'),
            VOField('cd', 'double', '*', 'VOX:WCS_CDMatrix'),
            VOField('filesize', 'int', None, 'VOX:Image_FileSize'),

            # VOField('bandpass', 'char', '*', 'VOX:BandPass_ID'),
            # VOField('bandpass_unit', 'char', '*', 'VOX:BandPass_Unit'),
            # VOField('bandpass_ref', 'double', None, 'VOX:BandPass_RefValue'),
            # VOField('bandpass_hi', 'double', None, 'VOX:BandPass_HiLimit'),
            # VOField('bandpass_lo', 'double', None, 'VOX:BandPass_LoLimit'),

            VOField('pixflags', 'char', '*', 'VOX:Image_PixFlags'),

            # MAY
            # the Equinox (not required for ICRS) of the coordinate system used for the image world coordinate system (WCS).
            # This should match whatever is in the image WCS and may differ from the default ICRS coordinates used elsewhere.
            # VOField('equinox', 'double', None, 'VOX:STC_CoordEquinox')

            # the minimum time to live in seconds of the access reference.
            # VOField('ttl', 'int', None, 'VOX:Image_AccessRefTTL')
            ]
        for f in flds:
            self.add_field(f)

        params = [
            # MUST
            VOParam('naxes', 'int', None, 'VOX:Image_Naxes', 2),

            # SHOULD
            VOParam('proj', 'char', '3', 'VOX:STC_CoordProjection', 'TAN'),
            VOParam('refframe', 'char', '*', 'VOX:STC_CoordRefFrame', 'ICRS'),
        ]
        for p in params:
            self.add_param(p)


class PointedRow(VORow):
    # --------
    #   MUST
    # --------
    # a short (usually one line) description of the image
    # identifying the image source (e.g., survey name), object name or field coordinates,
    # bandpass/filter, and so forth.
    image_title = None

    # the MIME-type of the object associated with the image acref, e.g., "image/fits", "text/html", and so forth.
    image_format = None

    # specifying the URL to be used to access or retrieve the image.
    # Since the URL will often contain metacharacters the URL is normally enclosed in an XML CDATA section
    # (<![CDATA[...]]>) or otherwise encoded to escape any embedded metacharacters.
    image_url = None

    # the ICRS right-ascension of the center of the image.
    ra_center = None

    # the ICRS declination of the center of the image.
    dec_center = None

    # the length in pixels of each image axis.
    naxis = [ 0, 0 ]

    # the scale in degrees per pixel of each image axis.
    scale = [ 0, 0 ]
    
    # --------
    #  SHOULD
    # --------

    # the instrument or instruments used to make the observation, e.g., STScI.HST.WFPC2.
    instrument = ''

    # the mean modified Julian date of the observation.
    # By "mean" we mean the midpoint of the observation in terms of normalized exposure times:
    # this is the "characteristic observation time" and is independent of observation duration.
    jdate = 0

    # the image pixel coordinates of the WCS reference pixel. This is identical to "CRPIX" in FITS WCS.
    crpix = [ 0, 0 ]

    # the world coordinates of the WCS reference pixel. This is identical to "CRVAL" in FITS WCS.
    crval = [ 0, 0 ]

    # the WCS CD matrix. This is identical to the "CD" term in FITS WCS, and defines the scale and rotation
    # (among other things) of the image.
    # Matrix elements should be ordered as CD[i,j] = [1,1], [1,2], [2,1], [2,2].
    cd = [ 0, 0, 0, 0 ]

    # the actual or estimated size of the encoded image in bytes (not pixels!). This is useful for image selection
    # and for optimizing distributed computations.
    image_filesize = 0

    # the bandpass by name (e.g., "V", "SDSS_U", "K", "K-Band", etc.).
    # bandpass = None

    # the units used to represent spectral values, selected from "meters", "hertz", and "keV".
    # No other units are permitted here; the client application may of course present a wider
    # range of units in the user interface.
    # bandpass_units = None

    # the characteristic (reference) frequency, wavelength, or energy for the bandpass model.
    # bandpass_ref = None

    # the upper limit of the bandpass.
    # bandpass_hi = None

    # the upper limit of the bandpass.
    # bandpass_lo = None

    # the type of processing done by the image service to produce an output image pixel.
    # The string value should be formed from some combination of the following character codes:
    # - C -- The image pixels were copied from a source image without change, as when an atlas image or cutout is returned.
    # - F -- The image pixels were computed by resampling an existing image, e.g., to rescale or reproject the data, and were filtered by an interpolator.
    # - X -- The image pixels were computed by the service directly from a primary data set hence were not filtered by an interpolator.
    # - Z -- The image pixels contain valid flux (intensity) values, e.g., if the pixels were resampled a flux-preserving interpolator was used.
    # - V -- The image pixels contain some unspecified visualization of the data, hence are suitable for display but not for numerical analysis.
    #
    # For example, a typical image cutout service would have PixFlags="C", whereas a mosaicing service operating on precomputed images
    # might have PixFlags="FZ". A preview page, graphics image, or a pixel mask might have PixFlags="V".
    # An image produced by sampling and reprojecting a high energy event list might have PixFlags="X".
    # If not specified, PixFlags="C" is assumed. 
    pixflags = 'V'

    def __init__(self, voimage=None):
        super(PointedRow, self).__init__()
        if voimage:
            field = voimage.field
            #self.image_title = 'Field_%i' % field.id
            self.image_title = voimage.image_title
            self.image_format = field.content_type()
            self.image_url = get_image_url(voimage.id)
            self.ra_center  = voimage.ra_center
            self.dec_center = voimage.dec_center
            self.instrument = voimage.instrument
            self.jdate = voimage.jdate
            wcs = voimage.wcs
            self.naxis = [wcs.imagew, wcs.imageh]
            self.crval = [wcs.crval1, wcs.crval2]
            self.crpix = [wcs.crpix1, wcs.crpix2]
            self.cd = [wcs.cd11, wcs.cd12, wcs.cd21, wcs.cd22]
            self.image_filesize = file_size(field.filename())
            # ???
            self.scale = [math.hypot(wcs.cd11, wcs.cd12),
                          math.hypot(wcs.cd21, wcs.cd22)]
            self.pixflags = 'V'


    def get_children(self):
        # This array must be in the same order as the "flds" array in
        # PointedTable.
        coldata = [ self.image_title, self.image_format, self.image_url,
                    self.ra_center, self.dec_center, self.naxis,
                    self.scale, self.instrument, self.jdate, self.crpix,
                    self.crval, self.cd, self.image_filesize, self.pixflags ]
        children = []
        for c in coldata:
            children.append(VOColumn(c))
        return children

def get_image_url(imageid):
    # HACKEROO!!!
    return 'http://oven.cosmo.fas.nyu.edu:8888/vo/getimage/?voimageid=' + str(imageid)


floatre = r'[+-]?[0-9]*(\.[0-9]*)?([eE][+-]?[0-9]+)?'
csf_re = re.compile(r'^' + floatre + r'(,' + floatre + r')*' + r'$')

class CommaSeparatedFloatField(forms.RegexField):
    def __init__(self, max_length=None, min_length=None, *args, **kwargs):
        super(CommaSeparatedFloatField, self).__init__(
            csf_re, max_length, min_length,
            'Must be comma-separated floating-point values',
            *args, **kwargs)

    def clean(self, value):
        val = super(CommaSeparatedFloatField, self).clean(value)
        strvals = val.split(',')
        vals = map(float, strvals)
        return vals

class FormatField(forms.CharField):
    def clean(self, val):
        if val is None:
            return [ 'ALL' ]
        formats = []
        terms = str(val).split(',')
        ct_re = re.compile(r'^(image/(png|fits|jpeg|gif))|(text/html)$')
        for i in range(len(terms)):
            t = terms[i]
            if t in ['ALL', 'GRAPHIC', 'METADATA']:
                formats.append(t)
            elif ct_re.match(t):
                formats.append(t)
            elif t.startswith('GRAPHIC-'):
                rest = t[8:]
                if rest == 'ALL' and (i == len(terms)-1):
                    formats.append(t)
                    break
                subterms = [rest] + terms[i+1:]
                if (len(subterms) == 0) or (len(subterms) > 3):
                    raise ValidationError('GRAPHIC-x can contain only 1 to 3 terms, not %i' % len(subterms))
                for st in subterms:
                    if not st in ['jpeg', 'png', 'gif']:
                        raise ValidationError('GRAPHIC-x: x must be jpeg, png or gif, not "%s"' % st)
                formats.append('GRAPHIC-' + ','.join(subterms))
                break
            else:
                raise ValidationError('Term "%s" not understood.' % t)
        return formats

class SiapForm(forms.Form):
    POS = CommaSeparatedFloatField()
    SIZE = CommaSeparatedFloatField()
    FORMAT = FormatField(required=False, initial='ALL')

    def getclean(self, name):
        if not hasattr(self, 'cleaned_data'):
            return None
        if name in self.cleaned_data:
            return self.cleaned_data[name]
        return None

    def clean(self):
        pos = self.getclean('POS')
        if pos and len(pos) != 2:
            self._errors['POS'] = (
                'POS must be two floating-point values separated by a '
                'comma (got %i values)' % len(pos))
        sz = self.getclean('SIZE')
        if sz and not len(sz) in [1, 2]:
            self._errors['POS'] = (
                'SIZE must be one or two floating-point values separated by '
                'a comma (got %i values)' % len(sz))
        return self.cleaned_data


def siap_pointed_html(request):
    res = HttpResponse()
    if len(request.GET):
        form = SiapForm(request.GET)
    else:
        form = SiapForm()

    if len(request.GET) and form.is_valid():
        log('Form is valid:')
        for k,v in form.cleaned_data.items():
            log('  ', k, ' = ', v)
        res.write('Ok')
        return res

    elif form._errors:
        log('Form is invalid:')
        for k,v in form._errors.items():
            if isinstance(v, ErrorList):
                v = v.as_text()
            log('  ', k, ' = ', v)

    res.write('<html>'
              '<head><style type="text/css">'
              '.errorlist { color:red; list-style:none; }'
              '.cb { margin-left:auto; margin-right:auto; }'
              'th { text-align:right; }'
              '</style></head>'
              '<body><table border="1" cellpadding="5" class="cb">'
              '<form method="get">')
    res.write(str(form))
    res.write('<tr><td colspan="2" style="text-align:center;">'
              '<input type="submit" name="sub" value="Submit" />'
              '</td></tr>'
              '</table>'
              '</form>'
              '</body></html>')
    return res



def siap_pointed(request):
    res = HttpResponse()
    res['Content-Type'] = 'text/xml'

    doc = VOTableDocument()
    resource = VOResource()
    resource.args['type'] = 'result'
    doc.add_child(resource)
    qstatus = VOInfo('QUERY_STATUS')
    resource.add_child(qstatus)

    form = SiapForm(request.GET)
    if form.is_valid():
        log('Form is valid:')
        for k,v in form.cleaned_data.items():
            log('  ', k, ' = ', v)
    else:
        log('Form is invalid:')
        for k,v in form._errors.items():
            if isinstance(v, ErrorList):
                v = v.as_text()
            log('  ', k, ' = ', v)

        qstatus.args['value'] = 'ERROR'
        errstr = ''
        for k,v in form._errors.items():
            if isinstance(v, ErrorList):
                v = '; '.join(v)
            errstr += k + ': ' + str(v)
        qstatus.add_child(errstr)
        res.write(str(doc))
        return res

    pos = form.cleaned_data['POS']
    size = form.cleaned_data['SIZE']
    formats = form.cleaned_data['FORMAT']

    #intr = None
    #if 'INTERSECT' in request.GET:
    #    intr = request.GET['INTERSECT']
    #    if not (intr in intersects):
    #        intr = None
    #if not intr:
    #    intr = 'OVERLAPS'
    
    # For now, ignore INTERSECT, NAXIS, CFRAME, EQUINOX, CRPIX,
    # CRVAL, CDELT, ROTANG, PROJ, VERB

    #log('POS:', pos, 'SIZE:', size)
    #log('FORMAT:', formats)

    qstatus.args['value'] = 'OK'

    # METADATA queries require this to be present:
    posparam = VOParam('INPUT:POS', None, None, None, '0,0')
    posparam.add_child(VODescription(
        'Center of the query region in the form "RA,Dec" in decimal degrees ICRS'))
    resource.add_child(posparam)

    sizeparam = VOParam('INPUT:SIZE', None, None, None, '0')
    sizeparam.add_child(VODescription(
        'Size of the query region: "size" or "rasize,decsize" in decimal degrees.'))
    resource.add_child(sizeparam)

    formatparam = VOParam('INPUT:FORMAT', None, None, None, 'ALL')
    formatparam.add_child(VODescription(
        'Requested format of images.'))
    formatparam.add_child(VOValues(['image/fits', 'image/jpeg', 'image/png',
                                    'image/gif', 'text/html', 'ALL', 'GRAPHIC', 'METADATA']))
    resource.add_child(formatparam)

    intparam = VOParam('INPUT:INTERSECT', None, None, None, 'OVERLAPS')
    intparam.add_child(VODescription(
        'How the resulting images should overlap the requested region.'))
    intparam.add_child(VOValues(['COVERS', 'ENCLOSED', 'CENTER', 'OVERLAPS']))
    resource.add_child(intparam)

    verbparam = VOParam('INPUT:VERB', None, None, None, '1')
    verbparam.add_child(VODescription(
        'Verbosity: controls the number of columns returned.'))
    resource.add_child(verbparam)


    # Now the results....

    table = PointedTable()
    table.args['name'] = 'results'
    resource.add_child(table)

    imgs = []
    if len(formats) == 1 and formats[0] == 'METADATA':
        pass
    else:
        # query...
        imgs = voImage.objects.all()

    for voimg in imgs:
        row = PointedRow(voimg)
        table.add_row(row)

    res.write(str(doc))
    return res

def getimage(request):
    if 'voimageid' in request.GET:
        imgid = request.GET['voimageid']
        voimages = voImage.objects.all().filter(id=imgid)
        if not len(voimages):
            return HttpResponse('no such image')
        img = voimages[0]
        field = img.field
        fn = field.filename()
        if not os.path.exists(fn):
            return HttpResponse('no such file')
        #owner = field.user
        #prefs = UserPreferences.for_user(owner)
        if not field.redistributable():
            return HttpResponse('not redistributable')
        
        res = HttpResponse()
        res['Content-Type'] = field.content_type()
        res['Content-Length'] = file_size(fn)
        f = open(fn)
        res.write(f.read())
        f.close()
        return res
    
    return HttpResponse('no voimageid')

