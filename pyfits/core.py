#!/usr/bin/env python

# $Id: core.py 597 2010-05-07 19:15:56Z stsci_jtaylor2 $

"""
A module for reading and writing FITS files and manipulating their
contents.

A module for reading and writing Flexible Image Transport System
(FITS) files.  This file format was endorsed by the International
Astronomical Union in 1999 and mandated by NASA as the standard format
for storing high energy astrophysics data.  For details of the FITS
standard, see the NASA/Science Office of Standards and Technology
publication, NOST 100-2.0.

For detailed examples of usage, see the `PyFITS User's Manual
<http://stsdas.stsci.edu/download/wikidocs/The_PyFITS_Handbook.pdf>`_.

"""

from __future__ import division # confidence high

"""
        Do you mean: "Profits"?

                - Google Search, when asked for "PyFITS"
"""

import re, os, tempfile, exceptions
import operator
import __builtin__
import urllib
import tempfile
import gzip
import zipfile
import numpy as np
from numpy import char as chararray
import rec
from numpy import memmap as Memmap
from string import maketrans
import string
import types
import signal
import threading
import sys
import warnings
import weakref
import datetime
import textwrap
try:
    import pyfitsComp
    compressionSupported = 1
except:
    compressionSupported = 0

# Module variables
_blockLen = 2880         # the FITS block size
_python_mode = {'readonly':'rb', 'copyonwrite':'rb', 'update':'rb+', 'append':'ab+', 'ostream':'w'}  # open modes
_memmap_mode = {'readonly':'r', 'copyonwrite':'c', 'update':'r+'}

TRUE  = True    # deprecated
FALSE = False   # deprecated

_INDENT = "   "
DELAYED = "delayed"     # used for lazy instantiation of data
ASCIITNULL = 0          # value for ASCII table cell with value = TNULL
                        # this can be reset by user.
_isInt = "isinstance(val, (int, long, np.integer))"

# The following variable and function are used to support case sensitive
# values for the value of a EXTNAME card in an extension header.  By default,
# pyfits converts the value of EXTNAME cards to upper case when reading from
# a file.  By calling setExtensionNameCaseSensitive() the user may circumvent
# this process so that the EXTNAME value remains in the same case as it is
# in the file.

_extensionNameCaseSensitive = False

def setExtensionNameCaseSensitive(value=True):
    global _extensionNameCaseSensitive
    _extensionNameCaseSensitive = value

# Warnings routines

_showwarning = warnings.showwarning

def showwarning(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stdout
    _showwarning(message, category, filename, lineno, file)

def formatwarning(message, category, filename, lineno, line=None):
    return str(message)+'\n'

warnings.showwarning = showwarning
warnings.formatwarning = formatwarning
warnings.filterwarnings('always',category=UserWarning,append=True)

# Functions

def _padLength(stringLen):
    """
    Bytes needed to pad the input stringLen to the next FITS block.
    """
    return (_blockLen - stringLen%_blockLen) % _blockLen

def _tmpName(input):
    """
    Create a temporary file name which should not already exist.  Use
    the directory of the input file and the base name of the mktemp()
    output.
    """
    dirName = os.path.dirname(input)
    if dirName != '':
        dirName += '/'
    _name = dirName + os.path.basename(tempfile.mktemp())
    if not os.path.exists(_name):
        return _name
    else:
        raise RuntimeError("%s exists" % _name)

def _fromfile(infile, dtype, count, sep):
    if isinstance(infile, file):
        return np.fromfile(infile, dtype=dtype, count=count, sep=sep)
    else: # treat as file-like object with "read" method
        read_size=np.dtype(dtype).itemsize * count
        str=infile.read(read_size)
        return np.fromstring(str, dtype=dtype, count=count, sep=sep)

def _tofile(arr, outfile):
    if isinstance(outfile, file):
        arr.tofile(outfile)
    else: # treat as file-like object with "write" method
        str=arr.tostring()
        outfile.write(str)

def _chunk_array(arr, CHUNK_SIZE=2 ** 25):
    """
    Yields subviews of the given array.  The number of rows is
    selected so it is as close to CHUNK_SIZE (bytes) as possible.
    """
    if len(arr) == 0:
        return
    if isinstance(arr, FITS_rec):
        arr = np.asarray(arr)
    row_size = arr[0].size
    rows_per_chunk = max(min(CHUNK_SIZE // row_size, len(arr)), 1)
    for i in range(0, len(arr), rows_per_chunk):
        yield arr[i:i+rows_per_chunk,...]

def _unsigned_zero(dtype):
    """
    Given a numpy dtype, finds it's "zero" point, which is exactly in
    the middle of its range.
    """
    assert dtype.kind == 'u'
    return 1 << (dtype.itemsize * 8 - 1)

def _is_pseudo_unsigned(dtype):
    return dtype.kind == 'u' and dtype.itemsize >= 2

class VerifyError(exceptions.Exception):
    """
    Verify exception class.
    """
    pass

class _ErrList(list):
    """
    Verification errors list class.  It has a nested list structure
    constructed by error messages generated by verifications at
    different class levels.
    """

    def __init__(self, val, unit="Element"):
        list.__init__(self, val)
        self.unit = unit

    def __str__(self, tab=0):
        """
        Print out nested structure with corresponding indentations.

        A tricky use of `__str__`, since normally `__str__` has only
        one argument.
        """
        result = ""
        element = 0

        # go through the list twice, first time print out all top level messages
        for item in self:
            if not isinstance(item, _ErrList):
                result += _INDENT*tab+"%s\n" % item

        # second time go through the next level items, each of the next level
        # must present, even it has nothing.
        for item in self:
            if isinstance(item, _ErrList):
                _dummy = item.__str__(tab=tab+1)

                # print out a message only if there is something
                if _dummy.strip():
                    if self.unit:
                        result += _INDENT*tab+"%s %s:\n" % (self.unit, element)
                    result += _dummy
                element += 1

        return result

class _Verify(object):
    """
    Shared methods for verification.
    """

    def run_option(self, option="warn", err_text="", fix_text="Fixed.", fix = "pass", fixable=1):
        """
        Execute the verification with selected option.
        """
        _text = err_text
        if not fixable:
            option = 'unfixable'
        if option in ['warn', 'exception']:
            #raise VerifyError, _text
        #elif option == 'warn':
            pass

        # fix the value
        elif option == 'unfixable':
            _text = "Unfixable error: %s" % _text
        else:
            exec(fix)
            #if option != 'silentfix':
            _text += '  ' + fix_text
        return _text

    def verify(self, option='warn'):
        """
        Verify all values in the instance.

        Parameters
        ----------
        option : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.
        """

        _option = option.lower()
        if _option not in ['fix', 'silentfix', 'ignore', 'warn', 'exception']:
            raise ValueError, 'Option %s not recognized.' % option

        if (_option == "ignore"):
            return

        x = str(self._verify(_option)).rstrip()
        if _option in ['fix', 'silentfix'] and x.find('Unfixable') != -1:
            raise VerifyError, '\n'+x
        if (_option != "silentfix"and _option != 'exception') and x:
            warnings.warn('Output verification result:')
            warnings.warn(x)
        if _option == 'exception' and x:
            raise VerifyError, '\n'+x

def _pad(input):
    """
    Pad blank space to the input string to be multiple of 80.
    """
    _len = len(input)
    if _len == Card.length:
        return input
    elif _len > Card.length:
        strlen = _len % Card.length
        if strlen == 0:
            return input
        else:
            return input + ' ' * (Card.length-strlen)

    # minimum length is 80
    else:
        strlen = _len % Card.length
        return input + ' ' * (Card.length-strlen)

def _floatFormat(value):
    """
    Format the floating number to make sure it gets the decimal point.
    """
    valueStr = "%.16G" % value
    if "." not in valueStr and "E" not in valueStr:
        valueStr += ".0"
    return valueStr

class Undefined:
    """
    Undefined value.
    """
    pass

class Delayed:
    """
    Delayed file-reading data.
    """
    def __init__(self, hdu=None, field=None):
        self.hdu = weakref.ref(hdu)
        self.field = field

    def __getitem__(self, key):
        # This forces the data for the HDU to be read, which will replace
        # the corresponding Delayed objects in the Tables Columns to be
        # transformed into ndarrays.  It will also return the value of the
        # requested data element.
        return self.hdu().data[key][self.field]

# translation table for floating value string
_fix_table = maketrans('de', 'DE')
_fix_table2 = maketrans('dD', 'eE')

class Card(_Verify):

    # string length of a card
    length = 80

    # String for a FITS standard compliant (FSC) keyword.
    _keywd_FSC = r'[A-Z0-9_-]* *$'
    _keywd_FSC_RE = re.compile(_keywd_FSC)

    # A number sub-string, either an integer or a float in fixed or
    # scientific notation.  One for FSC and one for non-FSC (NFSC) format:
    # NFSC allows lower case of DE for exponent, allows space between sign,
    # digits, exponent sign, and exponents
    _digits_FSC = r'(\.\d+|\d+(\.\d*)?)([DE][+-]?\d+)?'
    _digits_NFSC = r'(\.\d+|\d+(\.\d*)?) *([deDE] *[+-]? *\d+)?'
    _numr_FSC = r'[+-]?' + _digits_FSC
    _numr_NFSC = r'[+-]? *' + _digits_NFSC

    # This regex helps delete leading zeros from numbers, otherwise
    # Python might evaluate them as octal values.
    _number_FSC_RE = re.compile(r'(?P<sign>[+-])?0*(?P<digt>' + _digits_FSC+')')
    _number_NFSC_RE = re.compile(r'(?P<sign>[+-])? *0*(?P<digt>' + _digits_NFSC + ')')

    # FSC commentary card string which must contain printable ASCII characters.
    _ASCII_text = r'[ -~]*$'
    _comment_FSC_RE = re.compile(_ASCII_text)

    # Checks for a valid value/comment string.  It returns a match object
    # for a valid value/comment string.
    # The valu group will return a match if a FITS string, boolean,
    # number, or complex value is found, otherwise it will return
    # None, meaning the keyword is undefined.  The comment field will
    # return a match if the comment separator is found, though the
    # comment maybe an empty string.
    _value_FSC_RE = re.compile(
        r'(?P<valu_field> *'
            r'(?P<valu>'

                #  The <strg> regex is not correct for all cases, but
                #  it comes pretty darn close.  It appears to find the
                #  end of a string rather well, but will accept
                #  strings with an odd number of single quotes,
                #  instead of issuing an error.  The FITS standard
                #  appears vague on this issue and only states that a
                #  string should not end with two single quotes,
                #  whereas it should not end with an even number of
                #  quotes to be precise.
                #
                #  Note that a non-greedy match is done for a string,
                #  since a greedy match will find a single-quote after
                #  the comment separator resulting in an incorrect
                #  match.
                r'\'(?P<strg>([ -~]+?|\'\'|)) *?\'(?=$|/| )|'
                r'(?P<bool>[FT])|'
                r'(?P<numr>' + _numr_FSC + ')|'
                r'(?P<cplx>\( *'
                    r'(?P<real>' + _numr_FSC + ') *, *(?P<imag>' + _numr_FSC + ') *\))'
            r')? *)'
        r'(?P<comm_field>'
            r'(?P<sepr>/ *)'
            r'(?P<comm>[!-~][ -~]*)?'
        r')?$')

    _value_NFSC_RE = re.compile(
        r'(?P<valu_field> *'
            r'(?P<valu>'
                r'\'(?P<strg>([ -~]+?|\'\'|)) *?\'(?=$|/| )|'
                r'(?P<bool>[FT])|'
                r'(?P<numr>' + _numr_NFSC + ')|'
                r'(?P<cplx>\( *'
                    r'(?P<real>' + _numr_NFSC + ') *, *(?P<imag>' + _numr_NFSC + ') *\))'
            r')? *)'
        r'(?P<comm_field>'
            r'(?P<sepr>/ *)'
            r'(?P<comm>.*)'
        r')?$')

    # keys of commentary cards
    _commentaryKeys = ['', 'COMMENT', 'HISTORY']

    def __init__(self, key='', value='', comment=''):
        """
        Construct a card from `key`, `value`, and (optionally)
        `comment`.  Any specifed arguments, except defaults, must be
        compliant to FITS standard.

        Parameters
        ----------
        key : str, optional
            keyword name

        value : str, optional
            keyword value

        comment : str, optional
            comment
        """

        if key != '' or value != '' or comment != '':
            self._setkey(key)
            self._setvalue(value)
            self._setcomment(comment)

            # for commentary cards, value can only be strings and there
            # is no comment
            if self.key in Card._commentaryKeys:
                if not isinstance(self.value, str):
                    raise ValueError, 'Value in a commentary card must be a string'
        else:
            self.__dict__['_cardimage'] = ' '*80

    def __repr__(self):
        return self._cardimage

    def __getattr__(self, name):
        """
        Instantiate specified attribute object.
        """

        if name == '_cardimage':
            self.ascardimage()
        elif name == 'key':
            self._extractKey()
        elif name in ['value', 'comment']:
            self._extractValueComment(name)
        else:
            raise AttributeError, name

        return getattr(self, name)

    def _setkey(self, val):
        """
        Set the key attribute, surrogate for the `__setattr__` key case.
        """

        if isinstance(val, str):
            val = val.strip()
            if len(val) <= 8:
                val = val.upper()
                if val == 'END':
                    raise ValueError, "keyword 'END' not allowed"
                self._checkKey(val)
            else:
                if val[:8].upper() == 'HIERARCH':
                    val = val[8:].strip()
                    self.__class__ = _Hierarch
                else:
                    raise ValueError, 'keyword name %s is too long (> 8), use HIERARCH.' % val
        else:
            raise ValueError, 'keyword name %s is not a string' % val
        self.__dict__['key'] = val

    def _setvalue(self, val):
        """
        Set the value attribute.
        """

        if isinstance(val, (str, int, long, float, complex, bool, Undefined,
                            np.floating, np.integer, np.complexfloating)):
            if isinstance(val, str):
                self._checkText(val)
            self.__dict__['_valueModified'] = 1
        else:
            raise ValueError, 'Illegal value %s' % str(val)
        self.__dict__['value'] = val

    def _setcomment(self, val):
        """
        Set the comment attribute.
        """

        if isinstance(val,str):
            self._checkText(val)
        else:
            if val is not None:
                raise ValueError, 'comment %s is not a string' % val
        self.__dict__['comment'] = val

    def __setattr__(self, name, val):
        if name == 'key':
            raise SyntaxError, 'keyword name cannot be reset.'
        elif name == 'value':
            self._setvalue(val)
        elif name == 'comment':
            self._setcomment(val)
        elif name == '__class__':
            _Verify.__setattr__(self, name, val)
            return
        else:
            raise AttributeError, name

        # When an attribute (value or comment) is changed, will reconstructe
        # the card image.
        self._ascardimage()

    def ascardimage(self, option='silentfix'):
        """
        Generate a (new) card image from the attributes: `key`, `value`,
        and `comment`, or from raw string.

        Parameters
        ----------
        option : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.
        """

        # Only if the card image already exist (to avoid infinite loop),
        # fix it first.
        if self.__dict__.has_key('_cardimage'):
            self._check(option)
        self._ascardimage()
        return self.__dict__['_cardimage']

    def _ascardimage(self):
        """
        Generate a (new) card image from the attributes: `key`, `value`,
        and `comment`.  Core code for `ascardimage`.
        """

        # keyword string
        if self.__dict__.has_key('key') or self.__dict__.has_key('_cardimage'):
            if isinstance(self, _Hierarch):
                keyStr = 'HIERARCH %s ' % self.key
            else:
                keyStr = '%-8s' % self.key
        else:
            keyStr = ' '*8

        # value string

        # check if both value and _cardimage attributes are missing,
        # to avoid infinite loops
        if not (self.__dict__.has_key('value') or self.__dict__.has_key('_cardimage')):
            valStr = ''

        # string value should occupies at least 8 columns, unless it is
        # a null string
        elif isinstance(self.value, str):
            if self.value == '':
                valStr = "''"
            else:
                _expValStr = self.value.replace("'","''")
                valStr = "'%-8s'" % _expValStr
                valStr = '%-20s' % valStr
        # must be before int checking since bool is also int
        elif isinstance(self.value ,(bool,np.bool_)):
            valStr = '%20s' % `self.value`[0]
        elif isinstance(self.value , (int, long, np.integer)):
            valStr = '%20d' % self.value

        # XXX need to consider platform dependence of the format (e.g. E-009 vs. E-09)
        elif isinstance(self.value, (float, np.floating)):
            if self._valueModified:
                valStr = '%20s' % _floatFormat(self.value)
            else:
                valStr = '%20s' % self._valuestring
        elif isinstance(self.value, (complex,np.complexfloating)):
            if self._valueModified:
                _tmp = '(' + _floatFormat(self.value.real) + ', ' + _floatFormat(self.value.imag) + ')'
                valStr = '%20s' % _tmp
            else:
                valStr = '%20s' % self._valuestring
        elif isinstance(self.value, Undefined):
            valStr = ''

        # conserve space for HIERARCH cards
        if isinstance(self, _Hierarch):
            valStr = valStr.strip()

        # comment string
        if keyStr.strip() in Card._commentaryKeys:  # do NOT use self.key
            commentStr = ''
        elif self.__dict__.has_key('comment') or self.__dict__.has_key('_cardimage'):
            if self.comment in [None, '']:
                commentStr = ''
            else:
                commentStr = ' / ' + self.comment
        else:
            commentStr = ''

        # equal sign string
        eqStr = '= '
        if keyStr.strip() in Card._commentaryKeys:  # not using self.key
            eqStr = ''
            if self.__dict__.has_key('value'):
                valStr = str(self.value)

        # put all parts together
        output = keyStr + eqStr + valStr + commentStr

        # need this in case card-with-continue's value is shortened
        if not isinstance(self, _Hierarch) and \
           not isinstance(self, RecordValuedKeywordCard):
            self.__class__ = Card
        else:
            # does not support CONTINUE for HIERARCH
            if len(keyStr + eqStr + valStr) > Card.length:
                raise ValueError, "The keyword %s with its value is too long." % self.key
        if len(output) <= Card.length:
            output = "%-80s" % output

        # longstring case (CONTINUE card)
        else:
            # try not to use CONTINUE if the string value can fit in one line.
            # Instead, just truncate the comment
            if isinstance(self.value, str) and len(valStr) > (Card.length-10):
                self.__class__ = _Card_with_continue
                output = self._breakup_strings()
            else:
                warnings.warn('card is too long, comment is truncated.')
                output = output[:Card.length]

        self.__dict__['_cardimage'] = output

    def _checkText(self, val):
        """
        Verify `val` to be printable ASCII text.
        """
        if Card._comment_FSC_RE.match(val) is None:
            self.__dict__['_err_text'] = 'Unprintable string %s' % repr(val)
            self.__dict__['_fixable'] = 0
            raise ValueError, self._err_text

    def _checkKey(self, val):
        """
        Verify the keyword `val` to be FITS standard.
        """
        # use repr (not str) in case of control character
        if Card._keywd_FSC_RE.match(val) is None:
            self.__dict__['_err_text'] = 'Illegal keyword name %s' % repr(val)
            self.__dict__['_fixable'] = 0
            raise ValueError, self._err_text

    def _extractKey(self):
        """
        Returns the keyword name parsed from the card image.
        """
        head = self._getKeyString()
        if isinstance(self, _Hierarch):
            self.__dict__['key'] = head.strip()
        else:
            self.__dict__['key'] = head.strip().upper()

    def _extractValueComment(self, name):
        """
        Extract the keyword value or comment from the card image.
        """
        # for commentary cards, no need to parse further
        if self.key in Card._commentaryKeys:
            self.__dict__['value'] = self._cardimage[8:].rstrip()
            self.__dict__['comment'] = ''
            return

        valu = self._check(option='parse')

        if name == 'value':
            if valu is None:
                raise ValueError, "Unparsable card (" + self.key + \
                                  "), fix it first with .verify('fix')."
            if valu.group('bool') != None:
                _val = valu.group('bool')=='T'
            elif valu.group('strg') != None:
                _val = re.sub("''", "'", valu.group('strg'))
            elif valu.group('numr') != None:

                #  Check for numbers with leading 0s.
                numr = Card._number_NFSC_RE.match(valu.group('numr'))
                _digt = numr.group('digt').translate(_fix_table2, ' ')
                if numr.group('sign') == None:
                    _val = eval(_digt)
                else:
                    _val = eval(numr.group('sign')+_digt)
            elif valu.group('cplx') != None:

                #  Check for numbers with leading 0s.
                real = Card._number_NFSC_RE.match(valu.group('real'))
                _rdigt = real.group('digt').translate(_fix_table2, ' ')
                if real.group('sign') == None:
                    _val = eval(_rdigt)
                else:
                    _val = eval(real.group('sign')+_rdigt)
                imag  = Card._number_NFSC_RE.match(valu.group('imag'))
                _idigt = imag.group('digt').translate(_fix_table2, ' ')
                if imag.group('sign') == None:
                    _val += eval(_idigt)*1j
                else:
                    _val += eval(imag.group('sign') + _idigt)*1j
            else:
                _val = UNDEFINED

            self.__dict__['value'] = _val
            if '_valuestring' not in self.__dict__:
                self.__dict__['_valuestring'] = valu.group('valu')
            if '_valueModified' not in self.__dict__:
                self.__dict__['_valueModified'] = 0

        elif name == 'comment':
            self.__dict__['comment'] = ''
            if valu is not None:
                _comm = valu.group('comm')
                if isinstance(_comm, str):
                    self.__dict__['comment'] = _comm.rstrip()

    def _fixValue(self, input):
        """
        Fix the card image for fixable non-standard compliance.
        """
        _valStr = None

        # for the unparsable case
        if input is None:
            _tmp = self._getValueCommentString()
            try:
                slashLoc = _tmp.index("/")
                self.__dict__['value'] = _tmp[:slashLoc].strip()
                self.__dict__['comment'] = _tmp[slashLoc+1:].strip()
            except:
                self.__dict__['value'] = _tmp.strip()

        elif input.group('numr') != None:
            numr = Card._number_NFSC_RE.match(input.group('numr'))
            _valStr = numr.group('digt').translate(_fix_table, ' ')
            if numr.group('sign') is not None:
                _valStr = numr.group('sign')+_valStr

        elif input.group('cplx') != None:
            real  = Card._number_NFSC_RE.match(input.group('real'))
            _realStr = real.group('digt').translate(_fix_table, ' ')
            if real.group('sign') is not None:
                _realStr = real.group('sign')+_realStr

            imag  = Card._number_NFSC_RE.match(input.group('imag'))
            _imagStr = imag.group('digt').translate(_fix_table, ' ')
            if imag.group('sign') is not None:
                _imagStr = imag.group('sign') + _imagStr
            _valStr = '(' + _realStr + ', ' + _imagStr + ')'

        self.__dict__['_valuestring'] = _valStr
        self._ascardimage()

    def _locateEq(self):
        """
        Locate the equal sign in the card image before column 10 and
        return its location.  It returns `None` if equal sign is not
        present, or it is a commentary card.
        """
        # no equal sign for commentary cards (i.e. part of the string value)
        _key = self._cardimage[:8].strip().upper()
        if _key in Card._commentaryKeys:
            eqLoc = None
        else:
            if _key == 'HIERARCH':
                _limit = Card.length
            else:
                _limit = 10
            try:
                eqLoc = self._cardimage[:_limit].index("=")
            except:
                eqLoc = None
        return eqLoc

    def _getKeyString(self):
        """
        Locate the equal sign in the card image and return the string
        before the equal sign.  If there is no equal sign, return the
        string before column 9.
        """
        eqLoc = self._locateEq()
        if eqLoc is None:
            eqLoc = 8
        _start = 0
        if self._cardimage[:8].upper() == 'HIERARCH':
            _start = 8
            self.__class__ = _Hierarch
        return self._cardimage[_start:eqLoc]

    def _getValueCommentString(self):
        """
        Locate the equal sign in the card image and return the string
        after the equal sign.  If there is no equal sign, return the
        string after column 8.
        """
        eqLoc = self._locateEq()
        if eqLoc is None:
            eqLoc = 7
        return self._cardimage[eqLoc+1:]

    def _check(self, option='ignore'):
        """
        Verify the card image with the specified option.
        """
        self.__dict__['_err_text'] = ''
        self.__dict__['_fix_text'] = ''
        self.__dict__['_fixable'] = 1

        if option == 'ignore':
            return
        elif option == 'parse':

            # check the value only, no need to check key and comment for 'parse'
            result = Card._value_NFSC_RE.match(self._getValueCommentString())

            # if not parsable (i.e. everything else) result = None
            return result
        else:

            # verify the equal sign position
            if self.key not in Card._commentaryKeys and self._cardimage.find('=') != 8:
                if option in ['exception', 'warn']:
                    self.__dict__['_err_text'] = 'Card image is not FITS standard (equal sign not at column 8).'
                    raise ValueError, self._err_text + '\n%s' % self._cardimage
                elif option in ['fix', 'silentfix']:
                    result = self._check('parse')
                    self._fixValue(result)
                    if option == 'fix':
                        self.__dict__['_fix_text'] = 'Fixed card to be FITS standard.: %s' % self.key

            # verify the key, it is never fixable
            # always fix silently the case where "=" is before column 9,
            # since there is no way to communicate back to the _keylist.
            self._checkKey(self.key)

            # verify the value, it may be fixable
            result = Card._value_FSC_RE.match(self._getValueCommentString())
            if result is not None or self.key in Card._commentaryKeys:
                return result
            else:
                if option in ['fix', 'silentfix']:
                    result = self._check('parse')
                    self._fixValue(result)
                    if option == 'fix':
                        self.__dict__['_fix_text'] = 'Fixed card to be FITS standard.: %s' % self.key
                else:
                    self.__dict__['_err_text'] = 'Card image is not FITS standard (unparsable value string).'
                    raise ValueError, self._err_text + '\n%s' % self._cardimage

            # verify the comment (string), it is never fixable
            if result is not None:
                _str = result.group('comm')
                if _str is not None:
                    self._checkText(_str)

    def fromstring(self, input):
        """
        Construct a `Card` object from a (raw) string. It will pad the
        string if it is not the length of a card image (80 columns).
        If the card image is longer than 80 columns, assume it
        contains ``CONTINUE`` card(s).
        """
        self.__dict__['_cardimage'] = _pad(input)

        if self._cardimage[:8].upper() == 'HIERARCH':
            self.__class__ = _Hierarch
        # for card image longer than 80, assume it contains CONTINUE card(s).
        elif len(self._cardimage) > Card.length:
            self.__class__ = _Card_with_continue

        # remove the key/value/comment attributes, some of them may not exist
        for name in ['key', 'value', 'comment', '_valueModified']:
            if self.__dict__.has_key(name):
                delattr(self, name)
        return self

    def _ncards(self):
        return len(self._cardimage) // Card.length

    def _verify(self, option='warn'):
        """
        Card class verification method.
        """
        _err = _ErrList([])
        try:
            self._check(option)
        except ValueError:
            # Trapping the ValueError raised by _check method.  Want execution to continue while printing
            # exception message.
            pass
        _err.append(self.run_option(option, err_text=self._err_text, fix_text=self._fix_text, fixable=self._fixable))

        return _err

class RecordValuedKeywordCard(Card):
    """
    Class to manage record-valued keyword cards as described in the
    FITS WCS Paper IV proposal for representing a more general
    distortion model.

    Record-valued keyword cards are string-valued cards where the
    string is interpreted as a definition giving a record field name,
    and its floating point value.  In a FITS header they have the
    following syntax::

        keyword = 'field-specifier: float'

    where `keyword` is a standard eight-character FITS keyword name,
    `float` is the standard FITS ASCII representation of a floating
    point number, and these are separated by a colon followed by a
    single blank.  The grammar for field-specifier is::

        field-specifier:
            field
            field-specifier.field

        field:
            identifier
            identifier.index

    where `identifier` is a sequence of letters (upper or lower case),
    underscores, and digits of which the first character must not be a
    digit, and `index` is a sequence of digits.  No blank characters
    may occur in the field-specifier.  The `index` is provided
    primarily for defining array elements though it need not be used
    for that purpose.

    Multiple record-valued keywords of the same name but differing
    values may be present in a FITS header.  The field-specifier may
    be viewed as part of the keyword name.

    Some examples follow::

        DP1     = 'NAXIS: 2'
        DP1     = 'AXIS.1: 1'
        DP1     = 'AXIS.2: 2'
        DP1     = 'NAUX: 2'
        DP1     = 'AUX.1.COEFF.0: 0'
        DP1     = 'AUX.1.POWER.0: 1'
        DP1     = 'AUX.1.COEFF.1: 0.00048828125'
        DP1     = 'AUX.1.POWER.1: 1'
    """
    #
    # A group of class level regular expression definitions that allow the
    # extraction of the key, field-specifier, value, and comment from a
    # card string.
    #
    identifier = r'[a-zA-Z_]\w*'
    field = identifier + r'(\.\d+)?'
    field_specifier_s = field + r'(\.' + field + r')*'
    field_specifier_val = r'(?P<keyword>' + field_specifier_s + r'): (?P<val>' \
                          + Card._numr_FSC + r'\s*)'
    field_specifier_NFSC_val = r'(?P<keyword>' + field_specifier_s + \
                               r'): (?P<val>' + Card._numr_NFSC + r'\s*)'
    keyword_val = r'\'' + field_specifier_val + r'\''
    keyword_NFSC_val = r'\'' + field_specifier_NFSC_val + r'\''
    keyword_val_comm = r' +' + keyword_val + r' *(/ *(?P<comm>[ -~]*))?$'
    keyword_NFSC_val_comm = r' +' + keyword_NFSC_val + \
                            r' *(/ *(?P<comm>[ -~]*))?$'
    #
    # regular expression to extract the field specifier and value from
    # a card image (ex. 'AXIS.1: 2'), the value may not be FITS Standard
    # Complient
    #
    field_specifier_NFSC_image_RE = re.compile(field_specifier_NFSC_val)
    #
    # regular expression to extract the field specifier and value from
    # a card value; the value may not be FITS Standard Complient
    # (ex. 'AXIS.1: 2.0e5')
    #
    field_specifier_NFSC_val_RE = re.compile(field_specifier_NFSC_val+'$')
    #
    # regular expression to extract the key and the field specifier from a
    # string that is being used to index into a card list that contains
    # record value keyword cards (ex. 'DP1.AXIS.1')
    #
    keyword_name_RE = re.compile(r'(?P<key>' + identifier + r')\.' + \
                                 r'(?P<field_spec>' + field_specifier_s + r')$')
    #
    # regular expression to extract the field specifier and value and comment
    # from the string value of a record value keyword card
    # (ex "'AXIS.1: 1' / a comment")
    #
    keyword_val_comm_RE = re.compile(keyword_val_comm)
    #
    # regular expression to extract the field specifier and value and comment
    # from the string value of a record value keyword card  that is not FITS
    # Standard Complient (ex "'AXIS.1: 1.0d12' / a comment")
    #
    keyword_NFSC_val_comm_RE = re.compile(keyword_NFSC_val_comm)

    #
    # class method definitins
    #

    def coerce(cls,card):
        """
        Coerces an input `Card` object to a `RecordValuedKeywordCard`
        object if the value of the card meets the requirements of this
        type of card.

        Parameters
        ----------
        card : `Card` object
            A `Card` object to coerce

        Returns
        -------
        card
            - If the input card is coercible:

                a new `RecordValuedKeywordCard` constructed from the
                `key`, `value`, and `comment` of the input card.

            - If the input card is not coercible:

                the input card
        """
        mo = cls.field_specifier_NFSC_val_RE.match(card.value)
        if mo:
            return cls(card.key, card.value, card.comment)
        else:
            return card

    coerce = classmethod(coerce)

    def upperKey(cls, key):
        """
        `classmethod` to convert a keyword value that may contain a
        field-specifier to uppercase.  The effect is to raise the
        key to uppercase and leave the field specifier in its original
        case.

        Parameters
        ----------
        key : int or str
            A keyword value that could be an integer, a key, or a
            `key.field-specifier` value

        Returns
        -------
        Integer input
            the original integer key

        String input
            the converted string
        """
        if isinstance(key, (int, long,np.integer)):
            return key

        mo = cls.keyword_name_RE.match(key)

        if mo:
            return mo.group('key').strip().upper() + '.' + \
                   mo.group('field_spec')
        else:
            return key.strip().upper()

    upperKey = classmethod(upperKey)

    def validKeyValue(cls, key, value=0):
        """
        Determine if the input key and value can be used to form a
        valid `RecordValuedKeywordCard` object.  The `key` parameter
        may contain the key only or both the key and field-specifier.
        The `value` may be the value only or the field-specifier and
        the value together.  The `value` parameter is optional, in
        which case the `key` parameter must contain both the key and
        the field specifier.

        Parameters
        ----------
        key : str
            The key to parse

        value : str or float-like, optional
            The value to parse

        Returns
        -------
        valid input : A list containing the key, field-specifier, value

        invalid input : An empty list

        Examples
        --------

        >>> validKeyValue('DP1','AXIS.1: 2')
        >>> validKeyValue('DP1.AXIS.1', 2)
        >>> validKeyValue('DP1.AXIS.1')
        """

        rtnKey = rtnFieldSpec = rtnValue = ''
        myKey = cls.upperKey(key)

        if isinstance(myKey, str):
            validKey = cls.keyword_name_RE.match(myKey)

            if validKey:
               try:
                   rtnValue = float(value)
               except ValueError:
                   pass
               else:
                   rtnKey = validKey.group('key')
                   rtnFieldSpec = validKey.group('field_spec')
            else:
                if isinstance(value, str) and \
                Card._keywd_FSC_RE.match(myKey) and len(myKey) < 9:
                    validValue = cls.field_specifier_NFSC_val_RE.match(value)
                    if validValue:
                        rtnFieldSpec = validValue.group('keyword')
                        rtnValue = validValue.group('val')
                        rtnKey = myKey

        if rtnFieldSpec:
            return [rtnKey, rtnFieldSpec, rtnValue]
        else:
            return []

    validKeyValue = classmethod(validKeyValue)

    def createCard(cls, key='', value='', comment=''):
        """
        Create a card given the input `key`, `value`, and `comment`.
        If the input key and value qualify for a
        `RecordValuedKeywordCard` then that is the object created.
        Otherwise, a standard `Card` object is created.

        Parameters
        ----------
        key : str, optional
            The key

        value : str, optional
            The value

        comment : str, optional
            The comment

        Returns
        -------
        card
            Either a `RecordValuedKeywordCard` or a `Card` object.
        """
        if cls.validKeyValue(key, value):
            objClass = cls
        else:
            objClass = Card

        return objClass(key, value, comment)

    createCard = classmethod(createCard)

    def createCardFromString(cls, input):
        """
        Create a card given the `input` string.  If the `input` string
        can be parsed into a key and value that qualify for a
        `RecordValuedKeywordCard` then that is the object created.
        Otherwise, a standard `Card` object is created.

        Parameters
        ----------
        input : str
            The string representing the card

        Returns
        -------
        card
            either a `RecordValuedKeywordCard` or a `Card` object
        """
        idx1 = string.find(input, "'")+1
        idx2 = string.rfind(input, "'")

        if idx2 > idx1 and idx1 >= 0 and \
           cls.validKeyValue('',value=input[idx1:idx2]):
            objClass = cls
        else:
            objClass = Card

        return objClass().fromstring(input)

    createCardFromString = classmethod(createCardFromString)

    def __init__(self, key='', value='', comment=''):
        """
        Parameters
        ----------
        key : str, optional
            The key, either the simple key or one that contains
            a field-specifier

        value : str, optional
            The value, either a simple value or one that contains a
            field-specifier

        comment : str, optional
            The comment
        """

        mo = self.keyword_name_RE.match(key)

        if mo:
            self.__dict__['field_specifier'] = mo.group('field_spec')
            key = mo.group('key')
        else:
            if isinstance(value, str):
                if value != '':
                    mo = self.field_specifier_NFSC_val_RE.match(value)

                    if mo:
                        self.__dict__['field_specifier'] = mo.group('keyword')
                        value = float(mo.group('val'))
                    else:
                        raise ValueError, \
                              "value %s must be in the form " % value + \
                              "field_specifier: value (ex. 'NAXIS: 2')"
            else:
                raise ValueError, 'value %s is not a string' % value

        Card.__init__(self, key, value, comment)

    def __getattr__(self, name):

        if name == 'field_specifier':
            self._extractValueComment('value')
        else:
            Card.__getattr__(self, name)

        return getattr(self, name)

    def __setattr__(self, name, val):
        if name == 'field_specifier':
            raise SyntaxError, 'field_specifier cannot be reset.'
        else:
            if not isinstance(val, float):
                try:
                    val = float(val)
                except ValueError:
                    raise ValueError, 'value %s is not a float' % val
            Card.__setattr__(self,name,val)

    def _ascardimage(self):
        """
        Generate a (new) card image from the attributes: `key`, `value`,
        `field_specifier`, and `comment`.  Core code for `ascardimage`.
        """
        Card._ascardimage(self)
        eqloc = self._cardimage.index("=")
        slashloc = self._cardimage.find("/")

        if '_valueModified' in self.__dict__ and self._valueModified:
            valStr = _floatFormat(self.value)
        else:
            valStr = self._valuestring

        valStr = "'" + self.field_specifier + ": " + valStr + "'"
        valStr = '%-20s' % valStr

        output = self._cardimage[:eqloc+2] + valStr

        if slashloc > 0:
            output = output + self._cardimage[slashloc-1:]

        if len(output) <= Card.length:
            output = "%-80s" % output

        self.__dict__['_cardimage'] = output


    def _extractValueComment(self, name):
        """
        Extract the keyword value or comment from the card image.
        """
        valu = self._check(option='parse')

        if name == 'value':
            if valu is None:
                raise ValueError, \
                         "Unparsable card, fix it first with .verify('fix')."

            self.__dict__['field_specifier'] = valu.group('keyword')
            self.__dict__['value'] = \
                           eval(valu.group('val').translate(_fix_table2, ' '))

            if '_valuestring' not in self.__dict__:
                self.__dict__['_valuestring'] = valu.group('val')
            if '_valueModified' not in self.__dict__:
                self.__dict__['_valueModified'] = 0

        elif name == 'comment':
            Card._extractValueComment(self, name)


    def strvalue(self):
        """
        Method to extract the field specifier and value from the card
        image.  This is what is reported to the user when requesting
        the value of the `Card` using either an integer index or the
        card key without any field specifier.
        """

        mo = self.field_specifier_NFSC_image_RE.search(self._cardimage)
        return self._cardimage[mo.start():mo.end()]

    def _fixValue(self, input):
        """
        Fix the card image for fixable non-standard compliance.
        """
        _valStr = None

        if input is None:
            tmp = self._getValueCommentString()

            try:
                slashLoc = tmp.index("/")
            except:
                slashLoc = len(tmp)

            self.__dict__['_err_text'] = 'Illegal value %s' % tmp[:slashLoc]
            self.__dict__['_fixable'] = 0
            raise ValueError, self._err_text
        else:
            self.__dict__['_valuestring'] = \
                                 input.group('val').translate(_fix_table, ' ')
            self._ascardimage()


    def _check(self, option='ignore'):
        """
        Verify the card image with the specified `option`.
        """
        self.__dict__['_err_text'] = ''
        self.__dict__['_fix_text'] = ''
        self.__dict__['_fixable'] = 1

        if option == 'ignore':
            return
        elif option == 'parse':
            return self.keyword_NFSC_val_comm_RE.match(self._getValueCommentString())
        else:
            # verify the equal sign position

            if self._cardimage.find('=') != 8:
                if option in ['exception', 'warn']:
                    self.__dict__['_err_text'] = 'Card image is not FITS ' + \
                                       'standard (equal sign not at column 8).'
                    raise ValueError, self._err_text + '\n%s' % self._cardimage
                elif option in ['fix', 'silentfix']:
                    result = self._check('parse')
                    self._fixValue(result)

                    if option == 'fix':
                        self.__dict__['_fix_text'] = \
                           'Fixed card to be FITS standard. : %s' % self.key

            # verify the key

            self._checkKey(self.key)

            # verify the value

            result = \
              self.keyword_val_comm_RE.match (self._getValueCommentString())

            if result is not None:
                return result
            else:
                if option in ['fix', 'silentfix']:
                    result = self._check('parse')
                    self._fixValue(result)

                    if option == 'fix':
                        self.__dict__['_fix_text'] = \
                              'Fixed card to be FITS standard.: %s' % self.key
                else:
                    self.__dict__['_err_text'] = \
                    'Card image is not FITS standard (unparsable value string).'
                    raise ValueError, self._err_text + '\n%s' % self._cardimage

            # verify the comment (string), it is never fixable
            if result is not None:
                _str = result.group('comm')
                if _str is not None:
                    self._checkText(_str)

def createCard(key='', value='', comment=''):
    return RecordValuedKeywordCard.createCard(key, value, comment)
createCard.__doc__ = RecordValuedKeywordCard.createCard.__doc__

def createCardFromString(input):
    return RecordValuedKeywordCard.createCardFromString(input)
createCardFromString.__doc__ = \
    RecordValuedKeywordCard.createCardFromString.__doc__

def upperKey(key):
    return RecordValuedKeywordCard.upperKey(key)
upperKey.__doc__ = RecordValuedKeywordCard.upperKey.__doc__

class _Hierarch(Card):
    """
    Cards begins with ``HIERARCH`` which allows keyword name longer
    than 8 characters.
    """
    def _verify(self, option='warn'):
        """No verification (for now)."""
        return _ErrList([])


class _Card_with_continue(Card):
    """
    Cards having more than one 80-char "physical" cards, the cards after
    the first one must start with ``CONTINUE`` and the whole card must have
    string value.
    """

    def __str__(self):
        """
        Format a list of cards into a printable string.
        """
        kard = self._cardimage
        output = ''
        for i in range(len(kard)//80):
            output += kard[i*80:(i+1)*80] + '\n'
        return output[:-1]

    def _extractValueComment(self, name):
        """
        Extract the keyword value or comment from the card image.
        """
        longstring = ''

        ncards = self._ncards()
        for i in range(ncards):
            # take each 80-char card as a regular card and use its methods.
            _card = Card().fromstring(self._cardimage[i*80:(i+1)*80])
            if i > 0 and _card.key != 'CONTINUE':
                raise ValueError, 'Long card image must have CONTINUE cards after the first card.'
            if not isinstance(_card.value, str):
                raise ValueError, 'Cards with CONTINUE must have string value.'



            if name == 'value':
                _val = re.sub("''", "'", _card.value).rstrip()

                # drop the ending "&"
                if _val[-1] == '&':
                    _val = _val[:-1]
                longstring = longstring + _val

            elif name == 'comment':
                _comm = _card.comment
                if isinstance(_comm, str) and _comm != '':
                    longstring = longstring + _comm.rstrip() + ' '

            self.__dict__[name] = longstring.rstrip()

    def _breakup_strings(self):
        """
        Break up long string value/comment into ``CONTINUE`` cards.
        This is a primitive implementation: it will put the value
        string in one block and the comment string in another.  Also,
        it does not break at the blank space between words.  So it may
        not look pretty.
        """
        val_len = 67
        comm_len = 64
        output = ''

        # do the value string
        valfmt = "'%-s&'"
        val = self.value.replace("'", "''")
        val_list = self._words_group(val, val_len)
        for i in range(len(val_list)):
            if i == 0:
                headstr = "%-8s= " % self.key
            else:
                headstr = "CONTINUE  "
            valstr = valfmt % val_list[i]
            output = output + '%-80s' % (headstr + valstr)

        # do the comment string
        if self.comment is None:
            comm = ''
        else:
            comm = self.comment
        commfmt = "%-s"
        if not comm == '':
            comm_list = self._words_group(comm, comm_len)
            for i in comm_list:
                commstr = "CONTINUE  '&' / " + commfmt % i
                output = output + '%-80s' % commstr

        return output

    def _words_group(self, input, strlen):
        """
        Split a long string into parts where each part is no longer
        than `strlen` and no word is cut into two pieces.  But if
        there is one single word which is longer than `strlen`, then
        it will be split in the middle of the word.
        """
        list = []
        _nblanks = input.count(' ')
        nmax = max(_nblanks, len(input)//strlen+1)
        arr = chararray.array(input+' ', itemsize=1)

        # locations of the blanks
        blank_loc = np.nonzero(arr == ' ')[0]
        offset = 0
        xoffset = 0
        for i in range(nmax):
            try:
                loc = np.nonzero(blank_loc >= strlen+offset)[0][0]
                offset = blank_loc[loc-1] + 1
                if loc == 0:
                    offset = -1
            except:
                offset = len(input)

            # check for one word longer than strlen, break in the middle
            if offset <= xoffset:
                offset = xoffset + strlen

            # collect the pieces in a list
            tmp = input[xoffset:offset]
            list.append(tmp)
            if len(input) == offset:
                break
            xoffset = offset

        return list

class _Header_iter:
    """
    Iterator class for a FITS header object.

    Returns the key values of the cards in the header.  Duplicate key
    values are not returned.
    """

    def __init__(self, header):
        self._lastIndex = -1  # last index into the card list
                              # accessed by the class iterator
        self.keys = header.keys()  # the unique keys from the header

    def __iter__(self):
        return self

    def next(self):
        self._lastIndex += 1

        if self._lastIndex >= len(self.keys):
            self._lastIndex = -1
            raise StopIteration()

        return self.keys[self._lastIndex]


class Header:
    """
    FITS header class.

    The purpose of this class is to present the header like a
    dictionary as opposed to a list of cards.

    The attribute `ascard` supplies the header like a list of cards.

    The header class uses the card's keyword as the dictionary key and
    the cards value is the dictionary value.

    The `has_key`, `get`, and `keys` methods are implemented to
    provide the corresponding dictionary functionality.  The header
    may be indexed by keyword value and like a dictionary, the
    associated value will be returned.  When the header contains cards
    with duplicate keywords, only the value of the first card with the
    given keyword will be returned.

    The header may also be indexed by card list index number.  In that
    case, the value of the card at the given index in the card list
    will be returned.

    A delete method has been implemented to allow deletion from the
    header.  When `del` is called, all cards with the given keyword
    are deleted from the header.

    The `Header` class has an associated iterator class `_Header_iter`
    which will allow iteration over the unique keywords in the header
    dictionary.
    """
    def __init__(self, cards=[], txtfile=None):
        """
        Construct a `Header` from a `CardList` and/or text file.

        Parameters
        ----------
        cards : A list of `Card` objects, optional
            The cards to initialize the header with.

        txtfile : file path, file object or file-like object, optional
            Input ASCII header parameters file.
        """

        # populate the cardlist
        self.ascard = CardList(cards)

        if txtfile:
            # get the cards from the input ASCII file
            self.fromTxtFile(txtfile, not len(self.ascard))
            self._mod = 0
        else:
            # decide which kind of header it belongs to
            self._updateHDUtype()

    def _updateHDUtype(self):
        cards = self.ascard

        try:
            if cards[0].key == 'SIMPLE':
                if 'GROUPS' in cards._keylist and cards['GROUPS'].value == True:
                    self._hdutype = GroupsHDU
                elif cards[0].value == True:
                    self._hdutype = PrimaryHDU
                elif cards[0].value == False:
                    self._hdutype = _NonstandardHDU
                else:
                    self._hdutype = _CorruptedHDU
            elif cards[0].key == 'XTENSION':
                xtension = cards[0].value.rstrip()
                if xtension == 'TABLE':
                    self._hdutype = TableHDU
                elif xtension == 'IMAGE':
                    self._hdutype = ImageHDU
                elif xtension in ('BINTABLE', 'A3DTABLE'):
                    try:
                        if self.ascard['ZIMAGE'].value == True:
                            global compressionSupported

                            if compressionSupported == 1:
                                self._hdutype = CompImageHDU
                            else:
                                if compressionSupported == 0:
                                    print "Failure creating a header for a " + \
                                          "compressed image HDU."
                                    print "The pyfitsComp module is not " + \
                                          "available."
                                    print "The HDU will be treated as a " + \
                                          "Binary Table HDU."
                                    compressionSupported = -1

                                raise KeyError
                    except KeyError:
                        self._hdutype = BinTableHDU
                else:
                    self._hdutype = _NonstandardExtHDU
            else:
                self._hdutype = _ValidHDU
        except:
            self._hdutype = _CorruptedHDU

    def __contains__(self, item):
        return self.has_key(item)

    def __iter__(self):
        return _Header_iter(self)

    def __getitem__ (self, key):
        """
        Get a header keyword value.
        """
        card = self.ascard[key]

        if isinstance(card, RecordValuedKeywordCard) and \
           (not isinstance(key, types.StringType) or string.find(key,'.') < 0):
            returnVal = card.strvalue()
        elif isinstance(card, CardList):
            returnVal = card
        else:
            returnVal = card.value

        return returnVal

    def __setitem__ (self, key, value):
        """
        Set a header keyword value.
        """
        self.ascard[key].value = value
        self._mod = 1

    def __delitem__(self, key):
        """
        Delete card(s) with the name `key`.
        """
        # delete ALL cards with the same keyword name
        if isinstance(key, str):
            while 1:
                try:
                    del self.ascard[key]
                    self._mod = 1
                except:
                    return

        # for integer key only delete once
        else:
            del self.ascard[key]
            self._mod = 1

    def __str__(self):
        return self.ascard.__str__()

    def ascardlist(self):
        """
        Returns a `CardList` object.
        """
        return self.ascard

    def items(self):
        """
        Return a list of all keyword-value pairs from the `CardList`.
        """
        pairs = []
        for card in self.ascard:
            pairs.append((card.key, card.value))
        return pairs

    def has_key(self, key):
        """
        Check for existence of a keyword.

        Parameters
        ----------
        key : str or int
           Keyword name.  If given an index, always returns 0.

        Returns
        -------
        has_key : bool
            Returns `True` if found, otherwise, `False`.
        """
        try:
            key = upperKey(key)

            if key[:8] == 'HIERARCH':
                key = key[8:].strip()
            _index = self.ascard[key]
            return True
        except:
            return False

    def rename_key(self, oldkey, newkey, force=0):
        """
        Rename a card's keyword in the header.

        Parameters
        ----------
        oldkey : str or int
            old keyword

        newkey : str
            new keyword

        force : bool
            When `True`, if new key name already exists, force to have
            duplicate name.
        """
        oldkey = upperKey(oldkey)
        newkey = upperKey(newkey)

        if newkey == 'CONTINUE':
            raise ValueError, 'Can not rename to CONTINUE'
        if newkey in Card._commentaryKeys or oldkey in Card._commentaryKeys:
            if not (newkey in Card._commentaryKeys and oldkey in Card._commentaryKeys):
                raise ValueError, 'Regular and commentary keys can not be renamed to each other.'
        elif (force == 0) and self.has_key(newkey):
            raise ValueError, 'Intended keyword %s already exists in header.' % newkey
        _index = self.ascard.index_of(oldkey)
        _comment = self.ascard[_index].comment
        _value = self.ascard[_index].value
        self.ascard[_index] = createCard(newkey, _value, _comment)


#        self.ascard[_index].__dict__['key']=newkey
#        self.ascard[_index].ascardimage()
#        self.ascard._keylist[_index] = newkey

    def keys(self):
        """
        Return a list of keys with duplicates removed.
        """
        rtnVal = []

        for key in self.ascard.keys():
            if not key in rtnVal:
                rtnVal.append(key)

        return rtnVal

    def get(self, key, default=None):
        """
        Get a keyword value from the `CardList`.  If no keyword is
        found, return the default value.

        Parameters
        ----------
        key : str or int
            keyword name or index

        default : object, optional
            if no keyword is found, the value to be returned.
        """

        try:
            return self[key]
        except KeyError:
            return default

    def update(self, key, value, comment=None, before=None, after=None,
               savecomment=False):
        """
        Update one header card.

        If the keyword already exists, it's value and/or comment will
        be updated.  If it does not exist, a new card will be created
        and it will be placed before or after the specified location.
        If no `before` or `after` is specified, it will be appended at
        the end.

        Parameters
        ----------
        key : str
            keyword

        value : str
            value to be used for updating

        comment : str, optional
            to be used for updating, default=None.

        before : str or int, optional
            name of the keyword, or index of the `Card` before which
            the new card will be placed.  The argument `before` takes
            precedence over `after` if both specified.

        after : str or int, optional
            name of the keyword, or index of the `Card` after which
            the new card will be placed.

        savecomment : bool, optional
            When `True`, preserve the current comment for an existing
            keyword.  The argument `savecomment` takes precedence over
            `comment` if both specified.  If `comment` is not
            specified then the current comment will automatically be
            preserved.
        """
        keylist = RecordValuedKeywordCard.validKeyValue(key,value)

        if keylist:
            keyword = keylist[0] + '.' + keylist[1]
        else:
            keyword = key

        if self.has_key(keyword):
            j = self.ascard.index_of(keyword)
            if not savecomment and comment is not None:
                _comment = comment
            else:
                _comment = self.ascard[j].comment
            self.ascard[j] = createCard(key, value, _comment)
        elif before != None or after != None:
            _card = createCard(key, value, comment)
            self.ascard._pos_insert(_card, before=before, after=after)
        else:
            self.ascard.append(createCard(key, value, comment))

        self._mod = 1

        # If this header is associated with a compImageHDU then update
        # the objects underlying header (_tableHeader) unless the update was
        # made to a card that describes the data.

        if self.__dict__.has_key('_tableHeader') and \
           key not in ('XTENSION','BITPIX','PCOUNT','GCOUNT','TFIELDS',
                       'ZIMAGE','ZBITPIX','ZCMPTYPE') and \
           key[:4] not in ('ZVAL') and \
           key[:5] not in ('NAXIS','TTYPE','TFORM','ZTILE','ZNAME') and \
           key[:6] not in ('ZNAXIS'):
            self._tableHeader.update(key,value,comment,before,after)

    def add_history(self, value, before=None, after=None):
        """
        Add a ``HISTORY`` card.

        Parameters
        ----------
        value : str
            history text to be added.

        before : str or int, optional
            same as in `Header.update`

        after : str or int, optional
            same as in `Header.update`
        """
        self._add_commentary('history', value, before=before, after=after)

        # If this header is associated with a compImageHDU then update
        # the objects underlying header (_tableHeader).

        if self.__dict__.has_key('_tableHeader'):
            self._tableHeader.add_history(value,before,after)

    def add_comment(self, value, before=None, after=None):
        """
        Add a ``COMMENT`` card.

        Parameters
        ----------
        value : str
            text to be added.

        before : str or int, optional
            same as in `Header.update`

        after : str or int, optional
            same as in `Header.update`
        """
        self._add_commentary('comment', value, before=before, after=after)

        # If this header is associated with a compImageHDU then update
        # the objects underlying header (_tableHeader).

        if self.__dict__.has_key('_tableHeader'):
            self._tableHeader.add_comment(value,before,after)

    def add_blank(self, value='', before=None, after=None):
        """
        Add a blank card.

        Parameters
        ----------
        value : str, optional
            text to be added.

        before : str or int, optional
            same as in `Header.update`

        after : str or int, optional
            same as in `Header.update`
        """
        self._add_commentary(' ', value, before=before, after=after)

        # If this header is associated with a compImageHDU then update
        # the objects underlying header (_tableHeader).

        if self.__dict__.has_key('_tableHeader'):
            self._tableHeader.add_blank(value,before,after)

    def get_history(self):
        """
        Get all history cards as a list of string texts.
        """
        output = []
        for _card in self.ascardlist():
            if _card.key == 'HISTORY':
                output.append(_card.value)
        return output

    def get_comment(self):
        """
        Get all comment cards as a list of string texts.
        """
        output = []
        for _card in self.ascardlist():
            if _card.key == 'COMMENT':
                output.append(_card.value)
        return output



    def _add_commentary(self, key, value, before=None, after=None):
        """
        Add a commentary card.

        If `before` and `after` are `None`, add to the last occurrence
        of cards of the same name (except blank card).  If there is no
        card (or blank card), append at the end.
        """

        new_card = Card(key, value)
        if before != None or after != None:
            self.ascard._pos_insert(new_card, before=before, after=after)
        else:
            if key[0] == ' ':
                useblanks = new_card._cardimage != ' '*80
                self.ascard.append(new_card, useblanks=useblanks, bottom=1)
            else:
                try:
                    _last = self.ascard.index_of(key, backward=1)
                    self.ascard.insert(_last+1, new_card)
                except:
                    self.ascard.append(new_card, bottom=1)

        self._mod = 1

    def copy(self):
        """
        Make a copy of the `Header`.
        """
        tmp = Header(self.ascard.copy())

        # also copy the class
        tmp._hdutype = self._hdutype
        return tmp

    def _strip(self):
        """
        Strip cards specific to a certain kind of header.

        Strip cards like ``SIMPLE``, ``BITPIX``, etc. so the rest of
        the header can be used to reconstruct another kind of header.
        """
        try:

            # have both SIMPLE and XTENSION to accomodate Extension
            # and Corrupted cases
            del self['SIMPLE']
            del self['XTENSION']
            del self['BITPIX']

            if self.has_key('NAXIS'):
                _naxis = self['NAXIS']
            else:
                _naxis = 0

            if issubclass(self._hdutype, _TableBaseHDU):
                if self.has_key('TFIELDS'):
                    _tfields = self['TFIELDS']
                else:
                    _tfields = 0

            del self['NAXIS']
            for i in range(_naxis):
                del self['NAXIS'+`i+1`]

            if issubclass(self._hdutype, PrimaryHDU):
                del self['EXTEND']
            del self['PCOUNT']
            del self['GCOUNT']

            if issubclass(self._hdutype, PrimaryHDU):
                del self['GROUPS']

            if issubclass(self._hdutype, _ImageBaseHDU):
                del self['BSCALE']
                del self['BZERO']

            if issubclass(self._hdutype, _TableBaseHDU):
                del self['TFIELDS']
                for name in ['TFORM', 'TSCAL', 'TZERO', 'TNULL', 'TTYPE', 'TUNIT']:
                    for i in range(_tfields):
                        del self[name+`i+1`]

            if issubclass(self._hdutype, BinTableHDU):
                for name in ['TDISP', 'TDIM', 'THEAP']:
                    for i in range(_tfields):
                        del self[name+`i+1`]

            if issubclass(self._hdutype, TableHDU):
                for i in range(_tfields):
                    del self['TBCOL'+`i+1`]

        except KeyError:
            pass

    def toTxtFile(self, outFile, clobber=False):
        """
        Output the header parameters to a file in ASCII format.

        Parameters
        ----------
        outFile : file path, file object or file-like object
            Output header parameters file.

        clobber : bool
            When `True`, overwrite the output file if it exists.
        """

        closeFile = False

        # check if the output file already exists
        if (isinstance(outFile,types.StringType) or
            isinstance(outFile,types.UnicodeType)):
            if (os.path.exists(outFile) and os.path.getsize(outFile) != 0):
                if clobber:
                    warnings.warn( "Overwrite existing file '%s'." % outFile)
                    os.remove(outFile)
                else:
                    raise IOError, "File '%s' already exist." % outFile

            outFile = __builtin__.open(outFile,'w')
            closeFile = True

        lines = []   # lines to go out to the header parameters file

        # Add the card image for each card in the header to the lines list

        for j in range(len(self.ascardlist())):
            lines.append(self.ascardlist()[j].__str__()+'\n')

        # Write the header parameter lines out to the ASCII header
        # parameter file
        outFile.writelines(lines)

        if closeFile:
            outFile.close()

    def fromTxtFile(self, inFile, replace=False):
        """
        Input the header parameters from an ASCII file.

        The input header cards will be used to update the current
        header.  Therefore, when an input card key matches a card key
        that already exists in the header, that card will be updated
        in place.  Any input cards that do not already exist in the
        header will be added.  Cards will not be deleted from the
        header.

        Parameters
        ----------
        inFile : file path, file object or file-like object
            Input header parameters file.

        replace : bool, optional
            When `True`, indicates that the entire header should be
            replaced with the contents of the ASCII file instead of
            just updating the current header.
        """

        closeFile = False

        if isinstance(inFile, types.StringType) or \
           isinstance(inFile, types.UnicodeType):
            inFile = __builtin__.open(inFile,'r')
            closeFile = True

        lines = inFile.readlines()

        if closeFile:
            inFile.close()

        if len(self.ascardlist()) > 0 and not replace:
            prevKey = 0
        else:
            if replace:
                self.ascard = CardList([])

            prevKey = 0

        for line in lines:
            card = Card().fromstring(line[:min(80,len(line)-1)])
            card.verify('silentfix')

            if card.key == 'SIMPLE':
                if self.get('EXTENSION'):
                    del self.ascardlist()['EXTENSION']

                self.update(card.key, card.value, card.comment, before=0)
                prevKey = 0
            elif card.key == 'EXTENSION':
                if self.get('SIMPLE'):
                    del self.ascardlist()['SIMPLE']

                self.update(card.key, card.value, card.comment, before=0)
                prevKey = 0
            elif card.key == 'HISTORY':
                if not replace:
                    items = self.items()
                    idx = 0

                    for item in items:
                        if item[0] == card.key and item[1] == card.value:
                            break
                        idx += 1

                    if idx == len(self.ascardlist()):
                        self.add_history(card.value, after=prevKey)
                        prevKey += 1
                else:
                    self.add_history(card.value, after=prevKey)
                    prevKey += 1
            elif card.key == 'COMMENT':
                if not replace:
                    items = self.items()
                    idx = 0

                    for item in items:
                        if item[0] == card.key and item[1] == card.value:
                            break
                        idx += 1

                    if idx == len(self.ascardlist()):
                        self.add_comment(card.value, after=prevKey)
                        prevKey += 1
                else:
                    self.add_comment(card.value, after=prevKey)
                    prevKey += 1
            elif card.key == '        ':
                if not replace:
                    items = self.items()
                    idx = 0

                    for item in items:
                        if item[0] == card.key and item[1] == card.value:
                            break
                        idx += 1

                    if idx == len(self.ascardlist()):
                        self.add_blank(card.value, after=prevKey)
                        prevKey += 1
                else:
                    self.add_blank(card.value, after=prevKey)
                    prevKey += 1
            else:
                if isinstance(card, _Hierarch):
                    prefix = 'hierarch '
                else:
                    prefix = ''

                self.update(prefix + card.key,
                                     card.value,
                                     card.comment,
                                     after=prevKey)
                prevKey += 1

        # update the hdu type of the header to match the parameters read in
        self._updateHDUtype()


class CardList(list):
    """
    FITS header card list class.
    """

    def __init__(self, cards=[], keylist=None):
        """
        Construct the `CardList` object from a list of `Card` objects.

        Parameters
        ----------
        cards
            A list of `Card` objects.
        """

        list.__init__(self, cards)
        self._cards = cards

        # if the key list is not supplied (as in reading in the FITS file),
        # it will be constructed from the card list.
        if keylist is None:
            self._keylist = [k.upper() for k in self._keys()]
        else:
            self._keylist = keylist

        # find out how many blank cards are *directly* before the END card
        self._blanks = 0
        self.count_blanks()

    def _hasFilterChar(self, key):
        """
        Return `True` if the input key contains one of the special filtering
        characters (``*``, ``?``, or ...).
        """
        if isinstance(key, types.StringType) and (key.endswith('...') or \
           key.find('*') > 0 or key.find('?') > 0):
            return True
        else:
            return False

    def filterList(self, key):
        """
        Construct a `CardList` that contains references to all of the cards in
        this `CardList` that match the input key value including any special
        filter keys (``*``, ``?``, and ``...``).

        Parameters
        ----------
        key : str
            key value to filter the list with

        Returns
        -------
        cardlist :
            A `CardList` object containing references to all the
            requested cards.
        """
        outCl = CardList()

        mykey = upperKey(key)
        reStr = string.replace(mykey,'*','\w*')+'$'
        reStr = string.replace(reStr,'?','\w')
        reStr = string.replace(reStr,'...','\S*')
        match_RE = re.compile(reStr)

        for card in self:
            if isinstance(card, RecordValuedKeywordCard):
                matchStr = card.key + '.' + card.field_specifier
            else:
                matchStr = card.key

            if match_RE.match(matchStr):
                outCl.append(card)

        return outCl

    def __getitem__(self, key):
        """
        Get a `Card` by indexing or by the keyword name.
        """
        if self._hasFilterChar(key):
            return self.filterList(key)
        else:
            _key = self.index_of(key)
            return super(CardList, self).__getitem__(_key)

    def __getslice__(self, start, end):
        _cards = super(CardList, self).__getslice__(start,end)
        result = CardList(_cards, self._keylist[start:end])
        return result

    def __setitem__(self, key, value):
        """
        Set a `Card` by indexing or by the keyword name.
        """
        if isinstance (value, Card):
            _key = self.index_of(key)

            # only set if the value is different from the old one
            if str(self[_key]) != str(value):
                super(CardList, self).__setitem__(_key, value)
                self._keylist[_key] = value.key.upper()
                self.count_blanks()
                self._mod = 1
        else:
            raise SyntaxError, "%s is not a Card" % str(value)

    def __delitem__(self, key):
        """
        Delete a `Card` from the `CardList`.
        """
        if self._hasFilterChar(key):
            cardlist = self.filterList(key)

            if len(cardlist) == 0:
                raise KeyError, "Keyword '%s' not found/" % key

            for card in cardlist:
                if isinstance(card, RecordValuedKeywordCard):
                    mykey = card.key + '.' + card.field_specifier
                else:
                    mykey = card.key

                del self[mykey]
        else:
            _key = self.index_of(key)
            super(CardList, self).__delitem__(_key)
            del self._keylist[_key]  # update the keylist
            self.count_blanks()
            self._mod = 1

    def count_blanks(self):
        """
        Returns how many blank cards are *directly* before the ``END``
        card.
        """
        for i in range(1, len(self)):
            if str(self[-i]) != ' '*Card.length:
                self._blanks = i - 1
                break

    def append(self, card, useblanks=True, bottom=False):
        """
        Append a `Card` to the `CardList`.

        Parameters
        ----------
        card : `Card` object
            The `Card` to be appended.

        useblanks : bool, optional
            Use any *extra* blank cards?

            If `useblanks` is `True`, and if there are blank cards
            directly before ``END``, it will use this space first,
            instead of appending after these blank cards, so the total
            space will not increase.  When `useblanks` is `False`, the
            card will be appended at the end, even if there are blank
            cards in front of ``END``.

        bottom : bool, optional
           If `False` the card will be appended after the last
           non-commentary card.  If `True` the card will be appended
           after the last non-blank card.
        """

        if isinstance (card, Card):
            nc = len(self) - self._blanks
            i = nc - 1
            if not bottom:
                for i in range(nc-1, -1, -1): # locate last non-commentary card
                    if self[i].key not in Card._commentaryKeys:
                        break

            super(CardList, self).insert(i+1, card)
            self._keylist.insert(i+1, card.key.upper())
            if useblanks:
                self._use_blanks(card._ncards())
            self.count_blanks()
            self._mod = 1
        else:
            raise SyntaxError, "%s is not a Card" % str(card)

    def _pos_insert(self, card, before, after, useblanks=1):
        """
        Insert a `Card` to the location specified by before or after.

        The argument `before` takes precedence over `after` if both
        specified.  They can be either a keyword name or index.
        """

        if before != None:
            loc = self.index_of(before)
            self.insert(loc, card, useblanks=useblanks)
        elif after != None:
            loc = self.index_of(after)
            self.insert(loc+1, card, useblanks=useblanks)

    def insert(self, pos, card, useblanks=True):
        """
        Insert a `Card` to the `CardList`.

        Parameters
        ----------
        pos : int
            The position (index, keyword name will not be allowed) to
            insert. The new card will be inserted before it.

        card : `Card` object
            The card to be inserted.

        useblanks : bool, optional
            If `useblanks` is `True`, and if there are blank cards
            directly before ``END``, it will use this space first,
            instead of appending after these blank cards, so the total
            space will not increase.  When `useblanks` is `False`, the
            card will be appended at the end, even if there are blank
            cards in front of ``END``.
        """

        if isinstance (card, Card):
            super(CardList, self).insert(pos, card)
            self._keylist.insert(pos, card.key)  # update the keylist
            self.count_blanks()
            if useblanks:
                self._use_blanks(card._ncards())

            self.count_blanks()
            self._mod = 1
        else:
            raise SyntaxError, "%s is not a Card" % str(card)

    def _use_blanks(self, how_many):
        if self._blanks > 0:
            for i in range(min(self._blanks, how_many)):
                del self[-1] # it also delete the keylist item

    def keys(self):
        """
        Return a list of all keywords from the `CardList`.

        Keywords include ``field_specifier`` for
        `RecordValuedKeywordCard` objects.
        """
        rtnVal = []

        for card in self:
            if isinstance(card, RecordValuedKeywordCard):
                key = card.key+'.'+card.field_specifier
            else:
                key = card.key

            rtnVal.append(key)

        return rtnVal

    def _keys(self):
        """
        Return a list of all keywords from the `CardList`.
        """
        return map(lambda x: getattr(x,'key'), self)

    def values(self):
        """
        Return a list of the values of all cards in the `CardList`.

        For `RecordValuedKeywordCard` objects, the value returned is
        the floating point value, exclusive of the
        ``field_specifier``.
        """
        return map(lambda x: getattr(x,'value'), self)

    def index_of(self, key, backward=False):
        """
        Get the index of a keyword in the `CardList`.

        Parameters
        ----------
        key : str or int
            The keyword name (a string) or the index (an integer).

        backward : bool, optional
            When `True`, search the index from the ``END``, i.e.,
            backward.

        Returns
        -------
        index : int
            The index of the `Card` with the given keyword.
        """
        if isinstance(key, (int, long,np.integer)):
            return key
        elif isinstance(key, str):
            _key = key.strip().upper()
            if _key[:8] == 'HIERARCH':
                _key = _key[8:].strip()
            _keylist = self._keylist
            if backward:
                _keylist = self._keylist[:]  # make a copy
                _keylist.reverse()
            try:
                _indx = _keylist.index(_key)
            except ValueError:
                requestedKey = RecordValuedKeywordCard.validKeyValue(key)
                _indx = 0

                while requestedKey:
                    try:
                        i = _keylist[_indx:].index(requestedKey[0].upper())
                        _indx = i + _indx

                        if isinstance(self[_indx], RecordValuedKeywordCard) \
                        and requestedKey[1] == self[_indx].field_specifier:
                            break
                    except ValueError:
                        raise KeyError, 'Keyword %s not found.' % `key`

                    _indx = _indx + 1
                else:
                    raise KeyError, 'Keyword %s not found.' % `key`

            if backward:
                _indx = len(_keylist) - _indx - 1
            return _indx
        else:
            raise KeyError, 'Illegal key data type %s' % type(key)

    def copy(self):
        """
        Make a (deep)copy of the `CardList`.
        """
        return CardList([createCardFromString(repr(c)) for c in self])

    def __repr__(self):
        """
        Format a list of cards into a string.
        """
        return ''.join(map(repr,self))

    def __str__(self):
        """
        Format a list of cards into a printable string.
        """
        return '\n'.join(map(str,self))


# ----------------------------- HDU classes ------------------------------------

class _AllHDU(object):
    """
    Base class for all HDU (header data unit) classes.
    """
    def __init__(self, data=None, header=None):
        self._header = header

        if (data is DELAYED):
            return
        else:
            self.data = data

    def __getattr__(self, attr):
        if attr == 'header':
            return self.__dict__['_header']

        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        """
        Set an HDU attribute.
        """
        if attr == 'header':
            self._header = value
        else:
            object.__setattr__(self,attr,value)

class _CorruptedHDU(_AllHDU):
    """
    A Corrupted HDU class.

    This class is used when one or more mandatory `Card`s are
    corrupted (unparsable), such as the ``BITPIX``, ``NAXIS``, or
    ``END`` cards.  A corrupted HDU usually means that the data size
    cannot be calculated or the ``END`` card is not found.  In the case
    of a missing ``END`` card, the `Header` may also contain the binary
    data

    .. note::
       In future, it may be possible to decipher where the last block
       of the `Header` ends, but this task may be difficult when the
       extension is a `TableHDU` containing ASCII data.
    """
    def __init__(self, data=None, header=None):
        super(_CorruptedHDU, self).__init__(data, header)
        self._file, self._offset, self._datLoc = None, None, None
        self.name = None

    def size(self):
        """
        Returns the size (in bytes) of the HDU's data part.
        """
        self._file.seek(0, 2)
        return self._file.tell() - self._datLoc

    def _summary(self):
        return "%-10s  %-11s" % (self.name, "CorruptedHDU")

    def verify(self):
        pass


class _NonstandardHDU(_AllHDU, _Verify):
    """
    A Non-standard HDU class.

    This class is used for a Primary HDU when the ``SIMPLE`` Card has
    a value of `False`.  A non-standard HDU comes from a file that
    resembles a FITS file but departs from the standards in some
    significant way.  One example would be files where the numbers are
    in the DEC VAX internal storage format rather than the standard
    FITS most significant byte first.  The header for this HDU should
    be valid.  The data for this HDU is read from the file as a byte
    stream that begins at the first byte after the header ``END`` card
    and continues until the end of the file.
    """
    def __init__(self, data=None, header=None):
        super(_NonstandardHDU, self).__init__(data, header)
        self._file, self._offset, self._datLoc = None, None, None
        self.name = None

    def size(self):
        """
        Returns the size (in bytes) of the HDU's data part.
        """
        self._file.seek(0, 2)
        return self._file.tell() - self._datLoc

    def _summary(self):
        return "%-7s  %-11s  %5d" % (self.name, "NonstandardHDU",
                                     len(self._header.ascard))

    def __getattr__(self, attr):
        """
        Get the data attribute.
        """
        if attr == 'data':
            self.__dict__[attr] = None
            self._file.seek(self._datLoc)
            self.data = self._file.read()
        else:
            return _AllHDU.__getattr__(self, attr)

        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def _verify(self, option='warn'):
        _err = _ErrList([], unit='Card')

        # verify each card
        for _card in self._header.ascard:
            _err.append(_card._verify(option))

        return _err

    def writeto(self, name, output_verify='exception', clobber=False,
                classExtensions={}, checksum=False):
        """
        Write the HDU to a new file.  This is a convenience method to
        provide a user easier output interface if only one HDU needs
        to be written to a file.

        Parameters
        ----------
        name : file path, file object or file-like object
            Output FITS file.  If opened, must be opened for append
            ("ab+")).

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.

        clobber : bool
            Overwrite the output file if exists.

        classExtensions : dict
            A dictionary that maps pyfits classes to extensions of
            those classes.  When present in the dictionary, the
            extension class will be constructed in place of the pyfits
            class.

        checksum : bool
            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the header of the HDU when written to the file.
        """

        if classExtensions.has_key(HDUList):
            hdulist = classExtensions[HDUList]([self])
        else:
            hdulist = HDUList([self])

        hdulist.writeto(name, output_verify, clobber=clobber,
                        checksum=checksum, classExtensions=classExtensions)


class _ValidHDU(_AllHDU, _Verify):
    """
    Base class for all HDUs which are not corrupted.
    """

    # 0.6.5.5
    def size(self):
        """
        Size (in bytes) of the data portion of the HDU.
        """
        size = 0
        naxis = self._header.get('NAXIS', 0)
        if naxis > 0:
            size = 1
            for j in range(naxis):
                size = size * self._header['NAXIS'+`j+1`]
            bitpix = self._header['BITPIX']
            gcount = self._header.get('GCOUNT', 1)
            pcount = self._header.get('PCOUNT', 0)
            size = abs(bitpix) * gcount * (pcount + size) // 8
        return size

    def copy(self):
        """
        Make a copy of the HDU, both header and data are copied.
        """
        if self.data is not None:
            _data = self.data.copy()
        else:
            _data = None
        return self.__class__(data=_data, header=self._header.copy())

    def writeto(self, name, output_verify='exception', clobber=False,
                classExtensions={}, checksum=False):
        """
        Write the HDU to a new file.  This is a convenience method to
        provide a user easier output interface if only one HDU needs
        to be written to a file.

        Parameters
        ----------
        name : file path, file object or file-like object
            Output FITS file.  If opened, must be opened for append
            ("ab+")).

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.

        clobber : bool
            Overwrite the output file if exists, default = False.

        classExtensions : dict
           A dictionary that maps pyfits classes to extensions of
           those classes.  When present in the dictionary, the
           extension class will be constructed in place of the pyfits
           class.

        checksum : bool
            When `True`, adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the header of the HDU when written to the file.
        """

        if isinstance(self, _ExtensionHDU):
            if classExtensions.has_key(HDUList):
                hdulist = classExtensions[HDUList]([PrimaryHDU(),self])
            else:
                hdulist = HDUList([PrimaryHDU(), self])
        elif isinstance(self, PrimaryHDU):
            if classExtensions.has_key(HDUList):
                hdulist = classExtensions[HDUList]([self])
            else:
                hdulist = HDUList([self])
        hdulist.writeto(name, output_verify, clobber=clobber,
                        checksum=checksum, classExtensions=classExtensions)

    def _verify(self, option='warn'):
        _err = _ErrList([], unit='Card')

        isValid = "val in [8, 16, 32, 64, -32, -64]"

        # Verify location and value of mandatory keywords.
        # Do the first card here, instead of in the respective HDU classes,
        # so the checking is in order, in case of required cards in wrong order.
        if isinstance(self, _ExtensionHDU):
            firstkey = 'XTENSION'
            firstval = self._xtn
        else:
            firstkey = 'SIMPLE'
            firstval = True
        self.req_cards(firstkey, '== 0', '', firstval, option, _err)
        self.req_cards('BITPIX', '== 1', _isInt+" and "+isValid, 8, option, _err)
        self.req_cards('NAXIS', '== 2', _isInt+" and val >= 0 and val <= 999", 0, option, _err)

        naxis = self._header.get('NAXIS', 0)
        if naxis < 1000:
            for j in range(3, naxis+3):
                self.req_cards('NAXIS'+`j-2`, '== '+`j`, _isInt+" and val>= 0", 1, option, _err)
            # Remove NAXISj cards where j is not in range 1, naxis inclusive.
            for _card in self._header.ascard:
                if _card.key.startswith("NAXIS") and len(_card.key) > 5:
                    try:
                        number = int(_card.key[5:])
                        if number <= 0 or number > naxis:
                            raise ValueError
                    except ValueError:
                        _err.append(self.run_option(
                                option=option,
                                err_text=("NAXISj keyword out of range ('%s' when NAXIS == %d)" %
                                          (_card.key, naxis)),
                                fix="del self._header['%s']" % _card.key,
                                fix_text="Deleted."))

        # verify each card
        for _card in self._header.ascard:
            _err.append(_card._verify(option))

        return _err

    def req_cards(self, keywd, pos, test, fix_value, option, errlist):
        """
        Check the existence, location, and value of a required `Card`.

        TODO: Write about parameters

        If `pos` = `None`, it can be anywhere.  If the card does not exist,
        the new card will have the `fix_value` as its value when created.
        Also check the card's value by using the `test` argument.
        """
        _err = errlist
        fix = ''
        cards = self._header.ascard
        try:
            _index = cards.index_of(keywd)
        except:
            _index = None
        fixable = fix_value is not None

        insert_pos = len(cards)+1

        # if pos is a string, it must be of the syntax of "> n",
        # where n is an int
        if isinstance(pos, str):
            _parse = pos.split()
            if _parse[0] in ['>=', '==']:
                insert_pos = eval(_parse[1])

        # if the card does not exist
        if _index is None:
            err_text = "'%s' card does not exist." % keywd
            fix_text = "Fixed by inserting a new '%s' card." % keywd
            if fixable:

                # use repr to accomodate both string and non-string types
                # Boolean is also OK in this constructor
                _card = "Card('%s', %s)" % (keywd, `fix_value`)
                fix = "self._header.ascard.insert(%d, %s)" % (insert_pos, _card)
            _err.append(self.run_option(option, err_text=err_text, fix_text=fix_text, fix=fix, fixable=fixable))
        else:

            # if the supposed location is specified
            if pos is not None:
                test_pos = '_index '+ pos
                if not eval(test_pos):
                    err_text = "'%s' card at the wrong place (card %d)." % (keywd, _index)
                    fix_text = "Fixed by moving it to the right place (card %d)." % insert_pos
                    fix = "_cards=self._header.ascard; dummy=_cards[%d]; del _cards[%d];_cards.insert(%d, dummy)" % (_index, _index, insert_pos)
                    _err.append(self.run_option(option, err_text=err_text, fix_text=fix_text, fix=fix))

            # if value checking is specified
            if test:
                val = self._header[keywd]
                if not eval(test):
                    err_text = "'%s' card has invalid value '%s'." % (keywd, val)
                    fix_text = "Fixed by setting a new value '%s'." % fix_value
                    if fixable:
                        fix = "self._header['%s'] = %s" % (keywd, `fix_value`)
                    _err.append(self.run_option(option, err_text=err_text, fix_text=fix_text, fix=fix, fixable=fixable))

        return _err

    def _compute_checksum(self, bytes, sum32=0):
        """
        Compute the ones-complement checksum of a sequence of bytes.

        Parameters
        ----------
        bytes
            a memory region to checksum

        sum32
            incremental checksum value from another region

        Returns
        -------
        ones complement checksum
        """
        # Use uint32 literals as a hedge against type promotion to int64.
        u8 = np.array(8, dtype='uint32')
        u16 = np.array(16, dtype='uint32')
        uFFFF = np.array(0xFFFF, dtype='uint32')

        b0 = bytes[0::4].astype('uint32') << u8
        b1 = bytes[1::4].astype('uint32')
        b2 = bytes[2::4].astype('uint32') << u8
        b3 = bytes[3::4].astype('uint32')

        hi = np.array(sum32, dtype='uint32') >> u16
        lo = np.array(sum32, dtype='uint32') & uFFFF

        hi += np.add.reduce((b0 + b1)).astype('uint32')
        lo += np.add.reduce((b2 + b3)).astype('uint32')

        hicarry = hi >> u16
        locarry = lo >> u16

        while int(hicarry) or int(locarry):
            hi = (hi & uFFFF) + locarry
            lo = (lo & uFFFF) + hicarry
            hicarry = hi >> u16
            locarry = lo >> u16

        return (hi << u16) + lo


    # _MASK and _EXCLUDE used for encoding the checksum value into a character
    # string.
    _MASK = [ 0xFF000000,
              0x00FF0000,
              0x0000FF00,
              0x000000FF ]

    _EXCLUDE = [ 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40,
                 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60 ]

    def _encode_byte(self, byte):
        """
        Encode a single byte.
        """
        quotient = byte // 4 + ord('0')
        remainder = byte % 4

        ch = np.array(
            [(quotient + remainder), quotient, quotient, quotient],
            dtype='int32')

        check = True
        while check:
            check = False
            for x in self._EXCLUDE:
                for j in [0, 2]:
                    if ch[j] == x or ch[j+1] == x:
                        ch[j]   += 1
                        ch[j+1] -= 1
                        check = True
        return ch

    def _char_encode(self, value):
        """
        Encodes the checksum `value` using the algorithm described
        in SPR section A.7.2 and returns it as a 16 character string.

        Parameters
        ----------
        value
            a checksum

        Returns
        -------
        ascii encoded checksum
        """
        value = np.array(value, dtype='uint32')

        asc = np.zeros((16,), dtype='byte')
        ascii = np.zeros((16,), dtype='byte')

        for i in range(4):
            byte = (value & self._MASK[i]) >> ((3 - i) * 8)
            ch = self._encode_byte(byte)
            for j in range(4):
                asc[4*j+i] = ch[j]

        for i in range(16):
            ascii[i] = asc[(i+15) % 16]

        return ascii.tostring()

    def _datetime_str(self):
        """
        Time of now formatted like: 2007-05-30T19:05:11
        """
        now = str(datetime.datetime.now()).split()
        return now[0] + "T" + now[1].split(".")[0]

    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
        if (not self.__dict__.has_key('data')):
            # This is the case where the data has not been read from the file
            # yet.  We find the data in the file, read it, and calculate the
            # datasum.
            if self.size() > 0:
                self._file.seek(self._datLoc)
                raw_data = _fromfile(self._file, dtype='ubyte',
                                     count=self._datSpan, sep="")
                return self._compute_checksum(raw_data,0)
            else:
                return 0
        elif (self.data != None):
            return self._compute_checksum(
                                 np.fromstring(self.data, dtype='ubyte'),0)
        else:
            return 0

    def _calculate_checksum(self, datasum):
        """
        Calculate the value of the ``CHECKSUM`` card in the HDU.
        """
        oldChecksum = self.header['CHECKSUM']
        self.header.update('CHECKSUM', '0'*16);

        # Convert the header to a string.
        s = repr(self._header.ascard) + _pad('END')
        s = s + _padLength(len(s))*' '

        # Calculate the checksum of the Header and data.
        cs = self._compute_checksum(np.fromstring(s, dtype='ubyte'),datasum)

        # Encode the checksum into a string.
        s = self._char_encode(~cs)

        # Return the header card value.
        self.header.update("CHECKSUM", oldChecksum);

        return s

    def add_datasum(self, when=None):
        """
        Add the ``DATASUM`` card to this HDU with the value set to the
        checksum calculated for the data.

        Parameters
        ----------
        when : str, optional
            Comment string for the card that by default represents the
            time when the checksum was calculated

        Returns
        -------
        checksum : int
            The calculated datasum

        Notes
        -----
        For testing purposes, provide a `when` argument to enable the
        comment value in the card to remain consistent.  This will
        enable the generation of a ``CHECKSUM`` card with a consistent
        value.
        """
        cs = self._calculate_datasum()

        if when is None:
           when = "data unit checksum updated " + self._datetime_str()

        self.header.update("DATASUM", str(cs), when);
        return cs

    def add_checksum(self, when=None, override_datasum=False):
        """
        Add the ``CHECKSUM`` and ``DATASUM`` cards to this HDU with
        the values set to the checksum calculated for the HDU and the
        data respectively.  The addition of the ``DATASUM`` card may
        be overridden.

        Parameters
        ----------
        when : str, optional
           comment string for the cards; by default the comments
           will represent the time when the checksum was calculated

        override_datasum : bool, optional
           add the ``CHECKSUM`` card only

        Notes
        -----
        For testing purposes, first call `add_datasum` with a `when`
        argument, then call `add_checksum` with a `when` argument and
        `override_datasum` set to `True`.  This will provide
        consistent comments for both cards and enable the generation
        of a ``CHECKSUM`` card with a consistent value.
        """

        if not override_datasum:
           # Calculate and add the data checksum to the header.
           data_cs = self.add_datasum(when)
        else:
           # Just calculate the data checksum
           data_cs = self._calculate_datasum()

        if when is None:
            when = "HDU checksum updated " + self._datetime_str()

        # Add the CHECKSUM card to the header with a value of all zeros.
        if self.header.has_key("DATASUM"):
            self.header.update("CHECKSUM", "0"*16, when, before='DATASUM');
        else:
            self.header.update("CHECKSUM", "0"*16, when);

        s = self._calculate_checksum(data_cs)

        # Update the header card.
        self.header.update("CHECKSUM", s, when);

    def verify_datasum(self):
        """
        Verify that the value in the ``DATASUM`` keyword matches the value
        calculated for the ``DATASUM`` of the current HDU data.

        Returns
        -------
        valid : int
           - 0 - failure
           - 1 - success
           - 2 - no ``DATASUM`` keyword present
        """
        if self.header.has_key('DATASUM'):
            if self._calculate_datasum() == int(self.header['DATASUM']):
                return 1
            else:
                return 0
        else:
            return 2

    def verify_checksum(self):
        """
        Verify that the value in the ``CHECKSUM`` keyword matches the
        value calculated for the current HDU CHECKSUM.

        Returns
        -------
        valid : int
           - 0 - failure
           - 1 - success
           - 2 - no ``CHECKSUM`` keyword present
        """
        if self._header.has_key('CHECKSUM'):
            if self._header.has_key('DATASUM'):
                datasum = self._calculate_datasum()
            else:
                datasum = 0
            if self._calculate_checksum(datasum) == self.header['CHECKSUM']:
                return 1
            else:
                return 0
        else:
            return 2


class _TempHDU(_ValidHDU):
    """
    Temporary HDU, used when the file is first opened. This is to
    speed up the open.  Any header will not be initialized till the
    HDU is accessed.
    """

    def _getname(self):
        """
        Get the ``EXTNAME`` and ``EXTVER`` from the header.
        """
        re_extname = re.compile(r"EXTNAME\s*=\s*'([ -&(-~]*)'")
        re_extver = re.compile(r"EXTVER\s*=\s*(\d+)")

        mo = re_extname.search(self._raw)
        if mo:
            name = mo.group(1).rstrip()
        else:
            name = ''

        mo = re_extver.search(self._raw)
        if mo:
            extver = int(mo.group(1))
        else:
            extver = 1

        return name, extver

    def _getsize(self, block):
        """
        Get the size from the first block of the HDU.
        """
        re_simple = re.compile(r'SIMPLE  =\s*')
        re_bitpix = re.compile(r'BITPIX  =\s*(-?\d+)')
        re_naxis = re.compile(r'NAXIS   =\s*(\d+)')
        re_naxisn = re.compile(r'NAXIS(\d)  =\s*(\d+)')
        re_gcount = re.compile(r'GCOUNT  =\s*(-?\d+)')
        re_pcount = re.compile(r'PCOUNT  =\s*(-?\d+)')
        re_groups = re.compile(r'GROUPS  =\s*(T)')

        simple = re_simple.search(block[:80])
        mo = re_bitpix.search(block)
        if mo is not None:
            bitpix = int(mo.group(1))
        else:
            raise ValueError("BITPIX not found where expected")

        mo = re_gcount.search(block)
        if mo is not None:
            gcount = int(mo.group(1))
        else:
            gcount = 1

        mo = re_pcount.search(block)
        if mo is not None:
            pcount = int(mo.group(1))
        else:
            pcount = 0

        mo = re_groups.search(block)
        if mo and simple:
            groups = 1
        else:
            groups = 0

        mo = re_naxis.search(block)
        if mo is not None:
            naxis = int(mo.group(1))
            pos = mo.end(0)
        else:
            raise ValueError("NAXIS not found where expected")

        if naxis == 0:
            datasize = 0
        else:
            dims = [0]*naxis
            for i in range(naxis):
                mo = re_naxisn.search(block, pos)
                pos = mo.end(0)
                dims[int(mo.group(1))-1] = int(mo.group(2))
            datasize = reduce(operator.mul, dims[groups:])
        size = abs(bitpix) * gcount * (pcount + datasize) // 8

        if simple and not groups:
            name = 'PRIMARY'
        else:
            name = ''

        return size, name

    def setupHDU(self, classExtensions={}):
        """
        Read one FITS HDU, data portions are not actually read here,
        but the beginning locations are computed.
        """
        _cardList = []
        _keyList = []

        blocks = self._raw
        if (len(blocks) % _blockLen) != 0:
            raise IOError, 'Header size is not multiple of %d: %d' % (_blockLen, len(blocks))
        elif (blocks[:8] not in ['SIMPLE  ', 'XTENSION']):
            raise IOError, 'Block does not begin with SIMPLE or XTENSION'

        for i in range(0, len(blocks), Card.length):
            _card = createCardFromString(blocks[i:i+Card.length])
            _key = _card.key

            if _key == 'END':
                break
            else:
                _cardList.append(_card)
                _keyList.append(_key)

        # Deal with CONTINUE cards
        # if a long string has CONTINUE cards, the "Card" is considered
        # to be more than one 80-char "physical" cards.
        _max = _keyList.count('CONTINUE')
        _start = 0
        for i in range(_max):
            _where = _keyList[_start:].index('CONTINUE') + _start
            for nc in range(1, _max+1):
                if _where+nc >= len(_keyList):
                    break
                if _cardList[_where+nc]._cardimage[:10].upper() != 'CONTINUE  ':
                    break

            # combine contiguous CONTINUE cards with its parent card
            if nc > 0:
                _longstring = _cardList[_where-1]._cardimage
                for c in _cardList[_where:_where+nc]:
                    _longstring += c._cardimage
                _cardList[_where-1] = _Card_with_continue().fromstring(_longstring)
                del _cardList[_where:_where+nc]
                del _keyList[_where:_where+nc]
                _start = _where

            # if not the real CONTINUE card, skip to the next card to search
            # to avoid starting at the same CONTINUE card
            else:
                _start = _where + 1
            if _keyList[_start:].count('CONTINUE') == 0:
                break

        # construct the Header object, using the cards.
        try:
            header = Header(CardList(_cardList, keylist=_keyList))

            if classExtensions.has_key(header._hdutype):
                header._hdutype = classExtensions[header._hdutype]

            hdu = header._hdutype(data=DELAYED, header=header)

            # pass these attributes
            hdu._file = self._file
            hdu._hdrLoc = self._hdrLoc
            hdu._datLoc = self._datLoc
            hdu._datSpan = self._datSpan
            hdu._ffile = self._ffile
            hdu.name = self.name
            hdu._extver = self._extver
            hdu._new = 0
            hdu.header._mod = 0
            hdu.header.ascard._mod = 0
        except:
            pass

        return hdu

    def isPrimary(self):
        blocks = self._raw

        if (blocks[:8] == 'SIMPLE  '):
           return True
        else:
           return False

class _ExtensionHDU(_ValidHDU):
    """
    An extension HDU class.

    This class is the base class for the `TableHDU`, `ImageHDU`, and
    `BinTableHDU` classes.
    """

    def __init__(self, data=None, header=None):
        super(_ExtensionHDU, self).__init__(data, header)
        self._file, self._offset, self._datLoc = None, None, None
        self._xtn = ' '

    def __setattr__(self, attr, value):
        """
        Set an HDU attribute.
        """

        if attr == 'name' and value:
            if not isinstance(value, str):
                raise TypeError, 'bad value type'
            if not _extensionNameCaseSensitive:
                value = value.upper()
            if self._header.has_key('EXTNAME'):
                self._header['EXTNAME'] = value
            else:
                self._header.ascard.append(Card('EXTNAME', value, 'extension name'))

        _ValidHDU.__setattr__(self,attr,value)

    def _verify(self, option='warn'):
        _err = _ValidHDU._verify(self, option=option)

        # Verify location and value of mandatory keywords.
        naxis = self._header.get('NAXIS', 0)
        self.req_cards('PCOUNT', '== '+`naxis+3`, _isInt+" and val >= 0", 0, option, _err)
        self.req_cards('GCOUNT', '== '+`naxis+4`, _isInt+" and val == 1", 1, option, _err)
        return _err

class _NonstandardExtHDU(_ExtensionHDU):
    """
    A Non-standard Extension HDU class.

    This class is used for an Extension HDU when the ``XTENSION``
    `Card` has a non-standard value.  In this case, pyfits can figure
    out how big the data is but not what it is.  The data for this HDU
    is read from the file as a byte stream that begins at the first
    byte after the header ``END`` card and continues until the
    beginning of the next header or the end of the file.
    """
    def __init__(self, data=None, header=None):
        super(_NonstandardExtHDU, self).__init__(data, header)
        self._file, self._offset, self._datLoc = None, None, None
        self.name = None

    def _summary(self):
        return "%-6s  %-10s  %3d" % (self.name, "NonstandardExtHDU",
                                     len(self._header.ascard))

    def __getattr__(self, attr):
        """
        Get the data attribute.
        """
        if attr == 'data':
            self.__dict__[attr] = None
            self._file.seek(self._datLoc)
            self.data = self._file.read(self.size())
        else:
            return _ValidHDU.__getattr__(self, attr)

        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def writeto(self, name, output_verify='exception', clobber=False,
                classExtensions={}, checksum=False):
        """
        Write the HDU to a new file.  This is a convenience method to
        provide a user easier output interface if only one HDU needs
        to be written to a file.

        Parameters
        ----------
        name : file path, file object or file-like object
            Output FITS file.  If opened, must be opened for append
            (ab+)).

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.

        clobber : bool
            Overwrite the output file if exists.

        classExtensions : dict
            A dictionary that maps pyfits classes to extensions of
            those classes.  When present in the dictionary, the
            extension class will be constructed in place of the pyfits
            class.

        checksum : bool
            When `True`, adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the header of the HDU when written to the file.
        """

        if classExtensions.has_key(HDUList):
            hdulist = classExtensions[HDUList]([PrimaryHDU(),self])
        else:
            hdulist = HDUList([PrimaryHDU(),self])

        hdulist.writeto(name, output_verify, clobber=clobber,
                        checksum=checksum, classExtensions=classExtensions)


# 0.8.8
def _iswholeline(indx, naxis):
    if isinstance(indx, (int, long,np.integer)):
        if indx >= 0 and indx < naxis:
            if naxis > 1:
                return _SinglePoint(1, indx)
            elif naxis == 1:
                return _OnePointAxis(1, 0)
        else:
            raise IndexError, 'Index %s out of range.' % indx
    elif isinstance(indx, slice):
        indx = _normalize_slice(indx, naxis)
        if (indx.start == 0) and (indx.stop == naxis) and (indx.step == 1):
            return _WholeLine(naxis, 0)
        else:
            if indx.step == 1:
                return _LineSlice(indx.stop-indx.start, indx.start)
            else:
                return _SteppedSlice((indx.stop-indx.start)//indx.step, indx.start)
    else:
        raise IndexError, 'Illegal index %s' % indx


def _normalize_slice(input, naxis):
    """
    Set the slice's start/stop in the regular range.
    """

    def _normalize(indx, npts):
        if indx < -npts:
            indx = 0
        elif indx < 0:
            indx += npts
        elif indx > npts:
            indx = npts
        return indx

    _start = input.start
    if _start is None:
        _start = 0
    elif isinstance(_start, (int, long,np.integer)):
        _start = _normalize(_start, naxis)
    else:
        raise IndexError, 'Illegal slice %s, start must be integer.' % input

    _stop = input.stop
    if _stop is None:
        _stop = naxis
    elif isinstance(_stop, (int, long,np.integer)):
        _stop = _normalize(_stop, naxis)
    else:
        raise IndexError, 'Illegal slice %s, stop must be integer.' % input

    if _stop < _start:
        raise IndexError, 'Illegal slice %s, stop < start.' % input

    _step = input.step
    if _step is None:
        _step = 1
    elif isinstance(_step, (int, long, np.integer)):
        if _step <= 0:
            raise IndexError, 'Illegal slice %s, step must be positive.' % input
    else:
        raise IndexError, 'Illegal slice %s, step must be integer.' % input

    return slice(_start, _stop, _step)


class _KeyType:
    def __init__(self, npts, offset):
        self.npts = npts
        self.offset = offset


class _WholeLine(_KeyType):
    pass


class _SinglePoint(_KeyType):
    pass


class _OnePointAxis(_KeyType):
    pass


class _LineSlice(_KeyType):
    pass


class _SteppedSlice(_KeyType):
    pass


class Section:
    """
    Image section.

    TODO: elaborate
    """
    def __init__(self, hdu):
        self.hdu = hdu

    def __getitem__(self, key):
        dims = []
        if not isinstance(key, tuple):
            key = (key,)
        naxis = self.hdu.header['NAXIS']
        if naxis < len(key):
            raise IndexError, 'too many indices.'
        elif naxis > len(key):
            key = key + (slice(None),) * (naxis-len(key))

        offset = 0
        for i in range(naxis):
            _naxis = self.hdu.header['NAXIS'+`naxis-i`]
            indx = _iswholeline(key[i], _naxis)
            offset = offset * _naxis + indx.offset

            # all elements after the first WholeLine must be WholeLine or
            # OnePointAxis
            if isinstance(indx, (_WholeLine, _LineSlice)):
                dims.append(indx.npts)
                break
            elif isinstance(indx, _SteppedSlice):
                raise IndexError, 'Subsection data must be contiguous.'

        for j in range(i+1,naxis):
            _naxis = self.hdu.header['NAXIS'+`naxis-j`]
            indx = _iswholeline(key[j], _naxis)
            dims.append(indx.npts)
            if not isinstance(indx, _WholeLine):
                raise IndexError, 'Subsection data is not contiguous.'

            # the offset needs to multiply the length of all remaining axes
            else:
                offset *= _naxis

        if dims == []:
            dims = [1]
        npt = 1
        for n in dims:
            npt *= n

        # Now, get the data (does not include bscale/bzero for now XXX)
        _bitpix = self.hdu.header['BITPIX']
        code = _ImageBaseHDU.NumCode[_bitpix]
        self.hdu._file.seek(self.hdu._datLoc+offset*abs(_bitpix)//8)
        nelements = 1
        for dim in dims:
            nelements = nelements*dim
        raw_data = _fromfile(self.hdu._file, dtype=code, count=nelements, sep="")
        raw_data.shape = dims
#        raw_data._byteorder = 'big'
        raw_data.dtype = raw_data.dtype.newbyteorder(">")
        return raw_data


class _ImageBaseHDU(_ValidHDU):
    """FITS image HDU base class.

    Attributes
    ----------
    header
        image header

    data
        image data

    _file
        file associated with array

    _datLoc
        starting byte location of data block in file
    """

    # mappings between FITS and numpy typecodes
#    NumCode = {8:'int8', 16:'int16', 32:'int32', 64:'int64', -32:'float32', -64:'float64'}
#    ImgCode = {'<i2':8, '<i4':16, '<i8':32, '<i16':64, '<f8':-32, '<f16':-64}
    NumCode = {8:'uint8', 16:'int16', 32:'int32', 64:'int64', -32:'float32', -64:'float64'}
    ImgCode = {'uint8':8, 'int16':16, 'uint16':16, 'int32':32,
               'uint32':32, 'int64':64, 'uint64':64,
               'float32':-32, 'float64':-64}

    def __init__(self, data=None, header=None):
        self._file, self._datLoc = None, None

        if header is not None:
            if not isinstance(header, Header):
                raise ValueError, "header must be a Header object"


        if data is DELAYED:

            # this should never happen
            if header is None:
                raise ValueError, "No header to setup HDU."

            # if the file is read the first time, no need to copy, and keep it unchanged
            else:
                self._header = header
        else:

            # construct a list of cards of minimal header
            if isinstance(self, _ExtensionHDU):
                c0 = Card('XTENSION', 'IMAGE', 'Image extension')
            else:
                c0 = Card('SIMPLE', True, 'conforms to FITS standard')

            _list = CardList([
                c0,
                Card('BITPIX',    8, 'array data type'),
                Card('NAXIS',     0, 'number of array dimensions'),
                ])
            if isinstance(self, GroupsHDU):
                _list.append(Card('GROUPS', True, 'has groups'))

            if isinstance(self, (_ExtensionHDU, GroupsHDU)):
                _list.append(Card('PCOUNT',    0, 'number of parameters'))
                _list.append(Card('GCOUNT',    1, 'number of groups'))

            if header is not None:
                hcopy = header.copy()
                hcopy._strip()
                _list.extend(hcopy.ascardlist())

            self._header = Header(_list)

        self._bzero = self._header.get('BZERO', 0)
        self._bscale = self._header.get('BSCALE', 1)

        if (data is DELAYED): return

        self.data = data

        # update the header
        self.update_header()
        self._bitpix = self._header['BITPIX']

        # delete the keywords BSCALE and BZERO
        del self._header['BSCALE']
        del self._header['BZERO']

    def update_header(self):
        """
        Update the header keywords to agree with the data.
        """
        old_naxis = self._header.get('NAXIS', 0)

        if isinstance(self.data, GroupData):
            self._header['BITPIX'] = _ImageBaseHDU.ImgCode[
                      self.data.dtype.fields[self.data.dtype.names[0]][0].name]
            axes = list(self.data.data.shape)[1:]
            axes.reverse()
            axes = [0] + axes

        elif isinstance(self.data, np.ndarray):
            self._header['BITPIX'] = _ImageBaseHDU.ImgCode[self.data.dtype.name]
            axes = list(self.data.shape)
            axes.reverse()

        elif self.data is None:
            axes = []
        else:
            raise ValueError, "incorrect array type"

        self._header['NAXIS'] = len(axes)

        # add NAXISi if it does not exist
        for j in range(len(axes)):
            try:
                self._header['NAXIS'+`j+1`] = axes[j]
            except KeyError:
                if (j == 0):
                    _after = 'naxis'
                else :
                    _after = 'naxis'+`j`
                self._header.update('naxis'+`j+1`, axes[j], after = _after)

        # delete extra NAXISi's
        for j in range(len(axes)+1, old_naxis+1):
            try:
                del self._header.ascard['NAXIS'+`j`]
            except KeyError:
                pass

        if isinstance(self.data, GroupData):
            self._header.update('GROUPS', True, after='NAXIS'+`len(axes)`)
            self._header.update('PCOUNT', len(self.data.parnames), after='GROUPS')
            self._header.update('GCOUNT', len(self.data), after='PCOUNT')
            npars = len(self.data.parnames)
            (_scale, _zero)  = self.data._get_scale_factors(npars)[3:5]
            if _scale:
                self._header.update('BSCALE', self.data._coldefs.bscales[npars])
            if _zero:
                self._header.update('BZERO', self.data._coldefs.bzeros[npars])
            for i in range(npars):
                self._header.update('PTYPE'+`i+1`, self.data.parnames[i])
                (_scale, _zero)  = self.data._get_scale_factors(i)[3:5]
                if _scale:
                    self._header.update('PSCAL'+`i+1`, self.data._coldefs.bscales[i])
                if _zero:
                    self._header.update('PZERO'+`i+1`, self.data._coldefs.bzeros[i])

    def __getattr__(self, attr):
        """
        Get the data attribute.
        """
        if attr == 'section':
            return Section(self)
        elif attr == 'data':
            self.__dict__[attr] = None
            if self._header['NAXIS'] > 0:
                _bitpix = self._header['BITPIX']
                self._file.seek(self._datLoc)
                if isinstance(self, GroupsHDU):
                    dims = self.size()*8//abs(_bitpix)
                else:
                    dims = self._dimShape()

                code = _ImageBaseHDU.NumCode[self._header['BITPIX']]

                if self._ffile.memmap:
                    self._ffile.code = code
                    self._ffile.dims = dims
                    self._ffile.offset = self._datLoc
                    raw_data = self._ffile._mm
                else:

                    nelements = 1
                    for x in range(len(dims)):
                        nelements = nelements * dims[x]

                    raw_data = _fromfile(self._file, dtype=code,
                                         count=nelements,sep="")

                    raw_data.shape=dims

#                print "raw_data.shape: ",raw_data.shape
#                raw_data._byteorder = 'big'
                raw_data.dtype = raw_data.dtype.newbyteorder('>')

                if (self._bzero != 0 or self._bscale != 1):
                    data = None
                    # Handle "pseudo-unsigned" integers, if the user
                    # requested it.  In this case, we don't need to
                    # handle BLANK to convert it to NAN, since we
                    # can't do NaNs with integers, anyway, i.e. the
                    # user is responsible for managing blanks.
                    if self._ffile.uint and self._bscale == 1:
                        for bits, dtype in ((16, np.uint16),
                                            (32, np.uint32),
                                            (64, np.uint64)):
                            if _bitpix == bits and self._bzero == 1 << (bits - 1):
                                # Convert the input raw data into an unsigned
                                # integer array and then scale the data
                                # adjusting for the value of BZERO.  Note
                                # that we subtract the value of BZERO instead
                                # of adding because of the way numpy converts
                                # the raw signed array into an unsigned array.
                                data = np.array(raw_data, dtype=dtype)
                                data -= (1 << (bits - 1))
                                break

                    if data is None:
                        # In these cases, we end up with
                        # floating-point arrays and have to apply
                        # bscale and bzero. We may have to handle
                        # BLANK and convert to NaN in the resulting
                        # floating-point arrays.
                        if self._header.has_key('BLANK'):
                            nullDvals = np.array(self._header['BLANK'],
                                                 dtype='int64')
                            blanks = (raw_data == nullDvals)

                        if _bitpix > 16:  # scale integers to Float64
                            data = np.array(raw_data, dtype=np.float64)
                        elif _bitpix > 0:  # scale integers to Float32
                            data = np.array(raw_data, dtype=np.float32)
                        else:  # floating point cases
                            if self._ffile.memmap:
                                data = raw_data.copy()
                            # if not memmap, use the space already in memory
                            else:
                                data = raw_data

                        if self._bscale != 1:
                            np.multiply(data, self._bscale, data)
                        if self._bzero != 0:
                            data += self._bzero

                        if self._header.has_key('BLANK'):
                            data = np.where(blanks, np.nan, data)

                    self.data = data

                    # delete the keywords BSCALE and BZERO after scaling
                    del self._header['BSCALE']
                    del self._header['BZERO']
                    self._header['BITPIX'] = _ImageBaseHDU.ImgCode[self.data.dtype.name]
                else:
                    self.data = raw_data

        else:
            return _AllHDU.__getattr__(self, attr)

        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def _dimShape(self):
        """
        Returns a tuple of image dimensions, reverse the order of ``NAXIS``.
        """
        naxis = self._header['NAXIS']
        axes = naxis*[0]
        for j in range(naxis):
            axes[j] = self._header['NAXIS'+`j+1`]
        axes.reverse()
#        print "axes in _dimShape line 2081:",axes
        return tuple(axes)

    def _summary(self):
        """
        Summarize the HDU: name, dimensions, and formats.
        """
        class_name  = str(self.__class__)
        type  = class_name[class_name.rfind('.')+1:-2]

        if type.find('_') != -1:
            type = type[type.find('_')+1:]

        # if data is touched, use data info.
        if 'data' in dir(self):
            if self.data is None:
                _shape, _format = (), ''
            else:

                # the shape will be in the order of NAXIS's which is the
                # reverse of the numarray shape
                if isinstance(self, GroupsHDU):
                    _shape = list(self.data.data.shape)[1:]
                    _format = \
                       self.data.dtype.fields[self.data.dtype.names[0]][0].name
                else:
                    _shape = list(self.data.shape)
                    _format = self.data.dtype.name
                _shape.reverse()
                _shape = tuple(_shape)
                _format = _format[_format.rfind('.')+1:]

        # if data is not touched yet, use header info.
        else:
            _shape = ()
            for j in range(self._header['NAXIS']):
                if isinstance(self, GroupsHDU) and j == 0:
                    continue
                _shape += (self._header['NAXIS'+`j+1`],)
            _format = self.NumCode[self._header['BITPIX']]

        if isinstance(self, GroupsHDU):
            _gcount = '   %d Groups  %d Parameters' % (self._header['GCOUNT'], self._header['PCOUNT'])
        else:
            _gcount = ''
        return "%-10s  %-11s  %5d  %-12s  %s%s" % \
            (self.name, type, len(self._header.ascard), _shape, _format, _gcount)

    def scale(self, type=None, option="old", bscale=1, bzero=0):
        """
        Scale image data by using ``BSCALE``/``BZERO``.

        Call to this method will scale `data` and update the keywords
        of ``BSCALE`` and ``BZERO`` in `_header`.  This method should
        only be used right before writing to the output file, as the
        data will be scaled and is therefore not very usable after the
        call.

        Parameters
        ----------
        type : str, optional
            destination data type, use a string representing a numpy
            dtype name, (e.g. ``'uint8'``, ``'int16'``, ``'float32'``
            etc.).  If is `None`, use the current data type.

        option : str
            How to scale the data: if ``"old"``, use the original
            ``BSCALE`` and ``BZERO`` values when the data was
            read/created. If ``"minmax"``, use the minimum and maximum
            of the data to scale.  The option will be overwritten by
            any user specified `bscale`/`bzero` values.

        bscale, bzero : int, optional
            User-specified ``BSCALE`` and ``BZERO`` values.
        """

        if self.data is None:
            return

        # Determine the destination (numpy) data type
        if type is None:
            type = self.NumCode[self._bitpix]
        _type = getattr(np, type)

        # Determine how to scale the data
        # bscale and bzero takes priority
        if (bscale != 1 or bzero !=0):
            _scale = bscale
            _zero = bzero
        else:
            if option == 'old':
                _scale = self._bscale
                _zero = self._bzero
            elif option == 'minmax':
                if isinstance(_type, np.floating):
                    _scale = 1
                    _zero = 0
                else:

                    # flat the shape temporarily to save memory
                    dims = self.data.shape
                    self.data.shape = self.data.size
                    min = np.minimum.reduce(self.data)
                    max = np.maximum.reduce(self.data)
                    self.data.shape = dims

                    if _type == np.uint8:  # uint8 case
                        _zero = min
                        _scale = (max - min) / (2.**8 - 1)
                    else:
                        _zero = (max + min) / 2.

                        # throw away -2^N
                        _scale = (max - min) / (2.**(8*_type.bytes) - 2)

        # Do the scaling
        if _zero != 0:
            self.data += -_zero # 0.9.6.3 to avoid out of range error for BZERO = +32768
            self._header.update('BZERO', _zero)
        else:
            del self._header['BZERO']

        if _scale != 1:
            self.data /= _scale
            self._header.update('BSCALE', _scale)
        else:
            del self._header['BSCALE']

        if self.data.dtype.type != _type:
            self.data = np.array(np.around(self.data), dtype=_type) #0.7.7.1
        #
        # Update the BITPIX Card to match the data
        #
        self._header['BITPIX'] = _ImageBaseHDU.ImgCode[self.data.dtype.name]

    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
        if self.__dict__.has_key('data') and self.data != None:
            # We have the data to be used.
            d = self.data

            # First handle the special case where the data is unsigned integer
            # 16, 32 or 64
            if _is_pseudo_unsigned(self.data.dtype):
                d = np.array(self.data - _unsigned_zero(self.data.dtype),
                             dtype='i%d' % self.data.dtype.itemsize)

            # Check the byte order of the data.  If it is little endian we
            # must swap it before calculating the datasum.
            if d.dtype.str[0] != '>':
                byteswapped = True
                d = d.byteswap(True)
                d.dtype = d.dtype.newbyteorder('>')
            else:
                byteswapped = False

            cs = self._compute_checksum(np.fromstring(d, dtype='ubyte'),0)

            # If the data was byteswapped in this method then return it to
            # its original little-endian order.
            if byteswapped and not _is_pseudo_unsigned(self.data.dtype):
                d.byteswap(True)
                d.dtype = d.dtype.newbyteorder('<')

            return cs
        else:
            # This is the case where the data has not been read from the file
            # yet.  We can handle that in a generic manner so we do it in the
            # base class.  The other possibility is that there is no data at
            # all.  This can also be handled in a gereric manner.
            return super(_ImageBaseHDU,self)._calculate_datasum()

class PrimaryHDU(_ImageBaseHDU):
    """
    FITS primary HDU class.
    """
    def __init__(self, data=None, header=None):
        """
        Construct a primary HDU.

        Parameters
        ----------
        data : array or DELAYED, optional
            The data in the HDU.

        header : Header instance, optional
            The header to be used (as a template).  If `header` is
            `None`, a minimal header will be provided.
        """

        _ImageBaseHDU.__init__(self, data=data, header=header)
        self.name = 'PRIMARY'

        # insert the keywords EXTEND
        if header is None:
            dim = `self._header['NAXIS']`
            if dim == '0':
                dim = ''
            self._header.update('EXTEND', True, after='NAXIS'+dim)


class ImageHDU(_ExtensionHDU, _ImageBaseHDU):
    """
    FITS image extension HDU class.
    """

    def __init__(self, data=None, header=None, name=None):
        """
        Construct an image HDU.

        Parameters
        ----------
        data : array
            The data in the HDU.

        header : Header instance
            The header to be used (as a template).  If `header` is
            `None`, a minimal header will be provided.

        name : str, optional
            The name of the HDU, will be the value of the keyword
            ``EXTNAME``.
        """

        # no need to run _ExtensionHDU.__init__ since it is not doing anything.
        _ImageBaseHDU.__init__(self, data=data, header=header)
        self._xtn = 'IMAGE'

        self._header._hdutype = ImageHDU

        # insert the require keywords PCOUNT and GCOUNT
        dim = `self._header['NAXIS']`
        if dim == '0':
            dim = ''


        #  set extension name
        if (name is None) and self._header.has_key('EXTNAME'):
            name = self._header['EXTNAME']
        self.name = name

    def _verify(self, option='warn'):
        """
        ImageHDU verify method.
        """
        _err = _ValidHDU._verify(self, option=option)
        naxis = self.header.get('NAXIS', 0)
        self.req_cards('PCOUNT', '== '+`naxis+3`, _isInt+" and val == 0",
                       0, option, _err)
        self.req_cards('GCOUNT', '== '+`naxis+4`, _isInt+" and val == 1",
                       1, option, _err)
        return _err


class GroupsHDU(PrimaryHDU):
    """
    FITS Random Groups HDU class.
    """

    _dict = {8:'B', 16:'I', 32:'J', 64:'K', -32:'E', -64:'D'}

    def __init__(self, data=None, header=None, name=None):
        """
        TODO: Write me
        """
        PrimaryHDU.__init__(self, data=data, header=header)
        self._header._hdutype = GroupsHDU
        self.name = name

        if self._header['NAXIS'] <= 0:
            self._header['NAXIS'] = 1
        self._header.update('NAXIS1', 0, after='NAXIS')


    def __getattr__(self, attr):
        """
        Get the `data` or `columns` attribute.  The data of random
        group FITS file will be like a binary table's data.
        """

        if attr == 'data': # same code as in _TableBaseHDU
            size = self.size()
            if size:
                self._file.seek(self._datLoc)
                data = GroupData(_get_tbdata(self))
                data._coldefs = self.columns
                data.formats = self.columns.formats
                data.parnames = self.columns._pnames
            else:
                data = None
            self.__dict__[attr] = data

        elif attr == 'columns':
            _cols = []
            _pnames = []
            _pcount = self._header['PCOUNT']
            _format = GroupsHDU._dict[self._header['BITPIX']]
            for i in range(self._header['PCOUNT']):
                _bscale = self._header.get('PSCAL'+`i+1`, 1)
                _bzero = self._header.get('PZERO'+`i+1`, 0)
                _pnames.append(self._header['PTYPE'+`i+1`].lower())
                _cols.append(Column(name='c'+`i+1`, format = _format, bscale = _bscale, bzero = _bzero))
            data_shape = self._dimShape()[:-1]
            dat_format = `int(np.array(data_shape).sum())` + _format

            _bscale = self._header.get('BSCALE', 1)
            _bzero = self._header.get('BZERO', 0)
            _cols.append(Column(name='data', format = dat_format, bscale = _bscale, bzero = _bzero))
            _coldefs = ColDefs(_cols)
            _coldefs._shape = self._header['GCOUNT']
            _coldefs._dat_format = _fits2rec[_format]
            _coldefs._pnames = _pnames
            self.__dict__[attr] = _coldefs

        elif attr == '_theap':
            self.__dict__[attr] = 0
        else:
            return _AllHDU.__getattr__(self,attr)

        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    # 0.6.5.5
    def size(self):
        """
        Returns the size (in bytes) of the HDU's data part.
        """
        size = 0
        naxis = self._header.get('NAXIS', 0)

        # for random group image, NAXIS1 should be 0, so we skip NAXIS1.
        if naxis > 1:
            size = 1
            for j in range(1, naxis):
                size = size * self._header['NAXIS'+`j+1`]
            bitpix = self._header['BITPIX']
            gcount = self._header.get('GCOUNT', 1)
            pcount = self._header.get('PCOUNT', 0)
            size = abs(bitpix) * gcount * (pcount + size) // 8
        return size

    def _verify(self, option='warn'):
        _err = PrimaryHDU._verify(self, option=option)

        # Verify locations and values of mandatory keywords.
        self.req_cards('NAXIS', '== 2', _isInt+" and val >= 1 and val <= 999", 1, option, _err)
        self.req_cards('NAXIS1', '== 3', _isInt+" and val == 0", 0, option, _err)
        _after = self._header['NAXIS'] + 3

        # if the card EXTEND exists, must be after it.
        try:
            _dum = self._header['EXTEND']
            #_after += 1
        except KeyError:
            pass
        _pos = '>= '+`_after`
        self.req_cards('GCOUNT', _pos, _isInt, 1, option, _err)
        self.req_cards('PCOUNT', _pos, _isInt, 0, option, _err)
        self.req_cards('GROUPS', _pos, 'val == True', True, option, _err)
        return _err

    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
        if self.__dict__.has_key('data') and self.data != None:
            # We have the data to be used.
            # Check the byte order of the data.  If it is little endian we
            # must swap it before calculating the datasum.
            byteorder = \
                     self.data.dtype.fields[self.data.dtype.names[0]][0].str[0]

            if byteorder != '>':
                byteswapped = True
                d = self.data.byteswap(True)
                d.dtype = d.dtype.newbyteorder('>')
            else:
                byteswapped = False
                d = self.data

            cs = self._compute_checksum(np.fromstring(d, dtype='ubyte'),0)

            # If the data was byteswapped in this method then return it to
            # its original little-endian order.
            if byteswapped:
                d.byteswap(True)
                d.dtype = d.dtype.newbyteorder('<')

            return cs
        else:
            # This is the case where the data has not been read from the file
            # yet.  We can handle that in a generic manner so we do it in the
            # base class.  The other possibility is that there is no data at
            # all.  This can also be handled in a gereric manner.
            return super(GroupsHDU,self)._calculate_datasum()


# --------------------------Table related code----------------------------------

# lists of column/field definition common names and keyword names, make
# sure to preserve the one-to-one correspondence when updating the list(s).
# Use lists, instead of dictionaries so the names can be displayed in a
# preferred order.
_commonNames = ['name', 'format', 'unit', 'null', 'bscale', 'bzero', 'disp', 'start', 'dim']
_keyNames = ['TTYPE', 'TFORM', 'TUNIT', 'TNULL', 'TSCAL', 'TZERO', 'TDISP', 'TBCOL', 'TDIM']

# mapping from TFORM data type to numpy data type (code)

_booltype = 'i1'
_fits2rec = {'L':_booltype, 'B':'u1', 'I':'i2', 'E':'f4', 'D':'f8', 'J':'i4', 'A':'a', 'C':'c8', 'M':'c16', 'K':'i8'}

# the reverse dictionary of the above
_rec2fits = {}
for key in _fits2rec.keys():
    _rec2fits[_fits2rec[key]]=key


class _FormatX(str):
    """
    For X format in binary tables.
    """
    pass

class _FormatP(str):
    """
    For P format in variable length table.
    """
    pass

# TFORM regular expression
_tformat_re = re.compile(r'(?P<repeat>^[0-9]*)(?P<dtype>[A-Za-z])(?P<option>[!-~]*)')

# table definition keyword regular expression
_tdef_re = re.compile(r'(?P<label>^T[A-Z]*)(?P<num>[1-9][0-9 ]*$)')

def _parse_tformat(tform):
    """
    Parse the ``TFORM`` value into `repeat`, `dtype`, and `option`.
    """
    try:
        (repeat, dtype, option) = _tformat_re.match(tform.strip()).groups()
    except:
        warnings.warn('Format "%s" is not recognized.' % tform)


    if repeat == '': repeat = 1
    else: repeat = eval(repeat)

    return (repeat, dtype, option)

def _convert_format(input_format, reverse=0):
    """
    Convert FITS format spec to record format spec.  Do the opposite
    if reverse = 1.
    """
    if reverse and isinstance(input_format, np.dtype):
        shape = input_format.shape
        kind = input_format.base.kind
        option = str(input_format.base.itemsize)
        if kind == 'S':
            kind = 'a'
        dtype = kind

        ndims = len(shape)
        repeat = 1
        if ndims > 0:
            nel = np.array(shape, dtype='i8').prod()
            if nel > 1:
                repeat = nel
    else:
        fmt = input_format
        (repeat, dtype, option) = _parse_tformat(fmt)

    if reverse == 0:
        if dtype in _fits2rec.keys():                            # FITS format
            if dtype == 'A':
                output_format = _fits2rec[dtype]+`repeat`
                # to accomodate both the ASCII table and binary table column
                # format spec, i.e. A7 in ASCII table is the same as 7A in
                # binary table, so both will produce 'a7'.
                if fmt.lstrip()[0] == 'A' and option != '':
                    output_format = _fits2rec[dtype]+`int(option)` # make sure option is integer
            else:
                _repeat = ''
                if repeat != 1:
                    _repeat = `repeat`
                output_format = _repeat+_fits2rec[dtype]

        elif dtype == 'X':
            nbytes = ((repeat-1) // 8) + 1
            # use an array, even if it is only ONE u1 (i.e. use tuple always)
            output_format = _FormatX(`(nbytes,)`+'u1')
            output_format._nx = repeat

        elif dtype == 'P':
            output_format = _FormatP('2i4')
            output_format._dtype = _fits2rec[option[0]]
        elif dtype == 'F':
            output_format = 'f8'
        else:
            raise ValueError, "Illegal format %s" % fmt
    else:
        if dtype == 'a':
            # This is a kludge that will place string arrays into a
            # single field, so at least we won't lose data.  Need to
            # use a TDIM keyword to fix this, declaring as (slength,
            # dim1, dim2, ...)  as mwrfits does

            ntot = int(repeat)*int(option)

            output_format = str(ntot)+_rec2fits[dtype]
        elif isinstance(dtype, _FormatX):
            warnings.warn('X format')
        elif dtype+option in _rec2fits.keys():                    # record format
            _repeat = ''
            if repeat != 1:
                _repeat = `repeat`
            output_format = _repeat+_rec2fits[dtype+option]
        else:
            raise ValueError, "Illegal format %s" % fmt

    return output_format

def _convert_ASCII_format(input_format):
    """
    Convert ASCII table format spec to record format spec.
    """

    ascii2rec = {'A':'a', 'I':'i4', 'F':'f4', 'E':'f4', 'D':'f8'}
    _re = re.compile(r'(?P<dtype>[AIFED])(?P<width>[0-9]*)')

    # Parse the TFORM value into data type and width.
    try:
        (dtype, width) = _re.match(input_format.strip()).groups()
        dtype = ascii2rec[dtype]
        if width == '':
            width = None
        else:
            width = eval(width)
    except KeyError:
        raise ValueError, 'Illegal format `%s` for ASCII table.' % input_format

    return (dtype, width)

def _get_index(nameList, key):
    """
    Get the index of the `key` in the `nameList`.

    The `key` can be an integer or string.  If integer, it is the index
    in the list.  If string,

        a. Field (column) names are case sensitive: you can have two
           different columns called 'abc' and 'ABC' respectively.

        b. When you *refer* to a field (presumably with the field
           method), it will try to match the exact name first, so in
           the example in (a), field('abc') will get the first field,
           and field('ABC') will get the second field.

        If there is no exact name matched, it will try to match the
        name with case insensitivity.  So, in the last example,
        field('Abc') will cause an exception since there is no unique
        mapping.  If there is a field named "XYZ" and no other field
        name is a case variant of "XYZ", then field('xyz'),
        field('Xyz'), etc. will get this field.
    """

    if isinstance(key, (int, long,np.integer)):
        indx = int(key)
    elif isinstance(key, str):
        # try to find exact match first
        try:
            indx = nameList.index(key.rstrip())
        except ValueError:

            # try to match case-insentively,
            _key = key.lower().rstrip()
            _list = map(lambda x: x.lower().rstrip(), nameList)
            _count = operator.countOf(_list, _key) # occurrence of _key in _list
            if _count == 1:
                indx = _list.index(_key)
            elif _count == 0:
                raise KeyError, "Key '%s' does not exist." % key
            else:              # multiple match
                raise KeyError, "Ambiguous key name '%s'." % key
    else:
        raise KeyError, "Illegal key '%s'." % `key`

    return indx

def _unwrapx(input, output, nx):
    """
    Unwrap the X format column into a Boolean array.

    Parameters
    ----------
    input
        input ``Uint8`` array of shape (`s`, `nbytes`)

    output
        output Boolean array of shape (`s`, `nx`)

    nx
        number of bits
    """

    pow2 = [128, 64, 32, 16, 8, 4, 2, 1]
    nbytes = ((nx-1) // 8) + 1
    for i in range(nbytes):
        _min = i*8
        _max = min((i+1)*8, nx)
        for j in range(_min, _max):
            np.bitwise_and(input[...,i], pow2[j-i*8], output[...,j])

def _wrapx(input, output, nx):
    """
    Wrap the X format column Boolean array into an ``UInt8`` array.

    Parameters
    ----------
    input
        input Boolean array of shape (`s`, `nx`)

    output
        output ``Uint8`` array of shape (`s`, `nbytes`)

    nx
        number of bits
    """

    output[...] = 0 # reset the output
    nbytes = ((nx-1) // 8) + 1
    unused = nbytes*8 - nx
    for i in range(nbytes):
        _min = i*8
        _max = min((i+1)*8, nx)
        for j in range(_min, _max):
            if j != _min:
                np.left_shift(output[...,i], 1, output[...,i])
            np.add(output[...,i], input[...,j], output[...,i])

    # shift the unused bits
    np.left_shift(output[...,i], unused, output[...,i])

def _makep(input, desp_output, dtype):
    """
    Construct the P format column array, both the data descriptors and
    the data.  It returns the output "data" array of data type `dtype`.

    The descriptor location will have a zero offset for all columns
    after this call.  The final offset will be calculated when the file
    is written.

    Parameters
    ----------
    input
        input object array

    desp_output
        output "descriptor" array of data type ``Int32``

    dtype
        data type of the variable array
    """
    _offset = 0
    data_output = _VLF([None]*len(input))
    data_output._dtype = dtype

    if dtype == 'a':
        _nbytes = 1
    else:
        _nbytes = np.array([],dtype=np.typeDict[dtype]).itemsize

    for i in range(len(input)):
        if dtype == 'a':
            data_output[i] = chararray.array(input[i], itemsize=1)
        else:
            data_output[i] = np.array(input[i], dtype=dtype)

        desp_output[i,0] = len(data_output[i])
        desp_output[i,1] = _offset
        _offset += len(data_output[i]) * _nbytes

    return data_output

class _VLF(np.ndarray):
    """
    Variable length field object.
    """

    def __new__(subtype, input):
        """
        Parameters
        ----------
        input
            a sequence of variable-sized elements.
        """
        a = np.array(input,dtype=np.object)
        self = np.ndarray.__new__(subtype, shape=(len(input)), buffer=a,
                                  dtype=np.object)
        self._max = 0
        return self

    def __array_finalize__(self,obj):
        if obj is None:
            return
        self._max = obj._max

    def __setitem__(self, key, value):
        """
        To make sure the new item has consistent data type to avoid
        misalignment.
        """
        if isinstance(value, np.ndarray) and value.dtype == self.dtype:
            pass
        elif isinstance(value, chararray.chararray) and value.itemsize == 1:
            pass
        elif self._dtype == 'a':
            value = chararray.array(value, itemsize=1)
        else:
            value = np.array(value, dtype=self._dtype)
        np.ndarray.__setitem__(self, key, value)
        self._max = max(self._max, len(value))


class Column:
    """
    Class which contains the definition of one column, e.g.  `ttype`,
    `tform`, etc. and the array containing values for the column.
    Does not support `theap` yet.
    """
    def __init__(self, name=None, format=None, unit=None, null=None, \
                       bscale=None, bzero=None, disp=None, start=None, \
                       dim=None, array=None):
        """
        Construct a `Column` by specifying attributes.  All attributes
        except `format` can be optional.

        Parameters
        ----------
        name : str, optional
            column name, corresponding to ``TTYPE`` keyword

        format : str, optional
            column format, corresponding to ``TFORM`` keyword

        unit : str, optional
            column unit, corresponding to ``TUNIT`` keyword

        null : str, optional
            null value, corresponding to ``TNULL`` keyword

        bscale : int-like, optional
            bscale value, corresponding to ``TSCAL`` keyword

        bzero : int-like, optional
            bzero value, corresponding to ``TZERO`` keyword

        disp : str, optional
            display format, corresponding to ``TDISP`` keyword

        start : int, optional
            column starting position (ASCII table only), corresponding
            to ``TBCOL`` keyword

        dim : str, optional
            column dimension corresponding to ``TDIM`` keyword
        """
        # any of the input argument (except array) can be a Card or just
        # a number/string
        for cname in _commonNames:
            value = eval(cname)           # get the argument's value

            keyword = _keyNames[_commonNames.index(cname)]
            if isinstance(value, Card):
                setattr(self, cname, value.value)
            else:
                setattr(self, cname, value)

        # if the column data is not ndarray, make it to be one, i.e.
        # input arrays can be just list or tuple, not required to be ndarray
        if format is not None:
            # check format
            try:

                # legit FITS format? convert to record format (e.g. '3J'->'3i4')
                recfmt = _convert_format(format)
            except ValueError:
                try:
                    # legit recarray format?
                    recfmt = format
                    format = _convert_format(recfmt, reverse=1)
                except ValueError:
                    raise ValueError, "Illegal format `%s`." % format

            self.format = format

            # does not include Object array because there is no guarantee
            # the elements in the object array are consistent.
            if not isinstance(array, (np.ndarray, chararray.chararray, Delayed)):
                try: # try to convert to a ndarray first
                    if array is not None:
                        array = np.array(array)
                except:
                    try: # then try to conver it to a strings array
                        array = chararray.array(array, itemsize=eval(recfmt[1:]))

                    # then try variable length array
                    except:
                        if isinstance(recfmt, _FormatP):
                            try:
                                array=_VLF(array)
                            except:
                                try:
                                    # this handles ['abc'] and [['a','b','c']]
                                    # equally, beautiful!
                                    _func = lambda x: chararray.array(x, itemsize=1)
                                    array = _VLF(map(_func, array))
                                except:
                                    raise ValueError, "Inconsistent input data array: %s" % array
                            array._dtype = recfmt._dtype
                        else:
                            raise ValueError, "Data is inconsistent with the format `%s`." % format

        else:
            raise ValueError, "Must specify format to construct Column"

        # scale the array back to storage values if there is bscale/bzero
        if isinstance(array, np.ndarray):

            # boolean needs to be scaled too
            if recfmt[-2:] == _booltype:
                _out = np.zeros(array.shape, dtype=recfmt)
                array = np.where(array==0, ord('F'), ord('T'))

            # make a copy if scaled, so as not to corrupt the original array
            if bzero not in ['', None, 0] or bscale not in ['', None, 1]:
                array = array.copy()
                if bzero not in ['', None, 0]:
                    array += -bzero
                if bscale not in ['', None, 1]:
                    array /= bscale

        array = self.__checkValidDataType(array,self.format)
        self.array = array

    def __checkValidDataType(self,array,format):
        # Convert the format to a type we understand
        if isinstance(array,Delayed):
            return array
        elif (array is None):
            return array
        else:
            if (format.find('A') != -1 and format.find('P') == -1):
                if str(array.dtype).find('S') != -1:
                    # For ASCII arrays, reconstruct the array and ensure
                    # that all elements have enough characters to comply
                    # with the format.  The new array will have the data
                    # left justified in the field with trailing blanks
                    # added to complete the format requirements.
                    fsize=eval(_convert_format(format)[1:])
                    l = []

                    for i in range(len(array)):
                        al = len(array[i])
                        l.append(array[i][:min(fsize,array.itemsize)]+
                                 ' '*(fsize-al))
                    return chararray.array(l)
                else:
                    numpyFormat = _convert_format(format)
                    return array.astype(numpyFormat)
            elif (format.find('X') == -1 and format.find('P') == -1):
                (repeat, fmt, option) = _parse_tformat(format)
                numpyFormat = _convert_format(fmt)
                return array.astype(numpyFormat)
            elif (format.find('X') !=-1):
                return array.astype(np.uint8)
            else:
                return array

    def __repr__(self):
        text = ''
        for cname in _commonNames:
            value = getattr(self, cname)
            if value != None:
                text += cname + ' = ' + `value` + '\n'
        return text[:-1]

    def copy(self):
        """
        Return a copy of this `Column`.
        """
        tmp = Column(format='I') # just use a throw-away format
        tmp.__dict__=self.__dict__.copy()
        return tmp


class ColDefs(object):
    """
    Column definitions class.

    It has attributes corresponding to the `Column` attributes
    (e.g. `ColDefs` has the attribute `~ColDefs.names` while `Column`
    has `~Column.name`). Each attribute in `ColDefs` is a list of
    corresponding attribute values from all `Column` objects.
    """
    def __init__(self, input, tbtype='BinTableHDU'):
        """
        Parameters
        ----------

        input : sequence of `Column` objects
            an (table) HDU

        tbtype : str, optional
            which table HDU, ``"BinTableHDU"`` (default) or
            ``"TableHDU"`` (text table).
        """
        ascii_fmt = {'A':'A1', 'I':'I10', 'E':'E14.6', 'F':'F16.7', 'D':'D24.16'}
        self._tbtype = tbtype

        if isinstance(input, ColDefs):
            self.data = [col.copy() for col in input.data]

        # if the input is a list of Columns
        elif isinstance(input, (list, tuple)):
            for col in input:
                if not isinstance(col, Column):
                    raise TypeError(
                           "Element %d in the ColDefs input is not a Column."
                           % input.index(col))
            self.data = [col.copy() for col in input]

            # if the format of an ASCII column has no width, add one
            if tbtype == 'TableHDU':
                for i in range(len(self)):
                    (type, width) = _convert_ASCII_format(self.data[i].format)
                    if width is None:
                        self.data[i].format = ascii_fmt[self.data[i].format[0]]


        elif isinstance(input, _TableBaseHDU):
            hdr = input._header
            _nfields = hdr['TFIELDS']
            self._width = hdr['NAXIS1']
            self._shape = hdr['NAXIS2']

            # go through header keywords to pick out column definition keywords
            dict = [{} for i in range(_nfields)] # definition dictionaries for each field
            for _card in hdr.ascardlist():
                _key = _tdef_re.match(_card.key)
                try:
                    keyword = _key.group('label')
                except:
                    continue               # skip if there is no match
                if (keyword in _keyNames):
                    col = eval(_key.group('num'))
                    if col <= _nfields and col > 0:
                        cname = _commonNames[_keyNames.index(keyword)]
                        dict[col-1][cname] = _card.value

            # data reading will be delayed
            for col in range(_nfields):
                dict[col]['array'] = Delayed(input, col)

            # now build the columns
            tmp = [Column(**attrs) for attrs in dict]
            self.data = tmp
            self._listener = input
        else:
            raise TypeError, "input to ColDefs must be a table HDU or a list of Columns"

    def __getattr__(self, name):
        """
        Populate the attributes.
        """
        cname = name[:-1]
        if cname in _commonNames and name[-1] == 's':
            attr = [''] * len(self)
            for i in range(len(self)):
                val = getattr(self[i], cname)
                if val != None:
                    attr[i] = val
        elif name == '_arrays':
            attr = [col.array for col in self.data]
        elif name == '_recformats':
            if self._tbtype in ('BinTableHDU', 'CompImageHDU'):
                attr = [_convert_format(fmt) for fmt in self.formats]
            elif self._tbtype == 'TableHDU':
                self._Formats = self.formats
                if len(self) == 1:
                    dummy = []
                else:
                    dummy = map(lambda x, y: x-y, self.starts[1:], [self.starts[0]]+self.starts[1:-1])
                dummy.append(self._width-self.starts[-1]+1)
                attr = map(lambda y: 'a'+`y`, dummy)
        elif name == 'spans':
            # make sure to consider the case that the starting column of
            # a field may not be the column right after the last field
            if self._tbtype == 'TableHDU':
                last_end = 0
                attr = [0] * len(self)
                for i in range(len(self)):
                    (_format, _width) = _convert_ASCII_format(self.formats[i])
                    if self.starts[i] is '':
                        self.starts[i] = last_end + 1
                    _end = self.starts[i] + _width - 1
                    attr[i] = _width
                    last_end = _end
                self._width = _end
            else:
                raise KeyError, 'Attribute %s not defined.' % name
        else:
            raise KeyError, 'Attribute %s not defined.' % name

        self.__dict__[name] = attr
        return self.__dict__[name]


        """
                # make sure to consider the case that the starting column of
                # a field may not be the column right after the last field
                elif tbtype == 'TableHDU':
                    (_format, _width) = _convert_ASCII_format(self.formats[i])
                    if self.starts[i] is '':
                        self.starts[i] = last_end + 1
                    _end = self.starts[i] + _width - 1
                    self.spans[i] = _end - last_end
                    last_end = _end
                    self._Formats = self.formats

                self._arrays[i] = input[i].array
        """

    def __getitem__(self, key):
        x = self.data[key]
        if isinstance(key, (int, long, np.integer)):
            return x
        else:
            return ColDefs(x)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'ColDefs'+ `tuple(self.data)`

    def __add__(self, other, option='left'):
        if isinstance(other, Column):
            b = [other]
        elif isinstance(other, ColDefs):
            b = list(other.data)
        else:
            raise TypeError, 'Wrong type of input'
        if option == 'left':
            tmp = list(self.data) + b
        else:
            tmp = b + list(self.data)
        return ColDefs(tmp)

    def __radd__(self, other):
        return self.__add__(other, 'right')

    def __sub__(self, other):
        if not isinstance(other, (list, tuple)):
            other = [other]
        _other = [_get_index(self.names, key) for key in other]
        indx=range(len(self))
        for x in _other:
            indx.remove(x)
        tmp = [self[i] for i in indx]
        return ColDefs(tmp)

    def _update_listener(self):
        if hasattr(self, '_listener'):
            delattr(self._listener, 'data')

    def add_col(self, column):
        """
        Append one `Column` to the column definition.

        .. warning::

        *New in pyfits 2.3*: This function appends the new column to
        the `ColDefs` object in place.  Prior to pyfits 2.3, this
        function returned a new `ColDefs` with the new column at the
        end.
        """
        assert isinstance(column, Column)

        for cname in _commonNames:
            attr = getattr(self, cname+'s')
            attr.append(getattr(column, cname))

        self._arrays.append(column.array)
        # Obliterate caches of certain things
        if hasattr(self, '_recformats'):
            delattr(self, '_recformats')
        if hasattr(self, 'spans'):
            delattr(self, 'spans')

        self.data.append(column)
        # Force regeneration of self._Formats member
        ignored = self._recformats

        # If this ColDefs is being tracked by a Table, inform the
        # table that its data is now invalid.
        self._update_listener()
        return self

    def del_col(self, col_name):
        """
        Delete (the definition of) one `Column`.

        col_name : str or int
            The column's name or index
        """
        indx = _get_index(self.names, col_name)

        for cname in _commonNames:
            attr = getattr(self, cname+'s')
            del attr[indx]

        del self._arrays[indx]
        # Obliterate caches of certain things
        if hasattr(self, '_recformats'):
            delattr(self, '_recformats')
        if hasattr(self, 'spans'):
            delattr(self, 'spans')

        del self.data[indx]
        # Force regeneration of self._Formats member
        ignored = self._recformats

        # If this ColDefs is being tracked by a Table, inform the
        # table that its data is now invalid.
        self._update_listener()
        return self

    def change_attrib(self, col_name, attrib, new_value):
        """
        Change an attribute (in the commonName list) of a `Column`.

        col_name : str or int
            The column name or index to change

        attrib : str
            The attribute name

        value : object
            The new value for the attribute
        """
        indx = _get_index(self.names, col_name)
        getattr(self, attrib+'s')[indx] = new_value

        # If this ColDefs is being tracked by a Table, inform the
        # table that its data is now invalid.
        self._update_listener()

    def change_name(self, col_name, new_name):
        """
        Change a `Column`'s name.

        col_name : str
            The current name of the column

        new_name : str
            The new name of the column
        """
        if new_name != col_name and new_name in self.names:
            raise ValueError, 'New name %s already exists.' % new_name
        else:
            self.change_attrib(col_name, 'name', new_name)

        # If this ColDefs is being tracked by a Table, inform the
        # table that its data is now invalid.
        self._update_listener()

    def change_unit(self, col_name, new_unit):
        """
        Change a `Column`'s unit.

        col_name : str or int
            The column name or index

        new_unit : str
            The new unit for the column
        """
        self.change_attrib(col_name, 'unit', new_unit)

        # If this ColDefs is being tracked by a Table, inform the
        # table that its data is now invalid.
        self._update_listener()

    def info(self, attrib='all'):
        """
        Get attribute(s) information of the column definition.

        Parameters
        ----------
        attrib : str
           Can be one or more of the attributes listed in
           `_commonNames`.  The default is ``"all"`` which will print
           out all attributes.  It forgives plurals and blanks.  If
           there are two or more attribute names, they must be
           separated by comma(s).

        Notes
        -----
        This function doesn't return anything, it just prints to
        stdout.
        """

        if attrib.strip().lower() in ['all', '']:
            list = _commonNames
        else:
            list = attrib.split(',')
            for i in range(len(list)):
                list[i]=list[i].strip().lower()
                if list[i][-1] == 's':
                    list[i]=list[i][:-1]

        for att in list:
            if att not in _commonNames:
                print "'%s' is not an attribute of the column definitions."%att
                continue
            print "%s:" % att
            print '    ', getattr(self, att+'s')

    #def change_format(self, col_name, new_format):
        #new_format = _convert_format(new_format)
        #self.change_attrib(col_name, 'format', new_format)

def _get_tbdata(hdu):
    """
    Get the table data from input (an HDU object).
    """
    tmp = hdu.columns
    # get the right shape for the data part of the random group,
    # since binary table does not support ND yet
    if isinstance(hdu, GroupsHDU):
        tmp._recformats[-1] = `hdu._dimShape()[:-1]` + tmp._dat_format
    elif isinstance(hdu, TableHDU):
        # determine if there are duplicate field names and if there
        # are throw an exception
        _dup = rec.find_duplicate(tmp.names)

        if _dup:
            raise ValueError, "Duplicate field names: %s" % _dup

        itemsize = tmp.spans[-1]+tmp.starts[-1]-1
        dtype = {}

        for j in range(len(tmp)):
            data_type = 'S'+str(tmp.spans[j])

            if j == len(tmp)-1:
                if hdu._header['NAXIS1'] > itemsize:
                    data_type = 'S'+str(tmp.spans[j]+ \
                                hdu._header['NAXIS1']-itemsize)
            dtype[tmp.names[j]] = (data_type,tmp.starts[j]-1)

    if hdu._ffile.memmap:
        if isinstance(hdu, TableHDU):
            hdu._ffile.code = dtype
        else:
            hdu._ffile.code = rec.format_parser(",".join(tmp._recformats),
                                                 tmp.names,None)._descr

        hdu._ffile.dims = tmp._shape
        hdu._ffile.offset = hdu._datLoc
        _data = rec.recarray(shape=hdu._ffile.dims, buf=hdu._ffile._mm,
                             dtype=hdu._ffile.code, names=tmp.names)
    else:
        if isinstance(hdu, TableHDU):
            _data = rec.array(hdu._file, dtype=dtype, names=tmp.names,
                              shape=tmp._shape)
        else:
            _data = rec.array(hdu._file, formats=",".join(tmp._recformats),
                              names=tmp.names, shape=tmp._shape)

    if isinstance(hdu._ffile, _File):
#        _data._byteorder = 'big'
        _data.dtype = _data.dtype.newbyteorder(">")

    # pass datLoc, for P format
    _data._heapoffset = hdu._theap + hdu._datLoc
    _data._file = hdu._file
    _tbsize = hdu._header['NAXIS1']*hdu._header['NAXIS2']
    _data._gap = hdu._theap - _tbsize
    # comment out to avoid circular reference of _pcount

    # pass the attributes
    for attr in ['formats', 'names']:
        setattr(_data, attr, getattr(tmp, attr))
    for i in range(len(tmp)):
       # get the data for each column object from the rec.recarray
        tmp.data[i].array = _data.field(i)

    # delete the _arrays attribute so that it is recreated to point to the
    # new data placed in the column object above
    if tmp.__dict__.has_key('_arrays'):
        del tmp.__dict__['_arrays']

    # TODO: Probably a benign change, but I'd still like to get to
    # the bottom of the root cause...
    #return FITS_rec(_data)
    return _data.view(FITS_rec)

def new_table(input, header=None, nrows=0, fill=False, tbtype='BinTableHDU'):
    """
    Create a new table from the input column definitions.

    Parameters
    ----------
    input : sequence of Column or ColDefs objects
        The data to create a table from.

    header : Header instance
        Header to be used to populate the non-required keywords.

    nrows : int
        Number of rows in the new table.

    fill : bool
        If `True`, will fill all cells with zeros or blanks.  If
        `False`, copy the data from input, undefined cells will still
        be filled with zeros/blanks.

    tbtype : str
        Table type to be created ("BinTableHDU" or "TableHDU").
    """
    # construct a table HDU
    hdu = eval(tbtype)(header=header)

    if isinstance(input, ColDefs):
        if input._tbtype == tbtype:
            # Create a new ColDefs object from the input object and assign
            # it to the ColDefs attribute of the new hdu.
            tmp = hdu.columns = ColDefs(input, tbtype)
        else:
            raise ValueError, 'column definitions have a different table type'
    elif isinstance(input, FITS_rec): # input is a FITS_rec
        # Create a new ColDefs object from the input FITS_rec's ColDefs
        # object and assign it to the ColDefs attribute of the new hdu.
        tmp = hdu.columns = ColDefs(input._coldefs, tbtype)
    elif isinstance(input, np.ndarray):
        tmp = hdu.columns = eval(tbtype)(input).data._coldefs
    else:                 # input is a list of Columns
        # Create a new ColDefs object from the input list of Columns and
        # assign it to the ColDefs attribute of the new hdu.
        tmp = hdu.columns = ColDefs(input, tbtype)

    # read the delayed data
    for i in range(len(tmp)):
        _arr = tmp._arrays[i]
        if isinstance(_arr, Delayed):
            if _arr.hdu().data == None:
                tmp._arrays[i] = None
            else:
                tmp._arrays[i] = rec.recarray.field(_arr.hdu().data,_arr.field)

    # use the largest column shape as the shape of the record
    if nrows == 0:
        for arr in tmp._arrays:
            if (arr is not None):
                dim = arr.shape[0]
            else:
                dim = 0
            if dim > nrows:
                nrows = dim

    if tbtype == 'TableHDU':
        _itemsize = tmp.spans[-1]+tmp.starts[-1]-1
        dtype = {}

        for j in range(len(tmp)):
           data_type = 'S'+str(tmp.spans[j])
           dtype[tmp.names[j]] = (data_type,tmp.starts[j]-1)

        hdu.data = FITS_rec(rec.array(' '*_itemsize*nrows, dtype=dtype,
                                      shape=nrows))
        hdu.data.setflags(write=True)
    else:
        hdu.data = FITS_rec(rec.array(None, formats=",".join(tmp._recformats),
                                      names=tmp.names, shape=nrows))

    hdu.data._coldefs = hdu.columns
    hdu.data.formats = hdu.columns.formats

    # Populate data to the new table from the ndarrays in the input ColDefs
    # object.
    for i in range(len(tmp)):
        # For each column in the ColDef object, determine the number
        # of rows in that column.  This will be either the number of
        # rows in the ndarray associated with the column, or the
        # number of rows given in the call to this function, which
        # ever is smaller.  If the input FILL argument is true, the
        # number of rows is set to zero so that no data is copied from
        # the original input data.
        if tmp._arrays[i] is None:
            size = 0
        else:
            size = len(tmp._arrays[i])

        n = min(size, nrows)
        if fill:
            n = 0

        # Get any scale factors from the FITS_rec
        (_scale, _zero, bscale, bzero) = hdu.data._get_scale_factors(i)[3:]

        if n > 0:
            # Only copy data if there is input data to copy
            # Copy all of the data from the input ColDefs object for this
            # column to the new FITS_rec data array for this column.
            if isinstance(tmp._recformats[i], _FormatX):
                # Data is a bit array
                if tmp._arrays[i][:n].shape[-1] == tmp._recformats[i]._nx:
                    _wrapx(tmp._arrays[i][:n],
                           rec.recarray.field(hdu.data,i)[:n],
                           tmp._recformats[i]._nx)
                else: # from a table parent data, just pass it
                    rec.recarray.field(hdu.data,i)[:n] = tmp._arrays[i][:n]
            elif isinstance(tmp._recformats[i], _FormatP):
                hdu.data._convert[i] = _makep(tmp._arrays[i][:n],
                                            rec.recarray.field(hdu.data,i)[:n],
                                            tmp._recformats[i]._dtype)
            elif tmp._recformats[i][-2:] == _booltype and \
                 tmp._arrays[i].dtype == bool:
                # column is boolean 
                rec.recarray.field(hdu.data,i)[:n] = \
                           np.where(tmp._arrays[i]==False, ord('F'), ord('T'))
            else:
                if tbtype == 'TableHDU':

                    # string no need to convert,
                    if isinstance(tmp._arrays[i], chararray.chararray):
                        rec.recarray.field(hdu.data,i)[:n] = tmp._arrays[i][:n]
                    else:
                        hdu.data._convert[i] = np.zeros(nrows,
                                                    dtype=tmp._arrays[i].dtype)
                        if _scale or _zero:
                            _arr = tmp._arrays[i].copy()
                        else:
                            _arr = tmp._arrays[i]
                        if _scale:
                            _arr *= bscale
                        if _zero:
                            _arr += bzero
                        hdu.data._convert[i][:n] = _arr[:n]
                else:
                    rec.recarray.field(hdu.data,i)[:n] = tmp._arrays[i][:n]

        if n < nrows:
            # If there are additional rows in the new table that were not
            # copied from the input ColDefs object, initialize the new data
            if tbtype == 'BinTableHDU':
                if isinstance(rec.recarray.field(hdu.data,i), np.ndarray):
                    # make the scaled data = 0
                    rec.recarray.field(hdu.data,i)[n:] = -bzero/bscale
                else:
                    rec.recarray.field(hdu.data,i)[n:] = ''
            else:
                rec.recarray.field(hdu.data,i)[n:] = \
                                                 ' '*hdu.data._coldefs.spans[i]

    # Update the HDU header to match the data
    hdu.update()

    # Make the ndarrays in the Column objects of the ColDefs object of the HDU
    # reference the same ndarray as the HDU's FITS_rec object.
    for i in range(len(tmp)):
        hdu.columns.data[i].array = hdu.data.field(i)

    # Delete the _arrays attribute so that it is recreated to point to the
    # new data placed in the column objects above
    if hdu.columns.__dict__.has_key('_arrays'):
        del hdu.columns.__dict__['_arrays']

    return hdu

class FITS_record(object):
    """
    FITS record class.

    `FITS_record` is used to access records of the `FITS_rec` object.
    This will allow us to deal with scaled columns.  The `FITS_record`
    class expects a `FITS_rec` object as input.
    """
    def __init__(self, input, row=0, startColumn=0, endColumn=0):
        """
        Parameters
        ----------
        input : array
           The array to wrap.

        row : int, optional
           The starting logical row of the array.

        startColumn : int, optional
           The starting column in the row associated with this object.
           Used for subsetting the columns of the FITS_rec object.

        endColumn : int, optional
           The ending column in the row associated with this object.
           Used for subsetting the columns of the FITS_rec object.
        """
        self.array = input
        self.row = row
        len = self.array._nfields

        if startColumn > len:
            self.start = len + 1
        else:
            self.start = startColumn

        if endColumn <= 0 or endColumn > len:
            self.end = len
        else:
            self.end = endColumn

    def field(self, fieldName):
        """
        Get the field data of the record.
        """
        return self.__getitem__(fieldName)


    def setfield(self, fieldName, value):
        """
        Set the field data of the record.
        """
        self.__setitem__(fieldName, value)

    def __str__(self):
        """
        Print one row.
        """
        if isinstance(self.row, (str, unicode)):
            return repr(np.asarray(self.array)[self.row])
        else:
            outlist = []
            for i in range(self.array._nfields):
                if i >= self.start and i < self.end:
                    outlist.append(`self.array.field(i)[self.row]`)
            return "(" + ", ".join(outlist) + ")"


    def __repr__(self):
        return self.__str__()

    def __getitem__(self,key):
        if isinstance(key, (str, unicode)):
            indx = _get_index(self.array._coldefs.names, key)

            if indx < self.start or indx > self.end - 1:
                raise KeyError("Key '%s' does not exist."%key)
        else:
            indx = key + self.start

            if indx > self.end - 1:
                raise IndexError("index out of bounds")

        return self.array.field(indx)[self.row]

    def __setitem__(self,fieldName,value):
        if isinstance(fieldName, (str, unicode)):
            indx = _get_index(self.array._coldefs.names, fieldName)

            if indx < self.start or indx > self.end - 1:
                raise KeyError("Key '%s' does not exist."%fieldName)
        else:
            indx = fieldName + self.start

            if indx > self.end - 1:
                raise IndexError("index out of bounds")

        self.array.field(indx)[self.row] = value

    def __len__(self):
        return min(self.end - self.start, self.array._nfields)

    def __getslice__(self, i, j):
        return FITS_record(self.array,self.row,i,j)

class FITS_rec(rec.recarray):
    """
    FITS record array class.

    `FITS_rec` is the data part of a table HDU's data part.  This is a
    layer over the `recarray`, so we can deal with scaled columns.

    It inherits all of the standard methods from `numpy.ndarray`.
    """

    def __new__(subtype, input):
        """
        Construct a FITS record array from a recarray.
        """
        # input should be a record array
        if input.dtype.subdtype is None:
            self = rec.recarray.__new__(subtype, input.shape, input.dtype,
                                        buf=input.data,heapoffset=input._heapoffset,file=input._file)
        else:
            self = rec.recarray.__new__(subtype, input.shape, input.dtype,
                                        buf=input.data, strides=input.strides,heapoffset=input._heapoffset,file=input._file)

        self._nfields = len(self.dtype.names)
        self._convert = [None]*len(self.dtype.names)
        self._coldefs = None
        self._gap = 0
        self.names = self.dtype.names
        self._names = self.dtype.names # This attribute added for backward compatibility with numarray version of FITS_rec
        self.formats = None
        return self

    def __array_finalize__(self,obj):
        if obj is None:
            return

        # This will allow regular ndarrays with fields, rather than
        # just other FITS_rec objects
        self._nfields = len(obj.dtype.names)
        self._convert = [None]*len(obj.dtype.names)

        self._heapoffset = getattr(obj,'_heapoffset',0)
        self._file = getattr(obj,'_file', None)

        self._coldefs = None
        self._gap = 0
        self.names = obj.dtype.names
        self._names = obj.dtype.names # This attribute added for backward compatibility with numarray version of FITS_rec
        self.formats = None

        attrs=['_convert', '_coldefs', 'names', '_names', '_gap', 'formats']
        for attr in attrs:
            if hasattr(obj, attr):
                value = getattr(obj, attr, None)
                if value is None:
                    warnings.warn('Setting attribute %s as None' % attr)
                setattr(self, attr, value)

        if self._coldefs == None:
            # The data does not have a _coldefs attribute so
            # create one from the underlying recarray.
            columns = []
            formats = []

            for i in range(len(obj.dtype.names)):
                cname = obj.dtype.names[i]

                format = _convert_format(obj.dtype[i], reverse=True)

                formats.append(format)

                c = Column(name=cname,format=format)
                columns.append(c)

            tbtype = 'BinTableHDU'
            try:
                if self._xtn == 'TABLE':
                    tbtype = 'TableHDU'
            except AttributeError:
                pass

            self.formats = formats
            self._coldefs = ColDefs(columns, tbtype=tbtype)

    def _clone(self, shape):
        """
        Overload this to make mask array indexing work properly.
        """
        hdu = new_table(self._coldefs, nrows=shape[0])
        return hdu.data

    def __repr__(self):
        tmp = rec.recarray.__repr__(self)
        return tmp

    def __getslice__(self, i, j):
        key = slice(i,j)
        return self.__getitem__(key)

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key,np.ndarray):
            out = rec.recarray.__getitem__(self, key)
            out._coldefs = ColDefs(self._coldefs)
            arrays = []
            out._convert = [None]*len(self.dtype.names)
            for i in range(len(self.dtype.names)):
                #
                # Store the new arrays for the _coldefs object
                #
                arrays.append( self._coldefs._arrays[i][key])

                # touch all fields to expand the original ._convert list
                # so the sliced FITS_rec will view the same scaled columns as
                # the original
                dummy = self.field(i)
                if self._convert[i] is not None:
                    out._convert[i] = np.ndarray.__getitem__(self._convert[i], key)
            del dummy

            out._coldefs._arrays = arrays
            out._coldefs._shape = len(arrays[0])

            return out

        # if not a slice, do this because Record has no __getstate__.
        # also more efficient.
        else:
            if isinstance(key, int) and key >= len(self):
                raise IndexError("index out of bounds")

            newrecord = FITS_record(self,key)
            return newrecord

    def __setitem__(self,row,value):
        if isinstance(value, FITS_record):
            for i in range(self._nfields):
                self.field(self.names[i])[row] = value.field(self.names[i])
        elif isinstance(value, (tuple, list)):
            if self._nfields == len(value):
                for i in range (self._nfields):
                    self.field(i)[row] = value[i]
            else:
               raise ValueError, \
                     "input tuple or list required to have %s elements" \
                     % self._nfields
        else:
            raise TypeError, \
                  "assignment requires a FITS_record, tuple, or list as input"

    def __setslice__(self,start,end,value):
        _end = min(len(self),end)
        _end = max(0,_end)
        _start = max(0,start)
        _end = min(_end, _start+len(value))

        for i in range(_start,_end):
            self.__setitem__(i,value[i-_start])

    def _get_scale_factors(self, indx):
        """
        Get the scaling flags and factors for one field.

        `indx` is the index of the field.
        """
        if self._coldefs._tbtype == 'BinTableHDU':
            _str = 'a' in self._coldefs.formats[indx]
            _bool = self._coldefs._recformats[indx][-2:] == _booltype
        else:
            _str = self._coldefs.formats[indx][0] == 'A'
            _bool = 0             # there is no boolean in ASCII table
        _number = not(_bool or _str)
        bscale = self._coldefs.bscales[indx]
        bzero = self._coldefs.bzeros[indx]
        _scale = bscale not in ['', None, 1]
        _zero = bzero not in ['', None, 0]
        # ensure bscale/bzero are numbers
        if not _scale:
            bscale = 1
        if not _zero:
            bzero = 0

        return (_str, _bool, _number, _scale, _zero, bscale, bzero)

    def field(self, key):
        """
        A view of a `Column`'s data as an array.
        """
        indx = _get_index(self._coldefs.names, key)

        if (self._convert[indx] is None):
            # for X format
            if isinstance(self._coldefs._recformats[indx], _FormatX):
                _nx = self._coldefs._recformats[indx]._nx
                dummy = np.zeros(self.shape+(_nx,), dtype=np.bool_)
                _unwrapx(rec.recarray.field(self,indx), dummy, _nx)
                self._convert[indx] = dummy
                return self._convert[indx]

            (_str, _bool, _number, _scale, _zero, bscale, bzero) = self._get_scale_factors(indx)

            # for P format
            if isinstance(self._coldefs._recformats[indx], _FormatP):
                dummy = _VLF([None]*len(self))
                dummy._dtype = self._coldefs._recformats[indx]._dtype
                for i in range(len(self)):
                    _offset = rec.recarray.field(self,indx)[i,1] + self._heapoffset
                    self._file.seek(_offset)
                    if self._coldefs._recformats[indx]._dtype is 'a':
                        count = rec.recarray.field(self,indx)[i,0]
                        da = _fromfile(self._file, dtype=self._coldefs._recformats[indx]._dtype+str(1),count =count,sep="")
                        dummy[i] = chararray.array(da,itemsize=count)
                    else:
#                       print type(self._file)
#                       print "type =",self._coldefs._recformats[indx]._dtype
                        count = rec.recarray.field(self,indx)[i,0]
                        dummy[i] = _fromfile(self._file, dtype=self._coldefs._recformats[indx]._dtype,count =count,sep="")
                        dummy[i].dtype = dummy[i].dtype.newbyteorder(">")

                # scale by TSCAL and TZERO
                if _scale or _zero:
                    for i in range(len(self)):
                        dummy[i][:] = dummy[i]*bscale+bzero

                # Boolean (logical) column
                if self._coldefs._recformats[indx]._dtype is _booltype:
                    for i in range(len(self)):
                        dummy[i] = np.equal(dummy[i], ord('T'))

                self._convert[indx] = dummy
                return self._convert[indx]

            if _str:
                return rec.recarray.field(self,indx)

            # ASCII table, convert strings to numbers
            if self._coldefs._tbtype == 'TableHDU':
                _dict = {'I':np.int32, 'F':np.float32, 'E':np.float32, 'D':np.float64}
                _type = _dict[self._coldefs._Formats[indx][0]]

                # if the string = TNULL, return ASCIITNULL
                nullval = self._coldefs.nulls[indx].strip()
                dummy = np.zeros(len(self), dtype=_type)
                dummy[:] = ASCIITNULL
                self._convert[indx] = dummy
                for i in range(len(self)):
                    if rec.recarray.field(self,indx)[i].strip() != nullval:
                        dummy[i] = float(rec.recarray.field(self,indx)[i].replace('D', 'E'))
            else:
                dummy = rec.recarray.field(self,indx)

            # further conversion for both ASCII and binary tables
            if _number and (_scale or _zero):

                # only do the scaling the first time and store it in _convert
                self._convert[indx] = np.array(dummy, dtype=np.float64)
                if _scale:
                    np.multiply(self._convert[indx], bscale, self._convert[indx])
                if _zero:
                    self._convert[indx] += bzero
            elif _bool:
                self._convert[indx] = np.equal(dummy, ord('T'))
            else:
                return dummy

        return self._convert[indx]

    def _scale_back(self):
        """
        Update the parent array, using the (latest) scaled array.
        """
        _dict = {'A':'s', 'I':'d', 'F':'f', 'E':'E', 'D':'E'}
        # calculate the starting point and width of each field for ASCII table
        if self._coldefs._tbtype == 'TableHDU':
            _loc = self._coldefs.starts
            _width = []
            for i in range(len(self.dtype.names)):
                _width.append(_convert_ASCII_format(self._coldefs._Formats[i])[1])
            _loc.append(_loc[-1]+rec.recarray.field(self,i).itemsize)

        self._heapsize = 0
        for indx in range(len(self.dtype.names)):
            if (self._convert[indx] is not None):
                if isinstance(self._coldefs._recformats[indx], _FormatX):
                    _wrapx(self._convert[indx], rec.recarray.field(self,indx), self._coldefs._recformats[indx]._nx)
                    continue

                (_str, _bool, _number, _scale, _zero, bscale, bzero) = self._get_scale_factors(indx)

                # add the location offset of the heap area for each
                # variable length column
                if isinstance(self._coldefs._recformats[indx], _FormatP):
                    desc = rec.recarray.field(self,indx)
                    desc[:] = 0 # reset
                    _npts = map(len, self._convert[indx])
                    desc[:len(_npts),0] = _npts
                    _dtype= np.array([],dtype=self._coldefs._recformats[indx]._dtype)
                    desc[1:,1] = np.add.accumulate(desc[:-1,0])*_dtype.itemsize

                    desc[:,1][:] += self._heapsize
                    self._heapsize += desc[:,0].sum()*_dtype.itemsize

                # conversion for both ASCII and binary tables
                if _number or _str:
                    if _number and (_scale or _zero):
                        dummy = self._convert[indx].copy()
                        if _zero:
                            dummy -= bzero
                        if _scale:
                            dummy /= bscale
                    elif self._coldefs._tbtype == 'TableHDU':
                        dummy = self._convert[indx]
                    else:
                        continue

                    # ASCII table, convert numbers to strings
                    if self._coldefs._tbtype == 'TableHDU':
                        _format = self._coldefs._Formats[indx].strip()
                        _lead = self._coldefs.starts[indx] - _loc[indx]
                        if _lead < 0:
                            raise ValueError, "column `%s` starting point overlaps to the previous column" % indx+1
                        _trail = _loc[indx+1] - _width[indx] - self._coldefs.starts[indx]
                        if _trail < 0:
                            raise ValueError, "column `%s` ending point overlaps to the next column" % indx+1
                        if 'A' in _format:
                            _pc = '%-'
                        else:
                            _pc = '%'
                        _fmt = ' '*_lead + _pc + _format[1:] + _dict[_format[0]] + ' '*_trail

                        # not using numarray.strings's num2char because the
                        # result is not allowed to expand (as C/Python does).
                        for i in range(len(dummy)):
                            x = _fmt % dummy[i]
                            if len(x) > (_loc[indx+1]-_loc[indx]):
                                raise ValueError, "number `%s` does not fit into the output's itemsize of %s" % (x, _width[indx])
                            else:
                                rec.recarray.field(self,indx)[i] = x
                        if 'D' in _format:
                            rec.recarray.field(self,indx).replace('E', 'D')


                    # binary table
                    else:
                        if isinstance(rec.recarray.field(self,indx)[0], np.integer):
                            dummy = np.around(dummy)
                        rec.recarray.field(self,indx)[:] = dummy.astype(rec.recarray.field(self,indx).dtype)

                    del dummy

                # ASCII table does not have Boolean type
                elif _bool:
                    rec.recarray.field(self,indx)[:] = np.choose(self._convert[indx],
                                                    (np.array([ord('F')],dtype=np.int8)[0],
                                                    np.array([ord('T')],dtype=np.int8)[0]))


class GroupData(FITS_rec):
    """
    Random groups data object.

    Allows structured access to FITS Group data in a manner analogous
    to tables.
    """

    def __new__(subtype, input=None, bitpix=None, pardata=None, parnames=[],
                 bscale=None, bzero=None, parbscales=None, parbzeros=None):
        """
        Parameters
        ----------
        input : array or FITS_rec instance
            input data, either the group data itself (a
            `numpy.ndarray`) or a record array (`FITS_rec`) which will
            contain both group parameter info and the data.  The rest
            of the arguments are used only for the first case.

        bitpix : int
            data type as expressed in FITS ``BITPIX`` value (8, 16, 32,
            64, -32, or -64)

        pardata : sequence of arrays
            parameter data, as a list of (numeric) arrays.

        parnames : sequence of str
            list of parameter names.

        bscale : int
            ``BSCALE`` of the data

        bzero : int
            ``BZERO`` of the data

        parbscales : sequence of int
            list of bscales for the parameters

        parbzeros : sequence of int
            list of bzeros for the parameters
        """
        if not isinstance(input, FITS_rec):
            _formats = ''
            _cols = []
            if pardata is None:
                npars = 0
            else:
                npars = len(pardata)

            if parbscales is None:
                parbscales = [None]*npars
            if parbzeros is None:
                parbzeros = [None]*npars

            if bitpix is None:
                bitpix = _ImageBaseHDU.ImgCode[input.dtype.name]
            fits_fmt = GroupsHDU._dict[bitpix] # -32 -> 'E'
            _fmt = _fits2rec[fits_fmt] # 'E' -> 'f4'
            _formats = (_fmt+',') * npars
            data_fmt = '%s%s' % (`input.shape[1:]`, _fmt)
            _formats += data_fmt
            gcount = input.shape[0]
            for i in range(npars):
                _cols.append(Column(name='c'+`i+1`,
                                    format = fits_fmt,
                                    bscale = parbscales[i],
                                    bzero = parbzeros[i]))
            _cols.append(Column(name='data',
                                format = fits_fmt,
                                bscale = bscale,
                                bzero = bzero))
            _coldefs = ColDefs(_cols)

            self = FITS_rec.__new__(subtype,
                                    rec.array(None,
                                              formats=_formats,
                                              names=_coldefs.names,
                                              shape=gcount))
            self._coldefs = _coldefs
            self.parnames = [i.lower() for i in parnames]

            for i in range(npars):
                (_scale, _zero)  = self._get_scale_factors(i)[3:5]
                if _scale or _zero:
                    self._convert[i] = pardata[i]
                else:
                    rec.recarray.field(self,i)[:] = pardata[i]
            (_scale, _zero)  = self._get_scale_factors(npars)[3:5]
            if _scale or _zero:
                self._convert[npars] = input
            else:
                rec.recarray.field(self,npars)[:] = input
        else:
             self = FITS_rec.__new__(subtype,input)
        return self

    def __getattribute__(self, attr):
        if attr == 'data':
            return self.field('data')
        else:
            return super(GroupData, self).__getattribute__(attr)

    def __getattr__(self, attr):
        if attr == '_unique':
            _unique = {}
            for i in range(len(self.parnames)):
                _name = self.parnames[i]
                if _name in _unique:
                    _unique[_name].append(i)
                else:
                    _unique[_name] = [i]
            self.__dict__[attr] = _unique
        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def par(self, parName):
        """
        Get the group parameter values.
        """
        if isinstance(parName, (int, long, np.integer)):
            result = self.field(parName)
        else:
            indx = self._unique[parName.lower()]
            if len(indx) == 1:
                result = self.field(indx[0])

            # if more than one group parameter have the same name
            else:
                result = self.field(indx[0]).astype('f8')
                for i in indx[1:]:
                    result += self.field(i)

        return result

    def _getitem(self, key):
        row = (offset - self._byteoffset) // self._strides[0]
        return _Group(self, row)

    def __getitem__(self, key):
        return _Group(self,key,self.parnames)

class _Group(FITS_record):
    """
    One group of the random group data.
    """
    def __init__(self, input, row, parnames):
        super(_Group, self).__init__(input, row)
        self.parnames = parnames

    def __getattr__(self, attr):
        if attr == '_unique':
            _unique = {}
            for i in range(len(self.parnames)):
                _name = self.parnames[i]
                if _name in _unique:
                    _unique[_name].append(i)
                else:
                    _unique[_name] = [i]
            self.__dict__[attr] = _unique
        try:
             return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def __str__(self):
        """
        Print one row.
        """
        if isinstance(self.row, slice):
            if self.row.step:
                step = self.row.step
            else:
                step = 1

            if self.row.stop > len(self.array):
                stop = len(self.array)
            else:
                stop = self.row.stop

            outlist = []

            for i in range(self.row.start, stop, step):
                rowlist = []

                for j in range(self.array._nfields):
                    rowlist.append(`self.array.field(j)[i]`)

                outlist.append(" (" + ", ".join(rowlist) + ")")

            return "[" + ",\n".join(outlist) + "]"
        else:
            return super(_Group, self).__str__()

    def par(self, parName):
        """
        Get the group parameter value.
        """
        if isinstance(parName, (int, long, np.integer)):
            result = self.array[self.row][parName]
        else:
            indx = self._unique[parName.lower()]
            if len(indx) == 1:
                result = self.array[self.row][indx[0]]

            # if more than one group parameter have the same name
            else:
                result = self.array[self.row][indx[0]].astype('f8')
                for i in indx[1:]:
                    result += self.array[self.row][i]

        return result


    def setpar(self, parName, value):
        """
        Set the group parameter value.
        """
        if isinstance(parName, (int, long, np.integer)):
            self.array[self.row][parName] = value
        else:
            indx = self._unique[parName.lower()]
            if len(indx) == 1:
                self.array[self.row][indx[0]] = value

            # if more than one group parameter have the same name, the
            # value must be a list (or tuple) containing arrays
            else:
                if isinstance(value, (list, tuple)) and len(indx) == len(value):
                    for i in range(len(indx)):
                        self.array[self.row][indx[i]] = value[i]
                else:
                    raise ValueError, "parameter value must be a sequence " + \
                                      "with %d arrays/numbers." % len(indx)



class _TableBaseHDU(_ExtensionHDU):
    """
    FITS table extension base HDU class.
    """
    def __init__(self, data=None, header=None, name=None):
        """
        Parameters
        ----------
        header : Header instance
            header to be used

        data : array
            data to be used

        name : str
            name to be populated in ``EXTNAME`` keyword
        """

        if header is not None:
            if not isinstance(header, Header):
                raise ValueError, "header must be a Header object"

        if data is DELAYED:

            # this should never happen
            if header is None:
                raise ValueError, "No header to setup HDU."

            # if the file is read the first time, no need to copy, and keep it unchanged
            else:
                self._header = header
        else:

            # construct a list of cards of minimal header
            _list = CardList([
                Card('XTENSION',      '', ''),
                Card('BITPIX',         8, 'array data type'),
                Card('NAXIS',          2, 'number of array dimensions'),
                Card('NAXIS1',         0, 'length of dimension 1'),
                Card('NAXIS2',         0, 'length of dimension 2'),
                Card('PCOUNT',         0, 'number of group parameters'),
                Card('GCOUNT',         1, 'number of groups'),
                Card('TFIELDS',        0, 'number of table fields')
                ])

            if header is not None:

                # Make a "copy" (not just a view) of the input header, since it
                # may get modified.  the data is still a "view" (for now)
                hcopy = header.copy()
                hcopy._strip()
                _list.extend(hcopy.ascardlist())

            self._header = Header(_list)

        if (data is not DELAYED):
            if isinstance(data,np.ndarray) and not data.dtype.fields == None:
                if isinstance(data, FITS_rec):
                    self.data = data
                elif isinstance(data, rec.recarray):
                    self.data = FITS_rec(data)
                else:
                    self.data = data.view(FITS_rec)

                self._header['NAXIS1'] = self.data.itemsize
                self._header['NAXIS2'] = self.data.shape[0]
                self._header['TFIELDS'] = self.data._nfields

                if self.data._coldefs == None:
                    #
                    # The data does not have a _coldefs attribute so
                    # create one from the underlying recarray.
                    #
                    columns = []

                    for i in range(len(data.dtype.names)):
                       cname = data.dtype.names[i]

                       if data.dtype.fields[cname][0].type == np.string_:
                           format = \
                            'A'+str(data.dtype.fields[cname][0].itemsize)
                       else:
                           format = \
                            _convert_format(data.dtype.fields[cname][0].str[1:],
                            True)

                       c = Column(name=cname,format=format,array=data[cname])
                       columns.append(c)

                    try:
                        tbtype = 'BinTableHDU'

                        if self._xtn == 'TABLE':
                            tbtype = 'TableHDU'
                    except AttributeError:
                        pass

                    self.data._coldefs = ColDefs(columns,tbtype=tbtype)

                self.columns = self.data._coldefs
                self.update()

                try:
                   # Make the ndarrays in the Column objects of the ColDefs
                   # object of the HDU reference the same ndarray as the HDU's
                   # FITS_rec object.
                    for i in range(len(self.columns)):
                        self.columns.data[i].array = self.data.field(i)

                    # Delete the _arrays attribute so that it is recreated to
                    # point to the new data placed in the column objects above
                    if self.columns.__dict__.has_key('_arrays'):
                        del self.columns.__dict__['_arrays']
                except (TypeError, AttributeError), e:
                    pass
            elif data is None:
                pass
            else:
                raise TypeError, "table data has incorrect type"

        #  set extension name
        if not name and self._header.has_key('EXTNAME'):
            name = self._header['EXTNAME']
        self.name = name

    def __getattr__(self, attr):
        """
        Get the `data` or `columns` attribute.
        """
        if attr == 'data':
            size = self.size()
            if size:
                self._file.seek(self._datLoc)
                data = _get_tbdata(self)
                data._coldefs = self.columns
                data.formats = self.columns.formats
#                print "Got data?"
            else:
                data = None
            self.__dict__[attr] = data

        elif attr == 'columns':
            class_name = str(self.__class__)
            class_name = class_name[class_name.rfind('.')+1:-2]
            self.__dict__[attr] = ColDefs(self, tbtype=class_name)

        elif attr == '_theap':
            self.__dict__[attr] = self._header.get('THEAP', self._header['NAXIS1']*self._header['NAXIS2'])
        elif attr == '_pcount':
            self.__dict__[attr] = self._header.get('PCOUNT', 0)
        else:
            return _AllHDU.__getattr__(self,attr)

        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)


    def _summary(self):
        """
        Summarize the HDU: name, dimensions, and formats.
        """
        class_name  = str(self.__class__)
        type  = class_name[class_name.rfind('.')+1:-2]

        # if data is touched, use data info.
        if 'data' in dir(self):
            if self.data is None:
                _shape, _format = (), ''
                _nrows = 0
            else:
                _nrows = len(self.data)

            _ncols = len(self.columns.formats)
            _format = self.columns.formats

        # if data is not touched yet, use header info.
        else:
            _shape = ()
            _nrows = self._header['NAXIS2']
            _ncols = self._header['TFIELDS']
            _format = '['
            for j in range(_ncols):
                _format += self._header['TFORM'+`j+1`] + ', '
            _format = _format[:-2] + ']'
        _dims = "%dR x %dC" % (_nrows, _ncols)

        return "%-10s  %-11s  %5d  %-12s  %s" % \
            (self.name, type, len(self._header.ascard), _dims, _format)

    def get_coldefs(self):
        """
        Returns the table's column definitions.
        """
        return self.columns

    def update(self):
        """
        Update header keywords to reflect recent changes of columns.
        """
        _update = self._header.update
        _append = self._header.ascard.append
        _cols = self.columns
        _update('naxis1', self.data.itemsize, after='naxis')
        _update('naxis2', self.data.shape[0], after='naxis1')
        _update('tfields', len(_cols), after='gcount')

        # Wipe out the old table definition keywords.  Mark them first,
        # then delete from the end so as not to confuse the indexing.
        _list = []
        for i in range(len(self._header.ascard)-1,-1,-1):
            _card = self._header.ascard[i]
            _key = _tdef_re.match(_card.key)
            try: keyword = _key.group('label')
            except: continue                # skip if there is no match
            if (keyword in _keyNames):
                _list.append(i)
        for i in _list:
            del self._header.ascard[i]
        del _list

        # populate the new table definition keywords
        for i in range(len(_cols)):
            for cname in _commonNames:
                val = getattr(_cols, cname+'s')[i]
                if val != '':
                    keyword = _keyNames[_commonNames.index(cname)]+`i+1`
                    if cname == 'format' and isinstance(self, BinTableHDU):
                        val = _cols._recformats[i]
                        if isinstance(val, _FormatX):
                            val = `val._nx` + 'X'
                        elif isinstance(val, _FormatP):
                            VLdata = self.data.field(i)
                            VLdata._max = max(map(len, VLdata))
                            if val._dtype == 'a':
                                fmt = 'A'
                            else:
                                fmt = _convert_format(val._dtype, reverse=1)
                            val = 'P' + fmt + '(%d)' %  VLdata._max
                        else:
                            val = _convert_format(val, reverse=1)
                    #_update(keyword, val)
                    _append(Card(keyword, val))

    def copy(self):
        """
        Make a copy of the table HDU, both header and data are copied.
        """
        # touch the data, so it's defined (in the case of reading from a
        # FITS file)
        self.data
        return new_table(self.columns, header=self._header, tbtype=self.columns._tbtype)

    def _verify(self, option='warn'):
        """
        _TableBaseHDU verify method.
        """
        _err = _ExtensionHDU._verify(self, option=option)
        self.req_cards('NAXIS', None, 'val == 2', 2, option, _err)
        self.req_cards('BITPIX', None, 'val == 8', 8, option, _err)
        self.req_cards('TFIELDS', '== 7', _isInt+" and val >= 0 and val <= 999", 0, option, _err)
        tfields = self._header['TFIELDS']
        for i in range(tfields):
            self.req_cards('TFORM'+`i+1`, None, None, None, option, _err)
        return _err


class TableHDU(_TableBaseHDU):
    """
    FITS ASCII table extension HDU class.
    """
    __format_RE = re.compile(
        r'(?P<code>[ADEFI])(?P<width>\d+)(?:\.(?P<prec>\d+))?')

    def __init__(self, data=None, header=None, name=None):
        """
        Parameters
        ----------
        data : array
            data of the table

        header : Header instance
            header to be used for the HDU

        name : str
            the ``EXTNAME`` value
        """
        self._xtn = 'TABLE'
        _TableBaseHDU.__init__(self, data=data, header=header, name=name)
        if self._header[0].rstrip() != self._xtn:
            self._header[0] = self._xtn
            self._header.ascard[0].comment = 'ASCII table extension'
    '''
    def format(self):
        strfmt, strlen = '', 0
        for j in range(self._header['TFIELDS']):
            bcol = self._header['TBCOL'+`j+1`]
            valu = self._header['TFORM'+`j+1`]
            fmt  = self.__format_RE.match(valu)
            if fmt:
                code, width, prec = fmt.group('code', 'width', 'prec')
            else:
                raise ValueError, valu
            size = eval(width)+1
            strfmt = strfmt + 's'+str(size) + ','
            strlen = strlen + size
        else:
            strfmt = '>' + strfmt[:-1]
        return strfmt
    '''

    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
        if self.__dict__.has_key('data') and self.data != None:
            # We have the data to be used.
            # We need to pad the data to a block length before calculating
            # the datasum.

            if self.size() > 0:
                d = np.append(np.fromstring(self.data, dtype='ubyte'),
                              np.fromstring(_padLength(self.size())*' ',
                                            dtype='ubyte'))

            cs = self._compute_checksum(np.fromstring(d, dtype='ubyte'),0)
            return cs
        else:
            # This is the case where the data has not been read from the file
            # yet.  We can handle that in a generic manner so we do it in the
            # base class.  The other possibility is that there is no data at
            # all.  This can also be handled in a gereric manner.
            return super(TableHDU,self)._calculate_datasum()

    def _verify(self, option='warn'):
        """
        `TableHDU` verify method.
        """
        _err = _TableBaseHDU._verify(self, option=option)
        self.req_cards('PCOUNT', None, 'val == 0', 0, option, _err)
        tfields = self._header['TFIELDS']
        for i in range(tfields):
            self.req_cards('TBCOL'+`i+1`, None, _isInt, None, option, _err)
        return _err


class BinTableHDU(_TableBaseHDU):
    """
    Binary table HDU class.
    """
    def __init__(self, data=None, header=None, name=None):
        """
        Parameters
        ----------
        data : array
            data of the table

        header : Header instance
            header to be used for the HDU

        name : str
            the ``EXTNAME`` value
        """

        self._xtn = 'BINTABLE'
        _TableBaseHDU.__init__(self, data=data, header=header, name=name)
        hdr = self._header
        if hdr[0] != self._xtn:
            hdr[0] = self._xtn
            hdr.ascard[0].comment = 'binary table extension'

        self._header._hdutype = BinTableHDU

    def _calculate_datasum_from_data(self, data):
        """
        Calculate the value for the ``DATASUM`` card given the input data
        """
        # Check the byte order of the data.  If it is little endian we
        # must swap it before calculating the datasum.
        for i in range(data._nfields):
            coldata = data.field(i)

            if not isinstance(coldata, chararray.chararray):
                if isinstance(coldata, _VLF):
                    k = 0
                    for j in coldata:
                        if not isinstance(j, chararray.chararray):
                            if j.itemsize > 1:
                                if j.dtype.str[0] != '>':
                                    j[:] = j.byteswap()
                                    j.dtype = j.dtype.newbyteorder('>')
                        if rec.recarray.field(data,i)[k:k+1].dtype.str[0]!='>':
                            rec.recarray.field(data,i)[k:k+1].byteswap(True)
                        k = k + 1
                else:
                    if coldata.itemsize > 1:
                        if data.field(i).dtype.str[0] != '>':
                            data.field(i)[:] = data.field(i).byteswap()
        data.dtype = data.dtype.newbyteorder('>')

        dout=np.fromstring(data, dtype='ubyte')

        for i in range(data._nfields):
            if isinstance(data._coldefs._recformats[i], _FormatP):
                for j in range(len(data.field(i))):
                    coldata = data.field(i)[j]
                    if len(coldata) > 0:
                        dout = np.append(dout,
                                    np.fromstring(coldata,dtype='ubyte'))

        cs = self._compute_checksum(dout,0)
        return cs

    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
        if self.__dict__.has_key('data') and self.data != None:
            # We have the data to be used.
            return self._calculate_datasum_from_data(self.data)
        else:
            # This is the case where the data has not been read from the file
            # yet.  We can handle that in a generic manner so we do it in the
            # base class.  The other possibility is that there is no data at
            # all.  This can also be handled in a gereric manner.
            return super(BinTableHDU,self)._calculate_datasum()

    def tdump(self, datafile=None, cdfile=None, hfile=None, clobber=False):
        """
        Dump the table HDU to a file in ASCII format.  The table may be dumped
        in three separate files, one containing column definitions, one
        containing header parameters, and one for table data.

        Parameters
        ----------
        datafile : file path, file object or file-like object, optional
            Output data file.  The default is the root name of the
            fits file associated with this HDU appended with the
            extension ``.txt``.

        cdfile : file path, file object or file-like object, optional
            Output column definitions file.  The default is `None`, no
            column definitions output is produced.

        hfile : file path, file object or file-like object, optional
            Output header parameters file.  The default is `None`,
            no header parameters output is produced.

        clobber : bool
            Overwrite the output files if they exist.

        Notes
        -----
        The primary use for the `tdump` method is to allow editing in a
        standard text editor of the table data and parameters.  The
        `tcreate` method can be used to reassemble the table from the
        three ASCII files.
        """
        # check if the output files already exist
        exceptMessage = 'File '
        files = [datafile, cdfile, hfile]

        for f in files:
            if (isinstance(f,types.StringType) or
                isinstance(f,types.UnicodeType)):
                if (os.path.exists(f) and os.path.getsize(f) != 0):
                    if clobber:
                        warnings.warn(" Overwrite existing file '%s'." % f)
                        os.remove(f)
                    else:
                        exceptMessage = exceptMessage + "'%s', " % f

        if exceptMessage != 'File ':
            exceptMessage = exceptMessage[:-2] + ' already exist.'
            raise IOError, exceptMessage

        # Process the data

        if not datafile:
            root,ext = os.path.splitext(self._file.name)
            datafile = root + '.txt'

        closeDfile = False

        if isinstance(datafile, types.StringType) or \
           isinstance(datafile, types.UnicodeType):
            datafile = __builtin__.open(datafile,'w')
            closeDfile = True

        dlines = []   # lines to go out to the data file

        # Process each row of the table and output the result to the dlines
        # list.

        for i in range(len(self.data)):
            line = ''   # the line for this row of the table

            # Process each column of the row.

            for name in self.columns.names:
                VLA_format = None   # format of data in a variable length array
                                    # where None means it is not a VLA
                fmt = _convert_format(
                      self.columns.formats[self.columns.names.index(name)])

                if isinstance(fmt, _FormatP):
                    # P format means this is a variable length array so output
                    # the length of the array for this row and set the format
                    # for the VLA data
                    line = line + "VLA_Length= %-21d " % \
                                  len(self.data.field(name)[i])
                    (repeat,dtype,option) = _parse_tformat(
                         self.columns.formats[self.columns.names.index(name)])
                    VLA_format =  _fits2rec[option[0]][0]

                if self.data.dtype.fields[name][0].subdtype:
                    # The column data is an array not a single element

                    if VLA_format:
                        arrayFormat = VLA_format
                    else:
                        arrayFormat = \
                              self.data.dtype.fields[name][0].subdtype[0].char

                    # Output the data for each element in the array

                    for val in self.data.field(name)[i].flat:
                        if arrayFormat == 'S':
                            # output string

                            if len(string.split(val)) != 1:
                                # there is whitespace in the string so put it
                                # in quotes
                                width = val.itemsize+3
                                str = '"' + val + '" '
                            else:
                                # no whitespace
                                width = val.itemsize+1
                                str = val

                            line = line + '%-*s'%(width,str)
                        elif arrayFormat in np.typecodes['AllInteger']:
                            # output integer
                            line = line + '%21d ' % val
                        elif arrayFormat in np.typecodes['AllFloat']:
                            # output floating point
                            line = line + '%#21.15g ' % val
                else:
                    # The column data is a single element
                    arrayFormat = self.data.dtype.fields[name][0].char

                    if arrayFormat == 'S':
                        # output string

                        if len(string.split(self.data.field(name)[i])) != 1:
                            # there is whitespace in the string so put it
                            # in quotes
                            width = self.data.dtype.fields[name][0].itemsize+3
                            str = '"' + self.data.field(name)[i] + '" '
                        else:
                            # no whitespace
                            width = self.data.dtype.fields[name][0].itemsize+1
                            str = self.data.field(name)[i]

                        line = line + '%-*s'%(width,str)
                    elif arrayFormat in np.typecodes['AllInteger']:
                        # output integer
                        line = line + '%21d ' % self.data.field(name)[i]
                    elif arrayFormat in np.typecodes['AllFloat']:
                        # output floating point
                        line = line + '%21.15g ' % self.data.field(name)[i]

            # Replace the trailing blank in the line with a new line
            # and append the line for this row to the list of data lines
            line = line[:-1] + '\n'
            dlines.append(line)

        # Write the data lines out to the ASCII data file
        datafile.writelines(dlines)

        if closeDfile:
            datafile.close()

        # Process the column definitions

        if cdfile:
            closeCdfile = False

            if isinstance(cdfile, types.StringType) or \
               isinstance(cdfile, types.UnicodeType):
                cdfile = __builtin__.open(cdfile,'w')
                closeCdfile = True

            cdlines = []   # lines to go out to the column definitions file

            # Process each column of the table and output the result to the
            # cdlines list

            for j in range(len(self.columns.formats)):
                disp = self.columns.disps[j]

                if disp == '':
                    disp = '""'  # output "" if value is not set

                unit = self.columns.units[j]

                if unit == '':
                    unit = '""'

                dim = self.columns.dims[j]

                if dim == '':
                    dim = '""'

                null = self.columns.nulls[j]

                if null == '':
                    null = '""'

                bscale = self.columns.bscales[j]

                if bscale == '':
                    bscale = '""'

                bzero = self.columns.bzeros[j]

                if bzero == '':
                    bzero = '""'

                #Append the line for this column to the list of output lines
                cdlines.append(
                   "%-16s %-16s %-16s %-16s %-16s %-16s %-16s %-16s\n" %
                   (self.columns.names[j],self.columns.formats[j],
                    disp, unit, dim, null, bscale, bzero))

            # Write the column definition lines out to the ASCII column
            # definitions file
            cdfile.writelines(cdlines)

            if closeCdfile:
                cdfile.close()

        # Process the header parameters

        if hfile:
            self.header.toTxtFile(hfile)

    tdumpFileFormat = """

- **datafile:** Each line of the data file represents one row of table
  data.  The data is output one column at a time in column order.  If
  a column contains an array, each element of the column array in the
  current row is output before moving on to the next column.  Each row
  ends with a new line.

  Integer data is output right-justified in a 21-character field
  followed by a blank.  Floating point data is output right justified
  using 'g' format in a 21-character field with 15 digits of
  precision, followed by a blank.  String data that does not contain
  whitespace is output left-justified in a field whose width matches
  the width specified in the ``TFORM`` header parameter for the
  column, followed by a blank.  When the string data contains
  whitespace characters, the string is enclosed in quotation marks
  (``""``).  For the last data element in a row, the trailing blank in
  the field is replaced by a new line character.

  For column data containing variable length arrays ('P' format), the
  array data is preceded by the string ``'VLA_Length= '`` and the
  integer length of the array for that row, left-justified in a
  21-character field, followed by a blank.

  For column data representing a bit field ('X' format), each bit
  value in the field is output right-justified in a 21-character field
  as 1 (for true) or 0 (for false).

- **cdfile:** Each line of the column definitions file provides the
  definitions for one column in the table.  The line is broken up into
  8, sixteen-character fields.  The first field provides the column
  name (``TTYPEn``).  The second field provides the column format
  (``TFORMn``).  The third field provides the display format
  (``TDISPn``).  The fourth field provides the physical units
  (``TUNITn``).  The fifth field provides the dimensions for a
  multidimensional array (``TDIMn``).  The sixth field provides the
  value that signifies an undefined value (``TNULLn``).  The seventh
  field provides the scale factor (``TSCALn``).  The eighth field
  provides the offset value (``TZEROn``).  A field value of ``""`` is
  used to represent the case where no value is provided.

- **hfile:** Each line of the header parameters file provides the
  definition of a single HDU header card as represented by the card
  image.
"""

    tdump.__doc__ += tdumpFileFormat.replace("\n", "\n        ")

    def tcreate(self, datafile, cdfile=None, hfile=None, replace=False):
        """
        Create a table from the input ASCII files.  The input is from up to
        three separate files, one containing column definitions, one containing
        header parameters, and one containing column data.  The column
        definition and header parameters files are not required.  When absent
        the column definitions and/or header parameters are taken from the
        current values in this HDU.

        Parameters
        ----------
        datafile : file path, file object or file-like object
            Input data file containing the table data in ASCII format.

        cdfile : file path, file object, file-like object, optional
            Input column definition file containing the names,
            formats, display formats, physical units, multidimensional
            array dimensions, undefined values, scale factors, and
            offsets associated with the columns in the table.  If
            `None`, the column definitions are taken from the current
            values in this object.

        hfile : file path, file object, file-like object, optional
            Input parameter definition file containing the header
            parameter definitions to be associated with the table.  If
            `None`, the header parameter definitions are taken from
            the current values in this objects header.

        replace : bool
            When `True`, indicates that the entire header should be
            replaced with the contents of the ASCII file instead of
            just updating the current header.

        Notes
        -----
        The primary use for the `tcreate` method is to allow the input
        of ASCII data that was edited in a standard text editor of the
        table data and parameters.  The `tdump` method can be used to
        create the initial ASCII files.
        """
        # Process the column definitions file

        if cdfile:
            closeCdfile = False

            if isinstance(cdfile, types.StringType) or \
               isinstance(cdfile, types.UnicodeType):
                cdfile = __builtin__.open(cdfile,'r')
                closeCdfile = True

            cdlines = cdfile.readlines()

            if closeCdfile:
                cdfile.close()

            self.columns.names = []
            self.columns.formats = []
            self.columns.disps = []
            self.columns.units = []
            self.columns.dims = []
            self.columns.nulls = []
            self.columns.bscales = []
            self.columns.bzeros = []

            for line in cdlines:
                words = string.split(line[:-1])
                self.columns.names.append(words[0])
                self.columns.formats.append(words[1])
                self.columns.disps.append(string.replace(words[2],'""',''))
                self.columns.units.append(string.replace(words[3],'""',''))
                self.columns.dims.append(string.replace(words[4],'""',''))
                null = string.replace(words[5],'""','')

                if null != '':
                    self.columns.nulls.append(eval(null))
                else:
                    self.columns.nulls.append(null)

                bscale = string.replace(words[6],'""','')

                if bscale != '':
                    self.columns.bscales.append(eval(bscale))
                else:
                    self.columns.bscales.append(bscale)

                bzero = string.replace(words[7],'""','')

                if bzero != '':
                    self.columns.bzeros.append(eval(bzero))
                else:
                    self.columns.bzeros.append(bzero)

        # Process the parameter file

        if hfile:
            self._header.fromTxtFile(hfile, replace)

        # Process the data file

        closeDfile = False

        if isinstance(datafile, types.StringType) or \
           isinstance(datafile, types.UnicodeType):
            datafile = __builtin__.open(datafile,'r')
            closeDfile = True

        dlines = datafile.readlines()

        if closeDfile:
            datafile.close()

        arrays = []
        VLA_formats = []
        X_format_size = []
        recFmts = []

        for i in range(len(self.columns.names)):
            arrayShape = len(dlines)
            recFmt = _convert_format(self.columns.formats[i])
            recFmts.append(recFmt[0])
            X_format_size = X_format_size + [-1]

            if isinstance(recFmt, _FormatP):
                recFmt = 'O'
                (repeat,dtype,option) = _parse_tformat(self.columns.formats[i])
                VLA_formats = VLA_formats + [_fits2rec[option[0]]]
            elif isinstance(recFmt, _FormatX):
                recFmt = np.uint8
                (X_format_size[i],dtype,option) = \
                                     _parse_tformat(self.columns.formats[i])
                arrayShape = (len(dlines),X_format_size[i])

            arrays.append(np.empty(arrayShape,recFmt))

        lineNo = 0

        for line in dlines:
            words = []
            idx = 0
            VLA_Lengths = []

            while idx < len(line):

                if line[idx:idx+12] == 'VLA_Length= ':
                    VLA_Lengths = VLA_Lengths + [int(line[idx+12:idx+34])]
                    idx += 34

                idx1 = string.find(line[idx:],'"')

                if idx1 >=0:
                    words = words + string.split(line[idx:idx+idx1])
                    idx2 = string.find(line[idx+idx1+1:],'"')
                    words = words + [line[idx1+1:idx1+idx2+1]]
                    idx = idx + idx1 + idx2 + 2
                else:
                    idx2 = string.find(line[idx:],'VLA_Length= ')

                    if idx2 < 0:
                        words = words + string.split(line[idx:])
                        idx = len(line)
                    else:
                        words = words + string.split(line[idx:idx+idx2])
                        idx = idx + idx2

            idx = 0
            VLA_idx = 0

            for i in range(len(self.columns.names)):

                if arrays[i].dtype == 'object':
                    arrays[i][lineNo] = np.array(
                     words[idx:idx+VLA_Lengths[VLA_idx]],VLA_formats[VLA_idx])
                    idx += VLA_Lengths[VLA_idx]
                    VLA_idx += 1
                elif X_format_size[i] >= 0:
                    arrays[i][lineNo] = words[idx:idx+X_format_size[i]]
                    idx += X_format_size[i]
                elif isinstance(arrays[i][lineNo], np.ndarray):
                    arrays[i][lineNo] = words[idx:idx+arrays[i][lineNo].size]
                    idx += arrays[i][lineNo].size
                else:
                    if recFmts[i] == 'a':
                        # make sure character arrays are blank filled
                        arrays[i][lineNo] = words[idx]+(arrays[i].itemsize-
                                                        len(words[idx]))*' '
                    else:
                        arrays[i][lineNo] = words[idx]

                    idx += 1

            lineNo += 1

        columns = []

        for i in range(len(self.columns.names)):
            columns.append(Column(name=self.columns.names[i],
                                  format=self.columns.formats[i],
                                  disp=self.columns.disps[i],
                                  unit=self.columns.units[i],
                                  null=self.columns.nulls[i],
                                  bscale=self.columns.bscales[i],
                                  bzero=self.columns.bzeros[i],
                                  dim=self.columns.dims[i],
                                  array=arrays[i]))

        tmp = new_table(columns, self.header)
        self.__dict__ = tmp.__dict__

    tcreate.__doc__ += tdumpFileFormat.replace("\n", "\n        ")

if compressionSupported:
    # If compression object library imports properly then define the
    # CompImageHDU class.

    # Default compression parameter values

    def_compressionType = 'RICE_1'
    def_quantizeLevel = 16.
    def_hcompScale = 0.
    def_hcompSmooth = 0
    def_blockSize = 32
    def_bytePix = 4

    class CompImageHDU(BinTableHDU):
        """
        Compressed Image HDU class.
        """
        def __init__(self, data=None, header=None, name=None,
                     compressionType=def_compressionType,
                     tileSize=None,
                     hcompScale=def_hcompScale,
                     hcompSmooth=def_hcompSmooth,
                     quantizeLevel=def_quantizeLevel):
            """
            Parameters
            ----------
            data : array, optional
                data of the image

            header : Header instance, optional
                header to be associated with the image; when reading
                the HDU from a file ( `data` = "DELAYED" ), the header
                read from the file

            name : str, optional
                the ``EXTNAME`` value; if this value is `None`, then
                the name from the input image header will be used; if
                there is no name in the input image header then the
                default name ``COMPRESSED_IMAGE`` is used.

            compressionType : str, optional
                compression algorithm 'RICE_1', 'PLIO_1', 'GZIP_1',
                'HCOMPRESS_1'

            tileSize : int, optional
                compression tile sizes.  Default treats each row of
                image as a tile.

            hcompScale : float, optional
                HCOMPRESS scale parameter

            hcompSmooth : float, optional
                HCOMPRESS smooth parameter

            quantizeLevel : float, optional
                floating point quantization level; see note below

            Notes
            -----
            The pyfits module supports 2 methods of image compression.

                1) The entire FITS file may be externally compressed
                   with the gzip or pkzip utility programs, producing
                   a ``*.gz`` or ``*.zip`` file, respectively.  When
                   reading compressed files of this type, pyfits first
                   uncompresses the entire file into a temporary file
                   before performing the requested read operations.
                   The pyfits module does not support writing to these
                   types of compressed files.  This type of
                   compression is supported in the `_File` class, not
                   in the `CompImageHDU` class.  The file compression
                   type is recognized by the ``.gz`` or ``.zip`` file
                   name extension.

                2) The `CompImageHDU` class supports the FITS tiled
                   image compression convention in which the image is
                   subdivided into a grid of rectangular tiles, and
                   each tile of pixels is individually compressed.
                   The details of this FITS compression convention are
                   described at the `FITS Support Office web site
                   <http://fits.gsfc.nasa.gov/registry/tilecompression.html>`_.
                   Basically, the compressed image tiles are stored in
                   rows of a variable length arrray column in a FITS
                   binary table.  The pyfits module recognizes that
                   this binary table extension contains an image and
                   treats it as if it were an image extension.  Under
                   this tile-compression format, FITS header keywords
                   remain uncompressed.  At this time, pyfits does not
                   support the ability to extract and uncompress
                   sections of the image without having to uncompress
                   the entire image.

            The `pyfits` module supports 3 general-purpose compression
            algorithms plus one other special-purpose compression
            technique that is designed for data masks with positive
            integer pixel values.  The 3 general purpose algorithms
            are GZIP, Rice, and HCOMPRESS, and the special-purpose
            technique is the IRAF pixel list compression technique
            (PLIO).  The `compressionType` parameter defines the
            compression algorithm to be used.

            The FITS image can be subdivided into any desired
            rectangular grid of compression tiles.  With the GZIP,
            Rice, and PLIO algorithms, the default is to take each row
            of the image as a tile.  The HCOMPRESS algorithm is
            inherently 2-dimensional in nature, so the default in this
            case is to take 16 rows of the image per tile.  In most
            cases, it makes little difference what tiling pattern is
            used, so the default tiles are usually adequate.  In the
            case of very small images, it could be more efficient to
            compress the whole image as a single tile.  Note that the
            image dimensions are not required to be an integer
            multiple of the tile dimensions; if not, then the tiles at
            the edges of the image will be smaller than the other
            tiles.  The `tileSize` parameter may be provided as a list
            of tile sizes, one for each dimension in the image.  For
            example a `tileSize` value of ``[100,100]`` would divide a
            300 X 300 image into 9 100 X 100 tiles.

            The 4 supported image compression algorithms are all
            'loss-less' when applied to integer FITS images; the pixel
            values are preserved exactly with no loss of information
            during the compression and uncompression process.  In
            addition, the HCOMPRESS algorithm supports a 'lossy'
            compression mode that will produce larger amount of image
            compression.  This is achieved by specifying a non-zero
            value for the `hcompScale` parameter.  Since the amount of
            compression that is achieved depends directly on the RMS
            noise in the image, it is usually more convenient to
            specify the `hcompScale` factor relative to the RMS noise.
            Setting `hcompScale` = 2.5 means use a scale factor that
            is 2.5 times the calculated RMS noise in the image tile.
            In some cases it may be desirable to specify the exact
            scaling to be used, instead of specifying it relative to
            the calculated noise value.  This may be done by
            specifying the negative of the desired scale value
            (typically in the range -2 to -100).

            Very high compression factors (of 100 or more) can be
            achieved by using large `hcompScale` values, however, this
            can produce undesireable 'blocky' artifacts in the
            compressed image.  A variation of the HCOMPRESS algorithm
            (called HSCOMPRESS) can be used in this case to apply a
            small amount of smoothing of the image when it is
            uncompressed to help cover up these artifacts.  This
            smoothing is purely cosmetic and does not cause any
            significant change to the image pixel values.  Setting the
            `hcompSmooth` parameter to 1 will engage the smoothing
            algorithm.

            Floating point FITS images (which have ``BITPIX`` = -32 or
            -64) usually contain too much 'noise' in the least
            significant bits of the mantissa of the pixel values to be
            effectively compressed with any lossless algorithm.
            Consequently, floating point images are first quantized
            into scaled integer pixel values (and thus throwing away
            much of the noise) before being compressed with the
            specified algorithm (either GZIP, RICE, or HCOMPRESS).
            This technique produces much higher compression factors
            than simply using the GZIP utility to externally compress
            the whole FITS file, but it also means that the original
            floating point value pixel values are not exactly
            perserved.  When done properly, this integer scaling
            technique will only discard the insignificant noise while
            still preserving all the real imformation in the image.
            The amount of precision that is retained in the pixel
            values is controlled by the `quantizeLevel` parameter.
            Larger values will result in compressed images whose
            pixels more closely match the floating point pixel values,
            but at the same time the amount of compression that is
            achieved will be reduced.  Users should experiment with
            different values for this parameter to determine the
            optimal value that preserves all the useful information in
            the image, without needlessly preserving all the 'noise'
            which will hurt the compression efficiency.

            The default value for the `quantizeLevel` scale factor is
            16, which means that scaled integer pixel values will be
            quantized such that the difference between adjacent
            integer values will be 1/16th of the noise level in the
            image background.  An optimized algorithm is used to
            accurately estimate the noise in the image.  As an
            example, if the RMS noise in the background pixels of an
            image = 32.0, then the spacing between adjacent scaled
            integer pixel values will equal 2.0 by default.  Note that
            the RMS noise is independently calculated for each tile of
            the image, so the resulting integer scaling factor may
            fluctuate slightly for each tile.  In some cases, it may
            be desireable to specify the exact quantization level to
            be used, instead of specifying it relative to the
            calculated noise value.  This may be done by specifying
            the negative of desired quantization level for the value
            of `quantizeLevel`.  In the previous example, one could
            specify `quantizeLevel`=-2.0 so that the quantized integer
            levels differ by 2.0.  Larger negative values for
            `quantizeLevel` means that the levels are more
            coarsely-spaced, and will produce higher compression
            factors.
            """
            self._file, self._datLoc = None, None

            if data is DELAYED:
                # Reading the HDU from a file
                BinTableHDU.__init__(self, data=data, header=header)
            else:
                # Create at least a skeleton HDU that matches the input
                # header and data (if any were input)
                BinTableHDU.__init__(self, data=None, header=header)

                # Store the input image data
                self.data = data

                # Update the table header (_header) to the compressed
                # image format and to match the input data (if any);
                # Create the image header (_imageHeader) from the input
                # image header (if any) and ensure it matches the input
                # data; Create the initially empty table data array to
                # hold the compressed data.
                self.updateHeaderData(header, name, compressionType,
                                      tileSize, hcompScale, hcompSmooth,
                                      quantizeLevel)

            # store any scale factors from the table header
            self._bzero = self._header.get('BZERO', 0)
            self._bscale = self._header.get('BSCALE', 1)
            self._bitpix = self._header['ZBITPIX']

            # Maintain a reference to the table header in the image header.
            # This reference will be used to update the table header whenever
            # a card in the image header is updated.
            self.header._tableHeader = self._header

        def updateHeaderData(self, imageHeader,
                             name=None,
                             compressionType=None,
                             tileSize=None,
                             hcompScale=None,
                             hcompSmooth=None,
                             quantizeLevel=None):
            """
            Update the table header (`_header`) to the compressed
            image format and to match the input data (if any).  Create
            the image header (`_imageHeader`) from the input image
            header (if any) and ensure it matches the input
            data. Create the initially-empty table data array to hold
            the compressed data.

            This method is mainly called internally, but a user may wish to
            call this method after assigning new data to the `CompImageHDU`
            object that is of a different type.

            Parameters
            ----------
            imageHeader : Header instance
                header to be associated with the image

            name : str, optional
                the ``EXTNAME`` value; if this value is `None`, then
                the name from the input image header will be used; if
                there is no name in the input image header then the
                default name 'COMPRESSED_IMAGE' is used

            compressionType : str, optional
                compression algorithm 'RICE_1', 'PLIO_1', 'GZIP_1',
                'HCOMPRESS_1'; if this value is `None`, use value
                already in the header; if no value already in the
                header, use 'RICE_1'

            tileSize : sequence of int, optional
                compression tile sizes as a list; if this value is
                `None`, use value already in the header; if no value
                already in the header, treat each row of image as a
                tile

            hcompScale : float, optional
                HCOMPRESS scale parameter; if this value is `None`,
                use the value already in the header; if no value
                already in the header, use 1

            hcompSmooth : float, optional
                HCOMPRESS smooth parameter; if this value is `None`,
                use the value already in the header; if no value
                already in the header, use 0

            quantizeLevel : float, optional
                floating point quantization level; if this value
                is `None`, use the value already in the header; if
                no value already in header, use 16
            """

            # Construct an ImageBaseHDU object using the input header
            # and data so that we can ensure that the input image header
            # matches the input image data.  Store the header from this
            # temporary HDU object as the image header for this object.

            self._imageHeader = \
              ImageHDU(data=self.data, header=imageHeader).header
            self._imageHeader._tableHeader = self._header

            # Update the extension name in the table header

            if not name and not self._header.has_key('EXTNAME'):
                name = 'COMPRESSED_IMAGE'

            if name:
                self._header.update('EXTNAME', name,
                                    'name of this binary table extension',
                                    after='TFIELDS')
                self.name = name
            else:
                self.name = self._header['EXTNAME']

            # Set the compression type in the table header.

            if compressionType:
                if compressionType not in ['RICE_1','GZIP_1','PLIO_1',
                                           'HCOMPRESS_1']:
                    warnings.warn('Warning: Unknown compression type provided.'+
                                  '  Default RICE_1 compression used.')
                    compressionType = 'RICE_1'

                self._header.update('ZCMPTYPE', compressionType,
                                    'compression algorithm',
                                    after='TFIELDS')
            else:
                compressionType = self._header.get('ZCMPTYPE', 'RICE_1')

            # If the input image header had BSCALE/BZERO cards, then insert
            # them in the table header.

            if imageHeader:
                bzero = imageHeader.get('BZERO', 0.0)
                bscale = imageHeader.get('BSCALE', 1.0)
                afterCard = 'EXTNAME'

                if bscale != 1.0:
                    self._header.update('BSCALE',bscale,after=afterCard)
                    afterCard = 'BSCALE'

                if bzero != 0.0:
                    self._header.update('BZERO',bzero,after=afterCard)

                bitpix_comment = imageHeader.ascardlist()['BITPIX'].comment
                naxis_comment =  imageHeader.ascardlist()['NAXIS'].comment
            else:
                bitpix_comment = 'data type of original image'
                naxis_comment = 'dimension of original image'

            # Set the label for the first column in the table

            self._header.update('TTYPE1', 'COMPRESSED_DATA',
                                'label for field 1', after='TFIELDS')

            # Set the data format for the first column.  It is dependent
            # on the requested compression type.

            if compressionType == 'PLIO_1':
                tform1 = '1PI'
            else:
                tform1 = '1PB'

            self._header.update('TFORM1', tform1,
                                'data format of field: variable length array',
                                after='TTYPE1')

            # Create the first column for the table.  This column holds the
            # compressed data.
            col1 = Column(name=self._header['TTYPE1'], format=tform1)

            # Create the additional columns required for floating point
            # data and calculate the width of the output table.

            if self._imageHeader['BITPIX'] < 0:
                # floating point image has 'COMPRESSED_DATA',
                # 'UNCOMPRESSED_DATA', 'ZSCALE', and 'ZZERO' columns.
                ncols = 4

                # Set up the second column for the table that will hold
                # any uncompressable data.
                self._header.update('TTYPE2', 'UNCOMPRESSED_DATA',
                                    'label for field 2', after='TFORM1')

                if self._imageHeader['BITPIX'] == -32:
                    tform2 = '1PE'
                else:
                    tform2 = '1PD'

                self._header.update('TFORM2', tform2,
                                 'data format of field: variable length array',
                                 after='TTYPE2')
                col2 = Column(name=self._header['TTYPE2'],format=tform2)

                # Set up the third column for the table that will hold
                # the scale values for quantized data.
                self._header.update('TTYPE3', 'ZSCALE',
                                    'label for field 3', after='TFORM2')
                self._header.update('TFORM3', '1D',
                                 'data format of field: 8-byte DOUBLE',
                                 after='TTYPE3')
                col3 = Column(name=self._header['TTYPE3'],
                              format=self._header['TFORM3'])

                # Set up the fourth column for the table that will hold
                # the zero values for the quantized data.
                self._header.update('TTYPE4', 'ZZERO',
                                    'label for field 4', after='TFORM3')
                self._header.update('TFORM4', '1D',
                                 'data format of field: 8-byte DOUBLE',
                                 after='TTYPE4')
                after = 'TFORM4'
                col4 = Column(name=self._header['TTYPE4'],
                              format=self._header['TFORM4'])

                # Create the ColDefs object for the table
                cols = ColDefs([col1, col2, col3, col4])
            else:
                # default table has just one 'COMPRESSED_DATA' column
                ncols = 1
                after = 'TFORM1'

                # remove any header cards for the additional columns that
                # may be left over from the previous data
                keyList = ['TTYPE2', 'TFORM2', 'TTYPE3', 'TFORM3', 'TTYPE4',
                           'TFORM4']

                for k in keyList:
                    del self._header[k]

                # Create the ColDefs object for the table
                cols = ColDefs([col1])

            # Update the table header with the width of the table, the
            # number of fields in the table, the indicator for a compressed
            # image HDU, the data type of the image data and the number of
            # dimensions in the image data array.
            self._header.update('NAXIS1', ncols*8, 'width of table in bytes')
            self._header.update('TFIELDS', ncols,
                                'number of fields in each row')
            self._header.update('ZIMAGE', True,
                                'extension contains compressed image',
                                after = after)
            self._header.update('ZBITPIX', self._imageHeader['BITPIX'],
                                bitpix_comment,
                                after = 'ZIMAGE')
            self._header.update('ZNAXIS', self._imageHeader['NAXIS'],
                                naxis_comment,
                                after = 'ZBITPIX')

            # Strip the table header of all the ZNAZISn and ZTILEn keywords
            # that may be left over from the previous data

            i = 1

            while 1:
                try:
                    del self._header.ascardlist()['ZNAXIS'+`i`]
                    del self._header.ascardlist()['ZTILE'+`i`]
                    i += 1
                except KeyError:
                    break

            # Verify that any input tile size parameter is the appropriate
            # size to match the HDU's data.

            if not tileSize:
                tileSize = []
            elif len(tileSize) != self._imageHeader['NAXIS']:
                warnings.warn('Warning: Provided tile size not appropriate ' +
                              'for the data.  Default tile size will be used.')
                tileSize = []

            # Set default tile dimensions for HCOMPRESS_1

            if compressionType == 'HCOMPRESS_1':
                if self._imageHeader['NAXIS'] < 2:
                    raise ValueError, 'Hcompress cannot be used with ' + \
                                      '1-dimensional images.'
                elif self._imageHeader['NAXIS1'] < 4 or \
                self._imageHeader['NAXIS2'] < 4:
                    raise ValueError, 'Hcompress minimum image dimension is' + \
                                      ' 4 pixels'
                elif tileSize and (tileSize[0] < 4 or tileSize[1] < 4):
                    # user specified tile size is too small
                    raise ValueError, 'Hcompress minimum tile dimension is' + \
                                      ' 4 pixels'

                if tileSize and (tileSize[0] == 0 and tileSize[1] == 0):
                    #compress the whole image as a single tile
                    tileSize[0] = self._imageHeader['NAXIS1']
                    tileSize[1] = self._imageHeader['NAXIS2']

                    for i in range(2, self._imageHeader['NAXIS']):
                        # set all higher tile dimensions = 1
                        tileSize[i] = 1
                elif not tileSize:
                    # The Hcompress algorithm is inherently 2D in nature, so
                    # the row by row tiling that is used for other compression
                    # algorithms is not appropriate.  If the image has less
                    # than 30 rows, then the entire image will be compressed
                    # as a single tile.  Otherwise the tiles will consist of
                    # 16 rows of the image.  This keeps the tiles to a
                    # reasonable size, and it also includes enough rows to
                    # allow good compression efficiency.  It the last tile of
                    # the image happens to contain less than 4 rows, then find
                    # another tile size with between 14 and 30 rows
                    # (preferably even), so that the last tile has at least
                    # 4 rows.

                    # 1st tile dimension is the row length of the image
                    tileSize.append(self._imageHeader['NAXIS1'])

                    if self._imageHeader['NAXIS2'] <= 30:
                        tileSize.append(self._imageHeader['NAXIS1'])
                    else:
                        # look for another good tile dimension
                        if self._imageHeader['NAXIS2'] % 16 == 0 or \
                        self._imageHeader['NAXIS2'] % 16 > 3:
                            tileSize.append(16)
                        elif self._imageHeader['NAXIS2'] % 24 == 0 or \
                        self._imageHeader['NAXIS2'] % 24 > 3:
                            tileSize.append(24)
                        elif self._imageHeader['NAXIS2'] % 20 == 0 or \
                        self._imageHeader['NAXIS2'] % 20 > 3:
                            tileSize.append(20)
                        elif self._imageHeader['NAXIS2'] % 30 == 0 or \
                        self._imageHeader['NAXIS2'] % 30 > 3:
                            tileSize.append(30)
                        elif self._imageHeader['NAXIS2'] % 28 == 0 or \
                        self._imageHeader['NAXIS2'] % 28 > 3:
                            tileSize.append(28)
                        elif self._imageHeader['NAXIS2'] % 26 == 0 or \
                        self._imageHeader['NAXIS2'] % 26 > 3:
                            tileSize.append(26)
                        elif self._imageHeader['NAXIS2'] % 22 == 0 or \
                        self._imageHeader['NAXIS2'] % 22 > 3:
                            tileSize.append(22)
                        elif self._imageHeader['NAXIS2'] % 18 == 0 or \
                        self._imageHeader['NAXIS2'] % 18 > 3:
                            tileSize.append(18)
                        elif self._imageHeader['NAXIS2'] % 14 == 0 or \
                        self._imageHeader['NAXIS2'] % 14 > 3:
                            tileSize.append(14)
                        else:
                            tileSize.append(17)
                # check if requested tile size causes the last tile to have
                # less than 4 pixels

                remain = self._imageHeader['NAXIS1'] % tileSize[0] # 1st dimen

                if remain > 0 and remain < 4:
                    tileSize[0] += 1 # try increasing tile size by 1

                    remain = self._imageHeader['NAXIS1'] % tileSize[0]

                    if remain > 0 and remain < 4:
                        raise ValueError, 'Last tile along 1st dimension ' + \
                                          'has less than 4 pixels'

                remain = self._imageHeader['NAXIS2'] % tileSize[1] # 2nd dimen

                if remain > 0 and remain < 4:
                    tileSize[1] += 1 # try increasing tile size by 1

                    remain = self._imageHeader['NAXIS2'] % tileSize[1]

                    if remain > 0 and remain < 4:
                        raise ValueError, 'Last tile along 2nd dimension ' + \
                                          'has less than 4 pixels'

            # Set up locations for writing the next cards in the header.
            after = 'ZNAXIS'

            if self._imageHeader['NAXIS'] > 0:
                after1 = 'ZNAXIS1'
            else:
                after1 = 'ZNAXIS'

            # Calculate the number of rows in the output table and
            # write the ZNAXISn and ZTILEn cards to the table header.
            nrows = 1

            for i in range(0, self._imageHeader['NAXIS']):
                if tileSize:
                    ts = tileSize[i]
                elif not self._header.has_key('ZTILE'+`i+1`):
                    # Default tile size
                    if not i:
                        ts = self._imageHeader['NAXIS1']
                    else:
                        ts = 1
                else:
                    ts = self._header['ZTILE'+`i+1`]

                naxisn = self._imageHeader['NAXIS'+`i+1`]
                nrows = nrows * ((naxisn - 1) // ts + 1)

                if imageHeader and imageHeader.has_key('NAXIS'+`i+1`):
                    self._header.update('ZNAXIS'+`i+1`, naxisn,
                              imageHeader.ascardlist()['NAXIS'+`i+1`].comment,
                              after=after)
                else:
                    self._header.update('ZNAXIS'+`i+1`, naxisn,
                              'length of original image axis',
                              after=after)

                self._header.update('ZTILE'+`i+1`, ts,
                                    'size of tiles to be compressed',
                                    after=after1)
                after = 'ZNAXIS'+`i+1`
                after1 = 'ZTILE'+`i+1`

            # Set the NAXIS2 header card in the table hdu to the number of
            # rows in the table.
            self._header.update('NAXIS2', nrows, 'number of rows in table')

            # Create the record array to be used for the table data.
            self.columns = cols
            self.compData = FITS_rec(rec.array(None,
                                             formats=",".join(cols._recformats),
                                             names=cols.names, shape=nrows))
            self.compData._coldefs = self.columns
            self.compData.formats = self.columns.formats

            # Set up and initialize the variable length columns.  There will
            # either be one (COMPRESSED_DATA) or two (COMPRESSED_DATA,
            # UNCOMPRESSED_DATA) depending on whether we have floating point
            # data or not.  Note: the ZSCALE and ZZERO columns are fixed
            # length columns.
            for i in range(min(2,len(cols))):
                self.columns._arrays[i] = rec.recarray.field(self.compData,i)
                rec.recarray.field(self.compData,i)[0:] = 0
                self.compData._convert[i] = _makep(self.columns._arrays[i],
                                            rec.recarray.field(self.compData,i),
                                            self.columns._recformats[i]._dtype)

            # Set the compression parameters in the table header.

            # First, setup the values to be used for the compression parameters
            # in case none were passed in.  This will be either the value
            # already in the table header for that parameter or the default
            # value.
            i = 1

            while self._header.has_key('ZNAME'+`i`):
                if self._header['ZNAME'+`i`] == 'NOISEBIT':
                    if quantizeLevel == None:
                        quantizeLevel = self._header['ZVAL'+`i`]
                if self._header['ZNAME'+`i`] == 'SCALE   ':
                    if hcompScale == None:
                        hcompScale = self._header['ZVAL'+`i`]
                if self._header['ZNAME'+`i`] == 'SMOOTH  ':
                    if hcompSmooth == None:
                        hcompSmooth = self._header['ZVAL'+`i`]
                i += 1

            if quantizeLevel == None:
                quantizeLevel = def_quantizeLevel

            if hcompScale == None:
                hcompScale = def_hcompScale

            if hcompSmooth == None:
                hcompSmooth = def_hcompScale

            # Next, strip the table header of all the ZNAMEn and ZVALn keywords
            # that may be left over from the previous data

            i = 1

            while self._header.has_key('ZNAME'+`i`):
                del self._header.ascardlist()['ZNAME'+`i`]
                del self._header.ascardlist()['ZVAL'+`i`]
                i += 1

            # Finally, put the appropriate keywords back based on the
            # compression type.

            afterCard = 'ZCMPTYPE'
            i = 1

            if compressionType == 'RICE_1':
                self._header.update('ZNAME1', 'BLOCKSIZE',
                                    'compression block size',
                                    after=afterCard)
                self._header.update('ZVAL1', def_blockSize,
                                    'pixels per block',
                                    after='ZNAME1')

                self._header.update('ZNAME2', 'BYTEPIX',
                                    'bytes per pixel (1, 2, 4, or 8)',
                                    after='ZVAL1')

                if self._header['ZBITPIX'] == 8:
                    bytepix = 1
                elif self._header['ZBITPIX'] == 16:
                    bytepix = 2
                else:
                    bytepix = def_bytePix

                self._header.update('ZVAL2', bytepix,
                                    'bytes per pixel (1, 2, 4, or 8)',
                                        after='ZNAME2')
                afterCard = 'ZVAL2'
                i = 3
            elif compressionType == 'HCOMPRESS_1':
                self._header.update('ZNAME1', 'SCALE',
                                    'HCOMPRESS scale factor',
                                    after=afterCard)
                self._header.update('ZVAL1', hcompScale,
                                    'HCOMPRESS scale factor',
                                    after='ZNAME1')
                self._header.update('ZNAME2', 'SMOOTH',
                                    'HCOMPRESS smooth option',
                                    after='ZVAL1')
                self._header.update('ZVAL2', hcompSmooth,
                                    'HCOMPRESS smooth option',
                                    after='ZNAME2')
                afterCard = 'ZVAL2'
                i = 3

            if self._imageHeader['BITPIX'] < 0:   # floating point image
                self._header.update('ZNAME'+`i`, 'NOISEBIT',
                                    'floating point quantization level',
                                    after=afterCard)
                self._header.update('ZVAL'+`i`, quantizeLevel,
                                    'floating point quantization level',
                                    after='ZNAME'+`i`)

            if imageHeader:
                # Move SIMPLE card from the image header to the
                # table header as ZSIMPLE card.

                if imageHeader.has_key('SIMPLE'):
                    self._header.update('ZSIMPLE',
                            imageHeader['SIMPLE'],
                            imageHeader.ascardlist()['SIMPLE'].comment)

                # Move EXTEND card from the image header to the
                # table header as ZEXTEND card.

                if imageHeader.has_key('EXTEND'):
                    self._header.update('ZEXTEND',
                            imageHeader['EXTEND'],
                            imageHeader.ascardlist()['EXTEND'].comment)

                # Move BLOCKED card from the image header to the
                # table header as ZBLOCKED card.

                if imageHeader.has_key('BLOCKED'):
                    self._header.update('ZBLOCKED',
                            imageHeader['BLOCKED'],
                            imageHeader.ascardlist()['BLOCKED'].comment)

                # Move XTENSION card from the image header to the
                # table header as ZTENSION card.

                # Since we only handle compressed IMAGEs, ZTENSION should
                # always be IMAGE, even if the caller has passed in a header
                # for some other type of extension.
                if imageHeader.has_key('XTENSION'):
                    self._header.update('ZTENSION',
                            'IMAGE',
                            imageHeader.ascardlist()['XTENSION'].comment)

                # Move PCOUNT and GCOUNT cards from image header to the table
                # header as ZPCOUNT and ZGCOUNT cards.

                if imageHeader.has_key('PCOUNT'):
                    self._header.update('ZPCOUNT',
                            imageHeader['PCOUNT'],
                            imageHeader.ascardlist()['PCOUNT'].comment)

                if imageHeader.has_key('GCOUNT'):
                    self._header.update('ZGCOUNT',
                            imageHeader['GCOUNT'],
                            imageHeader.ascardlist()['GCOUNT'].comment)

                # Move CHECKSUM and DATASUM cards from the image header to the
                # table header as XHECKSUM and XDATASUM cards.

                if imageHeader.has_key('CHECKSUM'):
                    self._header.update('ZHECKSUM',
                            imageHeader['CHECKSUM'],
                            imageHeader.ascardlist()['CHECKSUM'].comment)

                if imageHeader.has_key('DATASUM'):
                    self._header.update('ZDATASUM',
                            imageHeader['DATASUM'],
                            imageHeader.ascardlist()['DATASUM'].comment)
            else:
                # Move XTENSION card from the image header to the
                # table header as ZTENSION card.

                # Since we only handle compressed IMAGEs, ZTENSION should
                # always be IMAGE, even if the caller has passed in a header
                # for some other type of extension.
                if self._imageHeader.has_key('XTENSION'):
                    self._header.update('ZTENSION',
                            'IMAGE',
                            self._imageHeader.ascardlist()['XTENSION'].comment)

                # Move PCOUNT and GCOUNT cards from image header to the table
                # header as ZPCOUNT and ZGCOUNT cards.

                if self._imageHeader.has_key('PCOUNT'):
                    self._header.update('ZPCOUNT',
                            self._imageHeader['PCOUNT'],
                            self._imageHeader.ascardlist()['PCOUNT'].comment)

                if self._imageHeader.has_key('GCOUNT'):
                    self._header.update('ZGCOUNT',
                            self._imageHeader['GCOUNT'],
                            self._imageHeader.ascardlist()['GCOUNT'].comment)


            # When we have an image checksum we need to ensure that the same
            # number of blank cards exist in the table header as there were in
            # the image header.  This allows those blank cards to be carried
            # over to the image header when the hdu is uncompressed.

            if self._header.has_key('ZHECKSUM'):
                imageHeader.ascardlist().count_blanks()
                self._imageHeader.ascardlist().count_blanks()
                self._header.ascardlist().count_blanks()
                requiredBlankCount = imageHeader.ascardlist()._blanks
                imageBlankCount = self._imageHeader.ascardlist()._blanks
                tableBlankCount = self._header.ascardlist()._blanks

                for i in range(requiredBlankCount - imageBlankCount):
                    self._imageHeader.add_blank()
                    tableBlankCount = tableBlankCount + 1

                for i in range(requiredBlankCount - tableBlankCount):
                    self._header.add_blank()


        def __getattr__(self, attr):
            """
            Get an HDU attribute.
            """
            if attr == 'data':
                # The data attribute is the image data (not the table data).

                # First we will get the table data (the compressed
                # data) from the file, if there is any.
                self.compData = BinTableHDU.__getattr__(self, attr)

                # Now that we have the compressed data, we need to uncompress
                # it into the image data.
                dataList = []
                naxesList = []
                tileSizeList = []
                zvalList = []
                uncompressedDataList = []

                # Set up an array holding the integer value that represents
                # undefined pixels.  This could come from the ZBLANK column
                # from the table, or from the ZBLANK header card (if no
                # ZBLANK column (all null values are the same for each tile)),
                # or from the BLANK header card.
                if not 'ZBLANK' in self.compData.names:
                    if self._header.has_key('ZBLANK'):
                        nullDvals = np.array(self._header['ZBLANK'],
                                             dtype='int32')
                        cn_zblank = -1 # null value is a constant
                    elif self._header.has_key('BLANK'):
                        nullDvals = np.array(self._header['BLANK'],
                                             dtype='int32')
                        cn_zblank = -1 # null value is a constant
                    else:
                        cn_zblank = 0 # no null value given so don't check
                        nullDvals = np.array(0,dtype='int32')
                else:
                    cn_zblank = 1  # null value supplied as a column

                    #if sys.byteorder == 'little':
                    #    nullDvals = self.compData.field('ZBLANK').byteswap()
                    #else:
                    #    nullDvals = self.compData.field('ZBLANK')
                    nullDvals = self.compData.field('ZBLANK')

                # Set up an array holding the linear scale factor values
                # This could come from the ZSCALE column from the table, or
                # from the ZSCALE header card (if no ZSCALE column (all
                # linear scale factor values are the same for each tile)).

                if self._header.has_key('BSCALE'):
                    self._bscale = self._header['BSCALE']
                    del self._header['BSCALE']
                else:
                    self._bscale = 1.

                if not 'ZSCALE' in self.compData.names:
                    if self._header.has_key('ZSCALE'):
                        zScaleVals = np.array(self._header['ZSCALE'],
                                              dtype='float64')
                        cn_zscale = -1 # scale value is a constant
                    else:
                        cn_zscale = 0 # no scale factor given so don't scale
                        zScaleVals = np.array(1.0,dtype='float64')
                else:
                    cn_zscale = 1 # scale value supplied as a column

                    #if sys.byteorder == 'little':
                    #    zScaleVals = self.compData.field('ZSCALE').byteswap()
                    #else:
                    #    zScaleVals = self.compData.field('ZSCALE')
                    zScaleVals = self.compData.field('ZSCALE')

                # Set up an array holding the zero point offset values
                # This could come from the ZZERO column from the table, or
                # from the ZZERO header card (if no ZZERO column (all
                # zero point offset values are the same for each tile)).

                if self._header.has_key('BZERO'):
                    self._bzero = self._header['BZERO']
                    del self._header['BZERO']
                else:
                    self._bzero = 0.

                if not 'ZZERO' in self.compData.names:
                    if self._header.has_key('ZZERO'):
                        zZeroVals = np.array(self._header['ZZERO'],
                                             dtype='float64')
                        cn_zzero = -1 # zero value is a constant
                    else:
                        cn_zzero = 0 # no zero value given so don't scale
                        zZeroVals = np.array(1.0,dtype='float64')
                else:
                    cn_zzero = 1 # zero value supplied as a column

                    #if sys.byteorder == 'little':
                    #    zZeroVals = self.compData.field('ZZERO').byteswap()
                    #else:
                    #    zZeroVals = self.compData.field('ZZERO')
                    zZeroVals = self.compData.field('ZZERO')

                # Is uncompressed data supplied in a column?
                if not 'UNCOMPRESSED_DATA' in self.compData.names:
                    cn_uncompressed = 0 # no uncompressed data supplied
                else:
                    cn_uncompressed = 1 # uncompressed data supplied as column

                # Take the compressed data out of the array and put it into
                # a list as character bytes to pass to the decompression
                # routine.
                for i in range(0,len(self.compData)):
                    dataList.append(
                         self.compData[i].field('COMPRESSED_DATA').tostring())

                    # If we have a column with uncompressed data then create
                    # a list of lists of the data in the coulum.  Each
                    # underlying list contains the uncompressed data for a
                    # pixel in the tile.  There are one of these lists for
                    # each tile in the image.
                    if 'UNCOMPRESSED_DATA' in self.compData.names:
                        tileUncDataList = []

                        for j in range(0,
                             len(self.compData.field('UNCOMPRESSED_DATA')[i])):
                            tileUncDataList.append(
                             self.compData.field('UNCOMPRESSED_DATA')[i][j])

                        uncompressedDataList.append(tileUncDataList)

                # Calculate the total number of elements (pixels) in the
                # resulting image data array.  Create a list of the number
                # of pixels along each axis in the image and a list of the
                # number of pixels along each axis in the compressed tile.
                nelem = 1

                for i in range(0,self._header['ZNAXIS']):
                    naxesList.append(self._header['ZNAXIS'+`i+1`])
                    tileSizeList.append(self._header['ZTILE'+`i+1`])
                    nelem = nelem * self._header['ZNAXIS'+`i+1`]

                # Create a list for the compression parameters.  The contents
                # of the list is dependent on the compression type.

                if self._header['ZCMPTYPE'] == 'RICE_1':
                    i = 1
                    blockSize = def_blockSize
                    bytePix = def_bytePix

                    while self._header.has_key('ZNAME'+`i`):
                        if self._header['ZNAME'+`i`] == 'BLOCKSIZE':
                            blockSize = self._header['ZVAL'+`i`]
                        if self._header['ZNAME'+`i`] == 'BYTEPIX':
                            bytePix = self._header['ZVAL'+`i`]
                        i += 1

                    zvalList.append(blockSize)
                    zvalList.append(bytePix)
                elif self._header['ZCMPTYPE'] == 'HCOMPRESS_1':
                    i = 1
                    hcompSmooth = def_hcompSmooth

                    while self._header.has_key('ZNAME'+`i`):
                        if self._header['ZNAME'+`i`] == 'SMOOTH':
                            hcompSmooth = self._header['ZVAL'+`i`]
                        i += 1

                    zvalList.append(hcompSmooth)

                # Treat the NOISEBIT and SCALE parameters separately because
                # they are floats instead of integers

                quantizeLevel = def_quantizeLevel

                if self._header['ZBITPIX'] < 0:
                    i = 1

                    while self._header.has_key('ZNAME'+`i`):
                        if self._header['ZNAME'+`i`] == 'NOISEBIT':
                            quantizeLevel = self._header['ZVAL'+`i`]
                        i += 1

                hcompScale = def_hcompScale

                if self._header['ZCMPTYPE'] == 'HCOMPRESS_1':
                    i = 1

                    while self._header.has_key('ZNAME'+`i`):
                        if self._header['ZNAME'+`i`] == 'SCALE':
                            hcompScale = self._header['ZVAL'+`i`]
                        i += 1

                # Create an array to hold the decompressed data.
                naxesList.reverse()
                data = np.empty(shape=naxesList,
                           dtype=_ImageBaseHDU.NumCode[self._header['ZBITPIX']])
                naxesList.reverse()

                # Call the C decompression routine to decompress the data.
                # Note that any errors in this routine will raise an
                # exception.
                status = pyfitsComp.decompressData(dataList,
                                                 self._header['ZNAXIS'],
                                                 naxesList, tileSizeList,
                                                 zScaleVals, cn_zscale,
                                                 zZeroVals, cn_zzero,
                                                 nullDvals, cn_zblank,
                                                 uncompressedDataList,
                                                 cn_uncompressed,
                                                 quantizeLevel,
                                                 hcompScale,
                                                 zvalList,
                                                 self._header['ZCMPTYPE'],
                                                 self._header['ZBITPIX'], 1,
                                                 nelem, 0.0, data)

                # Scale the data if necessary
                if (self._bzero != 0 or self._bscale != 1):
                    if self.header['BITPIX'] == -32:
                        data = np.array(data,dtype=np.float32)
                    else:
                        data = np.array(data,dtype=np.float64)

                    if cn_zblank:
                        blanks = (data == nullDvals)

                    if self._bscale != 1:
                        np.multiply(data, self._bscale, data)
                    if self._bzero != 0:
                        data += self._bzero

                    if cn_zblank:
                        data = np.where(blanks, np.nan, data)

                self.__dict__[attr] = data

            elif attr == 'compData':
                # In order to create the compressed data we will reference the
                # image data.  Referencing the image data will cause the
                # compressed data to be read from the file.
                data = self.data
            elif attr == 'header':
                # The header attribute is the header for the image data.  It
                # is not actually stored in the object dictionary.  Instead,
                # the _imageHeader is stored.  If the _imageHeader attribute
                # has already been defined we just return it.  If not, we nust
                # create it from the table header (the _header attribute).
                if not hasattr(self, '_imageHeader'):
                    # Start with a copy of the table header.
                    self._imageHeader = self._header.copy()
                    cardList = self._imageHeader.ascardlist()

                    try:
                        # Set the extension type to IMAGE
                        cardList['XTENSION'].value = 'IMAGE'
                        cardList['XTENSION'].comment = 'extension type'
                    except KeyError:
                        pass

                    # Delete cards that are related to the table.  And move
                    # the values of those cards that relate to the image from
                    # their corresponding table cards.  These include
                    # ZBITPIX -> BITPIX, ZNAXIS -> NAXIS, and ZNAXISn -> NAXISn.
                    try:
                        del cardList['ZIMAGE']
                    except KeyError:
                        pass

                    try:
                        del cardList['ZCMPTYPE']
                    except KeyError:
                        pass

                    try:
                        del cardList['ZBITPIX']
                        _bitpix = self._header['ZBITPIX']
                        cardList['BITPIX'].value = self._header['ZBITPIX']

                        if (self._bzero != 0 or self._bscale != 1):
                            if _bitpix > 16:  # scale integers to Float64
                                cardList['BITPIX'].value = -64
                            elif _bitpix > 0:  # scale integers to Float32
                                cardList['BITPIX'].value = -32

                        cardList['BITPIX'].comment = \
                                   self._header.ascardlist()['ZBITPIX'].comment
                    except KeyError:
                        pass

                    try:
                        del cardList['ZNAXIS']
                        cardList['NAXIS'].value = self._header['ZNAXIS']
                        cardList['NAXIS'].comment = \
                                 self._header.ascardlist()['ZNAXIS'].comment

                        for i in range(cardList['NAXIS'].value):
                            del cardList['ZNAXIS'+`i+1`]
                            self._imageHeader.update('NAXIS'+`i+1`,
                              self._header['ZNAXIS'+`i+1`],
                              self._header.ascardlist()['ZNAXIS'+`i+1`].comment,
                              after='NAXIS'+`i`)
                            lastNaxisCard = 'NAXIS'+`i+1`

                        if lastNaxisCard == 'NAXIS1':
                            # There is only one axis in the image data so we
                            # need to delete the extra NAXIS2 card.
                            del cardList['NAXIS2']
                    except KeyError:
                        pass

                    try:
                        for i in range(self._header['ZNAXIS']):
                            del cardList['ZTILE'+`i+1`]

                    except KeyError:
                        pass

                    try:
                        del cardList['ZPCOUNT']
                        self._imageHeader.update('PCOUNT',
                                 self._header['ZPCOUNT'],
                                 self._header.ascardlist()['ZPCOUNT'].comment)
                    except KeyError:
                        try:
                            del cardList['PCOUNT']
                        except KeyError:
                            pass

                    try:
                        del cardList['ZGCOUNT']
                        self._imageHeader.update('GCOUNT',
                                 self._header['ZGCOUNT'],
                                 self._header.ascardlist()['ZGCOUNT'].comment)
                    except KeyError:
                        try:
                            del cardList['GCOUNT']
                        except KeyError:
                            pass

                    try:
                        del cardList['ZEXTEND']
                        self._imageHeader.update('EXTEND',
                                 self._header['ZEXTEND'],
                                 self._header.ascardlist()['ZEXTEND'].comment,
                                 after = lastNaxisCard)
                    except KeyError:
                        pass

                    try:
                        del cardList['ZBLOCKED']
                        self._imageHeader.update('BLOCKED',
                                 self._header['ZBLOCKED'],
                                 self._header.ascardlist()['ZBLOCKED'].comment)
                    except KeyError:
                        pass

                    try:
                        del cardList['TFIELDS']

                        for i in range(self._header['TFIELDS']):
                            del cardList['TFORM'+`i+1`]

                            if self._imageHeader.has_key('TTYPE'+`i+1`):
                                del cardList['TTYPE'+`i+1`]

                    except KeyError:
                        pass

                    i = 1

                    while 1:
                        try:
                            del cardList['ZNAME'+`i`]
                            del cardList['ZVAL'+`i`]
                            i += 1
                        except KeyError:
                            break

                    # delete the keywords BSCALE and BZERO

                    try:
                        del cardList['BSCALE']
                    except KeyError:
                        pass

                    try:
                        del cardList['BZERO']
                    except KeyError:
                        pass

                    # Move the ZHECKSUM and ZDATASUM cards to the image header
                    # as CHECKSUM and DATASUM
                    try:
                        del cardList['ZHECKSUM']
                        self._imageHeader.update('CHECKSUM',
                                self._header['ZHECKSUM'],
                                self._header.ascardlist()['ZHECKSUM'].comment)
                    except KeyError:
                        pass

                    try:
                        del cardList['ZDATASUM']
                        self._imageHeader.update('DATASUM',
                                self._header['ZDATASUM'],
                                self._header.ascardlist()['ZDATASUM'].comment)
                    except KeyError:
                        pass

                    try:
                        del cardList['ZSIMPLE']
                        self._imageHeader.update('SIMPLE',
                                self._header['ZSIMPLE'],
                                self._header.ascardlist()['ZSIMPLE'].comment,
                                before=1)
                        del cardList['XTENSION']
                    except KeyError:
                        pass

                    try:
                        del cardList['ZTENSION']
                        if self._header['ZTENSION'] != 'IMAGE':
                            warnings.warn("ZTENSION keyword in compressed extension != 'IMAGE'")
                        self._imageHeader.update('XTENSION',
                                'IMAGE',
                                self._header.ascardlist()['ZTENSION'].comment)
                    except KeyError:
                        pass

                    # Remove the EXTNAME card if the value in the table header
                    # is the default value of COMPRESSED_IMAGE.

                    if self._header.has_key('EXTNAME') and \
                       self._header['EXTNAME'] == 'COMPRESSED_IMAGE':
                           del cardList['EXTNAME']

                    # Look to see if there are any blank cards in the table
                    # header.  If there are, there should be the same number
                    # of blank cards in the image header.  Add blank cards to
                    # the image header to make it so.
                    self._header.ascardlist().count_blanks()
                    tableHeaderBlankCount = self._header.ascardlist()._blanks
                    self._imageHeader.ascardlist().count_blanks()
                    imageHeaderBlankCount=self._imageHeader.ascardlist()._blanks

                    for i in range(tableHeaderBlankCount-imageHeaderBlankCount):
                        self._imageHeader.add_blank()

                try:
                    return self._imageHeader
                except KeyError:
                    raise AttributeError(attr)
            else:
                # Call the base class __getattr__ method.
                return BinTableHDU.__getattr__(self,attr)

            try:
                return self.__dict__[attr]
            except KeyError:
                raise AttributeError(attr)

        def __setattr__(self, attr, value):
            """
            Set an HDU attribute.
            """
            if attr == 'data':
                if (value != None) and (not isinstance(value,np.ndarray) or
                                        value.dtype.fields != None):
                    raise TypeError, "CompImageHDU data has incorrect type"

            _ExtensionHDU.__setattr__(self,attr,value)

        def _summary(self):
            """
            Summarize the HDU: name, dimensions, and formats.
            """
            class_name  = str(self.__class__)
            type  = class_name[class_name.rfind('.')+1:-2]

            # if data is touched, use data info.

            if 'data' in dir(self):
                if self.data is None:
                    _shape, _format = (), ''
                else:

                    # the shape will be in the order of NAXIS's which is the
                    # reverse of the numarray shape
                    _shape = list(self.data.shape)
                    _format = self.data.dtype.name
                    _shape.reverse()
                    _shape = tuple(_shape)
                    _format = _format[_format.rfind('.')+1:]

            # if data is not touched yet, use header info.
            else:
                _shape = ()

                for j in range(self.header['NAXIS']):
                    _shape += (self.header['NAXIS'+`j+1`],)

                _format = _ImageBaseHDU.NumCode[self.header['BITPIX']]

            return "%-10s  %-12s  %4d  %-12s  %s" % \
               (self.name, type, len(self.header.ascard), _shape, _format)

        def updateCompressedData(self):
            """
            Compress the image data so that it may be written to a file.
            """
            naxesList = []
            tileSizeList = []
            zvalList = []

            # Check to see that the imageHeader matches the image data
            if self.header.get('NAXIS',0) != len(self.data.shape) or \
               self.header.get('BITPIX',0) != \
               _ImageBaseHDU.ImgCode[self.data.dtype.name]:
                self.updateHeaderData(self.header)

            # Create lists to hold the number of pixels along each axis of
            # the image data and the number of pixels in each tile of the
            # compressed image.
            for i in range(0,self._header['ZNAXIS']):
                naxesList.append(self._header['ZNAXIS'+`i+1`])
                tileSizeList.append(self._header['ZTILE'+`i+1`])

            # Indicate if the linear scale factor is from a column, a single
            # scale value, or not given.
            if 'ZSCALE' in self.compData.names:
                cn_zscale = 1 # there is a scaled column
            elif self._header.has_key('ZSCALE'):
                cn_zscale = -1 # scale value is a constant
            else:
                cn_zscale = 0 # no scale value given so don't scale

            # Indicate if the zero point offset value is from a column, a
            # single value, or not given.
            if 'ZZERO' in self.compData.names:
                cn_zzero = 1 # there is a scaled column
            elif self._header.has_key('ZZERO'):
                cn_zzero = -1 # zero value is a constant
            else:
                cn_zzero = 0 # no zero value given so don't scale

            # Indicate if there is a UNCOMPRESSED_DATA column in the
            # compressed data table.
            if 'UNCOMPRESSED_DATA' in self.compData.names:
                cn_uncompressed = 1 # there is a uncompressed data column
            else:
                cn_uncompressed = 0 # there is no uncompressed data column

            # Create a list for the compression parameters.  The contents
            # of the list is dependent on the compression type.
            if self._header['ZCMPTYPE'] == 'RICE_1':
                i = 1
                blockSize = def_blockSize
                bytePix = def_bytePix

                while self._header.has_key('ZNAME'+`i`):
                    if self._header['ZNAME'+`i`] == 'BLOCKSIZE':
                        blockSize = self._header['ZVAL'+`i`]
                    if self._header['ZNAME'+`i`] == 'BYTEPIX':
                        bytePix = self._header['ZVAL'+`i`]
                    i += 1

                zvalList.append(blockSize)
                zvalList.append(bytePix)
            elif self._header['ZCMPTYPE'] == 'HCOMPRESS_1':
                i = 1
                hcompSmooth = def_hcompSmooth

                while self._header.has_key('ZNAME'+`i`):
                    if self._header['ZNAME'+`i`] == 'SMOOTH':
                        hcompSmooth = self._header['ZVAL'+`i`]
                    i += 1

                zvalList.append(hcompSmooth)

            # Treat the NOISEBIT and SCALE parameters separately because
            # they are floats instead of integers

            quantizeLevel = def_quantizeLevel

            if self._header['ZBITPIX'] < 0:
                i = 1

                while self._header.has_key('ZNAME'+`i`):
                    if self._header['ZNAME'+`i`] == 'NOISEBIT':
                        quantizeLevel = self._header['ZVAL'+`i`]
                    i += 1

            hcompScale = def_hcompScale

            if self._header['ZCMPTYPE'] == 'HCOMPRESS_1':
                i = 1

                while self._header.has_key('ZNAME'+`i`):
                    if self._header['ZNAME'+`i`] == 'SCALE':
                        hcompScale = self._header['ZVAL'+`i`]
                    i += 1

            # Indicate if the null value is a constant or if no null value
            # is provided.
            if self._header.has_key('ZBLANK'):
                cn_zblank = -1 # null value is a constant
                zblank = self._header['ZBLANK']
            else:
                cn_zblank = 0 # no null value so don't use
                zblank = 0

            if self._header.has_key('BSCALE') and self.data.dtype.str[1] == 'f':
                # If this is scaled data (ie it has a BSCALE value and it is
                # floating point data) then pass in the BSCALE value so the C
                # code can unscale it before compressing.
                cn_bscale = self._header['BSCALE']
            else:
                cn_bscale = 1.0

            if self._header.has_key('BZERO') and self.data.dtype.str[1] == 'f':
                cn_bzero = self._header['BZERO']
            else:
                cn_bzero = 0.0

            # put data in machine native byteorder on little endian machines

            byteswapped = False

            if self.data.dtype.str[0] == '>' and sys.byteorder == 'little':
                byteswapped = True
                self.data = self.data.byteswap(True)
                self.data.dtype = self.data.dtype.newbyteorder('<')

            try:
                # Compress the data.
                status, compDataList, scaleList, zeroList, uncompDataList =  \
                   pyfitsComp.compressData(self.data,
                                           self._header['ZNAXIS'],
                                           naxesList, tileSizeList,
                                           cn_zblank, zblank,
                                           cn_bscale, cn_bzero, cn_zscale,
                                           cn_zzero, cn_uncompressed,
                                           quantizeLevel,
                                           hcompScale,
                                           zvalList,
                                           self._header['ZCMPTYPE'],
                                           self.header['BITPIX'], 1,
                                           self.data.size)
            finally:
                # if data was byteswapped return it to its original order

                if byteswapped:
                    self.data = self.data.byteswap(True)
                    self.data.dtype = self.data.dtype.newbyteorder('>')

            if status != 0:
                raise RuntimeError, 'Unable to write compressed image'

            # Convert the compressed data from a list of byte strings to
            # an array and set it in the COMPRESSED_DATA field of the table.
            colDType = 'uint8'

            if self._header['ZCMPTYPE'] == 'PLIO_1':
                colDType = 'i2'

            for i in range(0,len(compDataList)):
                self.compData[i].setfield('COMPRESSED_DATA',np.fromstring(
                                                            compDataList[i],
                                                            dtype=colDType))

            # Convert the linear scale factor values from a list to an
            # array and set it in the ZSCALE field of the table.
            if cn_zscale > 0:
                for i in range (0,len(scaleList)):
                    self.compData[i].setfield('ZSCALE',scaleList[i])

            # Convert the zero point offset values from a list to an
            # array and set it in the ZZERO field of the table.
            if cn_zzero > 0:
                for i in range (0,len(zeroList)):
                    self.compData[i].setfield('ZZERO',zeroList[i])

            # Convert the uncompressed data values from a list to an
            # array and set it in the UNCOMPRESSED_DATA field of the table.
            if cn_uncompressed > 0:
                for i in range(0,len(uncompDataList)):
                    self.compData[i].setfield('UNCOMPRESSED_DATA',
                                              uncompDataList[i])

            # Update the table header cards to match the compressed data.
            self.updateHeader()

        def updateHeader(self):
            """
            Update the table header cards to match the compressed data.
            """
            # Get the _heapsize attribute to match the data.
            self.compData._scale_back()

            # Check that TFIELDS and NAXIS2 match the data.
            self._header['TFIELDS'] = self.compData._nfields
            self._header['NAXIS2'] = self.compData.shape[0]

            # Calculate PCOUNT, for variable length tables.
            _tbsize = self._header['NAXIS1']*self._header['NAXIS2']
            _heapstart = self._header.get('THEAP', _tbsize)
            self.compData._gap = _heapstart - _tbsize
            _pcount = self.compData._heapsize + self.compData._gap

            if _pcount > 0:
                self._header['PCOUNT'] = _pcount

            # Update TFORM for variable length columns.
            for i in range(self.compData._nfields):
                if isinstance(self.compData._coldefs.formats[i], _FormatP):
                    key = self._header['TFORM'+`i+1`]
                    self._header['TFORM'+`i+1`] = key[:key.find('(')+1] + \
                                              `hdu.compData.field(i)._max` + ')'
            # Insure that for RICE_1 that the BLOCKSIZE and BYTEPIX cards
            # are present and set to the hard coded values used by the
            # compression algorithm.
            if self._header['ZCMPTYPE'] == 'RICE_1':
                self._header.update('ZNAME1', 'BLOCKSIZE',
                                    'compression block size',
                                    after='ZCMPTYPE')
                self._header.update('ZVAL1', def_blockSize,
                                    'pixels per block',
                                    after='ZNAME1')

                self._header.update('ZNAME2', 'BYTEPIX',
                                    'bytes per pixel (1, 2, 4, or 8)',
                                    after='ZVAL1')

                if self._header['ZBITPIX'] == 8:
                    bytepix = 1
                elif self._header['ZBITPIX'] == 16:
                    bytepix = 2
                else:
                    bytepix = def_bytePix

                self._header.update('ZVAL2', bytepix,
                                    'bytes per pixel (1, 2, 4, or 8)',
                                        after='ZNAME2')

        def scale(self, type=None, option="old", bscale=1, bzero=0):
            """
            Scale image data by using ``BSCALE`` and ``BZERO``.

            Calling this method will scale `self.data` and update the
            keywords of ``BSCALE`` and ``BZERO`` in `self._header` and
            `self._imageHeader`.  This method should only be used
            right before writing to the output file, as the data will
            be scaled and is therefore not very usable after the call.

            Parameters
            ----------

            type : str, optional
                destination data type, use a string representing a numpy
                dtype name, (e.g. ``'uint8'``, ``'int16'``, ``'float32'``
                etc.).  If is `None`, use the current data type.

            option : str, optional
                how to scale the data: if ``"old"``, use the original
                ``BSCALE`` and ``BZERO`` values when the data was
                read/created. If ``"minmax"``, use the minimum and maximum
                of the data to scale.  The option will be overwritten
                by any user-specified bscale/bzero values.

            bscale, bzero : int, optional
                user specified ``BSCALE`` and ``BZERO`` values.
            """

            if self.data is None:
                return

            # Determine the destination (numpy) data type
            if type is None:
                type = _ImageBaseHDU.NumCode[self._bitpix]
            _type = getattr(np, type)

            # Determine how to scale the data
            # bscale and bzero takes priority
            if (bscale != 1 or bzero !=0):
                _scale = bscale
                _zero = bzero
            else:
                if option == 'old':
                    _scale = self._bscale
                    _zero = self._bzero
                elif option == 'minmax':
                    if isinstance(_type, np.floating):
                        _scale = 1
                        _zero = 0
                    else:

                        # flat the shape temporarily to save memory
                        dims = self.data.shape
                        self.data.shape = self.data.size
                        min = np.minimum.reduce(self.data)
                        max = np.maximum.reduce(self.data)
                        self.data.shape = dims

                        if _type == np.uint8:  # uint8 case
                            _zero = min
                            _scale = (max - min) / (2.**8 - 1)
                        else:
                            _zero = (max + min) / 2.

                            # throw away -2^N
                            _scale = (max - min) / (2.**(8*_type.bytes) - 2)

            # Do the scaling
            if _zero != 0:
                self.data += -_zero # 0.9.6.3 to avoid out of range error for
                                    # BZERO = +32768

            if _scale != 1:
                self.data /= _scale

            if self.data.dtype.type != _type:
                self.data = np.array(np.around(self.data), dtype=_type) #0.7.7.1
            #
            # Update the BITPIX Card to match the data
            #
            self.header['BITPIX']=_ImageBaseHDU.ImgCode[self.data.dtype.name]

            #
            # Update the table header to match the scaled data
            #
            self.updateHeaderData(self.header)

            #
            # Set the BSCALE/BZERO header cards
            #
            if _zero != 0:
                self.header.update('BZERO', _zero)
            else:
                del self.header['BZERO']

            if _scale != 1:
                self.header.update('BSCALE', _scale)
            else:
                del self.header['BSCALE']

        def _calculate_datasum(self):
            """
            Calculate the value for the ``DATASUM`` card in the HDU.
            """
            if self.__dict__.has_key('data') and self.data != None:
                # We have the data to be used.
                return self._calculate_datasum_from_data(self.compData)
            else:
                # This is the case where the data has not been read from the
                # file yet.  We can handle that in a generic manner so we do
                # it in the base class.  The other possibility is that there
                # is no data at all.  This can also be handled in a gereric
                # manner.
                return super(CompImageHDU,self)._calculate_datasum()


else:
    # Compression object library failed to import so define it as an
    # empty BinTableHDU class.  This way the code will run when the object
    # library is not present.

    class CompImageHDU(BinTableHDU):
        pass


class StreamingHDU:
    """
    A class that provides the capability to stream data to a FITS file
    instead of requiring data to all be written at once.

    The following pseudocode illustrates its use::

        header = pyfits.Header()

        for all the cards you need in the header:
            header.update(key,value,comment)

        shdu = pyfits.StreamingHDU('filename.fits',header)

        for each piece of data:
            shdu.write(data)

        shdu.close()
    """
    def __init__(self, name, header):
        """
        Construct a `StreamingHDU` object given a file name and a header.

        Parameters
        ----------
        name : file path, file object, or file like object
            The file to which the header and data will be streamed.
            If opened, the file object must be opened for append
            (ab+).

        header : `Header` instance
            The header object associated with the data to be written
            to the file.

        Notes
        -----
        The file will be opened and the header appended to the end of
        the file.  If the file does not already exist, it will be
        created, and if the header represents a Primary header, it
        will be written to the beginning of the file.  If the file
        does not exist and the provided header is not a Primary
        header, a default Primary HDU will be inserted at the
        beginning of the file and the provided header will be added as
        the first extension.  If the file does already exist, but the
        provided header represents a Primary header, the header will
        be modified to an image extension header and appended to the
        end of the file.
        """

        if isinstance(name, gzip.GzipFile):
            raise TypeError, 'StreamingHDU not supported for GzipFile objects'

        self._header = header.copy()

        # handle a file object instead of a file name

        if isinstance(name, file):
           filename = name.name
        elif isinstance(name, types.StringType) or \
             isinstance(name, types.UnicodeType):
            filename = name
        else:
            filename = ''
#
#       Check if the file already exists.  If it does not, check to see
#       if we were provided with a Primary Header.  If not we will need
#       to prepend a default PrimaryHDU to the file before writing the
#       given header.
#
        newFile = False

        if filename:
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                newFile = True
        elif (hasattr(name,'len') and name.len == 0):
            newFile = True

        if newFile:
            if not self._header.has_key('SIMPLE'):
                hdulist = HDUList([PrimaryHDU()])
                hdulist.writeto(name, 'exception')
        else:
            if self._header.has_key('SIMPLE'):
#
#               This will not be the first extension in the file so we
#               must change the Primary header provided into an image
#               extension header.
#
                self._header.update('XTENSION','IMAGE','Image extension',
                                   after='SIMPLE')
                del self._header['SIMPLE']

                if not self._header.has_key('PCOUNT'):
                    dim = self._header['NAXIS']

                    if dim == 0:
                        dim = ''
                    else:
                        dim = str(dim)

                    self._header.update('PCOUNT', 0, 'number of parameters',                                            after='NAXIS'+dim)

                if not self._header.has_key('GCOUNT'):
                    self._header.update('GCOUNT', 1, 'number of groups',                                                after='PCOUNT')

        self._ffo = _File(name, 'append')
        self._ffo.getfile().seek(0,2)

        self._hdrLoc = self._ffo.writeHDUheader(self)
        self._datLoc = self._ffo.getfile().tell()
        self._size = self.size()

        if self._size != 0:
            self.writeComplete = 0
        else:
            self.writeComplete = 1

    def write(self,data):
        """
        Write the given data to the stream.

        Parameters
        ----------
        data : ndarray
            Data to stream to the file.

        Returns
        -------
        writeComplete : int
            Flag that when `True` indicates that all of the required
            data has been written to the stream.

        Notes
        -----
        Only the amount of data specified in the header provided to
        the class constructor may be written to the stream.  If the
        provided data would cause the stream to overflow, an `IOError`
        exception is raised and the data is not written.  Once
        sufficient data has been written to the stream to satisfy the
        amount specified in the header, the stream is padded to fill a
        complete FITS block and no more data will be accepted.  An
        attempt to write more data after the stream has been filled
        will raise an `IOError` exception.  If the dtype of the input
        data does not match what is expected by the header, a
        `TypeError` exception is raised.
        """

        curDataSize = self._ffo.getfile().tell() - self._datLoc

        if self.writeComplete or curDataSize + data.nbytes > self._size:
            raise IOError, \
            "Attempt to write more data to the stream than the header specified"

        if _ImageBaseHDU.NumCode[self._header['BITPIX']] != data.dtype.name:
            raise TypeError, \
            "Supplied data does not match the type specified in the header."

        if data.dtype.str[0] != '>':
#
#           byteswap little endian arrays before writing
#
            output = data.byteswap()
        else:
            output = data

        _tofile(output, self._ffo.getfile())

        if self._ffo.getfile().tell() - self._datLoc == self._size:
#
#           the stream is full so pad the data to the next FITS block
#
            self._ffo.getfile().write(_padLength(self._size)*'\0')
            self.writeComplete = 1

        self._ffo.getfile().flush()

        return self.writeComplete

    def size(self):
        """
        Return the size (in bytes) of the data portion of the HDU.
        """

        size = 0
        naxis = self._header.get('NAXIS', 0)

        if naxis > 0:
            simple = self._header.get('SIMPLE','F')
            randomGroups = self._header.get('GROUPS','F')

            if simple == 'T' and randomGroups == 'T':
                groups = 1
            else:
                groups = 0

            size = 1

            for j in range(groups,naxis):
                size = size * self._header['NAXIS'+`j+1`]
            bitpix = self._header['BITPIX']
            gcount = self._header.get('GCOUNT', 1)
            pcount = self._header.get('PCOUNT', 0)
            size = abs(bitpix) * gcount * (pcount + size) // 8
        return size

    def close(self):
        """
        Close the physical FITS file.
        """

        self._ffo.close()

    # Support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

class ErrorURLopener(urllib.FancyURLopener):
    """
    A class to use with `urlretrieve` to allow `IOError` exceptions to
    be raised when a file specified by a URL cannot be accessed.
    """
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise IOError, (errcode, errmsg, url)

urllib._urlopener = ErrorURLopener() # Assign the locally subclassed opener
                                     # class to the urllibrary
urllib._urlopener.tempcache = {} # Initialize tempcache with an empty
                                 # dictionary to enable file cacheing

class _File:
    """
    A file I/O class.
    """
    def __init__(self, name, mode='copyonwrite', memmap=0, **parms):
        if mode not in _python_mode.keys():
            raise ValueError, "Mode '%s' not recognized" % mode


        if isinstance(name, file):
            self.name = name.name
        elif isinstance(name, types.StringType) or \
             isinstance(name, types.UnicodeType):
            if mode != 'append' and not os.path.exists(name) and \
            not os.path.splitdrive(name)[0]:
                #
                # Not writing file and file does not exist on local machine and
                # name does not begin with a drive letter (Windows), try to
                # get it over the web.
                #
                try:
                    self.name, fileheader = urllib.urlretrieve(name)
                except IOError, e:
                    raise e
            else:
                self.name = name
        else:
            if hasattr(name, 'name'):
                self.name = name.name
            elif hasattr(name, 'filename'):
                self.name = name.filename
            elif hasattr(name, '__class__'):
                self.name = str(name.__class__)
            else:
                self.name = str(type(name))

        self.mode = mode
        self.memmap = memmap
        self.code = None
        self.dims = None
        self.offset = 0

        if parms.has_key('ignore_missing_end'):
            self.ignore_missing_end = parms['ignore_missing_end']
        else:
            self.ignore_missing_end = 0

        self.uint = parms.get('uint16', False) or parms.get('uint', False)

        if memmap and mode not in ['readonly', 'copyonwrite', 'update']:
            raise NotImplementedError(
                   "Memory mapping is not implemented for mode `%s`." % mode)
        else:
            if isinstance(name, file) or isinstance(name, gzip.GzipFile):
                if hasattr(name, 'closed'):
                    closed = name.closed
                    foMode = name.mode
                else:
                    if name.fileobj != None:
                        closed = name.fileobj.closed
                        foMode = name.fileobj.mode
                    else:
                        closed = True
                        foMode = _python_mode[mode]

                if not closed:
                    if _python_mode[mode] != foMode:
                        raise ValueError, "Input mode '%s' (%s) " \
                              % (mode, _python_mode[mode]) + \
                              "does not match mode of the input file (%s)." \
                              % name.mode
                    self.__file = name
                elif isinstance(name, file):
                    self.__file=__builtin__.open(self.name, _python_mode[mode])
                else:
                    self.__file=gzip.open(self.name, _python_mode[mode])
            elif isinstance(name, types.StringType) or \
                 isinstance(name, types.UnicodeType):
                if os.path.splitext(self.name)[1] == '.gz':
                    # Handle gzip files
                    if mode in ['update', 'append']:
                        raise NotImplementedError(
                              "Writing to gzipped fits files is not supported")
                    zfile = gzip.GzipFile(self.name)
                    self.tfile = tempfile.NamedTemporaryFile('rb+',-1,'.fits')
                    self.name = self.tfile.name
                    self.__file = self.tfile.file
                    self.__file.write(zfile.read())
                    zfile.close()
                elif os.path.splitext(self.name)[1] == '.zip':
                    # Handle zip files
                    if mode in ['update', 'append']:
                        raise NotImplementedError(
                              "Writing to zipped fits files is not supported")
                    zfile = zipfile.ZipFile(self.name)
                    namelist = zfile.namelist()
                    if len(namelist) != 1:
                        raise NotImplementedError(
                          "Zip files with multiple members are not supported.")
                    self.tfile = tempfile.NamedTemporaryFile('rb+',-1,'.fits')
                    self.name = self.tfile.name
                    self.__file = self.tfile.file
                    self.__file.write(zfile.read(namelist[0]))
                    zfile.close()
                else:
                    self.__file=__builtin__.open(self.name, _python_mode[mode])
            else:
                # We are dealing with a file like object.
                # Assume it is open.
                self.__file = name

                # If there is not seek or tell methods then set the mode to
                # output streaming.
                if not hasattr(self.__file, 'seek') or \
                   not hasattr(self.__file, 'tell'):
                    self.mode = mode = 'ostream';

            # For 'ab+' mode, the pointer is at the end after the open in
            # Linux, but is at the beginning in Solaris.

            if mode == 'ostream':
                # For output stream start with a truncated file.
                self._size = 0
            elif isinstance(self.__file,gzip.GzipFile):
                self.__file.fileobj.seek(0,2)
                self._size = self.__file.fileobj.tell()
                self.__file.fileobj.seek(0)
                self.__file.seek(0)
            elif hasattr(self.__file, 'seek'):
                self.__file.seek(0, 2)
                self._size = self.__file.tell()
                self.__file.seek(0)
            else:
                self._size = 0

    def __getattr__(self, attr):
        """
        Get the `_mm` attribute.
        """
        if attr == '_mm':
            return Memmap(self.name,offset=self.offset,mode=_memmap_mode[self.mode],dtype=self.code,shape=self.dims)
        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError(attr)

    def getfile(self):
        return self.__file

    def _readheader(self, cardList, keyList, blocks):
        """Read blocks of header, and put each card into a list of cards.
           Will deal with CONTINUE cards in a later stage as CONTINUE cards
           may span across blocks.
        """
        if len(block) != _blockLen:
            raise IOError, 'Block length is not %d: %d' % (_blockLen, len(block))
        elif (blocks[:8] not in ['SIMPLE  ', 'XTENSION']):
            raise IOError, 'Block does not begin with SIMPLE or XTENSION'

        for i in range(0, len(_blockLen), Card.length):
            _card = Card('').fromstring(block[i:i+Card.length])
            _key = _card.key

            cardList.append(_card)
            keyList.append(_key)
            if _key == 'END':
                break

    def _readHDU(self):
        """
        Read the skeleton structure of the HDU.
        """
        if not hasattr(self.__file, 'tell') or not hasattr(self.__file, 'read'):
            raise EOFError

        end_RE = re.compile('END'+' '*77)
        _hdrLoc = self.__file.tell()

        # Read the first header block.
        block = self.__file.read(_blockLen)
        if block == '':
            raise EOFError

        hdu = _TempHDU()
        hdu._raw = ''

        # continue reading header blocks until END card is reached
        while 1:

            # find the END card
            mo = end_RE.search(block)
            if mo is None:
                hdu._raw += block
                block = self.__file.read(_blockLen)
                if block == '':
                    break
            else:
                break
        hdu._raw += block

        if not end_RE.search(block) and not self.ignore_missing_end:
            raise IOError, "Header missing END card."

        _size, hdu.name = hdu._getsize(hdu._raw)

        # get extname and extver
        if hdu.name == '':
            hdu.name, hdu._extver = hdu._getname()
        elif hdu.name == 'PRIMARY':
            hdu._extver = 1

        hdu._file = self.__file
        hdu._hdrLoc = _hdrLoc                # beginning of the header area
        hdu._datLoc = self.__file.tell()     # beginning of the data area

        # data area size, including padding
        hdu._datSpan = _size + _padLength(_size)
        hdu._new = 0
        hdu._ffile = self
        if isinstance(hdu._file, gzip.GzipFile):
            pos = self.__file.tell()
            self.__file.seek(pos+hdu._datSpan)
        else:
            self.__file.seek(hdu._datSpan, 1)

            if self.__file.tell() > self._size:
                warnings.warn('Warning: File may have been truncated: actual file length (%i) is smaller than the expected size (%i)'  % (self._size, self.__file.tell()))

        return hdu

    def writeHDU(self, hdu, checksum=False):
        """
        Write *one* FITS HDU.  Must seek to the correct location
        before calling this method.
        """

        if isinstance(hdu, _ImageBaseHDU):
            hdu.update_header()
        elif isinstance(hdu, CompImageHDU):
            hdu.updateCompressedData()
        return (self.writeHDUheader(hdu,checksum),) + self.writeHDUdata(hdu)

    def writeHDUheader(self, hdu, checksum=False):
        """
        Write FITS HDU header part.
        """
        # If the data is unsigned int 16, 32, or 64 add BSCALE/BZERO
        # cards to header

        if 'data' in dir(hdu) and hdu.data is not None \
        and not isinstance(hdu, _NonstandardHDU) \
        and not isinstance(hdu, _NonstandardExtHDU) \
        and _is_pseudo_unsigned(hdu.data.dtype):
            hdu._header.update(
                'BSCALE', 1,
                after='NAXIS'+`hdu.header.get('NAXIS')`)
            hdu._header.update(
                'BZERO', _unsigned_zero(hdu.data.dtype),
                after='BSCALE')

        # Handle checksum
        if hdu._header.has_key('CHECKSUM'):
            del hdu.header['CHECKSUM']

        if hdu._header.has_key('DATASUM'):
            del hdu.header['DATASUM']

        if checksum == 'datasum':
            hdu.add_datasum()
        elif checksum == 'test':
            hdu.add_datasum(hdu._datasum_comment)
            hdu.add_checksum(hdu._checksum_comment,True)
        elif checksum:
            hdu.add_checksum()

        blocks = repr(hdu._header.ascard) + _pad('END')
        blocks = blocks + _padLength(len(blocks))*' '

        if len(blocks)%_blockLen != 0:
            raise IOError

        if hasattr(self.__file, 'flush'):
            self.__file.flush()

        try:
           if self.__file.mode == 'ab+':
               self.__file.seek(0,2)
        except AttributeError:
           pass

        try:
            loc = self.__file.tell()
        except (AttributeError, IOError):
            loc = 0

        self.__file.write(blocks)

        # flush, to make sure the content is written
        if hasattr(self.__file, 'flush'):
            self.__file.flush()

        # If data is unsigned integer 16, 32 or 64, remove the
        # BSCALE/BZERO cards
        if 'data' in dir(hdu) and hdu.data is not None \
        and not isinstance(hdu, _NonstandardHDU) \
        and not isinstance(hdu, _NonstandardExtHDU) \
        and _is_pseudo_unsigned(hdu.data.dtype):
            del hdu._header['BSCALE']
            del hdu._header['BZERO']

        return loc

    def writeHDUdata(self, hdu):
        """
        Write FITS HDU data part.
        """
        if hasattr(self.__file, 'flush'):
            self.__file.flush()

        try:
            loc = self.__file.tell()
        except (AttributeError, IOError):
            loc = 0

        _size = 0
        if isinstance(hdu, _NonstandardHDU) and hdu.data is not None:
            self.__file.write(hdu.data)

            # flush, to make sure the content is written
            self.__file.flush()

            # return both the location and the size of the data area
            return loc, len(hdu.data)
        elif isinstance(hdu, _NonstandardExtHDU) and hdu.data is not None:
            self.__file.write(hdu.data)
            _size = len(hdu.data)

            # pad the fits data block
            self.__file.write(_padLength(_size)*'\0')

            # flush, to make sure the content is written
            self.__file.flush()

            # return both the location and the size of the data area
            return loc, _size+_padLength(_size)
        elif hdu.data is not None:
            # Based on the system type, determine the byteorders that
            # would need to be swapped to get to big-endian output
            if sys.byteorder == 'little':
                swap_types = ('<', '=')
            else:
                swap_types = ('<',)

            # if image, need to deal with byte order
            if isinstance(hdu, _ImageBaseHDU):
                # deal with unsigned integer 16, 32 and 64 data
                if _is_pseudo_unsigned(hdu.data.dtype):
                    # Convert the unsigned array to signed
                    output = np.array(
                        hdu.data - _unsigned_zero(hdu.data.dtype),
                        dtype='>i%d' % hdu.data.dtype.itemsize)
                    should_swap = False
                else:
                    output = hdu.data

                    if isinstance(hdu.data, GroupData):
                        byteorder = \
                            output.dtype.fields[hdu.data.dtype.names[0]][0].str[0]
                    else:
                        byteorder = output.dtype.str[0]
                    should_swap = (byteorder in swap_types)

                if should_swap:
                    # If we need to do byteswapping, do it in chunks
                    # so the original array is not touched
                    # output_dtype = output.dtype.newbyteorder('>')
                    # for chunk in _chunk_array(output):
                    #     chunk = np.array(chunk, dtype=output_dtype, copy=True)
                    #     _tofile(output, self.__file)

                    output.byteswap(True)
                    try:
                        _tofile(output, self.__file)
                    finally:
                        output.byteswap(True)
                else:
                    _tofile(output, self.__file)

            # Binary table byteswap
            elif isinstance(hdu, BinTableHDU):
                if isinstance(hdu, CompImageHDU):
                    output = hdu.compData
                else:
                    output = hdu.data

                swapped = []
                try:
                    for i in range(output._nfields):
                        coldata = output.field(i)
                        if not isinstance(coldata, chararray.chararray):
                            # only swap unswapped
                            # deal with var length table
                            if isinstance(coldata, _VLF):
                                k = 0
                                for j in coldata:
                                    if (not isinstance(j, chararray.chararray) and
                                        j.itemsize > 1 and
                                        j.dtype.str[0] in swap_types):
                                        j.byteswap(True)
                                        swapped.append(j)
                                    if (rec.recarray.field(output,i)[k:k+1].dtype.str[0] in
                                        swap_types):
                                        rec.recarray.field(output,i)[k:k+1].byteswap(True)
                                        swapped.append(rec.recarray.field(output,i)[k:k+1])
                                    k = k + 1
                            else:
                                if (coldata.itemsize > 1 and
                                    coldata.dtype.str[0] in swap_types):
                                    rec.recarray.field(output, i).byteswap(True)
                                    swapped.append(rec.recarray.field(output, i))

                    _tofile(output, self.__file)

                    # write out the heap of variable length array columns
                    # this has to be done after the "regular" data is written
                    # (above)
                    nbytes = output._gap
                    self.__file.write(output._gap*'\0')

                    for i in range(output._nfields):
                        if isinstance(output._coldefs._recformats[i], _FormatP):
                            for j in range(len(output.field(i))):
                                coldata = output.field(i)[j]
                                if len(coldata) > 0:
                                    nbytes= nbytes + coldata.nbytes
                                    coldata.tofile(self.__file)

                    output._heapsize = nbytes - output._gap
                    _size = _size + nbytes
                finally:
                    for obj in swapped:
                        obj.byteswap(True)
            else:
                output = hdu.data
                _tofile(output, self.__file)

            _size = _size + output.size * output.itemsize

            # pad the FITS data block
            if _size > 0:
                if isinstance(hdu, TableHDU):
                    self.__file.write(_padLength(_size)*' ')
                else:
                    self.__file.write(_padLength(_size)*'\0')

        # flush, to make sure the content is written
        if hasattr(self.__file, 'flush'):
            self.__file.flush()

        # return both the location and the size of the data area
        return loc, _size+_padLength(_size)

    def close(self):
        """
        Close the 'physical' FITS file.
        """
        if hasattr(self.__file, 'close'):
            self.__file.close()

        if hasattr(self, 'tfile'):
            del self.tfile

    # Support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class HDUList(list, _Verify):
    """
    HDU list class.  This is the top-level FITS object.  When a FITS
    file is opened, a `HDUList` object is returned.
    """

    def __init__(self, hdus=[], file=None):
        """
        Construct a `HDUList` object.

        Parameters
        ----------
        hdus : sequence of HDU objects or single HDU, optional
            The HDU object(s) to comprise the `HDUList`.  Should be
            instances of `_AllHDU`.

        file : file object, optional
            The opened physical file associated with the `HDUList`.
        """
        self.__file = file
        if hdus is None:
            hdus = []

        # can take one HDU, as well as a list of HDU's as input
        if isinstance(hdus, _ValidHDU):
            hdus = [hdus]
        elif not isinstance(hdus, (HDUList, list)):
            raise TypeError, "Invalid input for HDUList."

        for hdu in hdus:
            if not isinstance(hdu, _AllHDU):
                raise TypeError(
                      "Element %d in the HDUList input is not an HDU."
                      % hdus.index(hdu))
        list.__init__(self, hdus)

    def __iter__(self):
        return [self[i] for i in range(len(self))].__iter__()

    def __getitem__(self, key, classExtensions={}):
        """
        Get an HDU from the `HDUList`, indexed by number or name.
        """
        key = self.index_of(key)
        _item = super(HDUList, self).__getitem__(key)
        if isinstance(_item, _TempHDU):
            super(HDUList, self).__setitem__(key,
                                             _item.setupHDU(classExtensions))

        return super(HDUList, self).__getitem__(key)

    def __getslice__(self, start, end):
        _hdus = super(HDUList, self).__getslice__(start,end)
        result = HDUList(_hdus)
        return result

    def __setitem__(self, key, hdu):
        """
        Set an HDU to the `HDUList`, indexed by number or name.
        """
        _key = self.index_of(key)
        if isinstance(hdu, (slice, list)):
            if isinstance(_key, (int,np.integer)):
                raise ValueError, "An element in the HDUList must be an HDU."
            for item in hdu:
                if not isinstance(item, _AllHDU):
                    raise ValueError, "%s is not an HDU." % item
        else:
            if not isinstance(hdu, _AllHDU):
                raise ValueError, "%s is not an HDU." % hdu

        try:
            super(HDUList, self).__setitem__(_key, hdu)
        except IndexError:
            raise IndexError, 'Extension %s is out of bound or not found.' % key
        self._resize = 1
        self._truncate = 0

    def __delitem__(self, key):
        """
        Delete an HDU from the `HDUList`, indexed by number or name.
        """
        key = self.index_of(key)

        endIndex = len(self)-1
        super(HDUList, self).__delitem__(key)

        if ( key == endIndex or key == -1 and not self._resize):
            self._truncate = 1
        else:
            self._truncate = 0
            self._resize = 1

    def __delslice__(self, i, j):
        """
        Delete a slice of HDUs from the `HDUList`, indexed by number only.
        """
        endIndex = len(self)
        super(HDUList, self).__delslice__(i, j)

        if ( j == endIndex or j == sys.maxint and not self._resize):
            self._truncate = 1
        else:
            self._truncate = 0
            self._resize = 1


    def _verify (self, option='warn'):
        _text = ''
        _err = _ErrList([], unit='HDU')

        # the first (0th) element must be a primary HDU
        if len(self) > 0 and (not isinstance(self[0], PrimaryHDU)) and \
                             (not isinstance(self[0], _NonstandardHDU)):
            err_text = "HDUList's 0th element is not a primary HDU."
            fix_text = 'Fixed by inserting one as 0th HDU.'
            fix = "self.insert(0, PrimaryHDU())"
            _text = self.run_option(option, err_text=err_text, fix_text=fix_text, fix=fix)
            _err.append(_text)

        # each element calls their own verify
        for i in range(len(self)):
            if i > 0 and (not isinstance(self[i], _ExtensionHDU)):
                err_text = "HDUList's element %s is not an extension HDU." % `i`
                _text = self.run_option(option, err_text=err_text, fixable=0)
                _err.append(_text)

            else:
                _result = self[i]._verify(option)
                if _result:
                    _err.append(_result)
        return _err

    def insert(self, index, hdu, classExtensions={}):
        """
        Insert an HDU into the `HDUList` at the given `index`.

        Parameters
        ----------
        index : int
            Index before which to insert the new HDU.

        hdu : _AllHDU instance
            The HDU object to insert

        classExtensions : dict
            A dictionary that maps pyfits classes to extensions of those
            classes.  When present in the dictionary, the extension class
            will be constructed in place of the pyfits class.
        """
        if isinstance(hdu, _AllHDU):
            num_hdus = len(self)

            if index == 0 or num_hdus == 0:
                if num_hdus != 0:
                    # We are inserting a new Primary HDU so we need to 
                    # make the current Primary HDU into an extension HDU.
                    if isinstance(self[0], GroupsHDU):
                       raise ValueError, \
                             "The current Primary HDU is a GroupsHDU.  " + \
                             "It can't be made into an extension HDU," + \
                             " so you can't insert another HDU in front of it."

                    if classExtensions.has_key(ImageHDU):
                        hdu1 = classExtensions[ImageHDU](self[0].data,
                                                         self[0].header)
                    else:
                        hdu1= ImageHDU(self[0].data, self[0].header)

                    # Insert it into position 1, then delete HDU at position 0.
                    super(HDUList, self).insert(1,hdu1)
                    super(HDUList, self).__delitem__(0)

                if not isinstance(hdu, PrimaryHDU):
                    # You passed in an Extension HDU but we need a Primary HDU.
                    # If you provided an ImageHDU then we can convert it to
                    # a primary HDU and use that.
                    if isinstance(hdu, ImageHDU):
                        if classExtensions.has_key(PrimaryHDU):
                            hdu = classExtensions[PrimaryHDU](hdu.data,
                                                              hdu.header)
                        else:
                            hdu = PrimaryHDU(hdu.data, hdu.header)
                    else:
                        # You didn't provide an ImageHDU so we create a
                        # simple Primary HDU and append that first before
                        # we append the new Extension HDU.
                        if classExtensions.has_key(PrimaryHDU):
                            phdu = classExtensions[PrimaryHDU]()
                        else:
                            phdu = PrimaryHDU()

                        super(HDUList, self).insert(0,phdu)
                        index = 1
            else:
                if isinstance(hdu, GroupsHDU):
                   raise ValueError, \
                         "A GroupsHDU must be inserted as a Primary HDU"

                if isinstance(hdu, PrimaryHDU):
                    # You passed a Primary HDU but we need an Extension HDU
                    # so create an Extension HDU from the input Primary HDU.
                    if classExtensions.has_key(ImageHDU):
                        hdu = classExtensions[ImageHDU](hdu.data,hdu.header)
                    else:
                        hdu = ImageHDU(hdu.data, hdu.header)

            super(HDUList, self).insert(index,hdu)
            self._resize = 1
            self._truncate = 0
        else:
            raise ValueError, "%s is not an HDU." % hdu

        # make sure the EXTEND keyword is in primary HDU if there is extension
        if len(self) > 1:
            self.update_extend()

    def append(self, hdu, classExtensions={}):
        """
        Append a new HDU to the `HDUList`.

        Parameters
        ----------
        hdu : instance of _AllHDU
            HDU to add to the `HDUList`.

        classExtensions : dict
            A dictionary that maps pyfits classes to extensions of those
            classes.  When present in the dictionary, the extension class
            will be constructed in place of the pyfits class.
        """
        if isinstance(hdu, _AllHDU):
            if not isinstance(hdu, _TempHDU):
                if len(self) > 0:
                    if isinstance(hdu, GroupsHDU):
                       raise ValueError, \
                             "Can't append a GroupsHDU to a non-empty HDUList"

                    if isinstance(hdu, PrimaryHDU):
                        # You passed a Primary HDU but we need an Extension HDU
                        # so create an Extension HDU from the input Primary HDU.
                        if classExtensions.has_key(ImageHDU):
                            hdu = classExtensions[ImageHDU](hdu.data,hdu.header)
                        else:
                            hdu = ImageHDU(hdu.data, hdu.header)
                else:
                    if not isinstance(hdu, PrimaryHDU):
                        # You passed in an Extension HDU but we need a Primary
                        # HDU.
                        # If you provided an ImageHDU then we can convert it to
                        # a primary HDU and use that.
                        if isinstance(hdu, ImageHDU):
                            if classExtensions.has_key(PrimaryHDU):
                                hdu = classExtensions[PrimaryHDU](hdu.data,
                                                                  hdu.header)
                            else:
                                hdu = PrimaryHDU(hdu.data, hdu.header)
                        else:
                            # You didn't provide an ImageHDU so we create a
                            # simple Primary HDU and append that first before
                            # we append the new Extension HDU.
                            if classExtensions.has_key(PrimaryHDU):
                                phdu = classExtensions[PrimaryHDU]()
                            else:
                                phdu = PrimaryHDU()

                            super(HDUList, self).append(phdu)

            super(HDUList, self).append(hdu)
            hdu._new = 1
            self._resize = 1
            self._truncate = 0
        else:
            raise ValueError, "HDUList can only append an HDU"

        # make sure the EXTEND keyword is in primary HDU if there is extension
        if len(self) > 1:
            self.update_extend()

    def index_of(self, key):
        """
        Get the index of an HDU from the `HDUList`.

        Parameters
        ----------
        key : int, str or tuple of (string, int)
           The key identifying the HDU.  If `key` is a tuple, it is of
           the form (`key`, `ver`) where `ver` is an ``EXTVER`` value
           that must match the HDU being searched for.

        Returns
        -------
        index : int
           The index of the HDU in the `HDUList`.
        """

        if isinstance(key, (int, np.integer,slice)):
            return key
        elif isinstance(key, tuple):
            _key = key[0]
            _ver = key[1]
        else:
            _key = key
            _ver = None

        if not isinstance(_key, str):
            raise KeyError, key
        _key = (_key.strip()).upper()

        nfound = 0
        for j in range(len(self)):
            _name = self[j].name
            if isinstance(_name, str):
                _name = _name.strip().upper()
            if _name == _key:

                # if only specify extname, can only have one extension with
                # that name
                if _ver == None:
                    found = j
                    nfound += 1
                else:

                    # if the keyword EXTVER does not exist, default it to 1
                    _extver = self[j]._extver
                    if _ver == _extver:
                        found = j
                        nfound += 1

        if (nfound == 0):
            raise KeyError, 'extension %s not found' % `key`
        elif (nfound > 1):
            raise KeyError, 'there are %d extensions of %s' % (nfound, `key`)
        else:
            return found

    def readall(self):
        """
        Read data of all HDUs into memory.
        """
        for i in range(len(self)):
            if self[i].data is not None:
                continue

    def update_tbhdu(self):
        """
        Update all table HDU's for scaled fields.
        """
        for hdu in self:
            if 'data' in dir(hdu) and not isinstance(hdu, CompImageHDU):
                if isinstance(hdu, (GroupsHDU, _TableBaseHDU)) and hdu.data is not None:
                    hdu.data._scale_back()
                if isinstance(hdu, _TableBaseHDU) and hdu.data is not None:

                    # check TFIELDS and NAXIS2
                    hdu.header['TFIELDS'] = hdu.data._nfields
                    hdu.header['NAXIS2'] = hdu.data.shape[0]

                    # calculate PCOUNT, for variable length tables
                    _tbsize = hdu.header['NAXIS1']*hdu.header['NAXIS2']
                    _heapstart = hdu.header.get('THEAP', _tbsize)
                    hdu.data._gap = _heapstart - _tbsize
                    _pcount = hdu.data._heapsize + hdu.data._gap
                    if _pcount > 0:
                        hdu.header['PCOUNT'] = _pcount

                    # update TFORM for variable length columns
                    for i in range(hdu.data._nfields):
                        if isinstance(hdu.data._coldefs.formats[i], _FormatP):
                            key = hdu.header['TFORM'+`i+1`]
                            hdu.header['TFORM'+`i+1`] = key[:key.find('(')+1] + `hdu.data.field(i)._max` + ')'


    def flush(self, output_verify='exception', verbose=False, classExtensions={}):
        """
        Force a write of the `HDUList` back to the file (for append and
        update modes only).

        Parameters
        ----------
        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.

        verbose : bool
            When `True`, print verbose messages

        classExtensions : dict
            A dictionary that maps pyfits classes to extensions of
            those classes.  When present in the dictionary, the
            extension class will be constructed in place of the pyfits
            class.
        """

        # Get the name of the current thread and determine if this is a single treaded application
        threadName = threading.currentThread()
        singleThread = (threading.activeCount() == 1) and (threadName.getName() == 'MainThread')

        # Define new signal interput handler
        if singleThread:
            keyboardInterruptSent = False
            def New_SIGINT(*args):
                warnings.warn("KeyboardInterrupt ignored until flush is complete!")
                keyboardInterruptSent = True

            # Install new handler
            old_handler = signal.signal(signal.SIGINT,New_SIGINT)

        if self.__file.mode not in ('append', 'update', 'ostream'):
            warnings.warn("flush for '%s' mode is not supported." % self.__file.mode)
            return

        self.update_tbhdu()
        self.verify(option=output_verify)

        if self.__file.mode in ('append', 'ostream'):
            for hdu in self:
                if (verbose):
                    try: _extver = `hdu.header['extver']`
                    except: _extver = ''

                # only append HDU's which are "new"
                if not hasattr(hdu, '_new') or hdu._new:
                    # only output the checksum if flagged to do so
                    if hasattr(hdu, '_output_checksum'):
                        checksum = hdu._output_checksum
                    else:
                        checksum = False

                    self.__file.writeHDU(hdu, checksum=checksum)
                    if (verbose):
                        print "append HDU", hdu.name, _extver
                    hdu._new = 0

        elif self.__file.mode == 'update':
            if not self._resize:

                # determine if any of the HDU is resized
                for hdu in self:

                    # Header:
                    # Add 1 to .ascard to include the END card
                    _nch80 = reduce(operator.add, map(Card._ncards, hdu.header.ascard))
                    _bytes = (_nch80+1) * Card.length
                    _bytes = _bytes + _padLength(_bytes)
                    if _bytes != (hdu._datLoc-hdu._hdrLoc):
                        self._resize = 1
                        self._truncate = 0
                        if verbose:
                            print "One or more header is resized."
                        break

                    # Data:
                    if 'data' not in dir(hdu):
                        continue
                    if hdu.data is None:
                        continue
                    _bytes = hdu.data.nbytes
                    _bytes = _bytes + _padLength(_bytes)
                    if _bytes != hdu._datSpan:
                        self._resize = 1
                        self._truncate = 0
                        if verbose:
                            print "One or more data area is resized."
                        break

                if self._truncate:
                   try:
                       self.__file.getfile().truncate(hdu._datLoc+hdu._datSpan)
                   except IOError:
                       self._resize = 1
                   self._truncate = 0

            # if the HDUList is resized, need to write out the entire contents
            # of the hdulist to the file.
            if self._resize or isinstance(self.__file.getfile(), gzip.GzipFile):
                oldName = self.__file.name
                oldMemmap = self.__file.memmap
                _name = _tmpName(oldName)

                if isinstance(self.__file.getfile(), file) or \
                   isinstance(self.__file.getfile(), gzip.GzipFile):
                    #
                    # The underlying file is an acutal file object.
                    # The HDUList is resized, so we need to write it to a tmp
                    # file, delete the original file, and rename the tmp
                    # file to the original file.
                    #
                    if isinstance(self.__file.getfile(), gzip.GzipFile):
                        newFile = gzip.GzipFile(_name, mode='ab+')
                    else:
                        newFile = _name

                    _hduList = open(newFile, mode="append",
                                    classExtensions=classExtensions)
                    if (verbose): print "open a temp file", _name

                    for hdu in self:
                        # only output the checksum if flagged to do so
                        if hasattr(hdu, '_output_checksum'):
                            checksum = hdu._output_checksum
                        else:
                            checksum = False

                        (hdu._hdrLoc, hdu._datLoc, hdu._datSpan) = \
                               _hduList.__file.writeHDU(hdu, checksum=checksum)
                    _hduList.__file.close()
                    self.__file.close()
                    os.remove(self.__file.name)
                    if (verbose): print "delete the original file", oldName

                    # reopen the renamed new file with "update" mode
                    os.rename(_name, oldName)

                    if isinstance(newFile, gzip.GzipFile):
                        oldFile = gzip.GzipFile(oldName, mode='rb+')
                    else:
                        oldFile = oldName

                    if classExtensions.has_key(_File):
                        ffo = classExtensions[_File](oldFile, mode="update",
                                                       memmap=oldMemmap)
                    else:
                        ffo = _File(oldFile, mode="update", memmap=oldMemmap)

                    self.__file = ffo
                    if (verbose): print "reopen the newly renamed file", oldName
                else:
                    #
                    # The underlying file is not a file object, it is a file
                    # like object.  We can't write out to a file, we must
                    # update the file like object in place.  To do this,
                    # we write out to a temporary file, then delete the
                    # contents in our file like object, then write the
                    # contents of the temporary file to the now empty file
                    # like object.
                    #
                    self.writeto(_name)
                    _hduList = open(_name)
                    ffo = self.__file

                    try:
                        ffo.getfile().truncate(0)
                    except AttributeError:
                        pass

                    for hdu in _hduList:
                        # only output the checksum if flagged to do so
                        if hasattr(hdu, '_output_checksum'):
                            checksum = hdu._output_checksum
                        else:
                            checksum = False

                        (hdu._hdrLoc, hdu._datLoc, hdu._datSpan) = \
                                            ffo.writeHDU(hdu, checksum=checksum)

                    # Close the temporary file and delete it.

                    _hduList.close()
                    os.remove(_hduList.__file.name)

                # reset the resize attributes after updating
                self._resize = 0
                self._truncate = 0
                for hdu in self:
                    hdu.header._mod = 0
                    hdu.header.ascard._mod = 0
                    hdu._new = 0
                    hdu._file = ffo.getfile()

            # if not resized, update in place
            else:
                for hdu in self:
                    if (verbose):
                        try: _extver = `hdu.header['extver']`
                        except: _extver = ''
                    if hdu.header._mod or hdu.header.ascard._mod:
                        # only output the checksum if flagged to do so
                        if hasattr(hdu, '_output_checksum'):
                            checksum = hdu._output_checksum
                        else:
                            checksum = False

                        hdu._file.seek(hdu._hdrLoc)
                        self.__file.writeHDUheader(hdu,checksum=checksum)
                        if (verbose):
                            print "update header in place: Name =", hdu.name, _extver
                    if 'data' in dir(hdu):
                        if hdu.data is not None:
                            if isinstance(hdu.data,Memmap):
                                hdu.data.sync()
                            else:
                                hdu._file.seek(hdu._datLoc)
                                self.__file.writeHDUdata(hdu)
                            if (verbose):
                                print "update data in place: Name =", hdu.name, _extver

                # reset the modification attributes after updating
                for hdu in self:
                    hdu.header._mod = 0
                    hdu.header.ascard._mod = 0
        if singleThread:
            if keyboardInterruptSent:
                raise KeyboardInterrupt

            if old_handler != None:
                signal.signal(signal.SIGINT,old_handler)
            else:
                signal.signal(signal.SIGINT, signal.SIG_DFL)

    def update_extend(self):
        """
        Make sure that if the primary header needs the keyword
        ``EXTEND`` that it has it and it is correct.
        """
        hdr = self[0].header
        if hdr.has_key('extend'):
            if (hdr['extend'] == False):
                hdr['extend'] = True
        else:
            if hdr['naxis'] == 0:
                hdr.update('extend', True, after='naxis')
            else:
                n = hdr['naxis']
                hdr.update('extend', True, after='naxis'+`n`)

    def writeto(self, name, output_verify='exception', clobber=False,
                classExtensions={}, checksum=False):
        """
        Write the `HDUList` to a new file.

        Parameters
        ----------
        name : file path, file object or file-like object
            File to write to.  If a file object, must be opened for
            append (ab+).

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.

        clobber : bool
            When `True`, overwrite the output file if exists.

        classExtensions : dict
            A dictionary that maps pyfits classes to extensions of
            those classes.  When present in the dictionary, the
            extension class will be constructed in place of the pyfits
            class.

        checksum : bool
            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the headers of all HDU's written to the file.
        """

        if (len(self) == 0):
            warnings.warn("There is nothing to write.")
            return

        self.update_tbhdu()


        if output_verify == 'warn':
            output_verify = 'exception'
        self.verify(option=output_verify)

        # check if the file object is closed
        closed = True
        fileMode = 'ab+'

        if isinstance(name, file):
            closed = name.closed
            filename = name.name

            if not closed:
                fileMode = name.mode

        elif isinstance(name, gzip.GzipFile):
            if name.fileobj != None:
                closed = name.fileobj.closed
            filename = name.filename

            if not closed:
                fileMode = name.fileobj.mode

        elif isinstance(name, types.StringType) or \
             isinstance(name, types.UnicodeType):
            filename = name
        else:
            if hasattr(name, 'closed'):
                closed = name.closed

            if hasattr(name, 'mode'):
                fileMode = name.mode

            if hasattr(name, 'name'):
                filename = name.name
            elif hasattr(name, 'filename'):
                filename = name.filename
            elif hasattr(name, '__class__'):
                filename = str(name.__class__)
            else:
                filename = str(type(name))

        # check if the output file already exists
        if (isinstance(name,types.StringType) or
            isinstance(name,types.UnicodeType) or isinstance(name,file) or
            isinstance(name,gzip.GzipFile)):
            if (os.path.exists(filename) and os.path.getsize(filename) != 0):
                if clobber:
                    warnings.warn( "Overwrite existing file '%s'." % filename)
                    if (isinstance(name,file) and not name.closed) or \
                       (isinstance(name,gzip.GzipFile) and name.fileobj != None
                        and not name.fileobj.closed):
                       name.close()
                    os.remove(filename)
                else:
                    raise IOError, "File '%s' already exist." % filename
        elif (hasattr(name,'len') and name.len > 0):
            if clobber:
                warnings.warn( "Overwrite existing file '%s'." % filename)
                name.truncate(0)
            else:
                raise IOError, "File '%s' already exist." % filename

        # make sure the EXTEND keyword is there if there is extension
        if len(self) > 1:
            self.update_extend()

        for key in _python_mode.keys():
            if _python_mode[key] == fileMode:
                mode = key
                break

        hduList = open(name, mode=mode, classExtensions=classExtensions)

        for hdu in self:
            hduList.__file.writeHDU(hdu, checksum)
        hduList.close(output_verify=output_verify,closed=closed)


    def close(self, output_verify='exception', verbose=False, closed=True):
        """
        Close the associated FITS file and memmap object, if any.

        Parameters
        ----------
        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  See :ref:`verify` for more info.

        verbose : bool
            When `True`, print out verbose messages.

        closed : bool
            When `True`, close the underlying file object.
        """

        if self.__file != None:
            if self.__file.mode in ['append', 'update']:
                self.flush(output_verify=output_verify, verbose=verbose)

            if closed and hasattr(self.__file, 'close'):
                self.__file.close()

        # close the memmap object, it is designed to use an independent
        # attribute of mmobject so if the HDUList object is created from files
        # other than FITS, the close() call can also close the mm object.
#        try:
#            self.mmobject.close()
#        except:
#            pass

    def info(self):
        """
        Summarize the info of the HDUs in this `HDUList`.

        Note that this function prints its results to the console---it
        does not return a value.
        """
        if self.__file is None:
            _name = '(No file associated with this HDUList)'
        else:
            _name = self.__file.name
        results = "Filename: %s\nNo.    Name         Type"\
                  "      Cards   Dimensions   Format\n" % _name

        for j in range(len(self)):
            results = results + "%-3d  %s\n"%(j, self[j]._summary())
        results = results[:-1]
        print results

    def filename(self):
        """
        Return the file name associated with the HDUList object if one exists.
        Otherwise returns None.

        Returns
        -------
        filename : a string containing the file name associated with the
                   HDUList object if an association exists.  Otherwise returns
                   None.
        """
        if self.__file is not None:
           if hasattr(self.__file, 'name'):
              return self.__file.name
        return None

    # Support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def open(name, mode="copyonwrite", memmap=False, classExtensions={}, **parms):
    """
    Factory function to open a FITS file and return an `HDUList` object.

    Parameters
    ----------
    name : file path, file object or file-like object
        File to be
        opened.

    mode : str
        Open mode, 'copyonwrite' (default), 'readonly', 'update', 
        'append', or 'ostream'.

        If `name` is a file object that is already opened, `mode` must
        match the mode the file was opened with, copyonwrite (rb),
        readonly (rb), update (rb+), append (ab+), ostream (w)).

    memmap : bool
        Is memory mapping to be used?

    classExtensions : dict
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    parms : dict
        optional keyword arguments, possible values are:

        - **uint** : bool

            Interpret signed integer data where ``BZERO`` is the
            central value and ``BSCALE == 1`` as unsigned integer
            data.  For example, `int16` data with ``BZERO = 32768``
            and ``BSCALE = 1`` would be treated as `uint16` data.

            Note, for backward compatibility, the kwarg **uint16** may
            be used instead.  The kwarg was renamed when support was
            added for integers of any size.

        - **ignore_missing_end** : bool

            Do not issue an exception when opening a file that is
            missing an ``END`` card in the last header.

        - **checksum** : bool

            If `True`, verifies that both ``DATASUM`` and
            ``CHECKSUM`` card values (when present in the HDU header)
            match the header and data of all HDU's in the file.

    Returns
    -------
        hdulist : an HDUList object
            `HDUList` containing all of the header data units in the
            file.
    """
    # instantiate a FITS file object (ffo)

    if classExtensions.has_key(_File):
        ffo = classExtensions[_File](name, mode=mode, memmap=memmap, **parms)
    else:
        ffo = _File(name, mode=mode, memmap=memmap, **parms)

    if classExtensions.has_key(HDUList):
        hduList = classExtensions[HDUList](file=ffo)
    else:
        hduList = HDUList(file=ffo)

    if mode != 'ostream':
        # read all HDU's
        while 1:
            try:
                hduList.append(ffo._readHDU(), classExtensions=classExtensions)
            except EOFError:
                break
            # check in the case there is extra space after the last HDU or
            # corrupted HDU
            except ValueError, e:
                warnings.warn('Warning:  Required keywords missing when trying to read HDU #%d.\n          %s\n          There may be extra bytes after the last HDU or the file is corrupted.' % (len(hduList),e))
                break
            except IOError, e:
                if isinstance(ffo.getfile(), gzip.GzipFile) and \
                   string.find(str(e),'on write-only GzipFile object'):
                    break
                else:
                    raise e

        # If we're trying to read only and no header units were found,
        # raise and exception
        if mode == 'readonly' and len(hduList) == 0:
            raise IOError("Empty FITS file")

        # For each HDU, verify the checksum/datasum value if the cards exist in
        # the header and we are opening with checksum=True.  Always remove the
        # checksum/datasum cards from the header.
        for i in range(len(hduList)):
            hdu = hduList.__getitem__(i, classExtensions)

            if hdu._header.has_key('CHECKSUM'):
                 hdu._checksum = hdu._header['CHECKSUM']
                 hdu._checksum_comment = \
                                hdu._header.ascardlist()['CHECKSUM'].comment

                 if 'checksum' in parms and parms['checksum'] and \
                 not hdu.verify_checksum():
                     warnings.warn('Warning:  Checksum verification failed for '
                                   'HDU #%d.\n' % i)

                 del hdu.header['CHECKSUM']
            else:
                 hdu._checksum = None
                 hdu._checksum_comment = None

            if hdu._header.has_key('DATASUM'):
                 hdu._datasum = hdu.header['DATASUM']
                 hdu._datasum_comment = \
                                   hdu.header.ascardlist()['DATASUM'].comment

                 if 'checksum' in parms and parms['checksum'] and \
                 not hdu.verify_datasum():
                     warnings.warn('Warning:  Datasum verification failed for '
                                   'HDU #%d.\n' % (len(hduList)))

                 del hdu.header['DATASUM']
            else:
                 hdu._checksum = None
                 hdu._checksum_comment = None
                 hdu._datasum = None
                 hdu._datasum_comment = None

    # initialize/reset attributes to be used in "update/append" mode
    # CardList needs its own _mod attribute since it has methods to change
    # the content of header without being able to pass it to the header object
    hduList._resize = 0
    hduList._truncate = 0

    return hduList

fitsopen = open

# Convenience functions

class _Zero(int):
    def __init__(self):
        self = 0

def _getext(filename, mode, *ext1, **ext2):
    """
    Open the input file, return the `HDUList` and the extension.
    """
    hdulist = open(filename, mode=mode, **ext2)

    # delete these from the variable keyword argument list so the extension
    # will properly validate
    if ext2.has_key('classExtensions'):
        del ext2['classExtensions']

    if ext2.has_key('ignore_missing_end'):
        del ext2['ignore_missing_end']

    if ext2.has_key('uint16'):
        del ext2['uint16']

    if ext2.has_key('uint'):
        del ext2['uint']

    n_ext1 = len(ext1)
    n_ext2 = len(ext2)
    keys = ext2.keys()

    # parse the extension spec
    if n_ext1 > 2:
        raise ValueError, "too many positional arguments"
    elif n_ext1 == 1:
        if n_ext2 == 0:
            ext = ext1[0]
        else:
            if isinstance(ext1[0], (int, np.integer, tuple)):
                raise KeyError, 'Redundant/conflicting keyword argument(s): %s' % ext2
            if isinstance(ext1[0], str):
                if n_ext2 == 1 and 'extver' in keys:
                    ext = ext1[0], ext2['extver']
                raise KeyError, 'Redundant/conflicting keyword argument(s): %s' % ext2
    elif n_ext1 == 2:
        if n_ext2 == 0:
            ext = ext1
        else:
            raise KeyError, 'Redundant/conflicting keyword argument(s): %s' % ext2
    elif n_ext1 == 0:
        if n_ext2 == 0:
            ext = _Zero()
        elif 'ext' in keys:
            if n_ext2 == 1:
                ext = ext2['ext']
            elif n_ext2 == 2 and 'extver' in keys:
                ext = ext2['ext'], ext2['extver']
            else:
                raise KeyError, 'Redundant/conflicting keyword argument(s): %s' % ext2
        else:
            if 'extname' in keys:
                if 'extver' in keys:
                    ext = ext2['extname'], ext2['extver']
                else:
                    ext = ext2['extname']
            else:
                raise KeyError, 'Insufficient keyword argument: %s' % ext2

    return hdulist, ext

def getheader(filename, *ext, **extkeys):
    """
    Get the header from an extension of a FITS file.

    Parameters
    ----------
    filename : file path, file object, or file like object
        File to get header from.  If an opened file object, its mode
        must be one of the following rb, rb+, or ab+).

    classExtensions : optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    ext
        The rest of the arguments are for extension specification.
        `getdata` for explanations/examples.

    Returns
    -------
    header : `Header` object
    """
    # allow file object to already be opened in any of the valid modes
    # and leave the file in the same state (opened or closed) as when
    # the function was called

    mode = 'readonly'
    closed = True

    if (isinstance(filename, file) and not filename.closed) or \
       (isinstance(filename, gzip.GzipFile) and filename.fileobj != None and
                                            not filename.fileobj.closed):

        if isinstance(filename, gzip.GzipFile):
            fileMode = filename.fileobj.mode
        else:
            fileMode = filename.mode

        for key in _python_mode.keys():
            if _python_mode[key] == fileMode:
                mode = key
                break

    if hasattr(filename, 'closed'):
        closed = filename.closed
    elif hasattr(filename, 'fileobj'):
        if filename.fileobj != None:
           closed = filename.fileobj.closed

    hdulist, _ext = _getext(filename, mode, *ext, **extkeys)
    hdu = hdulist[_ext]
    hdr = hdu.header

    hdulist.close(closed=closed)
    return hdr


def _fnames_changecase(data, func):
    """
    Convert case of field names.
    """
    if data.dtype.names is None:
        # this data does not have fields
        return

    if data.dtype.descr[0][0] == '':
        # this data does not have fields
        return

    data.dtype.names = [func(n) for n in data.dtype.names]


def getdata(filename, *ext, **extkeys):
    """
    Get the data from an extension of a FITS file (and optionally the
    header).

    Parameters
    ----------
    filename : file path, file object, or file like object
        File to get data from.  If opened, mode must be one of the
        following rb, rb+, or ab+.

    classExtensions : dict, optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    ext
        The rest of the arguments are for extension specification.
        They are flexible and are best illustrated by examples.

        No extra arguments implies the primary header::

            >>> getdata('in.fits')

        By extension number::

            >>> getdata('in.fits', 0)    # the primary header
            >>> getdata('in.fits', 2)    # the second extension
            >>> getdata('in.fits', ext=2) # the second extension

        By name, i.e., ``EXTNAME`` value (if unique)::

            >>> getdata('in.fits', 'sci')
            >>> getdata('in.fits', extname='sci') # equivalent

        Note ``EXTNAME`` values are not case sensitive

        By combination of ``EXTNAME`` and EXTVER`` as separate
        arguments or as a tuple::

            >>> getdata('in.fits', 'sci', 2) # EXTNAME='SCI' & EXTVER=2
            >>> getdata('in.fits', extname='sci', extver=2) # equivalent
            >>> getdata('in.fits', ('sci', 2)) # equivalent

        Ambiguous or conflicting specifications will raise an exception::

            >>> getdata('in.fits', ext=('sci',1), extname='err', extver=2)

    lower, upper : bool, optional
        If `lower` or `upper` are `True`, the field names in the
        returned data object will be converted to lower or upper case,
        respectively.

    view : ndarray view class, optional
        When given, the data will be turned wrapped in the given view
        class, by calling::

           data.view(view)

    Returns
    -------
    array : array, record array or groups data object
        Type depends on the type of the extension being referenced.

        If the optional keyword `header` is set to `True`, this
        function will return a (`data`, `header`) tuple.
    """
    if 'header' in extkeys:
        _gethdr = extkeys['header']
        del extkeys['header']
    else:
        _gethdr = False

    # Code further down rejects unkown keys
    lower=False
    if 'lower' in extkeys:
        lower=extkeys['lower']
        del extkeys['lower']
    upper=False
    if 'upper' in extkeys:
        upper=extkeys['upper']
        del extkeys['upper']
    view=None
    if 'view' in extkeys:
        view=extkeys['view']
        del extkeys['view']

    # allow file object to already be opened in any of the valid modes
    # and leave the file in the same state (opened or closed) as when
    # the function was called

    mode = 'readonly'
    closed = True

    if (isinstance(filename, file) and not filename.closed) or \
       (isinstance(filename, gzip.GzipFile) and filename.fileobj != None and
                                            not filename.fileobj.closed):

        if isinstance(filename, gzip.GzipFile):
            fileMode = filename.fileobj.mode
        else:
            fileMode = filename.mode

        for key in _python_mode.keys():
            if _python_mode[key] == fileMode:
                mode = key
                break

    if hasattr(filename, 'closed'):
        closed = filename.closed
    elif hasattr(filename, 'fileobj'):
        if filename.fileobj != None:
           closed = filename.fileobj.closed

    hdulist, _ext = _getext(filename, mode, *ext, **extkeys)
    hdu = hdulist[_ext]
    _data = hdu.data
    if _data is None and isinstance(_ext, _Zero):
        try:
            hdu = hdulist[1]
            _data = hdu.data
        except IndexError:
            raise IndexError, 'No data in this HDU.'
    if _data is None:
        raise IndexError, 'No data in this HDU.'
    if _gethdr:
        _hdr = hdu.header
    hdulist.close(closed=closed)

    # Change case of names if requested
    if lower:
        _fnames_changecase(_data, str.lower)
    elif upper:
        _fnames_changecase(_data, str.upper)

    # allow different views into the underlying ndarray.  Keep the original
    # view just in case there is a problem
    if view is not None:
        _data = _data.view(view)

    if _gethdr:
        return _data, _hdr
    else:
        return _data

def getval(filename, key, *ext, **extkeys):
    """
    Get a keyword's value from a header in a FITS file.

    Parameters
    ----------
    filename : file path, file object, or file like object
        Name of the FITS file, or file object (if opened, mode must be
        one of the following rb, rb+, or ab+).

    key : str
        keyword name

    classExtensions : (optional)
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    ext
        The rest of the arguments are for extension specification.
        See `getdata` for explanations/examples.

    Returns
    -------
    keyword value : string, integer, or float
    """

    _hdr = getheader(filename, *ext, **extkeys)
    return _hdr[key]

def setval(filename, key, value="", comment=None, before=None, after=None,
           savecomment=False, *ext, **extkeys):
    """
    Set a keyword's value from a header in a FITS file.

    If the keyword already exists, it's value/comment will be updated.
    If it does not exist, a new card will be created and it will be
    placed before or after the specified location.  If no `before` or
    `after` is specified, it will be appended at the end.

    When updating more than one keyword in a file, this convenience
    function is a much less efficient approach compared with opening
    the file for update, modifying the header, and closing the file.

    Parameters
    ----------
    filename : file path, file object, or file like object
        Name of the FITS file, or file object If opened, mode must be
        update (rb+).  An opened file object or `GzipFile` object will
        be closed upon return.

    key : str
        keyword name

    value : str, int, float
        Keyword value, default = ""

    comment : str
        Keyword comment, default = None

    before : str, int
        name of the keyword, or index of the `Card` before which
        the new card will be placed.  The argument `before` takes
        precedence over `after` if both specified. default=`None`.

    after : str, int
        name of the keyword, or index of the `Card` after which the
        new card will be placed. default=`None`.

    savecomment : bool
        when `True`, preserve the current comment for an existing
        keyword.  The argument `savecomment` takes precedence over
        `comment` if both specified.  If `comment` is not specified
        then the current comment will automatically be preserved.
        default=`False`

    classExtensions : dict, optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    ext
        The rest of the arguments are for extension specification.
        See `getdata` for explanations/examples.
    """

    hdulist, ext = _getext(filename, mode='update', *ext, **extkeys)
    hdulist[ext].header.update(key, value, comment, before, after, savecomment)

    # Ensure that data will not be scaled when the file is closed

    for hdu in hdulist:
       hdu._bscale = 1
       hdu._bzero = 0

    hdulist.close()

def delval(filename, key, *ext, **extkeys):
    """
    Delete all instances of keyword from a header in a FITS file.

    Parameters
    ----------

    filename : file path, file object, or file like object
        Name of the FITS file, or file object If opened, mode must be
        update (rb+).  An opened file object or `GzipFile` object will
        be closed upon return.

    key : str, int
        Keyword name or index

    classExtensions : optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    ext
        The rest of the arguments are for extension specification.
        See `getdata` for explanations/examples.
    """

    hdulist, ext = _getext(filename, mode='update', *ext, **extkeys)
    del hdulist[ext].header[key]

    # Ensure that data will not be scaled when the file is closed

    for hdu in hdulist:
       hdu._bscale = 1
       hdu._bzero = 0

    hdulist.close()


def _makehdu(data, header, classExtensions={}):
    if header is None:
        if ((isinstance(data, np.ndarray) and data.dtype.fields is not None)
            or isinstance(data, np.recarray)
            or isinstance(data, rec.recarray)):
            if classExtensions.has_key(BinTableHDU):
                hdu = classExtensions[BinTableHDU](data)
            else:
                hdu = BinTableHDU(data)
        elif isinstance(data, np.ndarray):
            if classExtensions.has_key(ImageHDU):
                hdu = classExtensions[ImageHDU](data)
            else:
                hdu = ImageHDU(data)
        else:
            raise KeyError, 'data must be numarray or table data.'
    else:
        if classExtensions.has_key(header._hdutype):
            header._hdutype = classExtensions[header._hdutype]

        hdu=header._hdutype(data=data, header=header)
    return hdu

def _stat_filename_or_fileobj(filename):
    closed = True
    name = ''

    if isinstance(filename, file):
        closed = filename.closed
        name = filename.name
    elif isinstance(filename, gzip.GzipFile):
        if filename.fileobj != None:
            closed = filename.fileobj.closed
        name = filename.filename
    elif isinstance(filename, types.StringType):
        name = filename
    else:
        if hasattr(filename, 'closed'):
            closed = filename.closed

        if hasattr(filename, 'name'):
            name = filename.name
        elif hasattr(filename, 'filename'):
            name = filename.filename

    try:
        loc = filename.tell()
    except AttributeError:
        loc = 0

    noexist_or_empty = \
        (name and ((not os.path.exists(name)) or (os.path.getsize(name)==0))) \
         or (not name and loc==0)

    return name, closed, noexist_or_empty

def writeto(filename, data, header=None, **keys):
    """
    Create a new FITS file using the supplied data/header.

    Parameters
    ----------
    filename : file path, file object, or file like object
        File to write to.  If opened, must be opened for append (ab+).

    data : array, record array, or groups data object
        data to write to the new file

    header : Header object, optional
        the header associated with `data`. If `None`, a header
        of the appropriate type is created for the supplied data. This
        argument is optional.

    classExtensions : dict, optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    clobber : bool, optional
        If `True`, and if filename already exists, it will overwrite
        the file.  Default is `False`.

    checksum : bool, optional
        If `True`, adds both ``DATASUM`` and ``CHECKSUM`` cards to the
        headers of all HDU's written to the file.
    """
    if header is None:
        if 'header' in keys:
            header = keys['header']

    clobber = keys.get('clobber', False)
    output_verify = keys.get('output_verify', 'exception')

    classExtensions = keys.get('classExtensions', {})
    hdu = _makehdu(data, header, classExtensions)
    if not isinstance(hdu, PrimaryHDU) and not isinstance(hdu, _TableBaseHDU):
        if classExtensions.has_key(PrimaryHDU):
            hdu = classExtensions[PrimaryHDU](data, header=header)
        else:
            hdu = PrimaryHDU(data, header=header)
    checksum = keys.get('checksum', False)
    hdu.writeto(filename, clobber=clobber, output_verify=output_verify,
                checksum=checksum, classExtensions=classExtensions)

def append(filename, data, header=None, classExtensions={}, checksum=False,
           **keys):
    """
    Append the header/data to FITS file if filename exists, create if not.

    If only `data` is supplied, a minimal header is created.

    Parameters
    ----------
    filename : file path, file object, or file like object
        File to write to.  If opened, must be opened for update (rb+)
        unless it is a new file, then it must be opened for append
        (ab+).  A file or `GzipFile` object opened for update will be
        closed after return.

    data : array, table, or group data object
        the new data used for appending

    header : Header object, optional
        The header associated with `data`.  If `None`, an appropriate
        header will be created for the data object supplied.

    classExtensions : dictionary, optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    checksum : bool, optional
        When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards to
        the header of the HDU when written to the file.
    """
    name, closed, noexist_or_empty = _stat_filename_or_fileobj(filename)

    if noexist_or_empty:
        #
        # The input file or file like object either doesn't exits or is
        # empty.  Use the writeto convenience function to write the
        # output to the empty object.
        #
        writeto(filename, data, header, classExtensions=classExtensions,
                checksum=checksum, **keys)
    else:
        hdu=_makehdu(data, header, classExtensions)
        if isinstance(hdu, PrimaryHDU):
            if classExtensions.has_key(ImageHDU):
                hdu = classExtensions[ImageHDU](data, header)
            else:
                hdu = ImageHDU(data, header)

        f = open(filename, mode='update', classExtensions=classExtensions)
        f.append(hdu, classExtensions=classExtensions)

        # Set a flag in the HDU so that only this HDU gets a checksum
        # when writing the file.
        hdu._output_checksum = checksum
        f.close(closed=closed)

def update(filename, data, *ext, **extkeys):
    """
    Update the specified extension with the input data/header.

    Parameters
    ----------
    filename : file path, file object, or file like object
        File to update.  If opened, mode must be update (rb+).  An
        opened file object or `GzipFile` object will be closed upon
        return.

    data : array, table, or group data object
        the new data used for updating

    classExtensions : dict, optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    ext
        The rest of the arguments are flexible: the 3rd argument can
        be the header associated with the data.  If the 3rd argument
        is not a `Header`, it (and other positional arguments) are
        assumed to be the extension specification(s).  Header and
        extension specs can also be keyword arguments.  For example::

            >>> update(file, dat, hdr, 'sci')  # update the 'sci' extension
            >>> update(file, dat, 3)  # update the 3rd extension
            >>> update(file, dat, hdr, 3)  # update the 3rd extension
            >>> update(file, dat, 'sci', 2)  # update the 2nd SCI extension
            >>> update(file, dat, 3, header=hdr)  # update the 3rd extension
            >>> update(file, dat, header=hdr, ext=5)  # update the 5th extension
    """

    # parse the arguments
    header = None
    if len(ext) > 0:
        if isinstance(ext[0], Header):
            header = ext[0]
            ext = ext[1:]
        elif not isinstance(ext[0], (int, long, np.integer, str, tuple)):
            raise KeyError, 'Input argument has wrong data type.'

    if 'header' in extkeys:
        header = extkeys['header']
        del extkeys['header']

    classExtensions = extkeys.get('classExtensions', {})

    new_hdu=_makehdu(data, header, classExtensions)

    if not isinstance(filename, file) and hasattr(filename, 'closed'):
        closed = filename.closed
    else:
        closed = True

    hdulist, _ext = _getext(filename, 'update', *ext, **extkeys)
    hdulist[_ext] = new_hdu

    hdulist.close(closed=closed)


def info(filename, classExtensions={}, **parms):
    """
    Print the summary information on a FITS file.

    This includes the name, type, length of header, data shape and type
    for each extension.

    Parameters
    ----------
    filename : file path, file object, or file like object
        FITS file to obtain info from.  If opened, mode must be one of
        the following: rb, rb+, or ab+.

    classExtensions : dict, optional
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    parms : optional keyword arguments

        - **uint** : bool

            Interpret signed integer data where ``BZERO`` is the
            central value and ``BSCALE == 1`` as unsigned integer
            data.  For example, `int16` data with ``BZERO = 32768``
            and ``BSCALE = 1`` would be treated as `uint16` data.

            Note, for backward compatibility, the kwarg **uint16** may
            be used instead.  The kwarg was renamed when support was
            added for integers of any size.

        - **ignore_missing_end** : bool

            Do not issue an exception when opening a file that is
            missing an ``END`` card in the last header.  Default is
            `True`.
    """

    # allow file object to already be opened in any of the valid modes
    # and leave the file in the same state (opened or closed) as when
    # the function was called

    mode = 'copyonwrite'
    closed = True

    if not isinstance(filename, types.StringType) and \
       not isinstance(filename, types.UnicodeType):
        if hasattr(filename, 'closed'):
            closed = filename.closed
        elif hasattr(filename, 'fileobj') and filename.fileobj != None:
            closed = filename.fileobj.closed

    if not closed and hasattr(filename, 'mode'):

        if isinstance(filename, gzip.GzipFile):
            fmode = filename.fileobj.mode
        else:
            fmode = filename.mode

        for key in _python_mode.keys():
            if _python_mode[key] == fmode:
                mode = key
                break

    # Set the default value for the ignore_missing_end parameter
    if not parms.has_key('ignore_missing_end'):
        parms['ignore_missing_end'] = True

    f = open(filename,mode=mode,classExtensions=classExtensions, **parms)
    f.info()

    if closed:
        f.close()

def tdump(fitsFile, datafile=None, cdfile=None, hfile=None, ext=1,
          clobber=False, classExtensions={}):
    """
    Dump a table HDU to a file in ASCII format.  The table may be
    dumped in three separate files, one containing column definitions,
    one containing header parameters, and one for table data.

    Parameters
    ----------
    fitsFile : file path, file object or file-like object
        Input fits file.

    datafile : file path, file object or file-like object, optional
        Output data file.  The default is the root name of the input
        fits file appended with an underscore, followed by the
        extension number (ext), followed by the extension ``.txt``.

    cdfile : file path, file object or file-like object, optional
        Output column definitions file.  The default is `None`,
        no column definitions output is produced.

    hfile : file path, file object or file-like object, optional
        Output header parameters file.  The default is `None`,
        no header parameters output is produced.

    ext : int
        The number of the extension containing the table HDU to be
        dumped.

    clobber : bool
        Overwrite the output files if they exist.

    classExtensions : dict
        A dictionary that maps pyfits classes to extensions of those
        classes.  When present in the dictionary, the extension class
        will be constructed in place of the pyfits class.

    Notes
    -----
    The primary use for the `tdump` function is to allow editing in a
    standard text editor of the table data and parameters.  The
    `tcreate` function can be used to reassemble the table from the
    three ASCII files.
    """

    # allow file object to already be opened in any of the valid modes
    # and leave the file in the same state (opened or closed) as when
    # the function was called

    mode = 'copyonwrite'
    closed = True

    if not isinstance(fitsFile, types.StringType) and \
       not isinstance(fitsFile, types.UnicodeType):
        if hasattr(fitsFile, 'closed'):
            closed = fitsFile.closed
        elif hasattr(fitsFile, 'fileobj') and fitsFile.fileobj != None:
            closed = fitsFile.fileobj.closed

    if not closed and hasattr(fitsFile, 'mode'):

        if isinstance(fitsFile, gzip.GzipFile):
            fmode = fitsFile.fileobj.mode
        else:
            fmode = fitsFile.mode

        for key in _python_mode.keys():
            if _python_mode[key] == fmode:
                mode = key
                break

    f = open(fitsFile,mode=mode,classExtensions=classExtensions)

    # Create the default data file name if one was not provided

    if not datafile:
        root,tail = os.path.splitext(f._HDUList__file.name)
        datafile = root + '_' + `ext` + '.txt'

    # Dump the data from the HDU to the files
    f[ext].tdump(datafile, cdfile, hfile, clobber)

    if closed:
        f.close()

tdump.__doc__ += BinTableHDU.tdumpFileFormat.replace("\n", "\n    ")

def tcreate(datafile, cdfile, hfile=None):
    """
    Create a table from the input ASCII files.  The input is from up
    to three separate files, one containing column definitions, one
    containing header parameters, and one containing column data.  The
    header parameters file is not required.  When the header
    parameters file is absent a minimal header is constructed.

    Parameters
    ----------
    datafile : file path, file object or file-like object
        Input data file containing the table data in ASCII format.

    cdfile : file path, file object or file-like object
        Input column definition file containing the names, formats,
        display formats, physical units, multidimensional array
        dimensions, undefined values, scale factors, and offsets
        associated with the columns in the table.

    hfile : file path, file object or file-like object, optional
        Input parameter definition file containing the header
        parameter definitions to be associated with the table.
        If `None`, a minimal header is constructed.

    Notes
    -----
    The primary use for the `tcreate` function is to allow the input of
    ASCII data that was edited in a standard text editor of the table
    data and parameters.  The tdump function can be used to create the
    initial ASCII files.
    """

    # Construct an empty HDU
    hdu = BinTableHDU()

    # Populate and return that HDU
    hdu.tcreate(datafile, cdfile, hfile, replace=True)
    return hdu

tcreate.__doc__ += BinTableHDU.tdumpFileFormat.replace("\n", "\n    ")

UNDEFINED = Undefined()

__credits__="""

Copyright (C) 2004 Association of Universities for Research in Astronomy (AURA)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
