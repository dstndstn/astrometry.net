#! /usr/bin/env python
"""
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

Convert an image in a variety of formats into a pnm file
"""
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import os.path
import tempfile

from astrometry.util.shell import shell_escape
from astrometry.util.filetype import filetype_short
import logging

fitstype = 'FITS image data'
fitsext = 'fits'
tiffext = 'tiff'

pgmcmd = 'pgmtoppm rgbi:1/1/1 %s > %s'
pgmext = 'pgm'

an_fitstopnm_ext_cmd = 'an-fitstopnm -e %i -i %%s > %%s'

imgcmds = {fitstype : (fitsext, 'an-fitstopnm -i %s > %s'),
       'JPEG image data'  : ('jpg',  'jpegtopnm %s > %s'),
       'PNG image data'       : ('png',      'pngtopnm %s > %s'),
       'PNG image'    : ('png',      'pngtopnm %s > %s'),
       'GIF image data'       : ('gif',      'giftopnm %s > %s'),
       'Netpbm PPM'       : ('ppm',      'ppmtoppm < %s > %s'),
       'Netpbm PPM "rawbits" image data' : ('ppm',  'cp %s %s'),
       'Netpbm PGM'       : ('pgm',      pgmcmd),
       'Netpbm PGM "rawbits" image data' : ('pgm',  pgmcmd),
       'TIFF image data'  : ('tiff',  'tifftopnm %s > %s'),
       'PC bitmap' : ('bmp', 'bmptopnm %s > %s'),
       # RAW is not recognized by 'file'; we have to use 'dcraw',
       # but we still store this here for convenience.
       'raw'              : ('raw', 'dcraw -4 -c %s > %s'),
       }

compcmds = {'gzip compressed data'    : ('gz',      'gunzip -c %s > %s'),
        "compress'd data 16 bits" : ('gz',      'gunzip -c %s > %s'),
        'bzip2 compressed data'   : ('bz2', 'bunzip2 -k -c %s > %s')
        }

# command to identify a RAW image.
raw_id_cmd = 'dcraw -i %s >/dev/null 2> /dev/null'

verbose = False

def do_command(cmd):
    logging.debug('Running: "%s"' % cmd)
    if os.system(cmd) != 0:
        print('Command failed: %s' % cmd, file=sys.stderr)
        sys.exit(-1)

def get_cmd(types, cmds):
    if types is None:
        return None
    ext=None
    cmd=None
    for t in types:
        (ext,cmd) = cmds.get(t, (None,None))
        if ext is not None:
            break
    return (ext,cmd)

def uncompress_file(infile, uncompressed, typeinfo=None):
    """
    infile: input filename.
    uncompressed: output filename.
    typeinfo: output from the 'file' command; if None we'll run 'file'.

    Returns: comptype
    comptype: None if the file wasn't compressed, or 'gz' or 'bz2'.
    """
    if typeinfo is None:
        typeinfo = filetype_short(infile)
        if typeinfo is None:
            logging.debug('Could not determine file type of "%s"' % infile)
            return None
    (ext,cmd) = get_cmd(typeinfo, compcmds)
    if ext is None:
        logging.debug('File is not compressed: "%s"' % '/'.join(typeinfo))
        return None
    assert uncompressed != infile
    logging.debug('Compressed file (type %s), dumping to: "%s"' % (ext, uncompressed))
    do_command(cmd % (shell_escape(infile), shell_escape(uncompressed)))
    return ext

def is_raw(fn):
    rtn = os.system(raw_id_cmd % shell_escape(fn))
    logging.debug('ran dcraw: return value %i' % rtn)
    return os.WIFEXITED(rtn) and (os.WEXITSTATUS(rtn) == 0)

# Returns (extension, command, error)
def get_image_type(infile):
    typeinfo = filetype_short(infile)
    if typeinfo is None:
        return (None, None, 'Could not determine file type (does the file exist?): %s' % infile)
    (ext,cmd) = get_cmd(typeinfo, imgcmds)
    logging.debug('ext: %s' % ext)
    # "file" recognizes some RAWs as TIFF, but tifftopnm can't read them...
    # run "dcraw" here if the type is TIFF.
    if ext == tiffext and is_raw(infile):
        (ext, cmd) = imgcmds['raw']
    if ext is not None:
        return (ext, cmd, None)
    if ext != tiffext and is_raw(infile):
        # it's a RAW image.
        (ext, cmd) = imgcmds['raw']
        return (ext, cmd, None)
    return (None, None, 'Unknown image type "%s"' % typeinfo)

def find_program(dirs, cmd):
    # pull off the executable name.
    parts = cmd.split(' ', 1)
    prog = parts[0]
    # try the same directory - this should work for installed
    # versions where image2pnm.py and an-fitstopnm are both in
    # "bin".
    for mydir in dirs:
        # If mydir is actually a file, get its dir
        if os.path.isfile(mydir):
            mydir = os.path.dirname(mydir)
        p = os.path.join(mydir, prog)
        if os.path.exists(p):
            return ' '.join([p, parts[1]])
        logging.info('path', p, 'does not exist.')
    return None

def image2pnm(infile, outfile, force_ppm=False, extension=None, mydir=None):
    """
    infile: input filename.
    outfile: output filename.
    force_ppm: boolean, convert PGM to PPM so that the output is always PPM.

    Returns: (type, error)

    - type: (string): image type: 'jpg', 'png', 'gif', etc., or None if
       image type isn't recognized.

    - error: (string): error string, or None
    """
    (ext, cmd, err) = get_image_type(infile)
    if ext is None:
        return (None, 'Image type not recognized: ' + err)

    tempfiles = []
    (outfile_dir, outfile_file) = os.path.split(outfile)

    if force_ppm:
        original_outfile = outfile
        outfile_dir = os.path.dirname(outfile)
        (f, outfile) = tempfile.mkstemp(suffix='.pnm',
                    dir=outfile_dir)
        # we might rename this file later, so don't add it to the list of
        # tempfiles to delete until later...
        os.close(f)
        logging.debug('temporary output file: %s' % outfile)

    if ext == fitsext and extension:
        cmd = an_fitstopnm_ext_cmd % extension

    if ext == fitsext and mydir:
        # an-fitstopnm: add explicit path...
        cmd = find_program(mydir, cmd)
        if cmd is None:
            return (None, 'Couldn\'t find the program "an-fitstopnm".')

    # Do the actual conversion
    do_command(cmd % (shell_escape(infile), shell_escape(outfile)))

    if force_ppm:
        if ext == pgmext:
            # Convert to PPM.
            do_command(pgmcmd % (shell_escape(outfile), shell_escape(original_outfile)))
            tempfiles.append(outfile)
        else:
            # print 'file type extension:', ext, '; renaming', outfile, 'to', original_outfile
            os.rename(outfile, original_outfile)

    for fn in tempfiles:
        os.unlink(fn)
    # Success
    return (ext, None)

def convert_image(infile, outfile, uncompressed=None, force_ppm=False,
                  extension=None, mydir=None):
    tempfiles = []
    # if the caller didn't specify where to put the uncompressed file,
    # create a tempfile.
    if uncompressed is None:
        (outfile_dir, outfile_file) = os.path.split(outfile)
        (f, uncompressed) = tempfile.mkstemp('', 'uncomp', outfile_dir)
        os.close(f)
        tempfiles.append(uncompressed)
    comp = uncompress_file(infile, uncompressed)
    if comp:
        print('compressed')
        print(comp)
        infile = uncompressed

    (imgtype, errstr) = image2pnm(infile, outfile, force_ppm=force_ppm,
                                  extension=extension, mydir=mydir)
    for fn in tempfiles:
        os.unlink(fn)

    if errstr:
        logging.error('ERROR: %s' % errstr)
        return -1
    print(imgtype)
    return 0

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-i', '--infile',
                      dest='infile',
                      help='input image FILE', metavar='FILE')
    parser.add_option('-u', '--uncompressed-outfile',
                      dest='uncompressed_outfile',
                      help='uncompressed temporary FILE', metavar='FILE',
                      default='')
    parser.add_option('-o', '--outfile',
                      dest='outfile',
                      help='output pnm image FILE', metavar='FILE')
    parser.add_option('-p', '--ppm',
                      action='store_true', dest='force_ppm',
                      help='convert the output to PPM')
    parser.add_option('-e', '--extension',
                      dest='extension', type='int',
                      help='FITS extension to read')
    parser.add_option('--mydir', help='Set directory to search for an-fitstopnm')
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose',
                      help='be chatty')

    (options, args) = parser.parse_args()

    if not options.infile:
        parser.error('required argument missing: infile')
    if not options.outfile:
        parser.error('required argument missing: outfile')

    # Find the path to this executable and use it to find other Astrometry.net
    # executables.
    dirs = []
    if options.mydir:
        dirs.append(options.mydir)
    if len(sys.argv):
        dirs.append(os.path.dirname(sys.argv[0]))
    # util/ -- useful when running from source directory
    dirs.append(os.path.dirname(__file__))
        
    global verbose
    verbose = options.verbose

    logformat = '%(message)s'
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=logformat)
    else:
        logging.basicConfig(level=logging.INFO, format=logformat)
    logging.raiseExceptions = False
        
    return convert_image(options.infile, options.outfile,
                         uncompressed=options.uncompressed_outfile,
                         force_ppm=options.force_ppm,
                         extension=options.extension,
                         mydir=dirs)

if __name__ == '__main__':
    sys.exit(main())
