#! /usr/bin/env python
"""
Convert an image in a variety of formats into a pnm file

Copyright Keir Mierle 2007, Dustin Lang 2008
"""
import sys
import os
import os.path
import tempfile

if __name__ == '__main__':
    try:
        import astrometry
        from astrometry.util.shell import shell_escape
        from astrometry.util.filetype import filetype_short
    except ImportError:
        me = sys.argv[0]
        #print 'i am', me
        path = os.path.realpath(me)
        #print 'my real path is', path
        utildir = os.path.dirname(path)
        assert(os.path.basename(utildir) == 'util')
        andir = os.path.dirname(utildir)
        assert(os.path.basename(andir) == 'astrometry')
        rootdir = os.path.dirname(andir)
        #print 'adding path', rootdir
        sys.path += [rootdir]

from astrometry.util.shell import shell_escape
from astrometry.util.filetype import filetype_short

fitstype = 'FITS image data'
fitsext = 'fits'

pgmcmd = 'pgmtoppm rgbi:1/1/1 %s > %s'
pgmext = 'pgm'

imgcmds = {fitstype : (fitsext, 'an-fitstopnm -i %s > %s'),
           'JPEG image data'  : ('jpg',  'jpegtopnm %s > %s'),
           'PNG image data'   : ('png',  'pngtopnm %s > %s'),
           'GIF image data'   : ('gif',  'giftopnm %s > %s'),
           'Netpbm PPM'       : ('ppm',  'ppmtoppm < %s > %s'),
           'Netpbm PPM "rawbits" image data' : ('ppm',  'cp %s %s'),
           'Netpbm PGM'       : ('pgm',  pgmcmd),
           'Netpbm PGM "rawbits" image data' : ('pgm',  pgmcmd),
           'TIFF image data'  : ('tiff',  'tifftopnm %s > %s'),
           # RAW is not recognized by 'file'; we have to use 'dcraw',
           # but we still store this here for convenience.
           'raw'              : ('raw', 'dcraw -4 -c %s > %s'),
           }

compcmds = {'gzip compressed data'    : ('gz',  'gunzip -c %s > %s'),
            'bzip2 compressed data'   : ('bz2', 'bunzip2 -k -c %s > %s')
           }

# command to identify a RAW image.
raw_id_cmd = 'dcraw -i %s >/dev/null 2> /dev/null'

verbose = False

def log(*x):
    print >> sys.stderr, 'image2pnm:', ' '.join(x)

def logverb(*x):
    if verbose:
        print >> sys.stderr, 'image2pnm:', ' '.join(x)

def do_command(cmd):
    logverb('Running: "%s"' % cmd)
    if os.system(cmd) != 0:
        print >>sys.stderr, 'Command failed: %s' % cmd
        sys.exit(-1)

def get_cmd(types, cmds):
    ext=None
    cmd=None
    for t in types:
        (ext,cmd) = cmds.get(t, (None,None))
        if ext is not None:
            break
    return (ext,cmd)

def uncompress_file(infile, uncompressed, typeinfo=None, quiet=True):
    """
    infile: input filename.
    uncompressed: output filename.
    typeinfo: output from the 'file' command; if None we'll run 'file'.
    quiet: don't print any informational messages.

    Returns: comptype
    comptype: None if the file wasn't compressed, or 'gz' or 'bz2'.
    """
    if typeinfo is None:
        typeinfo = filetype_short(infile)

    (ext,cmd) = get_cmd(typeinfo, compcmds)
    if ext is None:
        logverb('File is not compressed: "%s"' % '/'.join(typeinfo))
        return None
    assert uncompressed != infile
    logverb('Compressed file (type %s), dumping to: "%s"' % (ext, uncompressed))
    do_command(cmd % (shell_escape(infile), shell_escape(uncompressed)))
    return ext

# Returns (extension, command, error)
def get_image_type(infile):
    typeinfo = filetype_short(infile)
    (ext,cmd) = get_cmd(typeinfo, imgcmds)
    if ext is not None:
        return (ext, cmd, None)
    rtn = os.system(raw_id_cmd % shell_escape(infile))
    if os.WIFEXITED(rtn) and (os.WEXITSTATUS(rtn) == 0):
        # it's a RAW image.
        (ext, cmd) = imgcmds['raw']
        return (ext, cmd, None)
    return (None, None, 'Unknown image type "%s"' % typeinfo)

def find_program(mydir, cmd):
    # pull off the executable name.
    parts = cmd.split(' ', 1)
    prog = parts[0]
    # try the same directory - this should work for installed
    # versions where image2pnm.py and an-fitstopnm are both in
    # "bin".
    p = os.path.join(mydir, prog)
    if os.path.exists(p):
        return ' '.join([p, parts[1]])
    log('path', p, 'does not exist.')
    return None

def image2pnm(infile, outfile, sanitized, force_ppm, no_fits2fits,
              mydir, quiet):
    """
    infile: input filename.
    outfile: output filename.
    sanitized: for FITS images, output filename of sanitized (fits2fits'd) image.
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

    # If it's a FITS file we want to filter it first because of the many
    # misbehaved FITS files. fits2fits is a sanitizer.
    if (ext == fitsext) and (not no_fits2fits):

        from fits2fits import fits2fits as fits2fits

        if not sanitized:
            (outfile_dir, outfile_file) = os.path.split(outfile)
            (f, sanitized) = tempfile.mkstemp('sanitized', outfile_file, outfile_dir)
            os.close(f)
            tempfiles.append(sanitized)
        else:
            assert sanitized != infile
        errstr = fits2fits(infile, sanitized, not quiet)
        if errstr:
            return (None, errstr)
        infile = sanitized

    if force_ppm:
        original_outfile = outfile
        (outfile_dir, outfile_file) = os.path.split(outfile)
        (f, outfile) = tempfile.mkstemp('pnm', outfile_file, outfile_dir)
        # we might rename this file later, so don't add it to the list of
        # tempfiles to delete until later...
        os.close(f)
        if not quiet:
            log('temporary output file: ', outfile)

    # Do the actual conversion
    if ext == fitsext and mydir:
        # an-fitstopnm: add explicit path...
        cmd = find_program(mydir, cmd)
        if cmd is None:
            return (None, 'Couldn\'t find the program "an-fitstopnm".')
    if quiet:
        cmd += ' 2>/dev/null'
    do_command(cmd % (shell_escape(infile), shell_escape(outfile)))

    if force_ppm:
        if ext == pgmext:
            # Convert to PPM.
            do_command(pgmcmd % (shell_escape(outfile), shell_escape(original_outfile)))
            tempfiles.append(outfile)
        else:
            os.rename(outfile, original_outfile)

    for fn in tempfiles:
        os.unlink(fn)

    # Success
    return (ext, None)
    

def convert_image(infile, outfile, uncompressed, sanitized,
                  force_ppm, no_fits2fits, mydir, quiet):
    tempfiles = []
    # if the caller didn't specify where to put the uncompressed file,
    # create a tempfile.
    if not uncompressed:
        (outfile_dir, outfile_file) = os.path.split(outfile)
        (f, uncompressed) = tempfile.mkstemp(None, 'uncomp', outdir)
        os.close(f)
        tempfiles.append(uncompressed)

    comp = uncompress_file(infile, uncompressed, quiet=quiet)
                           
    if comp:
        print 'compressed'
        print comp
        infile = uncompressed

    (imgtype, errstr) = image2pnm(infile, outfile, sanitized, force_ppm, no_fits2fits, mydir, quiet)

    for fn in tempfiles:
        os.unlink(fn)

    if errstr:
        log('ERROR: %s' % errstr)
        return -1
    print imgtype
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
    parser.add_option('-s', '--sanitized-fits-outfile',
                      dest='sanitized_outfile',
                      help='sanitized temporary fits FILE', metavar='FILE',
                      default='')
    parser.add_option('-o', '--outfile',
                      dest='outfile',
                      help='output pnm image FILE', metavar='FILE')
    parser.add_option('-p', '--ppm',
                      action='store_true', dest='force_ppm',
                      help='convert the output to PPM');
    parser.add_option('-2', '--no-fits2fits',
                      action='store_true', dest='no_fits2fits',
                      help="don't sanitize FITS files");
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet',
                      help='only print errors');
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose',
                      help='be chatty');

    (options, args) = parser.parse_args()

    if not options.infile:
        parser.error('required argument missing: infile')
    if not options.outfile:
        parser.error('required argument missing: outfile')

    # Find the path to this executable and use it to find other Astrometry.net
    # executables.
    if (len(sys.argv) > 0):
        mydir = os.path.dirname(sys.argv[0])

    global verbose
    verbose = options.verbose

    return convert_image(options.infile, options.outfile,
                         options.uncompressed_outfile,
                         options.sanitized_outfile,
                         options.force_ppm,
                         options.no_fits2fits,
                         mydir, options.quiet)

if __name__ == '__main__':
    sys.exit(main())
