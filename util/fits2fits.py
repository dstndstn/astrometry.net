#! /usr/bin/env python
import os
import sys
import re

if __name__ == '__main__':
    try:
        import pyfits
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

import pyfits


def fits2fits(infile, outfile, verbose):
    """
    Returns: error string, or None on success.
    """
    # Read input file.
    fitsin = pyfits.open(infile)
    # Print out info about input file.
    if verbose:
        fitsin.info()
    # Create output list of HDUs
    fitsout = pyfits.HDUList()

    for i, hdu in enumerate(fitsin):
        # verify() fails when a keywords contains invalid characters,
        # so go through the primary header and fix them by converting invalid
        # characters to '_'
        hdr = fitsin[i].header
        cards = hdr.ascardlist()
        # allowed charactors (FITS standard section 5.1.2.1)
        pat = re.compile(r'[^A-Z0-9_\-]')
        for c in cards.keys():
            # new keyword:
            cnew = pat.sub('_', c)
            if (c != cnew):
                if verbose:
                    print 'Replacing illegal keyword ', c, ' by ', cnew
                # add the new header card
                hdr.update(cnew, cards[c].value, cards[c].comment, after=c)
                # remove the old one.
                del hdr[c]

        # Fix input header
        fitsin[i].verify('fix')
        # Copy fixed input header to output
        fitsout.append(fitsin[i])

    # Describe output file we're about to write...
    if verbose:
        print 'Outputting:'
        fitsout.info()

    try:
        fitsout.writeto(outfile)
    except IOError:
        # File probably exists
        if verbose:
            print 'File %s appears to already exist; deleting!' % outfile
        os.unlink(outfile)
        fitsout.writeto(outfile)
    except VerifyError:
        return 'Verification of output file failed: your FITS file is probably too broken to automatically fix.';
    return None

if __name__ == '__main__':
    if (len(sys.argv) == 3):
        infile = sys.argv[1]
        outfile = sys.argv[2]
        verbose = False
    elif (len(sys.argv) == 4) and (sys.argv[1] == '--verbose'):
        verbose = True
        infile = sys.argv[2]
        outfile = sys.argv[3]
    else:
        print 'Usage: fits2fits.py [--verbose] input.fits output.fits'
        sys.exit()

    errstr = fits2fits(infile, outfile, verbose)
    if errstr:
        print errstr
        sys.exit(-1)
    sys.exit(0)
