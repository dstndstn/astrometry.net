/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <assert.h>

#include "fitsfile.h"
#include "fitsioutils.h"
#include "errors.h"

int fitsfile_pad_with(FILE* fid, char pad) {
    return fits_pad_file_with(fid, pad);
}

int fitsfile_write_header(FILE* fid, qfits_header* hdr,
                          off_t* start_offset, off_t* end_offset,
                          int ext, const char* fn) {
    assert(fid);
    assert(hdr);
    assert(end_offset);
    // Pad out to FITS boundary.
    fits_pad_file(fid);
    *start_offset = ftello(fid);
    if (qfits_header_dump(hdr, fid)) {
        if (ext == -1)
            ERROR("Failed to write FITS extension header to file %s", fn);
        else
            ERROR("Failed to write header for extension %i to file %s", ext, fn);
        return -1;
    }
    *end_offset = ftello(fid);
    return 0;
}

int fitsfile_fix_header(FILE* fid, qfits_header* hdr,
                        off_t* start_offset, off_t* end_offset,
                        int ext, const char* fn) {
    off_t offset;
    off_t new_header_end;
    off_t old_header_end;

    offset = ftello(fid);
    fseeko(fid, *start_offset, SEEK_SET);
    old_header_end = *end_offset;

    if (fitsfile_write_header(fid, hdr, start_offset, end_offset, ext, fn))
        return -1;
    new_header_end = *end_offset;

    if (new_header_end != old_header_end) {
        if (ext == -1)
            ERROR("Error: FITS header for file %s, used to end at %lu, "
                  "now it ends at %lu.  Data loss is likely!", fn,
                  (unsigned long)old_header_end, (unsigned long)new_header_end);
        else
            ERROR("Error: FITS header for file %s, ext %i, used to end at %lu, "
                  "now it ends at %lu.  Data loss is likely!", fn, ext,
                  (unsigned long)old_header_end, (unsigned long)new_header_end);
        return -1;
    }
    fseeko(fid, offset, SEEK_SET);
    // Pad out to FITS boundary.
    fits_pad_file(fid);
    return 0;
}

int fitsfile_write_primary_header(FILE* fid, qfits_header* hdr,
                                  off_t* end_offset, const char* fn) {
    off_t zero = 0;
    return fitsfile_write_header(fid, hdr, &zero, end_offset, 0, fn);
}

int fitsfile_fix_primary_header(FILE* fid, qfits_header* hdr,
                                off_t* end_offset, const char* fn) {
    off_t zero = 0;
    return fitsfile_fix_header(fid, hdr, &zero, end_offset, 0, fn);
}


