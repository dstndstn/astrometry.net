/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "fitstable.h"
#include "fitsioutils.h"
#include "errors.h"
#include "ioutils.h"
#include "log.h"

int merge_index(quadfile_t* quad, codetree_t* code, startree_t* star,
                const char* indexfn) {
    FILE* fout;
    fitstable_t* tag = NULL;

    fout = fopen(indexfn, "wb");
    if (!fout) {
        SYSERROR("Failed to open output file");
        return -1;
    }

    if (quadfile_write_header_to(quad, fout)) {
        ERROR("Failed to write quadfile header to index file %s", indexfn);
        return -1;
    }
    if (quadfile_write_all_quads_to(quad, fout)) {
        ERROR("Failed to write quads to index file %s", indexfn);
        return -1;
    }
    if (fits_pad_file(fout)) {
        ERROR("Failed to pad index file %s", indexfn);
        return -1;
    }

    if (codetree_append_to(code, fout)) {
        ERROR("Failed to write code kdtree to index file %s", indexfn);
        return -1;
    }
    if (fits_pad_file(fout)) {
        ERROR("Failed to pad index file %s", indexfn);
        return -1;
    }

    if (startree_append_to(star, fout)) {
        ERROR("Failed to write star kdtree to index file %s", indexfn);
        return -1;
    }
    if (fits_pad_file(fout)) {
        ERROR("Failed to pad index file %s", indexfn);
        return -1;
    }

    if (startree_has_tagalong(star))
        tag = startree_get_tagalong(star);
    if (tag) {
        if (fitstable_append_to(tag, fout)) {
            ERROR("Failed to write star kdtree tag-along data to index file %s", indexfn);
            return -1;
        }
        if (fits_pad_file(fout)) {
            ERROR("Failed to pad index file %s", indexfn);
            return -1;
        }
    }

    if (fclose(fout)) {
        SYSERROR("Failed to close index file %s", indexfn);
        return -1;
    }
    return 0;
}

int merge_index_open_files(const char* quadfn, const char* ckdtfn, const char* skdtfn,
                           quadfile_t** quad, codetree_t** code, startree_t** star) {
    logmsg("Reading code tree from %s ...\n", ckdtfn);
    *code = codetree_open(ckdtfn);
    if (!*code) {
        ERROR("Failed to read code kdtree from %s", ckdtfn);
        return -1;
    }
    logmsg("Ok.\n");

    logmsg("Reading star tree from %s ...\n", skdtfn);
    *star = startree_open(skdtfn);
    if (!*star) {
        ERROR("Failed to read star kdtree from %s", skdtfn);
        return -1;
    }
    logmsg("Ok.\n");

    logmsg("Reading quads from %s ...\n", quadfn);
    *quad = quadfile_open(quadfn);
    if (!*quad) {
        ERROR("Failed to read quads from %s", quadfn);
        return -1;
    }
    logmsg("Ok.\n");
    return 0;
}

int merge_index_files(const char* quadfn, const char* ckdtfn, const char* skdtfn,
                      const char* indexfn) {
    quadfile_t* quad = NULL;
    codetree_t* code = NULL;
    startree_t* star = NULL;
    int rtn;

    if (merge_index_open_files(quadfn, ckdtfn, skdtfn, &quad, &code, &star)) {
        rtn = -1;
        goto cleanup;
    }

    logmsg("Writing index to %s ...\n", indexfn);
    rtn = merge_index(quad, code, star, indexfn);

 cleanup:
    if (code)
        codetree_close(code);
    if (star)
        startree_close(star);
    if (quad)
        quadfile_close(quad);

    return rtn;
}

