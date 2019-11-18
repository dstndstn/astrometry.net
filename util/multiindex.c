/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <assert.h>

#include "multiindex.h"
#include "index.h"
#include "log.h"
#include "errors.h"

void multiindex_unload_starkd(multiindex_t* mi) {
    int i;
    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        ind->starkd = NULL;
    }
    if (mi->starkd) {
        startree_close(mi->starkd);
        mi->starkd = NULL;
    }
}

int multiindex_reload_starkd(multiindex_t* mi) {
    int i;
    assert(mi->fits);
    if (mi->starkd) {
        // It's already loaded
        return 0;
    }
    mi->starkd = startree_open_fits(mi->fits);
    if (!mi->starkd) {
        ERROR("Failed to open multi-index star kdtree");
        return -1;
    }
    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        ind->starkd = mi->starkd;
    }
    return 0;
}

void multiindex_unload(multiindex_t* mi) {
    int i;
    multiindex_unload_starkd(mi);
    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        index_unload(ind);
    }
}

int multiindex_reload(multiindex_t* mi) {
    int i;
    if (multiindex_reload_starkd(mi)) {
        return -1;
    }
    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        if (index_reload(ind)) {
            return -1;
        }
    }
    return 0;
}

// How many indices?
int multiindex_n(const multiindex_t* mi) {
    return pl_size(mi->inds);
}
// Get an index
index_t* multiindex_get(const multiindex_t* mi, int i) {
    return pl_get(mi->inds, i);
}

multiindex_t* multiindex_new(const char* skdtfn) {
    multiindex_t* mi = calloc(1, sizeof(multiindex_t));
    logverb("Reading star KD tree from %s...\n", skdtfn);
    mi->fits = anqfits_open(skdtfn);
    if (!mi->fits) {
        ERROR("Failed to open multiindex file \"%s\"", skdtfn);
        goto bailout;
    }
    mi->inds = pl_new(16);
    if (multiindex_reload_starkd(mi)) {
        ERROR("Failed to open multiindex star kd-tree \"%s\"", skdtfn);
        goto bailout;
    }
    return mi;
 bailout:
    multiindex_free(mi);
    return NULL;
}

int multiindex_add_index(multiindex_t* mi, const char* fn, int flags) {
    anqfits_t* fits;
    quadfile_t* quads = NULL;
    codetree_t* codes = NULL;
    index_t* ind = NULL;
    logverb("Reading index file \"%s\"...\n", fn);
    fits = anqfits_open(fn);
    if (!fits) {
        ERROR("Failed to open FITS file \"%s\"", fn);
        goto bailout;
    }
    logverb("Reading quads from file \"%s\"...\n", fn);
    quads = quadfile_open_fits(fits);
    if (!quads) {
        ERROR("Failed to read quads from file \"%s\"", fn);
        anqfits_close(fits);
        goto bailout;
    }
    logverb("Reading codes from file \"%s\"...\n", fn);
    codes = codetree_open_fits(fits);
    if (!codes) {
        ERROR("Failed to read quads from file \"%s\"", fn);
        quadfile_close(quads);
        anqfits_close(fits);
        goto bailout;
    }
	
    ind = index_build_from(codes, quads, mi->starkd);
    ind->fits = fits;
    if (!ind->indexname)
        ind->indexname = strdup(fn);
    // shouldn't be needed, but set anyway
    ind->indexfn = strdup(fn);

    pl_append(mi->inds, ind);

    if (flags & INDEX_ONLY_LOAD_METADATA) {
        // don't let it unload the starkd!
        ind->starkd = NULL;
        index_unload(ind);
        ind->starkd = mi->starkd;
    }

    return 0;
 bailout:
    if (quads)
        quadfile_close(quads);
    if (codes)
        codetree_close(codes);
    if (fits)
        anqfits_close(fits);
    return -1;
}


multiindex_t* multiindex_open(const char* skdtfn, const sl* indfns,
                              int flags) {
    multiindex_t* mi = multiindex_new(skdtfn);
    if (!mi)
        return NULL;
    int i;
    for (i=0; i<sl_size(indfns); i++) {
        const char* fn = sl_get_const(indfns, i);
        if (multiindex_add_index(mi, fn, flags)) {
            goto bailout;
        }
    }
    if (flags & INDEX_ONLY_LOAD_METADATA) {
        multiindex_unload_starkd(mi);
    }
    return mi;
 bailout:
    multiindex_free(mi);
    return NULL;
}

void multiindex_close(multiindex_t* mi) {
    if (!mi)
        return;
    if (mi->starkd) {
        startree_close(mi->starkd);
        mi->starkd = NULL;
    }
    if (mi->inds) {
        int i;
        for (i=0; i<pl_size(mi->inds); i++) {
            index_t* ind = pl_get(mi->inds, i);
            ind->starkd = NULL;
            index_free(ind);
        }
        pl_free(mi->inds);
        mi->inds = NULL;
    }
    if (mi->fits) {
        anqfits_close(mi->fits);
        mi->fits = NULL;
    }
}

void multiindex_free(multiindex_t* mi) {
    multiindex_close(mi);
    free(mi);
}

