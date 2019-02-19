/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AN_MULTIINDEX_H
#define AN_MULTIINDEX_H

#include "astrometry/index.h"
#include "astrometry/bl.h"
#include "astrometry/starkd.h"

/**

 About unloading and reloading multiindexes:

 The multiindex object holds the star-kdtree, and the list of index
 files.
 
 The star-kdtree can be unloaded and reloaded.

 AFTER calling multiindex_unload_starkd(), you can then call
 index_unload() on individual index_t*s.

 To reload, FIRST call multiindex_reload_starkd(), THEN index_reload()
 on individual index_t*s.

 We use a bit of sneakiness:

 -- multiindex_unload_starkd() sets each of the index_t*'s starkd
 pointers NULL.  Then when index_unload() is called, it doesn't try to
 unload the starkd.

 -- likewise, for reloading, multiindex_reload_starkd() sets the
 index_t* starkd pointers, so upon index_reload(), it doesn't try to
 reload the starkd.

 -- we set index->fits to an anqfits_t for the file that contains the
 codekd and quadfile.  Upon index_reload, it doesn't try to load the
 starkd (if it did, it would try to read the star-kd from the wrong
 file), and uses the index->fits object to read the quadfile and
 codekd.

 */

typedef struct {
    pl* inds;
    startree_t* starkd;
    // for the starkd:
    anqfits_t* fits;
} multiindex_t;

/*
 * Opens a set of index files.
 *
 *   flags - If INDEX_ONLY_LOAD_METADATA, then only metadata will be
 *               loaded.
 */
multiindex_t* multiindex_open(const char* skdtfn, const sl* indfns,
                              int flags);

/*
 * Opens a single star-kdtree.
 */
multiindex_t* multiindex_new(const char* skdtfn);

/*
 * Adds an index files (quadfile and code-tree) to this multi-index.
 *
 *   flags - If INDEX_ONLY_LOAD_METADATA, then only metadata will be
 *               loaded.
 */
int multiindex_add_index(multiindex_t* mi, const char* indexfn,
                         int flags);

/* Unloads the shared star kdtree -- ie, closes mem-maps, etc.
 * None of the indices will be usable.
 */
void multiindex_unload_starkd(multiindex_t* mi);

/* Reloads a previously unloaded shared star kdtree.
 */
int multiindex_reload_starkd(multiindex_t* mi);

/* Calls multiindex_unload_starkd() and index_unload() on all
 contained indexes.*/
void multiindex_unload(multiindex_t* mi);

/* Calls multiindex_reload_starkd() and index_reload() on all
 contained indexes.*/
int multiindex_reload(multiindex_t* mi);

void multiindex_close(multiindex_t* mi);

// close and free
void multiindex_free(multiindex_t* mi);

// How many indices?
int multiindex_n(const multiindex_t* mi);
// Get an index
index_t* multiindex_get(const multiindex_t* mi, int i);

#endif
