#ifndef _INDEX_H
#define _INDEX_H

#include "idfile.h"
#include "quadfile.h"
#include "starkd.h"
#include "codekd.h"

/*
 * These routines handle loading and closing indexes which consist of several
 * files rather than a single large fire.
 */

#define DEFAULT_INDEX_JITTER 1.0  // arcsec

/**
 * A loaded index
 */
struct index_s {
	// Name of the current index.
	const char *indexname;

	// Unique id for this index.
	int indexid;
	int healpix;

	// The index
	codetree* codekd;
	idfile* id_file;
	quadfile* quads;
	startree* starkd;

	// Jitter in the index, in arcseconds.
	double index_jitter;

	// Does the index have the CIRCLE header - (codes live in the circle, not the box)?
	bool circle;
	// Does the index have the CX <= DX property
	bool cx_less_than_dx;

	// Limits of the size of quads in the index, in arcseconds.
	double index_scale_upper;
	double index_scale_lower;
};
typedef struct index_s index_t;

#define INDEX_USE_IDFILE 1
#define INDEX_ONLY_LOAD_METADATA 2
#define INDEX_ONLY_LOAD_SKDT 4
//#define INDEX_QUIET 8

/**
 * Load an index from disk
 *
 * Parameters:
 *
 *   indexname - the base name of the index files; for example, if the index is
 *               in files 'myindex.ckdt.fits' and 'myindex.skdt.fits', then
 *               the indexname is just 'myindex'
 *
 *   flags     - Either 0 or INDEX_USE_IDFILE. If INDEX_USE_IDFILE is
 *               specified, then the idfile will be loaded also.
 *               If INDEX_ONLY_LOAD_METADATA, then only metadata will be
 *               loaded.
 *
 * Returns:
 *
 *   A pointer to an index_t structure or NULL on error.
 *
 */
index_t* index_load(const char* indexname, int flags);

/**
 * Close an index and free associated data structures
 */
void index_close(index_t* index);

#endif // _INDEX_H
