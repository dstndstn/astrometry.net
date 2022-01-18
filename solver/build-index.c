/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "os-features.h"
#include "build-index.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"
#include "uniformize-catalog.h"
#include "startree.h"
#include "codetree.h"
#include "unpermute-quads.h"
#include "unpermute-stars.h"
#include "bl.h"
#include "ioutils.h"
#include "rdlist.h"
#include "kdtree.h"
#include "hpquads.h"
#include "sip.h"
#include "sip_qfits.h"
#include "codefile.h"
#include "codekd.h"
#include "merge-index.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "mathutil.h"

static void add_boilerplate(index_params_t* p, qfits_header* hdr) {
}

static int step_hpquads(index_params_t* p,
                        codefile_t** p_codes, quadfile_t** p_quads,
                        char** p_codefn, char** p_quadfn, 
                        startree_t* starkd, const char* skdtfn,
                        sl* tempfiles) {
    codefile_t* codes = NULL;
    quadfile_t* quads = NULL;
    char* quadfn = NULL;
    char* codefn = NULL;

    if (p->inmemory) {
        codes = codefile_open_in_memory();
        quads = quadfile_open_in_memory();
        if (hpquads(starkd, codes, quads, p->Nside,
                    p->qlo, p->qhi, p->dimquads, p->passes, p->Nreuse, p->Nloosen,
                    p->indexid, p->scanoccupied,
                    p->hpquads_sort_data, p->hpquads_sort_func, p->hpquads_sort_size,
                    p->args, p->argc)) {
            ERROR("hpquads failed");
            return -1;
        }
        if (!quadfile_nquads(quads)) {
            logmsg("Did not create any quads.  Perhaps your catalog does not have enough stars?\n");
            return -1;
        }
        if (quadfile_switch_to_reading(quads)) {
            ERROR("Failed to switch quadfile to read-mode");
            return -1;
        }
        if (codefile_switch_to_reading(codes)) {
            ERROR("Failed to switch codefile to read-mode");
            return -1;
        }

    } else {
        quadfn = create_temp_file("quad", p->tempdir);
        sl_append_nocopy(tempfiles, quadfn);
        codefn = create_temp_file("code", p->tempdir);
        sl_append_nocopy(tempfiles, codefn);

        if (hpquads_files(skdtfn, codefn, quadfn, p->Nside,
                          p->qlo, p->qhi, p->dimquads, p->passes, p->Nreuse, p->Nloosen,
                          p->indexid, p->scanoccupied, 
                          p->hpquads_sort_data, p->hpquads_sort_func, p->hpquads_sort_size,
                          p->args, p->argc)) {
            ERROR("hpquads failed");
            return -1;
        }
		
    }

    if (p_codes) *p_codes = codes;
    if (p_quads) *p_quads = quads;
    if (p_codefn) *p_codefn = codefn;
    if (p_quadfn) *p_quadfn = quadfn;
		
    return 0;
}

static int step_codetree(index_params_t* p,
                         codefile_t* codes, codetree_t** p_codekd,
                         const char* codefn, char** p_ckdtfn,
                         sl* tempfiles) {
    codetree_t* codekd = NULL;
    char* ckdtfn=NULL;

    if (p->inmemory) {
        logmsg("Building code kdtree from %i codes\n", codes->numcodes);
        logmsg("dim: %i\n", codefile_dimcodes(codes));
        codekd = codetree_build(codes, 0, 0, 0, 0, p->args, p->argc);
        if (!codekd) {
            ERROR("Failed to build code kdtree");
            return -1;
        }
        if (codefile_close(codes)) {
            ERROR("Failed to close codefile");
            return -1;
        }

    } else {
        ckdtfn = create_temp_file("ckdt", p->tempdir);
        sl_append_nocopy(tempfiles, ckdtfn);

        if (codetree_files(codefn, ckdtfn, 0, 0, 0, 0, p->args, p->argc)) {
            ERROR("codetree failed");
            return -1;
        }
    }
	
    if (p_codekd) *p_codekd = codekd;
    if (p_ckdtfn) *p_ckdtfn = ckdtfn;
    return 0;
}

static int step_unpermute_quads(index_params_t* p,
                                quadfile_t* quads2, codetree_t* codekd,
                                quadfile_t** p_quads3, codetree_t** p_codekd2,
                                const char* quad2fn, const char* ckdtfn,
                                char** p_quad3fn, char** p_ckdt2fn,
                                sl* tempfiles) {
    quadfile_t* quads3 = NULL;
    codetree_t* codekd2 = NULL;
    char* quad3fn=NULL;
    char* ckdt2fn=NULL;

    logmsg("Unpermute-quads...\n");
    if (p->inmemory) {
        quads3 = quadfile_open_in_memory();
        if (unpermute_quads(quads2, codekd, quads3, &codekd2, p->args, p->argc)) {
            ERROR("Failed to unpermute-quads");
            return -1;
        }
        // unpermute-quads makes a shallow copy of the tree, so don't just codetree_close(codekd)...
        free(codekd->tree->perm);
        free(codekd->tree);
        codekd->tree = NULL;
        codetree_close(codekd);

        if (quadfile_switch_to_reading(quads3)) {
            ERROR("Failed to switch quads3 to read-mode");
            return -1;
        }
        if (quadfile_close(quads2)) {
            ERROR("Failed to close quadfile quads2");
            return -1;
        }

    } else {
        ckdt2fn = create_temp_file("ckdt2", p->tempdir);
        sl_append_nocopy(tempfiles, ckdt2fn);
        quad3fn = create_temp_file("quad3", p->tempdir);
        sl_append_nocopy(tempfiles, quad3fn);
        logmsg("Unpermuting quads from %s and %s to %s and %s\n", quad2fn, ckdtfn, quad3fn, ckdt2fn);
        if (unpermute_quads_files(quad2fn, ckdtfn,
                                  quad3fn, ckdt2fn, p->args, p->argc)) {
            ERROR("Failed to unpermute-quads");
            return -1;
        }
    }

    if (p_quads3) *p_quads3 = quads3;
    if (p_codekd2) *p_codekd2 = codekd2;
    if (p_quad3fn) *p_quad3fn = quad3fn;
    if (p_ckdt2fn) *p_ckdt2fn = ckdt2fn;
    return 0;
}

static int step_merge_index(index_params_t* p,
                            codetree_t* codekd2, quadfile_t* quads3,
                            startree_t* starkd2,
                            index_t** p_index,
                            const char* ckdt2fn, const char* quad3fn,
                            const char* skdt2fn, const char* indexfn) {
    index_t* index = NULL;

    if (p->inmemory) {
        qfits_header* hdr;

        index = index_build_from(codekd2, quads3, starkd2);
        if (!index) {
            ERROR("Failed to create index from constituent parts");
            return -1;
        }
        hdr = quadfile_get_header(index->quads);
        if (hdr)
            add_boilerplate(p, hdr);

        /* When closing:
         kdtree_free(codekd2->tree);
         codekd2->tree = NULL;
         */
        *p_index = index;

    } else {
        quadfile_t* quad;
        codetree_t* code;
        startree_t* star;
        qfits_header* hdr;

        logmsg("Merging %s and %s and %s to %s\n", quad3fn, ckdt2fn, skdt2fn, indexfn);
        /*
         if (merge_index_files(quad3fn, ckdt2fn, skdt2fn, indexfn)) {
         ERROR("Failed to merge-index");
         return -1;
         }
         */
        if (merge_index_open_files(quad3fn, ckdt2fn, skdt2fn,
                                   &quad, &code, &star)) {
            ERROR("Failed to open index files for merging");
            return -1;
        }
        hdr = quadfile_get_header(quad);
        if (hdr)
            add_boilerplate(p, hdr);
        if (merge_index(quad, code, star, indexfn)) {
            ERROR("Failed to write merged index");
            return -1;
        }
        codetree_close(code);
        startree_close(star);
        quadfile_close(quad);
    }
    return 0;
}

static void step_delete_tempfiles(index_params_t* p, sl* tempfiles) {
    if (p->delete_tempfiles) {
        int i;
        for (i=0; i<sl_size(tempfiles); i++) {
            char* fn = sl_get(tempfiles, i);
            logverb("Deleting temp file %s\n", fn);
            if (unlink(fn))
                SYSERROR("Failed to delete temp file \"%s\"", fn);
        }
    }
}

int build_index_shared_skdt(const char* skdtfn,
                            startree_t* starkd, index_params_t* p,
                            index_t** p_index, const char* indexfn) {
    // assume we've got a final (ie, post-unpermute-stars) skdt
    // we use that skdt's stars to uniformize, along with a column pulled
    // from its tag-along data.  This yields a permutation array, which
    // we feed to hpquads to alter its idea of how to sort stars that could
    // be used to build quads.
    // Then we do unpermute-quads, skip unpermute-stars, and feed
    // the original skdt in to merge-index.
    // TODO - tweak the index-reading code to allow multiple indices
    // sharing an skdt in one FITS file.
    // --quads_X table
    // --header card in the .quads HDU saying what the name of its skdt is.
    double* sortdata = NULL;
    //int* uniperm = NULL;
    int rtn = -1;
    codefile_t* codes = NULL;
    quadfile_t* quads = NULL;
    char* quadfn=NULL;
    char* codefn=NULL;
    codetree_t* codekd = NULL;
    char* ckdtfn=NULL;

    startree_t* starkd2 = NULL;
    quadfile_t* quads2 = NULL;
    char* quad2fn=NULL;

    quadfile_t* quads3 = NULL;
    codetree_t* codekd2 = NULL;
    char* quad3fn=NULL;
    char* ckdt2fn=NULL;

    sl* tempfiles;

    if (!p->UNside)
        p->UNside = p->Nside;

    assert(p->Nside);

    if (p->inmemory && !p_index) {
        ERROR("If you set inmemory, you must set p_index");
        return -1;
    }
    if (!p->inmemory && !indexfn) {
        ERROR("If you set !inmemory, you must set indexfn");
        return -1;
    }
    assert(starkd->tree);
    assert(starkd->tree->perm == NULL);
    assert(p->sortcol);
    if (!p->sortcol) {
        ERROR("You must set the sort column\n");
        return -1;
    }

    tempfiles = sl_new(4);

    // OR, should we just strictly sort on the tag-along sortcol in hpquads?
    // Yes, probably.  Uniformization isn't going to change the ordering much
    // in deciding which stars to use in a quad -- they're all nearby.

    logverb("Grabbing tag-along column \"%s\" for sorting...\n", p->sortcol);
    sortdata = startree_get_data_column(starkd, p->sortcol, NULL, startree_N(starkd));
    if (!sortdata) {
        ERROR("Failed to find sort column data for sorting catalog");
        goto cleanup;
    }

    /*
     logverb("Uniformizing...\n");
     uniperm = uniformize_catalog_get_permutation(skdt, sortdata,
     p->bighp, p->bignside, p->margin,
     p->UNside, p->dedup, p->sweeps,
     p->args, p->argc);
     if (!uniperm) {
     ERROR("Failed to find uniformization permutation array");
     goto cleanup;
     }
     p->hpquads_sort = uniperm;
     p->hpquads_sortfunc = compare_ints_asc;
     */

    p->hpquads_sort_data = sortdata;
    p->hpquads_sort_func = (p->sortasc ? compare_doubles_asc : compare_doubles_desc);
    p->hpquads_sort_size = sizeof(double);

    // hpquads
    if (step_hpquads(p, &codes, &quads, &codefn, &quadfn,
                     starkd, skdtfn, tempfiles))
        return -1;

    // codetree
    if (step_codetree(p, codes, &codekd,
                      codefn, &ckdtfn, tempfiles))
        return -1;

    // no unpermute-stars...
    quads2 = quads;
    quad2fn = quadfn;
    starkd2 = starkd;

    // unpermute-quads...
    if (step_unpermute_quads(p, quads2, codekd, &quads3, &codekd2,
                             quad2fn, ckdtfn, &quad3fn, &ckdt2fn, tempfiles))
        return -1;

    // merge-index...
    if (step_merge_index(p, codekd2, quads3, starkd2, p_index,
                         ckdt2fn, quad3fn, skdtfn, indexfn))
        return -1;

    step_delete_tempfiles(p, tempfiles);

    sl_free2(tempfiles);

    rtn = 0;

 cleanup:
    //free(uniperm);
    free(sortdata);
    return rtn;
}

int build_index(fitstable_t* catalog, index_params_t* p,
                index_t** p_index, const char* indexfn) {

    fitstable_t* uniform;

    // star kdtree
    startree_t* starkd = NULL;
    fitstable_t* startag = NULL;

    // hpquads
    codefile_t* codes = NULL;
    quadfile_t* quads = NULL;

    // codetree
    codetree_t* codekd = NULL;

    // unpermute-stars
    startree_t* starkd2 = NULL;
    quadfile_t* quads2 = NULL;
    fitstable_t* startag2 = NULL;

    // unpermute-quads
    quadfile_t* quads3 = NULL;
    codetree_t* codekd2 = NULL;

    //index_t* index = NULL;

    sl* tempfiles;
    char* unifn=NULL;
    char* skdtfn=NULL;
    char* quadfn=NULL;
    char* codefn=NULL;
    char* ckdtfn=NULL;
    char* skdt2fn=NULL;
    char* quad2fn=NULL;
    char* quad3fn=NULL;
    char* ckdt2fn=NULL;

    if (!p->UNside)
        p->UNside = p->Nside;

    assert(p->Nside);

    if (p->inmemory && !p_index) {
        ERROR("If you set inmemory, you must set p_index");
        return -1;
    }
    if (!p->inmemory && !indexfn) {
        ERROR("If you set !inmemory, you must set indexfn");
        return -1;
    }

    tempfiles = sl_new(4);

    if (p->inmemory)
        uniform = fitstable_open_in_memory();
    else {
        unifn = create_temp_file("uniform", p->tempdir);
        sl_append_nocopy(tempfiles, unifn);
        uniform = fitstable_open_for_writing(unifn);
    }
    if (!uniform) {
        ERROR("Failed to open output table %s", unifn);
        return -1;
    }

    if (uniformize_catalog(catalog, uniform, p->racol, p->deccol,
                           p->sortcol, p->sortasc, p->brightcut,
                           p->bighp, p->bignside, p->margin,
                           p->UNside, p->dedup, p->sweeps, p->args, p->argc)) {
        return -1;
    }

    if (fitstable_fix_primary_header(uniform)) {
        ERROR("Failed to fix output table");
        return -1;
    }

    if (p->inmemory) {
        if (fitstable_switch_to_reading(uniform)) {
            ERROR("Failed to switch uniformized table to read-mode");
            return -1;
        }
    } else {
        if (fitstable_close(uniform)) {
            ERROR("Failed to close output table");
            return -1;
        }
    }
    fitstable_close(catalog);

    // startree
    if (!p->inmemory) {
        skdtfn = create_temp_file("skdt", p->tempdir);
        sl_append_nocopy(tempfiles, skdtfn);

        logverb("Reading uniformized catalog %s...\n", unifn);
        uniform = fitstable_open(unifn);
        if (!uniform) {
            ERROR("Failed to open uniformized catalog");
            return -1;
        }
    }

    // DEBUG -- print RA,Dec from uniform catalog.
    if (log_get_level() > LOG_VERB) {
        tfits_type dubl = fitscolumn_double_type();
        double* ra;
        double* dec;
        int i,N;
        ra = fitstable_read_column(uniform, p->racol, dubl);
        dec = fitstable_read_column(uniform, p->deccol, dubl);
        N = fitstable_nrows(uniform);
        logdebug("Checking %i columns of 'uniform' catalog\n", N);
        logdebug("  RA column: \"%s\"; Dec column: \"%s\"\n", p->racol, p->deccol);
        assert(ra && dec);
        for (i=0; i<N; i++)
            logdebug("  %i RA,Dec %g,%g\n", i, ra[i], dec[i]);
        free(ra);
        free(dec);
    }


    {
        int Nleaf = 25;
        int datatype = KDT_DATA_U32;
        int treetype = KDT_TREE_U32;
        int buildopts = KD_BUILD_SPLIT;

        logverb("Building star kdtree from %i stars\n", fitstable_nrows(uniform));
        starkd = startree_build(uniform, p->racol, p->deccol, datatype, treetype,
                                buildopts, Nleaf, p->args, p->argc);
        if (!starkd) {
            ERROR("Failed to create star kdtree");
            return -1;
        }

        if (p->jitter > 0.0) {
            startree_set_jitter(starkd, p->jitter);
        }

        if (!p->inmemory) {
            logverb("Writing star kdtree to %s\n", skdtfn);
            if (startree_write_to_file(starkd, skdtfn)) {
                ERROR("Failed to write star kdtree");
                return -1;
            }
            startree_close(starkd);
        }

        if (startree_has_tagalong_data(uniform)) {
            logverb("Adding star kdtree tag-along data...\n");
            if (p->inmemory) {
                startag = fitstable_open_in_memory();
            } else {
                startag = fitstable_open_for_appending(skdtfn);
                if (!startag) {
                    ERROR("Failed to re-open star kdtree file %s for appending", skdtfn);
                    return -1;
                }
            }
            if (startree_write_tagalong_table(uniform, startag, p->racol, p->deccol, NULL, p->drop_radec)) {
                ERROR("Failed to write tag-along table");
                return -1;
            }
            if (p->inmemory) {
                if (fitstable_switch_to_reading(startag)) {
                    ERROR("Failed to switch star tag-along data to read-mode");
                    return -1;
                }
                starkd->tagalong = startag;
            } else {
                if (fitstable_close(startag)) {
                    ERROR("Failed to close star kdtree tag-along data");
                    return -1;
                }
            }
        }
    }
    fitstable_close(uniform);

    // hpquads
    if (step_hpquads(p, &codes, &quads, &codefn, &quadfn, 
                     starkd, skdtfn,
                     tempfiles))
        return -1;

    // codetree
    if (step_codetree(p, codes, &codekd,
                      codefn, &ckdtfn, tempfiles))
        return -1;

    // unpermute-stars
    logmsg("Unpermute-stars...\n");
    if (p->inmemory) {
        quads2 = quadfile_open_in_memory();
        if (unpermute_stars(starkd, quads, &starkd2, quads2,
                            TRUE, FALSE, p->args, p->argc)) {
            ERROR("Failed to unpermute-stars");
            return -1;
        }
        if (quadfile_close(quads)) {
            ERROR("Failed to close in-memory quads");
            return -1;
        }
        if (quadfile_switch_to_reading(quads2)) {
            ERROR("Failed to switch quads2 to read-mode");
            return -1;
        }
        if (startag) {
            startag2 = fitstable_open_in_memory();
            startag2->table = fits_copy_table(startag->table);
            startag2->table->nr = 0;
            startag2->header = qfits_header_copy(startag->header);
            if (unpermute_stars_tagalong(starkd, startag2)) {
                ERROR("Failed to unpermute-stars tag-along data");
                return -1;
            }
            starkd2->tagalong = startag2;
        }

        // unpermute-stars makes a shallow copy of the tree, so don't just startree_close(starkd)...
        free(starkd->tree->perm);
        free(starkd->tree);
        starkd->tree = NULL;
        startree_close(starkd);

    } else {
        skdt2fn = create_temp_file("skdt2", p->tempdir);
        sl_append_nocopy(tempfiles, skdt2fn);
        quad2fn = create_temp_file("quad2", p->tempdir);
        sl_append_nocopy(tempfiles, quad2fn);

        logmsg("Unpermuting stars from %s and %s to %s and %s\n", skdtfn, quadfn, skdt2fn, quad2fn);
        if (unpermute_stars_files(skdtfn, quadfn, skdt2fn, quad2fn,
                                  TRUE, FALSE, p->args, p->argc)) {
            ERROR("Failed to unpermute-stars");
            return -1;
        }
    }

    // unpermute-quads
    if (step_unpermute_quads(p, quads2, codekd, &quads3, &codekd2,
                             quad2fn, ckdtfn, &quad3fn, &ckdt2fn, tempfiles))
        return -1;


    // index
    if (step_merge_index(p, codekd2, quads3, starkd2, p_index,
                         ckdt2fn, quad3fn, skdt2fn, indexfn))
        return -1;

    // FIXME -- close codekd2, quads3, starkd2?

    step_delete_tempfiles(p, tempfiles);

    sl_free2(tempfiles);
    return 0;
}


int build_index_files(const char* infn, int ext, const char* indexfn,
                      index_params_t* p) {
    fitstable_t* catalog;
    logmsg("Reading %s...\n", infn);
    if (ext)
        catalog = fitstable_open_extension_2(infn, ext);
    else
        catalog = fitstable_open(infn);
    if (!catalog) {
        ERROR("Couldn't read catalog %s", infn);
        return -1;
    }
    logmsg("Got %i stars\n", fitstable_nrows(catalog));

    if (p->inmemory) {
        index_t* index;
        if (build_index(catalog, p, &index, NULL)) {
            return -1;
        }
        logmsg("Writing to file %s\n", indexfn);
        if (merge_index(index->quads, index->codekd, index->starkd, indexfn)) {
            ERROR("Failed to write index file");
            return -1;
        }
        kdtree_free(index->codekd->tree);
        index->codekd->tree = NULL;
        index_close(index);

    } else {
        if (build_index(catalog, p, NULL, indexfn)) {
            return -1;
        }
    }

    return 0;
}

int build_index_shared_skdt_files(const char* starkdfn, const char* indexfn,
                                  index_params_t* p) {
    startree_t* skdt = NULL;

    logmsg("Reading %s...\n", starkdfn);
    skdt = startree_open(starkdfn);
    if (!skdt) {
        ERROR("Couldn't read star kdtree from \"%s\"", starkdfn);
        return -1;
    }
    logmsg("Got %i stars\n", startree_N(skdt));

    if (p->inmemory) {
        index_t* index;
        if (build_index_shared_skdt(starkdfn, skdt, p, &index, NULL)) {
            return -1;
        }
        logmsg("Writing to file %s\n", indexfn);
        if (merge_index(index->quads, index->codekd, index->starkd, indexfn)) {
            ERROR("Failed to write index file \"%s\"", indexfn);
            return -1;
        }
        // FIXME?  Why close codekd independently?
        kdtree_free(index->codekd->tree);
        index->codekd->tree = NULL;
        index_close(index);

    } else {
        if (build_index_shared_skdt(starkdfn, skdt, p, NULL, indexfn)) {
            return -1;
        }
    }
    return 0;
}


void build_index_defaults(index_params_t* p) {
    memset(p, 0, sizeof(index_params_t));
    p->sweeps = 10;
    p->racol = "RA";
    p->deccol = "DEC";
    p->drop_radec = TRUE;
    p->passes = 16;
    p->Nreuse = 8;
    p->Nloosen = 20;
    p->dimquads = 4;
    p->sortasc = TRUE;
    p->brightcut = -LARGE_VAL;
    // default to all-sky
    p->bighp = -1;
    //p->inmemory = TRUE;
    p->delete_tempfiles = TRUE;
    p->tempdir = "/tmp";
}

