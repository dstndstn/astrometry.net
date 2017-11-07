/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <assert.h>
#include "startree.h"
#include "kdtree.h"
#include "errors.h"
#include "log.h"
#include "starutil.h"
#include "an-bool.h"
#include "fitsioutils.h"
#include "boilerplate.h"
#include "fitstable.h"

anbool startree_has_tagalong_data(const fitstable_t* intab) {
    // don't include RA,Dec.
    return fitstable_n_fits_columns(intab) > 2;
}

int startree_write_tagalong_table(fitstable_t* intab, fitstable_t* outtab,
                                  const char* racol, const char* deccol,
                                  int* indices,
                                  anbool remove_radec_columns) {
    int i, R, NB, N;
    qfits_header* hdr;

    fitstable_clear_table(intab);
    fitstable_add_fits_columns_as_struct(intab);
    fitstable_copy_columns(intab, outtab);

    if (remove_radec_columns) {
        if (!racol)
            racol = "RA";
        if (!deccol)
            deccol = "DEC";
        fitstable_remove_column(outtab, racol);
        fitstable_remove_column(outtab, deccol);
    }
    fitstable_read_extension(intab, 1);
    hdr = fitstable_get_header(outtab);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_TAGALONG, "Extra data for stars", NULL);
    if (fitstable_write_header(outtab)) {
        ERROR("Failed to write tag-along data header");
        return -1;
    }
    N = fitstable_nrows(intab);
    R = fitstable_row_size(intab);

    if (indices) {
        if (!remove_radec_columns) {
            // row-by-row raw data copy; read whole tag-along array into memory
            char* data = malloc((size_t)N * (size_t)R);
            // FIXME -- could read row-by-row if the malloc fails.....
            if (!data) {
                ERROR("Failed to allocate enough memory to read full tag-along table");
                return -1;
            }
            printf("Reading tag-along table...\n");
            if (fitstable_read_nrows_data(intab, 0, N, data)) {
                ERROR("Failed to read tag-along table");
                free(data);
                return -1;
            }
            printf("Writing tag-along table...\n");
            for (i=0; i<N; i++) {
                if (fitstable_write_row_data(outtab, data + (size_t)indices[i]*(size_t)R)) {
                    ERROR("Failed to write a row of data");
                    free(data);
                    return -1;
                }
            }
            free(data);
            
        } else {
            if (fitstable_copy_rows_data(intab, indices, N, outtab)) {
                ERROR("Failed to copy tag-along table rows from input to output");
                return -1;
            }
        }
    } else {
        char* buf;
        
        NB = 1000;
        logverb("Input row size: %i, output row size: %i\n", R, fitstable_row_size(outtab));
        buf = malloc(NB * R);
	
        for (i=0; i<N; i+=NB) {
            int nr = NB;
            if (i+NB > N)
                nr = N - i;
            if (fitstable_read_structs(intab, buf, R, i, nr)) {
                ERROR("Failed to read tag-along data from catalog");
                return -1;
            }
            if (fitstable_write_structs(outtab, buf, R, nr)) {
                ERROR("Failed to write tag-along data");
                return -1;
            }
        }
        free(buf);
    }
    if (fitstable_fix_header(outtab)) {
        ERROR("Failed to fix tag-along data header");
        return -1;
    }
    return 0;
}

startree_t* startree_build(fitstable_t* intable,
                           const char* racol, const char* deccol,
                           // keep RA,Dec in the tag-along table?
                           //anbool keep_radec,
                           // KDT_DATA_*, KDT_TREE_*
                           int datatype, int treetype,
                           // KD_BUILD_*
                           int buildopts,
                           int Nleaf,
                           char** args, int argc) {
    double* ra = NULL;
    double* dec = NULL;
    double* xyz = NULL;
    int N;
    startree_t* starkd = NULL;
    int tt;
    int d;
    double low[3];
    double high[3];
    qfits_header* hdr;
    qfits_header* inhdr;
    int i;

    if (!racol)
        racol = "RA";
    if (!deccol)
        deccol = "DEC";
    if (!datatype)
        datatype = KDT_DATA_U32;
    if (!treetype)
        treetype = KDT_TREE_U32;
    if (!buildopts)
        buildopts = KD_BUILD_SPLIT;
    if (!Nleaf)
        Nleaf = 25;

    ra = fitstable_read_column(intable, racol, TFITS_BIN_TYPE_D);
    if (!ra) {
        ERROR("Failed to read RA from column %s", racol);
        goto bailout;
    }
    dec = fitstable_read_column(intable, deccol, TFITS_BIN_TYPE_D);
    if (!dec) {
        ERROR("Failed to read RA from column %s", racol);
        goto bailout;
    }
    printf("First RA,Dec: %g,%g\n", ra[0], dec[0]);
    N = fitstable_nrows(intable);
    xyz = malloc(N * 3 * sizeof(double));
    if (!xyz) {
        SYSERROR("Failed to malloc xyz array to build startree");
        goto bailout;
    }
    radecdeg2xyzarrmany(ra, dec, xyz, N);
    free(ra);
    ra = NULL;
    free(dec);
    dec = NULL;
    printf("First x,y,z: %g,%g,%g\n", xyz[0], xyz[1], xyz[2]);

    starkd = startree_new();
    if (!starkd) {
        ERROR("Failed to allocate startree");
        free(xyz);
        goto bailout;
    }
    tt = kdtree_kdtypes_to_treetype(KDT_EXT_DOUBLE, treetype, datatype);
    printf("Treetype: 0x%x\n", tt);
    starkd->tree = kdtree_new(N, 3, Nleaf);
    for (d=0; d<3; d++) {
        low[d] = -1.0;
        high[d] = 1.0;
    }
    kdtree_set_limits(starkd->tree, low, high);
    logverb("Building star kdtree...\n");
    starkd->tree = kdtree_build(starkd->tree, xyz, N, 3, Nleaf, tt, buildopts);
    if (!starkd->tree) {
        ERROR("Failed to build star kdtree");
        startree_close(starkd);
        starkd = NULL;
        free(xyz);
        goto bailout;
    }
    starkd->tree->name = strdup(STARTREE_NAME);

    printf("After kdtree_build:\n");
    kdtree_print(starkd->tree);
    {
        double* treed = kdtree_get_data(starkd->tree, 0);
        printf("First data elements in tree: %g,%g,%g\n", treed[0], treed[1], treed[2]);
    }

    inhdr = fitstable_get_primary_header(intable);
    hdr = startree_header(starkd);
    an_fits_copy_header(inhdr, hdr, "HEALPIX");
    an_fits_copy_header(inhdr, hdr, "HPNSIDE");
    an_fits_copy_header(inhdr, hdr, "ALLSKY");
    an_fits_copy_header(inhdr, hdr, "JITTER");
    an_fits_copy_header(inhdr, hdr, "CUTNSIDE");
    an_fits_copy_header(inhdr, hdr, "CUTMARG");
    an_fits_copy_header(inhdr, hdr, "CUTDEDUP");
    an_fits_copy_header(inhdr, hdr, "CUTNSWEP");
    //fits_copy_header(inhdr, hdr, "CUTBAND");
    //fits_copy_header(inhdr, hdr, "CUTMINMG");
    //fits_copy_header(inhdr, hdr, "CUTMAXMG");
    BOILERPLATE_ADD_FITS_HEADERS(hdr);
    qfits_header_add(hdr, "HISTORY", "This file was created by the command-line:", NULL, NULL);
    fits_add_args(hdr, args, argc);
    qfits_header_add(hdr, "HISTORY", "(end of command line)", NULL, NULL);
    qfits_header_add(hdr, "HISTORY", "** History entries copied from the input file:", NULL, NULL);
    fits_copy_all_headers(inhdr, hdr, "HISTORY");
    qfits_header_add(hdr, "HISTORY", "** End of history entries.", NULL, NULL);
    for (i=1;; i++) {
        char key[16];
        int n;
        sprintf(key, "SWEEP%i", i);
        n = qfits_header_getint(inhdr, key, -1);
        if (n == -1)
            break;
        an_fits_copy_header(inhdr, hdr, key);
    }

 bailout:
    if (ra)
        free(ra);
    if (dec)
        free(dec);
    // NOOO don't free xyz -- it belongs to the kdtree!
    //if (xyz)
    //free(xyz);
    return starkd;
}

