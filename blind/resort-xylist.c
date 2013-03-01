/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

// DEBUG
#include <sys/mman.h>

#include "qfits.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "an-bool.h"
#include "fitstable.h"
#include "errors.h"
#include "log.h"

int resort_xylist(const char* infn, const char* outfn,
                  const char* fluxcol, const char* backcol,
                  anbool ascending) {
	FILE* fin = NULL;
	FILE* fout = NULL;
    double *flux = NULL, *back = NULL;
    int *perm1 = NULL, *perm2 = NULL;
    anbool *used = NULL;
    int start, size, nextens, ext;
    int (*compare)(const void*, const void*);
    fitstable_t* tab = NULL;

    if (ascending)
        compare = compare_doubles_asc;
    else
        compare = compare_doubles_desc;

    if (!fluxcol)
        fluxcol = "FLUX";
    if (!backcol)
        backcol = "BACKGROUND";

    fin = fopen(infn, "rb");
    if (!fin) {
        SYSERROR("Failed to open input file %s", infn);
        return -1;
    }

    fout = fopen(outfn, "wb");
    if (!fout) {
        SYSERROR("Failed to open output file %s", outfn);
        goto bailout;
    }

	// copy the main header exactly.
	if (qfits_get_hdrinfo(infn, 0, &start, &size)) {
		ERROR("Failed to read primary FITS header.");
        goto bailout;
	}

    if (pipe_file_offset(fin, start, size, fout)) {
        ERROR("Failed to copy primary FITS header.");
        goto bailout;
    }

	nextens = qfits_query_n_ext(infn);

    tab = fitstable_open(infn);
    if (!tab) {
        ERROR("Failed to open FITS table in file %s", infn);
        goto bailout;
    }

	for (ext=1; ext<=nextens; ext++) {
		int hdrstart, hdrsize, datsize, datstart;
		int i, N;
        int rowsize;

		if (qfits_get_hdrinfo(infn, ext, &hdrstart, &hdrsize) ||
			qfits_get_datinfo(infn, ext, &datstart, &datsize)) {
			ERROR("Couldn't get extension %i header or data extent.", ext);
            goto bailout;
        }
		if (!qfits_is_table(infn, ext)) {
            ERROR("Extention %i isn't a table. Skipping", ext);
			continue;
		}
        // Copy the header as-is.
        if (pipe_file_offset(fin, hdrstart, hdrsize, fout)) {
            ERROR("Failed to copy the header of extension %i", ext);
			goto bailout;
        }

        if (fitstable_read_extension(tab, ext)) {
            ERROR("Failed to read FITS table from extension %i", ext);
            goto bailout;
        }
        rowsize = fitstable_row_size(tab);

        // read FLUX column as doubles.
        flux = fitstable_read_column(tab, fluxcol, TFITS_BIN_TYPE_D);
        if (!flux) {
            ERROR("Failed to read FLUX column from extension %i", ext);
            goto bailout;
        }
        // BACKGROUND
        back = fitstable_read_column(tab, backcol, TFITS_BIN_TYPE_D);
        if (!back) {
            ERROR("Failed to read BACKGROUND column from extension %i", ext);
            goto bailout;
        }

		debug("First 10 rows of input table:\n");
		for (i=0; i<10; i++)
			debug("flux %g, background %g\n", flux[i], back[i]);

        N = fitstable_nrows(tab);

        // set back = flux + back (ie, non-background-subtracted flux)
		for (i=0; i<N; i++)
            back[i] += flux[i];

        // Sort by flux...
		perm1 = permuted_sort(flux, sizeof(double), compare, NULL, N);

        // Sort by non-background-subtracted flux...
		perm2 = permuted_sort(back, sizeof(double), compare, NULL, N);

        used = malloc(N * sizeof(anbool));
        memset(used, 0, N * sizeof(anbool));

		// Check sort...
        for (i=0; i<N-1; i++) {
			if (ascending) {
				assert(flux[perm1[i]] <= flux[perm1[i+1]]);
				assert(back[perm2[i]] <= back[perm2[i+1]]);
			} else {
				assert(flux[perm1[i]] >= flux[perm1[i+1]]);
				assert(back[perm2[i]] >= back[perm2[i+1]]);
			}
		}

        for (i=0; i<N; i++) {
            int j;
            int inds[] = { perm1[i], perm2[i] };
            for (j=0; j<2; j++) {
                int index = inds[j];
				assert(index < N);
                if (used[index])
                    continue;
                used[index] = TRUE;
				debug("adding index %i: %s %g\n", index, j==0 ? "flux" : "bgsub", j==0 ? flux[index] : back[index]);
                if (pipe_file_offset(fin, datstart + index * rowsize, rowsize, fout)) {
                    ERROR("Failed to copy row %i", index);
                    goto bailout;
                }
            }
        }

        for (i=0; i<N; i++)
			assert(used[i]);

		if (fits_pad_file(fout)) {
			ERROR("Failed to add padding to extension %i", ext);
            goto bailout;
		}

        free(flux);
        flux = NULL;
        free(back);
        back = NULL;
        free(perm1);
        perm1 = NULL;
        free(perm2);
        perm2 = NULL;
        free(used);
        used = NULL;
    }

    fitstable_close(tab);
    tab = NULL;

	if (fclose(fout)) {
		SYSERROR("Failed to close output file %s", outfn);
        return -1;
    }
	fclose(fin);
    return 0;

 bailout:
    if (tab)
        fitstable_close(tab);
    if (fout)
        fclose(fout);
    if (fin)
        fclose(fin);
    free(flux);
    free(back);
    free(perm1);
    free(perm2);
    free(used);
	return -1;
}


