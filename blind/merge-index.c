/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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
#include <math.h>
#include <string.h>

#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "fitsioutils.h"
#include "errors.h"
#include "ioutils.h"
#include "log.h"

int merge_index(const char* quadfn, const char* ckdtfn, const char* skdtfn,
				const char* indexfn) {
    quadfile* quad;
	codetree* code;
	startree_t* star;
    FILE* fout;
	FILE* fin;
	int i;
    int N;

    fout = fopen(indexfn, "wb");
    if (!fout) {
        SYSERROR("Failed to open output file");
        return -1;
    }
	logmsg("Reading code tree from %s ...\n", ckdtfn);
	code = codetree_open(ckdtfn);
	if (!code) {
		ERROR("Failed to read code kdtree from %s", ckdtfn);
		return -1;
	}
    logmsg("Ok.\n");
    codetree_close(code);

	logmsg("Reading star tree from %s ...\n", skdtfn);
	star = startree_open(skdtfn);
	if (!star) {
		ERROR("Failed to read star kdtree from %s", skdtfn);
		return -1;
	}
    logmsg("Ok.\n");
    startree_close(star);

	logmsg("Reading quads from %s ...\n", quadfn);
	quad = quadfile_open(quadfn);
	if (!quad) {
		ERROR("Failed to read quads from %s", quadfn);
		return -1;
	}
    logmsg("Ok.\n");
    quadfile_close(quad);

    {
        const char* infiles[] = { quadfn, ckdtfn, skdtfn };
        int j;

        for (j=0; j<3; j++) {
            const char* fn = infiles[j];
            logmsg("Copying %s...\n", fn);

            fin = fopen(fn, "rb");
            if (!fin) {
                SYSERROR("Failed to open input file");
                return -1;
            }
            N = qfits_query_n_ext(fn);
            for (i=0; i<=N; i++) {
                int hdrstart, hdrlen, datastart, datalen;
                // skip the primary headers of code and star trees.
                if (j != 0 && i == 0)
                    continue;
                if (qfits_get_hdrinfo(fn, i, &hdrstart,  &hdrlen ) ||
                    qfits_get_datinfo(fn, i, &datastart, &datalen)) {
					ERROR("Error getting extents of extension %i", i);
                    return -1;
                }
                if (pipe_file_offset(fin, hdrstart,  hdrlen,  fout) ||
                    pipe_file_offset(fin, datastart, datalen, fout)) {
                    ERROR("Error writing extension %i", i);
                }
            }
            fclose(fin);
        }
    }
    if (fclose(fout)) {
        SYSERROR("Failed to close output file %s", indexfn);
        return -1;
    }
	return 0;
}
