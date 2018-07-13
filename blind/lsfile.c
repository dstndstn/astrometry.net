/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include "lsfile.h"
#include "ioutils.h"

char* get_next_line(FILE* fid) {
    char* line = read_string_terminated(fid, "\n", 1);
    if (!line) return line;
    // ignore comments...
    if (line[0] == '#') {
        free(line);
        return get_next_line(fid);
    }
    return line;
}

void free_all(pl* list) {
    int i,N;
    N = pl_size(list);
    for (i=0; i<N; i++) {
        dl* plist = (dl*)pl_get(list, i);
        if (plist) {
            dl_free(plist);
        }
    }
    pl_free(list);
}

void ls_file_free(pl* list) {
    free_all(list);
}

int read_ls_file_header(FILE* fid) {
    char* line;
    int numfields;

    line = get_next_line(fid);
    if (!line) return -1;

    // first line: numfields
    if (sscanf(line, "NumFields=%i\n", &numfields) != 1) {
        fprintf(stderr, "parse error: numfields\n");
        free(line);
        return -1;
    }
    free(line);
    return numfields;
}

dl* read_ls_file_field(FILE* fid, int dimension) {
    dl* pointlist;
    int offset;
    int inc;
    char* line;
    int npoints;
    int i;

    // first element: number of entries
    line = get_next_line(fid);
    if (!line) {
        fprintf(stderr, "premature end of file.\n");
        return NULL;
    }
    if (sscanf(line, "%i%n", &npoints, &offset) < 1) {
        fprintf(stderr, "parse error: npoints\n");
        free(line);
        return NULL;
    }

    pointlist = dl_new(32);

    for (i=0; i<(npoints * dimension); i++) {
        double val;
        if (sscanf(line+offset, ",%lf%n", &val, &inc) < 1) {
            fprintf(stderr, "parse error: point %i\n", i);
            dl_free(pointlist);
            free(line);
            return NULL;
        }
        dl_append(pointlist, val);
        offset += inc;
    }
    free(line);
    return pointlist;
}

pl* read_ls_file(FILE* fid, int dimension) {
    int j;
    int numfields;
    pl* list;

    numfields = read_ls_file_header(fid);
    if (numfields == -1)
        return NULL;

    list = pl_new(256);

    for (j=0; j<numfields; j++) {
        dl* pointlist;

        pointlist = read_ls_file_field(fid, dimension);

        if (!pointlist) {
            free_all(list);
            return NULL;
        }
        pl_append(list, pointlist);

    }
    return list;
}

