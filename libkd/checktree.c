/*
  This file is part of libkd.
  Copyright 2006-2008 Dustin Lang and Keir Mierle.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "kdtree.h"
#include "kdtree_fits_io.h"

void printHelp(char* progname) {
	printf("\nUsage: %s\n"
		   "     [-d]: print data\n"
		   "        [-n]: also print data in its native format (as stored in the kdtree file).\n"
		   "     <tree-filename>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

const char* OPTIONS = "hdn";

int main(int argc, char** args) {
    int argchar;
	char* progname = args[0];
	kdtree_t* kd;
	char* fn;
	int printData = 0;
	int treeData = 0;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'h':
			printHelp(progname);
			exit(-1);
		case 'd':
			printData = 1;
			break;
		case 'n':
			treeData = 1;
			break;
		}

    if (argc - optind == 1) {
		fn = args[optind];
		optind++;
    } else {
		printHelp(progname);
		exit(-1);
	}

	printf("Reading kdtree from file %s ...\n", fn);
	kd = kdtree_fits_read(fn, NULL, NULL);

    printf("Tree name: \"%s\"\n", kd->name);

	printf("Treetype: 0x%x\n", kd->treetype);

	printf("Data type:     %s\n", kdtree_kdtype_to_string(kdtree_datatype(kd)));
	printf("Tree type:     %s\n", kdtree_kdtype_to_string(kdtree_treetype(kd)));
	printf("External type: %s\n", kdtree_kdtype_to_string(kdtree_exttype(kd)));

	printf("N data points:  %i\n", kd->ndata);
	printf("Dimensions:     %i\n", kd->ndim);
	printf("Nodes:          %i\n", kd->nnodes);
	printf("Leaf nodes:     %i\n", kd->nbottom);
	printf("Non-leaf nodes: %i\n", kd->ninterior);
	printf("Tree levels:    %i\n", kd->nlevels);

	printf("Legacy nodes: %s\n", (kd->nodes  ? "yes" : "no"));
	printf("LR array:     %s\n", (kd->lr     ? "yes" : "no"));
	printf("Perm array:   %s\n", (kd->perm   ? "yes" : "no"));
	printf("Bounding box: %s\n", (kd->bb.any ? "yes" : "no"));
	printf("Split plane:  %s\n", (kd->split.any ? "yes" : "no"));
	printf("Split dim:    %s\n", (kd->splitdim  ? "yes" : "no"));
	printf("Data:         %s\n", (kd->data.any  ? "yes" : "no"));

	if (kd->minval && kd->maxval) {
		int d;
		printf("Data ranges:\n");
		for (d=0; d<kd->ndim; d++)
			printf("  %i: [%g, %g]\n", d, kd->minval[d], kd->maxval[d]);
	}

	printf("Running kdtree_check...\n");
	if (kdtree_check(kd)) {
		printf("kdtree_check failed.\n");
		exit(-1);
	}

	if (printData) {
		int i, d;
		int dt = kdtree_datatype(kd);
		double data[kd->ndim];
		for (i=0; i<kd->ndata; i++) {
			int iarray;
			kdtree_copy_data_double(kd, i, 1, data);
			printf("data[%i] = %n(", i, &iarray);
			for (d=0; d<kd->ndim; d++)
				printf("%s%g", d?", ":"", data[d]);
			printf(")\n");

			if (treeData) {
				printf("%*s(", iarray, "");
				for (d=0; d<kd->ndim; d++)
					switch (dt) {
					case KDT_DATA_DOUBLE:
						printf("%s%g", (d?", ":""),
							   kd->data.d[kd->ndim * i + d]);
						break;
					case KDT_DATA_FLOAT:
						printf("%s%g", (d?", ":""),
							   kd->data.f[kd->ndim * i + d]);
						break;
					case KDT_DATA_U32:
						printf("%s%u", (d?", ":""),
							   kd->data.u[kd->ndim * i + d]);
						break;
					case KDT_DATA_U16:
						printf("%s%u", (d?", ":""),
							   (unsigned int)kd->data.s[kd->ndim * i + d]);
						break;
					}
				printf(")\n");
			}
		}
	}

	kdtree_fits_close(kd);

	return 0;
}
