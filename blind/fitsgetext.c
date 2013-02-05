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
#include <errno.h>
#include <string.h>

#include "an-bool.h"
#include "anqfits.h"
#include "bl.h"
#include "ioutils.h"

char* OPTIONS = "he:i:o:baDHM";

void printHelp(char* progname) {
  fprintf(stderr, "%s    -i <input-file>\n"
	  "      -o <output-file>\n"
	  "      [-a]: write out ALL extensions; the output filename should be\n"
	  "            a \"sprintf\" pattern such as  \"extension-%%04i\".\n"
	  "      [-b]: print sizes and offsets in FITS blocks (of 2880 bytes)\n"
	  "      [-M]: print sizes in megabytes (using floor(), not round()!)\n"
	  "      [-D]: data blocks only\n"
	  "      [-H]: header blocks only\n"
	  "      -e <extension-number> ...\n\n",
	  progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
  int argchar;

  char* infn = NULL;
  char* outfn = NULL;
  bool tostdout = FALSE;
  FILE* fin = NULL;
  FILE* fout = NULL;
  il* exts;
  int i;
  char* progname = argv[0];
  bool inblocks = FALSE;
  bool inmegs = FALSE;
  int allexts = 0;
  int Next = -1;
  bool dataonly = FALSE;
  bool headeronly = FALSE;

  exts = il_new(16);

  while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
    switch (argchar) {
    case 'D':
      dataonly = TRUE;
      break;
    case 'H':
      headeronly = TRUE;
      break;
    case 'a':
      allexts = 1;
      break;
    case 'b':
      inblocks = TRUE;
      break;
    case 'M':
      inmegs = TRUE;
      break;
    case 'e':
      il_append(exts, atoi(optarg));
      break;
    case 'i':
      infn = optarg;
      break;
    case 'o':
      outfn = optarg;
      break;
    case '?':
    case 'h':
      printHelp(progname);
      return 0;
    default:
      return -1;
    }

  if (headeronly && dataonly) {
    fprintf(stderr, "Can't write data blocks only AND header blocks only!\n");
    exit(-1);
  }

  if (inblocks && inmegs) {
    fprintf(stderr, "Can't write sizes in FITS blocks and megabytes.\n");
    exit(-1);
  }

  if (infn) {
    Next = qfits_query_n_ext(infn);
    if (Next == -1) {
      fprintf(stderr, "Couldn't determine how many extensions are in file %s.\n", infn);
      exit(-1);
    } else {
      fprintf(stderr, "File %s contains %i FITS extensions.\n", infn, Next);
    }
  }

  if (infn && !outfn) {
    for (i=0; i<=Next; i++) {
      int hdrstart, hdrlen, datastart, datalen;
      if (qfits_get_hdrinfo(infn, i, &hdrstart,  &hdrlen ) ||
	  qfits_get_datinfo(infn, i, &datastart, &datalen)) {
	fprintf(stderr, "Error getting extents of extension %i.\n", i);
	exit(-1);
      }
      if (inblocks) {
	fprintf(stderr, "Extension %i : header start %i , length %i ; data start %i , length %i blocks.\n",
		i, hdrstart/FITS_BLOCK_SIZE, hdrlen/FITS_BLOCK_SIZE,
		datastart/FITS_BLOCK_SIZE, datalen/FITS_BLOCK_SIZE);
      } else if (inmegs) {
	int meg = 1024*1024;
	fprintf(stderr, "Extension %i : header start %i , length %i ; data start %i , length %i megabytes.\n",
		i, hdrstart/meg, hdrlen/meg,
		datastart/meg, datalen/meg);
      } else {
	fprintf(stderr, "Extension %i : header start %i , length %i ; data start %i , length %i .\n",
		i, hdrstart, hdrlen, datastart, datalen);
      }
    }
    exit(0);
  }

  if (!infn || !outfn || !(il_size(exts) || allexts)) {
    printHelp(progname);
    exit(-1);
  }

  if (!strcmp(outfn, "-")) {
    tostdout = TRUE;
    if (allexts) {
      fprintf(stderr, "Specify all extensions (-a) and outputting to stdout (-o -) doesn't make much sense...\n");
      exit(-1);
    }
  }

  if (infn) {
    fin = fopen(infn, "rb");
    if (!fin) {
      fprintf(stderr, "Failed to open input file %s: %s\n", infn, strerror(errno));
      exit(-1);
    }
  }

  if (tostdout)
    fout = stdout;
  else {
    if (allexts)
      for (i=0; i<=Next; i++)
	il_append(exts, i);
    else {
      // open the (single) output file.
      fout = fopen(outfn, "wb");
      if (!fout) {
	fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
	exit(-1);
      }
    }
  }

  for (i=0; i<il_size(exts); i++) {
    int hdrstart, hdrlen, datastart, datalen;
    int ext = il_get(exts, i);

    if (allexts) {
      char fn[256];
      snprintf(fn, sizeof(fn), outfn, ext);
      fout = fopen(fn, "wb");
      if (!fout) {
	fprintf(stderr, "Failed to open output file %s: %s\n", fn, strerror(errno));
	exit(-1);
      }
    }

    if (qfits_get_hdrinfo(infn, ext, &hdrstart,  &hdrlen ) ||
	qfits_get_datinfo(infn, ext, &datastart, &datalen)) {
      fprintf(stderr, "Error getting extents of extension %i.\n", ext);
      exit(-1);
    }
    if (inblocks)
      fprintf(stderr, "Writing extension %i: header start %i, length %i, data start %i, length %i blocks.\n",
	      ext, hdrstart/FITS_BLOCK_SIZE, hdrlen/FITS_BLOCK_SIZE, datastart/FITS_BLOCK_SIZE, datalen/FITS_BLOCK_SIZE);
    else if (inmegs) {
      int meg = 1024*1024;
      fprintf(stderr, "Writing extension %i: header start %i, length %i, data start %i, length %i megabytes.\n",
	      ext, hdrstart/meg, hdrlen/meg, datastart/meg, datalen/meg);
    } else
      fprintf(stderr, "Writing extension %i: header start %i, length %i, data start %i, length %i.\n",
	      ext, hdrstart, hdrlen, datastart, datalen);

    if (hdrlen && !dataonly) {
      if (pipe_file_offset(fin, hdrstart, hdrlen, fout)) {
	fprintf(stderr, "Failed to write header for extension %i: %s\n", ext, strerror(errno));
	exit(-1);
      }
    }
    if (datalen && !headeronly) {
      if (pipe_file_offset(fin, datastart, datalen, fout)) {
	fprintf(stderr, "Failed to write data for extension %i: %s\n", ext, strerror(errno));
	exit(-1);
      }
    }

    if (allexts)
      if (fclose(fout)) {
	fprintf(stderr, "Failed to close output file: %s\n", strerror(errno));
	exit(-1);
      }
  }

  fclose(fin);
  if (!allexts && !tostdout)
    fclose(fout);
  il_free(exts);
  return 0;
}
