/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, Dustin Lang.

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
#include <stdio.h>

#include "qfits.h"
#include "ioutils.h"
#include "bl.h"
#include "log.h"
#include "errors.h"

char* OPTIONS = "hflk:b";

void printHelp(char* progname) {
	fprintf(stderr, "Usage:   %s [options] <input-file>\n"
            "   [-f]: keep first instance of each duplicate header card\n"
            "   [-l]: keep last  instance of each duplicate header card\n"
            "   [-b]: blank out the duplicate line instead of COMMENT-ing out\n"
            "   [-k <keyword>]: look at only this header keyword\n"
            "\n"
            "Finds and removes duplicate FITS header cards.  The default is just to find\n"
            "and print duplicates.  Use -f or -l to remove all but the first / last\n"
            "instance of duplicate cards.  The duplicates will be turned into COMMENTs\n"
            "(or blanked out if -b is included)\n"
            "so that the length of the header does not change and the replacement can be\n"
            "written in-place.\n"
            "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int argchar;
	char* infn = NULL;
    anbool modify = FALSE;
    anbool first = FALSE;
    anbool last = FALSE;
    anbool blankout = FALSE;
    char* keyword = NULL;
    int i, Next;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'h':
        case '?':
            printHelp(args[0]);
            exit(0);
            break;
        case 'k':
            keyword = optarg;
            break;
        case 'f':
            first = TRUE;
            break;
        case 'l':
            last = TRUE;
            break;
        case 'b':
            blankout = TRUE;
            break;
        }

    log_init(LOG_MSG);

    if (first && last) {
        printHelp(args[0]);
        logerr("You must specify only one of -f and -l\n");
        exit(-1);
    }

    modify = first | last;

    if (optind != argc-1) {
        printHelp(args[0]);
        exit(-1);
    }

    infn = args[optind];

    Next = qfits_query_n_ext(infn);
    
    for (i=0; i<=Next; i++) {
        qfits_header* hdr;
        int j, Ncards;
        sl* keylist;
        il* indlist;
        anbool modified = FALSE;

        hdr = qfits_header_readext(infn, i);
        Ncards = qfits_header_n(hdr);

        /*
          printf("Old header:\n");
          qfits_header_list(hdr, stdout);
          printf("\n\n");
        */

        keylist = sl_new(36);
        indlist = il_new(36);

        for (j=0; j<Ncards; j++) {
            char key[FITS_LINESZ+1];
            char val[FITS_LINESZ+1];
            char comment[FITS_LINESZ+1];
            char line[FITS_LINESZ+1];
            int indx;
            char newline[FITS_LINESZ+1];
            int setind;

            qfits_header_getitem(hdr, j, key, val, comment, line);
            if (keyword && !streq(key, keyword))
                continue;

            if (streq(key, "HISTORY") || streq(key, "COMMENT") || streq(key, "        "))
                continue;

            if (last)
                indx = sl_last_index_of(keylist, key);
            else
                indx = sl_index_of(keylist, key);

            sl_append(keylist, key);
            il_append(indlist, j);

            if (indx == -1)
                continue;

            // found a duplicate!
            logmsg("Found a duplicate key \"%s\" in extension %i.\n", key, i);
            if (!modify)
                continue;

            modified = TRUE;

            // replace a duplicate!
            memset(newline, ' ', FITS_LINESZ+1);
            if (first) {
                setind = j;
            } else { // last
                setind = il_get(indlist, indx);
                // keep the last instance, replace the previous card.
                qfits_header_getitem(hdr, setind, key, val, comment, line);
            }
            if (blankout) {
                newline[80] = '\0';
            } else {
                if (line[0]) {
                    snprintf(newline, FITS_LINESZ+1, "COMMENT %s", line);
                } else {
                    snprintf(newline, FITS_LINESZ+1, "COMMENT %s%s%s%s%s",
                             key, strlen(key) ? "=" : "",
                             val, strlen(val) ? " " : "",
                             comment);
                }
            }

            if (line[0]) {
                logmsg("  Old: %.80s\n", line);
            } else {
                logmsg("  Old: %s = %s / %s\n", key, val, comment);
            }
            logmsg("  New: %s\n", newline);

            // replace '\0'-termination.
            newline[strlen(newline)] = ' ';
            qfits_header_setitem(hdr, setind, NULL, NULL, NULL, newline);
        }

        /*
          printf("\n");
          printf("Writing new header:\n");
          qfits_header_list(hdr, stdout);
          printf("\n\n");
        */

        sl_free2(keylist);
        il_free(indlist);

        if (modified) {
            FILE* fid;
            int offset, len;
            fid = fopen(infn, "r+");
            if (!fid) {
                SYSERROR("Failed to open file \"%s\"", infn);
                exit(-1);
            }
            if (qfits_get_hdrinfo(infn, i, &offset, &len)) {
                ERROR("Failed to find offset of header for extension %i in file \"%s\"", i, infn);
                exit(-1);
            }
            if (fseek(fid, offset, SEEK_SET)) {
                SYSERROR("Failed to seek to offset %i in \"%s\" to write updated header.", offset, infn);
                exit(-1);
            }
            qfits_header_dump(hdr, fid);
            if (fclose(fid)) {
                SYSERROR("Failed to close file \"%s\" after updating header.\n", infn);
                exit(-1);
            }
        }

        qfits_header_destroy(hdr);
    }


    return 0;
}

