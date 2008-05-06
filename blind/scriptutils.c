/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301 USA
 */

#include <string.h>
#include <libgen.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "scriptutils.h"
#include "ioutils.h"
#include "bl.h"
#include "errors.h"

int mkdir_p(const char* dirpath) {
    sl* tomake = sl_new(4);
    char* path = strdup(dirpath);
    while (!file_exists(path)) {
        char* dir;
        sl_push(tomake, path);
        dir = strdup(dirname(path));
        free(path);
        path = dir;
    }
    free(path);
    while (sl_size(tomake)) {
        char* path = sl_pop(tomake);
        if (mkdir(path, 0777)) {
            SYSERROR("Failed to mkdir(%s)", path);
            sl_free2(tomake);
            free(path);
            return -1;
        }
        free(path);
    }
    sl_free2(tomake);
    return 0;
}

char* shell_escape(const char* str) {
    char* escape = "|&;()<> \t\n\\'\"";
    int nescape = 0;
    int len = strlen(str);
    int i;
    char* result;
    int j;

    for (i=0; i<len; i++) {
        char* cp = strchr(escape, str[i]);
        if (!cp) continue;
        nescape++;
    }
    result = malloc(len + nescape + 1);
    for (i=0, j=0; i<len; i++, j++) {
        char* cp = strchr(escape, str[i]);
        if (!cp) {
            result[j] = str[i];
        } else {
            result[j] = '\\';
            j++;
            result[j] = str[i];
        }
    }
    assert(j == (len + nescape));
    result[j] = '\0';
    return result;
}

char* create_temp_file(char* fn, char* dir) {
    char* tempfile;
    int fid;
    asprintf_safe(&tempfile, "%s/tmp.%s.XXXXXX", dir, fn);
    fid = mkstemp(tempfile);
    if (fid == -1) {
        fprintf(stderr, "Failed to create temp file: %s\n", strerror(errno));
        exit(-1);
    }
    close(fid);
    //printf("Created temp file %s\n", tempfile);
    return tempfile;
}

int parse_positive_range_string(il* depths, const char* str,
                                int default_low, int default_high,
                                const char* valuename) {
    // -10,10-20,20-30,40-
    while (str && *str) {
        int check_range = 1;
        unsigned int lo, hi;
        int nread;
        char div[2];
        lo = default_low;
        hi = default_high;
        if (sscanf(str, "%u-%u", &lo, &hi) == 2) {
            sscanf(str, "%*u-%*u%n", &nread);
        } else if (sscanf(str, "%u%1[-]", &lo, div) == 2) {
            sscanf(str, "%*u%*1[-]%n", &nread);
            check_range = 0;
        } else if (sscanf(str, "-%u", &hi) == 1) {
            sscanf(str, "-%*u%n", &nread);
            check_range = 0;
        } else if (sscanf(str, "%u", &hi) == 1) {
            sscanf(str, "%*u%n", &nread);
        } else {
            fprintf(stderr, "Failed to parse %s range: \"%s\"\n", valuename, str);
            return -1;
        }
        if (check_range && (lo < 0)) {
            fprintf(stderr, "%s value %i is invalid: must be >= 0.\n", valuename, lo);
            return -1;
        }
        if (check_range && (lo > hi)) {
            fprintf(stderr, "%s range %i to %i is invalid: max must be >= min!\n", valuename, lo, hi);
            return -1;
        }
        il_append(depths, lo);
        il_append(depths, hi);
        str += nread;
        while ((*str == ',') || isspace(*str))
            str++;
    }
    return 0;
}
