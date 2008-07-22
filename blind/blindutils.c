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
#include <ctype.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "blindutils.h"
#include "ioutils.h"
#include "bl.h"
#include "errors.h"
#include "log.h"

int parse_depth_string(il* depths, const char* str) {
    unsigned int lasthi = 0;
    unsigned int lo, hi;

    while (str && *str) {
        int nread;
        char div[2];
        lo = 0;
        hi = 0;
        // 10-20
        if (sscanf(str, "%u-%u", &lo, &hi) == 2) {
            sscanf(str, "%*u-%*u%n", &nread);
            if (lo > hi) {
                logerr("Depth range %i to %i is invalid: max must be >= min!\n", lo, hi);
                return -1;
            }
            if (lo == 0) {
                logerr("Depth lower limit %i is invalid: depths must be >= 1.\n", lo);
                return -1;
            }
            // 30-
        } else if (sscanf(str, "%u%1[-]", &lo, div) == 2) {
            sscanf(str, "%*u%*1[-]%n", &nread);
            if (lo == 0) {
                logerr("Depth lower limit %i is invalid: depths must be >= 1.\n", lo);
                return -1;
            }
            // -100
        } else if (sscanf(str, "-%u", &hi) == 1) {
            sscanf(str, "-%*u%n", &nread);
            if (hi == 0) {
                logerr("Depth upper limit %i is invalid: depths must be >= 1.\n", hi);
                return -1;
            }
            lo = 1;
            // 7
        } else if (sscanf(str, "%u", &hi) == 1) {
            sscanf(str, "%*u%n", &nread);
            if (hi == 0) {
                logerr("Depth %i is invalid: depths must be >= 1.\n", hi);
                return -1;
            }
            lo = lasthi + 1;

        } else {
            logerr("Failed to parse depth range: \"%s\"\n", str);
            return -1;
        }

        il_append(depths, lo);
        il_append(depths, hi);
        lasthi = hi;
        str += nread;
        while ((*str == ',') || isspace(*str))
            str++;
    }
    return 0;
}
