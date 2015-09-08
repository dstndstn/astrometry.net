/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef BOILERPLATE_H
#define BOILERPLATE_H

#include <stdio.h>
#include "astrometry/qfits_header.h"

#define BOILERPLATE_HELP_HEADER(fid)                                         \
do {								             \
    fprintf(fid, "This program is part of the Astrometry.net suite.\n");     \
    fprintf(fid, "For details, visit http://astrometry.net.\n");	     \
    fprintf(fid, "Git URL %s\n", AN_GIT_URL);                      \
    fprintf(fid, "Revision %s, date %s.\n", AN_GIT_REVISION, AN_GIT_DATE); \
} while (0)

#define BOILERPLATE_ADD_FITS_HEADERS(hdr)                                        \
do {                                                                             \
    fits_add_long_history(hdr, "Created by the Astrometry.net suite.");          \
    fits_add_long_history(hdr, "For more details, see http://astrometry.net.");  \
    fits_add_long_history(hdr, "Git URL %s", AN_GIT_URL);          \
    fits_add_long_history(hdr, "Git revision %s", AN_GIT_REVISION); \
    fits_add_long_history(hdr, "Git date %s", AN_GIT_DATE);        \
} while (0)

#endif
