/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef SOLVEDCLIENT_H
#define SOLVEDCLIENT_H

#include "astrometry/bl.h"

int solvedclient_set_server(char* addr);

int solvedclient_get(int filenum, int fieldnum);

void solvedclient_set(int filenum, int fieldnum);

il* solvedclient_get_fields(int filenum, int firstfield, int lastfield,
							int maxnfields);

#endif
