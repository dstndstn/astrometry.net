/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

//#define NODE_NUMDATA(node) ((number*)NODE_DATA(node))

#define NLFGLUE2(n,f) n ## _ ## f
#define NLFGLUE(n,f) NLFGLUE2(n,f)
#define NLF(func) NLFGLUE(nl, func)

