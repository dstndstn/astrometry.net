/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <assert.h>

#include "intmap.h"

#define IMGLUE2(n,f) n ## _ ## f
#define IMGLUE(n,f) IMGLUE2(n,f)

#define key_t int
#define kl il
#define maptype intmap
#define map_t IMGLUE(maptype, t)

#define KL(x) IMGLUE(kl, x)
#define IMAP(x) IMGLUE(maptype, x)

#include "intmap.inc"

#undef key_t
#undef kl
#undef maptype

#define key_t int64_t
#define kl ll
#define maptype longmap

#include "intmap.inc"

#undef key_t
#undef kl
#undef maptype

#undef IMAP
#undef KL
#undef IMGLUE2
#undef IMGLUE
