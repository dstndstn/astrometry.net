/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef AN_THREAD_H
#define AN_THREAD_H

/**
 Some wrappers for threading functions.  We provide a pthreads
 implementation but since most of the programs in the package are
 single-threaded, a null implementation could be written if necessary.
 */

/*
 The implementation must define:

 -- declare (probably at the top of your source file)
 -- a named run-once function.  'name' should be a valid C
 -- variable identifier.

 AN_THREAD_DECLARE_ONCE(name);

 -- ensure that the named run-once function 'name' has been run.

 AN_THREAD_CALL_ONCE(name, void (*func)(void));

--

 AN_THREAD_DECLARE_MUTEX(name);

 AN_THREAD_LOCK(name);
 AN_THREAD_UNLOCK(name);

 */

#include "astrometry/an-thread-pthreads.h"



#endif
