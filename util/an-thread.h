/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

#include "an-thread-pthreads.h"



#endif
