/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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
#ifndef GNU_SPECIFIC_H
#define GNU_SPECIFIC_H

#include "gnu-specific-config.h"

// Declarations of GNU-specific functions that we use.

char* canonicalize_file_name(const char* fn);

// This is actually in POSIX1b but may or may not be available.
int fdatasync(int fd);

#if NEED_DECLARE_QSORT_R
//// NOTE: this declaration must match gnu-specific-test.c .
// This is in some BSDs, Mandriva 2009, and Ubuntu 8.10.
// (glibc 2.8)
void qsort_r(void *base, size_t nmemb, size_t sz,
             void *userdata,
             int (*compar)(void *, const void *, const void *));
#endif

// As suggested in http://gcc.gnu.org/onlinedocs/gcc-4.3.0/gcc/Function-Names.html
#if __STDC_VERSION__ < 199901L
# if __GNUC__ >= 2
#  define __func__ __FUNCTION__
# else
#  define __func__ "<unknown>"
# endif
#endif

#endif
