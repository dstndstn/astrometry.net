/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#ifndef OS_FEATURES_H
#define OS_FEATURES_H

#ifndef DONT_INCLUDE_OS_FEATURES_CONFIG_H
#include "astrometry/os-features-config.h"
#endif

// Features we use that aren't standard across all supported platforms

char* canonicalize_file_name(const char* fn);

// This is actually in POSIX1b but may or may not be available.
int fdatasync(int fd);

// Not POSIX; doesn't exist in Solaris 10
#include <sys/param.h>
#ifndef MIN
#define	MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define	MAX(a,b) (((a)>(b))?(a):(b))
#endif

// isfinite() on Solaris; from
// https://code.google.com/p/redis/issues/detail?id=20
#if defined(__sun) && defined(__GNUC__)

#undef isnan
#define isnan(x) \
      __extension__({ __typeof (x) __x_a = (x); \
      __builtin_expect(__x_a != __x_a, 0); })

#undef isfinite
#define isfinite(x) \
      __extension__ ({ __typeof (x) __x_f = (x); \
      __builtin_expect(!isnan(__x_f - __x_f), 1); })

#undef isinf
#define isinf(x) \
      __extension__ ({ __typeof (x) __x_i = (x); \
      __builtin_expect(!isnan(__x_i) && !isfinite(__x_i), 0); })

#endif

/**
   The qsort_r story:

   -qsort_r appears in BSD (including Mac OSX)
         void qsort_r(void *, size_t, size_t,
                      void *,
                      int (*)(void *, const void *, const void *));

   -qsort_r appears in glibc 2.8, but with a different argument order:
         void qsort_r(void*, size_t, size_t,
                      int (*)(const void*, const void*, void*),
                      void*);

   Notice that the "thunk" and "comparison function" arguments to qsort_r are
   swapped, and the "thunk" appears either at the beginning or end of the comparison
   function.

   We check a few things:
   -is qsort_r declared?
   -does qsort_r exist?
   -do we need to swap the arguments?

   Those using qsort_r in Astrometry.net should instead use the macro QSORT_R()
   to take advantage of these tests.

   Its signature is:

   void QSORT_R(void* base, size_t nmembers, size_t member_size,
                void* token, comparison_function);

   You should define the "comparison" function like this:

   static int QSORT_COMPARISON_FUNCTION(my_comparison, void* token, const void* v1, const void* v2) {
     ...
   }


   Distributions including glibc 2.8 include:
   -Mandriva 2009
   -Ubuntu 8.10
*/

#if NEED_DECLARE_QSORT_R
//// NOTE: this declaration must match os-features-test.c .
void qsort_r(void *base, size_t nmemb, size_t sz,
             void *userdata,
             int (*compar)(void *, const void *, const void *));
#endif

#if NEED_SWAP_QSORT_R
#define QSORT_R(a,b,c,d,e) qsort_r(a,b,c,e,d)
#define QSORT_COMPARISON_FUNCTION(func, thunk, v1, v2) func(v1, v2, thunk)

#else
#define QSORT_R qsort_r
#define QSORT_COMPARISON_FUNCTION(func, thunk, v1, v2) func(thunk, v1, v2)

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
