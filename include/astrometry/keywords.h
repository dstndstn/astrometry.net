/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

// borrowed from http://rlove.org/log/2005102601.

#ifndef ASTROMETRY_KEYWORDS_H
#define ASTROMETRY_KEYWORDS_H

#define ATTRIB_FORMAT(style,fmt,start) __attribute__ ((format(style,fmt,start)))

// this snippet borrowed from GNU libc features.h:
#if defined __GNUC__
# define GNUC_PREREQ(maj, min) \
         ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
# define GNUC_PREREQ(maj, min) 0
#endif

#if GNUC_PREREQ (3, 0)

// Clang masquerades as gcc but isn't compatible.  Someone should file a
// lawsuit.  Clang treats inlining differently; see
//    http://clang.llvm.org/compatibility.html#inline

#if defined __clang__ || GNUC_PREREQ (5, 0)

// After gcc 5.0, -std=gnu11 is the default (vs -std=gnu89 in previous
// versions).  This affects inlining semantics, among other things.

#define InlineDeclare
#define InlineDefineH
#define InlineDefineC

#else

// plain old gcc

#define INCLUDE_INLINE_SOURCE 1

#define InlineDeclare  extern inline
#define InlineDefineH  extern inline
#define InlineDefineC

#endif

// See:
//   http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html

# define Inline           inline
# define Pure             __attribute__ ((pure))
# define Const            __attribute__ ((const))
# define Noreturn         __attribute__ ((noreturn))
# define Malloc           __attribute__ ((malloc))
# define Used             __attribute__ ((used))
# define Unused           __attribute__ ((unused))
# define VarUnused        __attribute__ ((unused))
# define Packed           __attribute__ ((packed))
# define likely(x)        __builtin_expect (!!(x), 1)
# define unlikely(x)      __builtin_expect (!!(x), 0)
# define Noinline         __attribute__ ((noinline))
// alloc_size

// new in gcc-3.1:
#if GNUC_PREREQ (3, 1)
# define Deprecated       __attribute__ ((deprecated))
#else
# define Deprecated
#endif

// new in gcc-3.4:
#if GNUC_PREREQ (3, 4)
# define Must_check       __attribute__ ((warn_unused_result))
# define WarnUnusedResult __attribute__ ((warn_unused_result))
#else
# define Must_check
# define WarnUnusedResult
#endif

// new in gcc-4.1:
#if GNUC_PREREQ (4, 1)

#if defined __clang__
// clang complains very loudly about this being ignored...
# define Flatten
#else
# define Flatten          __attribute__ (( flatten))
#endif

#else
# define Flatten
#endif

#else

// not gnuc >= 3.0

# define Inline
# define Pure
# define Const
# define Noreturn
# define Malloc
# define Must_check
# define Deprecated
# define Used
# define Unused
# define VarUnused
# define Packed
# define likely(x)	(x)
# define unlikely(x)	(x)
# define Noinline
# define WarnUnusedResult
# define Flatten

#endif

#endif // ASTROMETRY_KEYWORDS_H
