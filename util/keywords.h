/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#if defined __clang__

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
