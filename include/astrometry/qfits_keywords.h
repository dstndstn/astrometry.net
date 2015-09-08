/*
 This file was added by the Astrometry.net team.
 Copyright 2007 Dustin Lang.

# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

 */

#ifndef QFITS_KEYWORDS_H
#define QFITS_KEYWORDS_H

// this snippet borrowed from GNU libc features.h:
#if defined __GNUC__
# define GNUC_PREREQ(maj, min) \
         ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
# define GNUC_PREREQ(maj, min) 0
#endif

// new in gcc-3.1:
#if GNUC_PREREQ (3, 1)
# define Deprecated       __attribute__ ((deprecated))
#else
# define Deprecated
#endif



#endif

