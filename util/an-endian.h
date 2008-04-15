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
#ifndef AN_ENDIAN_H
#define AN_ENDIAN_H

#include <stdint.h>

// MacOSX doesn't have endian.h
#if __APPLE__
# include <sys/types.h>
#elif __FreeBSD__
# include <sys/endian.h>
#else
# include <endian.h>
#endif

#if \
  (defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)) || \
  (defined( _BYTE_ORDER) && ( _BYTE_ORDER ==  _BIG_ENDIAN)) || \
  (defined(  BYTE_ORDER) && (  BYTE_ORDER ==   BIG_ENDIAN))
#define IS_BIG_ENDIAN 1
#else
#define IS_BIG_ENDIAN 0
#endif

inline uint32_t u32_letoh(uint32_t i);
inline uint32_t u32_htole(uint32_t i);

inline void v32_htole(void* p);
inline void v16_htole(void* p);

inline void v32_letoh(void* p);

inline void v64_ntoh(void* p);
inline void v32_ntoh(void* p);
inline void v16_ntoh(void* p);

inline void v64_hton(void* p);
inline void v32_hton(void* p);
inline void v16_hton(void* p);

#endif
