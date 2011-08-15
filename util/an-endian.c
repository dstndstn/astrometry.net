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

#include "an-endian.h"

/*
 #if IS_BIG_ENDIAN
 #warning "Big endian"
 #else
 #warning "Little endian"
 #endif
 */

int is_big_endian() {
    return IS_BIG_ENDIAN;
}

uint16_t u16_letoh(uint16_t i) {
#if IS_BIG_ENDIAN
    return (
		((i & 0x00ff) <<  8) |
		((i & 0xff00) >>  8));
#else
	return i;
#endif
}
uint16_t u16_htole(uint16_t i) {
    return u16_letoh(i);
}

// convert a u32 from little-endian to local.
inline uint32_t u32_letoh(uint32_t i) {
#if IS_BIG_ENDIAN
    return (
            ((i & 0x000000ff) << 24) |
            ((i & 0x0000ff00) <<  8) |
            ((i & 0x00ff0000) >>  8) |
            ((i & 0xff000000) >> 24));
#else
	return i;
#endif
}

// convert a u32 from local to little-endian.
inline uint32_t u32_htole(uint32_t i) {
    return u32_letoh(i);
}

static inline void v_swap(void* p, int nbytes) {
	int i;
	unsigned char* c = p;
	for (i=0; i<(nbytes/2); i++) {
		unsigned char tmp = c[i];
		c[i] = c[nbytes-(i+1)];
		c[nbytes-(i+1)] = tmp;
	}
}

void endian_swap(void* p, int nbytes) {
    v_swap(p, nbytes);
}

static inline void v_htole(void* p, int nbytes) {
#if IS_BIG_ENDIAN
    return v_swap(p, nbytes);
#else
    // nop.
#endif
}

static inline void v_ntoh(void* p, int nbytes) {
#if IS_BIG_ENDIAN
    // nop.
#else
    return v_swap(p, nbytes);
#endif
}

// convert a 32-bit object from local to little-endian.
inline void v32_htole(void* p) {
	return v_htole(p, 4);
}

// convert a 16-bit object from local to little-endian.
inline void v16_htole(void* p) {
	return v_htole(p, 2);
}

inline void v32_letoh(void* p) {
	return v32_htole(p);
}


// convert a 64-bit object from big-endian (network) to local.
inline void v64_ntoh(void* p) {
	return v_ntoh(p, 8);
}
// convert a 32-bit object from big-endian (network) to local.
inline void v32_ntoh(void* p) {
	return v_ntoh(p, 4);
}
// convert a 16-bit object from big-endian (network) to local.
inline void v16_ntoh(void* p) {
	return v_ntoh(p, 2);
}

// convert a 64-bit object from local to big-endian (network).
inline void v64_hton(void* p) {
    return v64_ntoh(p);
}
// convert a 32-bit object from local to big-endian (network).
inline void v32_hton(void* p) {
    return v32_ntoh(p);
}
// convert a 16-bit object from local to big-endian (network).
inline void v16_hton(void* p) {
    return v16_ntoh(p);
}


