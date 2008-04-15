/**
 This file includes code copied from three source files in the Linux kernel
 v 2.6.23.1:
 - include/asm-generic/bitops/fls.h
 - include/asm-i386/bitops.h
 - include/asm-x86_64/bitops.h

 The second and third files contain this notice:
 * Copyright 1992, Linus Torvalds.

 And of course the kernel is distributed under the terms of the GPL v2.
 */

#ifndef AN_FLS_H
#define AN_FLS_H

#include <stdint.h>
#include <assert.h>

/**
 * fls - find last (most-significant) bit set
 *
 * @x: the word to search
 *
 * This is defined the same way as ffs.
 *
 * Note fls(0) = 0, fls(1) = 1, fls(0x80000000) = 32.
 */
static inline int an_fls(int x);

/**
 * flsB()  =  fls() - 1.
 *
 * Note that x MUST be > 0.
 */
static inline uint8_t an_flsB(uint32_t x);



/**** Below this line are the implementations for different CPUs. ****/

#if AN_I386

static inline int an_fls(int x) {
    int r;
    __asm__("bsrl %1,%0\n\t"
            "jnz 1f\n\t"
            "movl $-1,%0\n"
            "1:" : "=r" (r) : "rm" (x));
    return r+1;
} 

static inline uint8_t an_flsB(uint32_t x) {
    int r;
    assert(x);
    __asm__("bsrl %1,%0\n\t"
            "jnz 1f\n\t"
            "movl $-1,%0\n"
            "1:" : "=r" (r) : "rm" (x));
    return r;
} 

#elif AN_X86_64

static __inline__ int an_fls(int x) {
    int r;
    __asm__("bsrl %1,%0\n\t"
            "cmovzl %2,%0"
            : "=&r" (r) : "rm" (x), "rm" (-1));
    return r+1;
} 

static inline uint8_t an_flsB(uint32_t x) {
    int r;
    assert(x);
    __asm__("bsrl %1,%0\n\t"
            "cmovzl %2,%0"
            : "=&r" (r) : "rm" (x), "rm" (-1));
    return r;
}

#else

static inline int an_fls(int x) {
    int r = 32;
    if (!x)
        return 0;
    if (!(x & 0xffff0000u)) {
        x <<= 16;
        r -= 16;
    }
    if (!(x & 0xff000000u)) {
        x <<= 8;
        r -= 8;
    }
    if (!(x & 0xf0000000u)) {
        x <<= 4;
        r -= 4;
    }
    if (!(x & 0xc0000000u)) {
        x <<= 2;
        r -= 2;
    }
    if (!(x & 0x80000000u)) {
        x <<= 1;
        r -= 1;
    }
    return r;
}

static inline uint8_t an_flsB(uint32_t x) {
    int r = 31;
    assert(x);
    if (!(x & 0xffff0000u)) {
        x <<= 16;
        r -= 16;
    }
    if (!(x & 0xff000000u)) {
        x <<= 8;
        r -= 8;
    }
    if (!(x & 0xf0000000u)) {
        x <<= 4;
        r -= 4;
    }
    if (!(x & 0xc0000000u)) {
        x <<= 2;
        r -= 2;
    }
    if (!(x & 0x80000000u)) {
        x <<= 1;
        r -= 1;
    }
    return r;
}

#endif

#endif


