/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

typedef u64 ttype;

// https://stackoverflow.com/questions/16088282/is-there-a-128-bit-integer-in-gcc
// gcc: 128-bit ints only available on 64-bit platforms, not 32-bit
#ifdef __SIZEOF_INT128__
// GCC only??
typedef __int128           int128_t;
typedef unsigned __int128 uint128_t;
static const uint128_t UINT128_MAX = (uint128_t)((int128_t)(-1L));
#define BIGTTYPE uint128_t
#define BIGTTYPE_MAX UINT128_MAX
typedef uint128_t bigttype;

#else
// Fall back to using just 64-bit types.  This *should* still work okay, because
// we're careful to check the max possible value before using BIGT types; search
// for "use_tmath" in the code.
#define BIGTTYPE uint64_t
#define BIGTTYPE_MAX UINT64_MAX
typedef uint64_t bigttype;

#endif

#define TTYPE_INTEGER 1

#define TTYPE_MIN 0
#define TTYPE_MAX UINT64_MAX
#define TTYPE_SQRT_MAX UINT32_MAX

#define TTYPE l
#define TTYPE_M l
