/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

typedef u64 ttype;

// GCC only??
typedef __int128           int128_t;
typedef unsigned __int128 uint128_t;
static const uint128_t UINT128_MAX = (uint128_t)((int128_t)(-1L));

#define BIGTTYPE uint128_t
#define BIGTTYPE_MAX UINT128_MAX
typedef uint128_t bigttype;

#define TTYPE_INTEGER 1

#define TTYPE_MIN 0
#define TTYPE_MAX UINT64_MAX
#define TTYPE_SQRT_MAX UINT32_MAX

#define TTYPE l
#define TTYPE_M l
