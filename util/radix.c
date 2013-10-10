/*
  Radix.cpp: a fast floating-point radix sort demo
  
  Copyright (C) Herf Consulting LLC 2001.  All Rights Reserved.
  Use for anything you want, just tell me what you do with it.
  Code provided "as-is" with no liabilities for anything that goes wrong.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "keywords.h"
#include "radix.h"

// ------------------------------------------------------------------------------------------------
// ---- Basic types

typedef int32_t  int32;
//typedef uint32_t uint32;
typedef uint8_t  uint8;
//typedef float    real32;
typedef double   real64;
//typedef const char* cpointer;

// ------------------------------------------------------------------------------------------------
// Configuration/Testing

// ---- number of elements to test (shows tradeoff of histogram size vs. sort size)
const uint32 ct = 65536;

// ---- really, a correctness check, not correctness itself ;)
#define CORRECTNESS 1

// ---- use SSE prefetch (needs compiler support), not really a problem on non-SSE machines.
//		need http://msdn.microsoft.com/vstudio/downloads/ppack/default.asp
//		or recent VC to use this

// __builtin_prefetch is new in gcc-3.1
#if GNUC_PREREQ (3, 1)
#define PREFETCH 1
#else
#define PREFETCH 0
#endif

#if PREFETCH
//#include <xmmintrin.h>	// for prefetch
#define pfval	64
#define pfval2	128
//#define pf(x)	_mm_prefetch(cpointer(x + i + pfval), 0)
//#define pf2(x)	_mm_prefetch(cpointer(x + i + pfval2), 0)
#define pf(x)	__builtin_prefetch((x + i + pfval))
#define pf2(x)	__builtin_prefetch((x + i + pfval2))
#else
#define pf(x)
#define pf2(x)
#endif

// ------------------------------------------------------------------------------------------------
// ---- Visual C++ eccentricities

#if _WINDOWS
#define finline __forceinline
#else
#define finline inline
#endif

// ================================================================================================
// flip a float for sorting
//  finds SIGN of fp number.
//  if it's 1 (negative float), it flips all bits
//  if it's 0 (positive float), it flips the sign only
// ================================================================================================
finline uint32 FloatFlip(uint32 f)
{
	uint32 mask = -(int32)(f >> 31) | 0x80000000;
	return f ^ mask;
}

finline void FloatFlipX(uint32* pf)
{
	uint32 mask = -(int32)(*pf >> 31) | 0x80000000;
	*pf ^= mask;
}

// ================================================================================================
// flip a float back (invert FloatFlip)
//  signed was flipped from above, so:
//  if sign is 1 (negative), it flips the sign bit back
//  if sign is 0 (positive), it flips all bits back
// ================================================================================================
finline uint32 IFloatFlip(uint32 f)
{
	uint32 mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}

// ---- utils for accessing 11-bit quantities
#define _0(x)	(x & 0x7FF)
#define _1(x)	(x >> 11 & 0x7FF)
#define _2(x)	(x >> 22 )

// ================================================================================================
// Main radix sort
// ================================================================================================
#if __cplusplus
extern "C" {
#endif
void RadixSort11(real32 *farray, real32 *sorted, uint32 elements);
#if __cplusplus
}
#endif
void RadixSort11(real32 *farray, real32 *sorted, uint32 elements)
{
	uint32 i;
	uint32 *sort = (uint32*)sorted;
	uint32 *array = (uint32*)farray;

	// 3 histograms on the stack:
	const uint32 kHist = 2048;
	uint32 b0[kHist * 3];

	uint32 *b1 = b0 + kHist;
	uint32 *b2 = b1 + kHist;

	for (i = 0; i < kHist * 3; i++) {
		b0[i] = 0;
	}
	//memset(b0, 0, kHist * 12);

	// 1.  parallel histogramming pass
	//
	for (i = 0; i < elements; i++) {
	  uint32 fi;
		
		pf(array);

		fi = FloatFlip((uint32)array[i]);

		b0[_0(fi)] ++;
		b1[_1(fi)] ++;
		b2[_2(fi)] ++;
	}
	
	// 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
	{
		uint32 sum0 = 0, sum1 = 0, sum2 = 0;
		uint32 tsum;
		for (i = 0; i < kHist; i++) {

			tsum = b0[i] + sum0;
			b0[i] = sum0 - 1;
			sum0 = tsum;

			tsum = b1[i] + sum1;
			b1[i] = sum1 - 1;
			sum1 = tsum;

			tsum = b2[i] + sum2;
			b2[i] = sum2 - 1;
			sum2 = tsum;
		}
	}

	// byte 0: floatflip entire value, read/write histogram, write out flipped
	for (i = 0; i < elements; i++) {
      uint32 fi,pos;

		fi = array[i];
		FloatFlipX(&fi);
		pos = _0(fi);
		
		pf2(array);
		sort[++b0[pos]] = fi;
	}

	// byte 1: read/write histogram, copy
	//   sorted -> array
	for (i = 0; i < elements; i++) {
		uint32 si = sort[i];
		uint32 pos = _1(si);
		pf2(sort);
		array[++b1[pos]] = si;
	}

	// byte 2: read/write histogram, copy & flip out
	//   array -> sorted
	for (i = 0; i < elements; i++) {
		uint32 ai = array[i];
		uint32 pos = _2(ai);

		pf2(array);
		sort[++b2[pos]] = IFloatFlip(ai);
	}

	// to write original:
	// memcpy(array, sorted, elements * 4);
}

