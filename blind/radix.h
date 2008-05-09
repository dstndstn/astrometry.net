/*
  Radix.cpp: a fast floating-point radix sort demo
  
  Copyright (C) Herf Consulting LLC 2001.  All Rights Reserved.
  Use for anything you want, just tell me what you do with it.
  Code provided "as-is" with no liabilities for anything that goes wrong.
*/
#ifndef RADIX_H_
#define RADIX_H_

#include <stdint.h>

typedef uint32_t uint32;
typedef float    real32;

//  **** WARNING *****
// Your input array will be modified.
void RadixSort11(real32 *farray, real32 *sorted, uint32 elements);

#endif
