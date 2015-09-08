/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef KDTREE_MEM_H
#define KDTREE_MEM_H

#include <stdlib.h>

// Are we tracking memory usage by libkd?
#if defined(KDTREE_MEM_TRACK)

void* CALLOC(size_t nmemb, size_t sz);
void* MALLOC(size_t sz);
void* REALLOC(void* ptr, size_t sz);
void  FREE(void* ptr);

#else

#define CALLOC calloc
#define MALLOC malloc
#define REALLOC realloc
#define FREE free

#endif

// Is memory-tracking enabled?
int kdtree_mem_enabled();

// Reset the memory usage counter
void kdtree_mem_reset();

// Get the current memory usage
int  kdtree_mem_get();

#endif
