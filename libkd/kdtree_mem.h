/*
  This file is part of libkd.
  Copyright 2007 Dustin Lang and Keir Mierle.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
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
