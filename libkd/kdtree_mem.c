/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <string.h>

#include "kdtree_mem.h"

static int memory_total = 0;

// Are we tracking memory usage by libkd?
#if defined(KDTREE_MEM_TRACK)

struct memblock {
    int nbytes;
};
typedef struct memblock memblock;

void* MALLOC(size_t sz) {
    memblock* mb;
    if (!sz) return NULL;
    mb = (memblock*)malloc(sz + sizeof(memblock));
    mb->nbytes = sz;
    memory_total += sz;
    return mb + 1;
}

void* CALLOC(size_t nmemb, size_t sz) {
    char* ptr = MALLOC(nmemb * sz);
    memset(ptr, 0, nmemb * sz);
    return ptr;
}

void* REALLOC(void* ptr, size_t sz) {
    memblock* mb;
    if (!ptr) {
        return MALLOC(sz);
    }
    if (!sz) {
        FREE(ptr);
        return NULL;
    }

    mb = ptr;
    mb--;
    memory_total += (sz - mb->nbytes);
    mb->nbytes = sz;
    mb = realloc(mb, sz + sizeof(memblock));
    return mb + 1;
}

void  FREE(void* ptr) {
    memblock* mb;
    int nfreed;
    if (!ptr) return;
    mb = ptr;
    mb--;
    nfreed = mb->nbytes;
    memory_total -= nfreed;
    free(mb);
}

#endif

// Is memory-tracking enabled?
int kdtree_mem_enabled() {
#if defined(KDTREE_MEM_TRACK)
    return 1;
#else
    return 0;
#endif
}


// Reset the memory usage counter
void kdtree_mem_reset() {
    memory_total = 0;
}

// Get the current memory usage
int  kdtree_mem_get() {
    return memory_total;
}

