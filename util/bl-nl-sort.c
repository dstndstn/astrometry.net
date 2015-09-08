/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */
static int NLF(compare_ascending)(const void* v1, const void* v2) {
    number i1 = *(number*)v1;
    number i2 = *(number*)v2;
    if (i1 > i2) return 1;
    else if (i1 < i2) return -1;
    else return 0;
}

static int NLF(compare_descending)(const void* v1, const void* v2) {
    number i1 = *(number*)v1;
    number i2 = *(number*)v2;
    if (i1 > i2) return -1;
    else if (i1 < i2) return 1;
    else return 0;
}

void NLF(sort)(nl* list, int ascending) {
	bl_sort(list, ascending ? NLF(compare_ascending) : NLF(compare_descending));
}

