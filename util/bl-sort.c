/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include "ioutils.h" // for QSORT_R

#include "bl-sort.h"
// for qsort_r
#include "os-features.h"
#include "bl.ph"

#define nl il
#define number int
#include "bl-nl-sort.c"
#undef nl
#undef number

#define nl ll
#define number int64_t
#include "bl-nl-sort.c"
#undef nl
#undef number

#define nl fl
#define number float
#include "bl-nl-sort.c"
#undef nl
#undef number

#define nl dl
#define number double
#include "bl-nl-sort.c"
#undef nl
#undef number

static void bl_sort_with_userdata(bl* list,
                                  int (*compare)(const void* v1, const void* v2, void* userdata),
                                  void* userdata);

struct funcandtoken {
    int (*compare)(const void* v1, const void* v2, void* userdata);
    void* userdata;
};
static int QSORT_COMPARISON_FUNCTION(qcompare, void* token, const void* v1, const void* v2) {
    struct funcandtoken* ft = token;
    return ft->compare(v1, v2, ft->userdata);
}

static void bl_sort_rec(bl* list, void* pivot,
                        int (*compare)(const void* v1, const void* v2, void* userdata),
                        void* userdata) {
    bl* less;
    bl* equal;
    bl* greater;
    int i;
    bl_node* node;

    // Empty case
    if (!list->head)
        return;

    // Base case: list with only one block.
    if (list->head == list->tail) {
        bl_node* node;
        struct funcandtoken ft;
        ft.compare = compare;
        ft.userdata = userdata;
        node = list->head;
        QSORT_R(NODE_DATA(node), node->N, list->datasize, &ft, qcompare);
        return;
    }

    less = bl_new(list->blocksize, list->datasize);
    equal = bl_new(list->blocksize, list->datasize);
    greater = bl_new(list->blocksize, list->datasize);
    for (node=list->head; node; node=node->next) {
        char* data = NODE_CHARDATA(node);
        for (i=0; i<node->N; i++) {
            int val = compare(data, pivot, userdata);
            if (val < 0)
                bl_append(less, data);
            else if (val > 0)
                bl_append(greater, data);
            else
                bl_append(equal, data);
            data += list->datasize;
        }
    }

    // recurse before freeing anything...
    bl_sort_with_userdata(less, compare, userdata);
    bl_sort_with_userdata(greater, compare, userdata);

    for (node=list->head; node;) {
        bl_node* next;
        next = node->next;
        bl_free_node(node);
        node = next;
    }
    list->head = NULL;
    list->tail = NULL;
    list->N = 0;
    list->last_access = NULL;
    list->last_access_n = 0;

    if (less->N) {
        list->head = less->head;
        list->tail = less->tail;
        list->N = less->N;
    }
    if (equal->N) {
        if (list->N) {
            list->tail->next = equal->head;
            list->tail = equal->tail;
        } else {
            list->head = equal->head;
            list->tail = equal->tail;
        }
        list->N += equal->N;
    }
    if (greater->N) {
        if (list->N) {
            list->tail->next = greater->head;
            list->tail = greater->tail;
        } else {
            list->head = greater->head;
            list->tail = greater->tail;
        }
        list->N += greater->N;
    }
    // note, these are supposed to be "free", not "bl_free"; we've stolen
    // the blocks, we're just freeing the headers.
    free(less);
    free(equal);
    free(greater);
}

static void bl_sort_with_userdata(bl* list,
                                  int (*compare)(const void* v1, const void* v2, void* userdata),
                                  void* userdata) {
    int ind;
    int N = list->N;
    if (N <= 1)
        return;
    // should do median-of-3/5/... to select pivot when N is large.
    ind = rand() % N;
    bl_sort_rec(list, bl_access(list, ind), compare, userdata);
}

static int sort_helper_bl(const void* v1, const void* v2, void* userdata) {
    int (*compare)(const void* v1, const void* v2) = userdata;
    return compare(v1, v2);
}

void bl_sort(bl* list, int (*compare)(const void* v1, const void* v2)) {
    bl_sort_with_userdata(list, sort_helper_bl, compare);
}

// dereference one level...
static int sort_helper_pl(const void* v1, const void* v2, void* userdata) {
    const void* p1 = *((const void**)v1);
    const void* p2 = *((const void**)v2);
    int (*compare)(const void* p1, const void* p2) = userdata;
    return compare(p1, p2);
}

void  pl_sort(pl* list, int (*compare)(const void* v1, const void* v2)) {
    bl_sort_with_userdata(list, sort_helper_pl, compare);
}

