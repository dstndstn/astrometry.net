/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ngc2000.h"
#include "ngcic-accurate.h"
#include "bl.h"
#include "ioutils.h"

struct ngc_name {
    anbool is_ngc;
    int id;
    char* name;
};
typedef struct ngc_name ngc_name;

ngc_name ngc_names[] = {
#include "ngc2000names.c"
};

ngc_entry ngc_entries[] = {
#include "ngc2000entries.c"
};

static int n_names() {
    return sizeof(ngc_names) / sizeof(ngc_name);
}

ngc_entry* ngc_get_entry_accurate(int i) {
    float ra, dec;
    ngc_entry* ngc = ngc_get_entry(i);
    if (ngcic_accurate_get_radec(ngc->is_ngc, ngc->id, &ra, &dec) == 0) {
        ngc->ra  = ra;
        ngc->dec = dec;
    }
    return ngc;
}

int ngc_num_entries() {
    return sizeof(ngc_entries) / sizeof(ngc_entry);
}

ngc_entry* ngc_get_entry(int i) {
    if (i < 0)
        return NULL;
    if (i >= ngc_num_entries())
        return NULL;
    return ngc_entries + i;
}

ngc_entry* ngc_get_ngcic_num(int is_ngc, int num) {
    int i, N;
    N = ngc_num_entries();
    for (i=0; i<N; i++) {
        ngc_entry* e = ngc_get_entry(i);
        if (e->is_ngc == is_ngc && e->id == num)
            return e;
    }
    return NULL;
}

ngc_entry* ngc_get_entry_named(const char* name) {
    if (starts_with(name, "NGC") || starts_with(name, "IC")) {
        int num;
        const char* cptr;
        anbool isngc;
        isngc = starts_with(name, "NGC");
        cptr = name + (isngc ? 3 : 2);
        if (*cptr == ' ')
            cptr++;
        num = atoi(cptr);
        if (!num)
            return NULL;
        return ngc_get_ngcic_num(isngc, num);
    } else {
        int i, N;
        N = n_names();
        for (i=0; i<N; i++) {
            char nsname[256];
            const char* src;
            char* dest;
            if (streq(name, ngc_names[i].name))
                return ngc_get_ngcic_num(ngc_names[i].is_ngc, ngc_names[i].id);
            // Try without spaces
            dest = nsname;
            src = ngc_names[i].name;
            for (; *src; src++) {
                if (*src == ' ')
                    continue;
                *dest = *src;
                dest++;
            }
            *dest = '\0';
            //printf("Spaceless name: '%s'\n", nsname);
            if (streq(name, nsname))
                return ngc_get_ngcic_num(ngc_names[i].is_ngc, ngc_names[i].id);
        }
    }
    return NULL;
}

char* ngc_get_name(ngc_entry* entry, int num) {
    int i;
    for (i=0; i<sizeof(ngc_names)/sizeof(ngc_name); i++) {
        if ((entry->is_ngc == ngc_names[i].is_ngc) &&
            (entry->id == ngc_names[i].id)) {
            if (num == 0)
                return ngc_names[i].name;
            else
                num--;
        }
    }
    return NULL;
}

sl* ngc_get_names(ngc_entry* entry, sl* lst) {
    int i;
    if (!lst)
        lst = sl_new(4);
    sl_appendf(lst, "%s %i", (entry->is_ngc ? "NGC" : "IC"), entry->id);
    for (i=0; i<sizeof(ngc_names)/sizeof(ngc_name); i++) {
        if ((entry->is_ngc == ngc_names[i].is_ngc) &&
            (entry->id == ngc_names[i].id)) {
            sl_append(lst, ngc_names[i].name);
        }
    }
    return lst;
}

char* ngc_get_name_list(ngc_entry* entry, const char* separator) {
    char* str;
    sl* lst = ngc_get_names(entry, NULL);
    str = sl_implode(lst, separator);
    sl_free2(lst);
    return str;
}

