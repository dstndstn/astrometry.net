struct maptype {
    // dense only:
    bl** dense;
    int ND;

    // sparse only:
    kl* keys;
    // list of bl*
    pl* lists;

    // common:
    // list blocksize
    int blocksize;
    // data size
    int datasize;
};
typedef struct maptype map_t;

// intmap_t* intmap_new(...)
map_t* IMAP(new)(int datasize, int subblocksize, int blocksize,
                     int Ndense) {
    map_t* im = calloc(1, sizeof(map_t));
    if (!blocksize)
        blocksize = 4096;
    im->blocksize = subblocksize;
    im->datasize = datasize;
    if (Ndense) {
        im->ND = Ndense;
        im->dense = calloc(im->ND, sizeof(bl*));
    } else {
        im->keys = KL(new)(blocksize);
        im->lists = pl_new(blocksize);
    }
    return im;
}

// void intmap_free(intmap_t*)
void IMAP(free)(map_t* im) {
    int i;
    if (im->lists) {
        for (i=0; i<pl_size(im->lists); i++) {
            bl* lst = pl_get(im->lists, i);
            bl_free(lst);
        }
        pl_free(im->lists);
    }
    if (im->dense) {
        for (i=0; i<im->ND; i++) {
            bl* lst = im->dense[i];
            if (!lst)
                continue;
            bl_free(lst);
        }
        free(im->dense);
    }
    if (im->keys)
        KL(free)(im->keys);
    free(im);
}

// bl* intmap_find(intmap_t* im, key_t key, anbool create);
bl* IMAP(find)(map_t* im, key_t key, anbool create) {
    key_t ind;
    assert(key >= 0);
    assert(im);
    if (!im->dense) {
        assert(im->keys);
        assert(im->lists);
        ind = KL(sorted_index_of)(im->keys, key);
        if (ind == -1) {
            bl* lst;
            if (!create)
                return NULL;
            lst = bl_new(im->blocksize, im->datasize);
            ind = KL(insert_unique_ascending)(im->keys, key);
            pl_insert(im->lists, ind, lst);
            return lst;
        }
        return pl_get(im->lists, ind);
    } else {
        bl* lst;
        assert(key < im->ND);
        assert(im->dense);
        lst = im->dense[key];
        if (lst)
            return lst;
        if (!create)
            return lst;
        lst = im->dense[key] = bl_new(im->blocksize, im->datasize);
        return lst;
    }
}

// void intmap_append(intmap_t* it, int key, void* pval)
void IMAP(append)(map_t* it, key_t key, void* pval) {
    bl* lst = IMAP(find)(it, key, TRUE);
    bl_append(lst, pval);
}

// anbool intmap_get_entry(intmap_t* im, int index, key_t* p_key, bl** p_list);
anbool IMAP(get_entry)(map_t* im, int index,
                       key_t* p_key, bl** p_list) {
    assert(im);
    assert(index >= 0);
    if (im->dense) {
        if (index >= im->ND)
            return FALSE;
        if (p_key)
            *p_key = index;
        if (p_list)
            *p_list = im->dense[index];
        return TRUE;
    }

    assert(im->keys);
    assert(im->lists);
    if (index >= KL(size)(im->keys))
        return FALSE;
    if (p_key)
        *p_key = KL(get)(im->keys, index);
    if (p_list)
        *p_list = pl_get(im->lists, index);
    return TRUE;
}
