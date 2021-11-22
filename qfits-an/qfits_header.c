/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_header.c,v 1.10 2006/11/22 13:33:42 yjung Exp $
 *
 * This file is part of the ESO QFITS Library
 * Copyright (C) 2001-2004 European Southern Observatory
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * $Author: yjung $
 * $Date: 2006/11/22 13:33:42 $
 * $Revision: 1.10 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include "qfits_header.h"
#include "qfits_std.h"
#include "qfits_tools.h"
#include "qfits_card.h"
#include "qfits_error.h"
#include "qfits_memory.h"

struct qfits_header {
    void    *   first;         /* Pointer to list start */
    void    *   last;          /* Pointer to list end */
    int         n;             /* Number of cards in list */
    /* For efficient looping internally */
    void    *   current;
    int         current_idx;
};


/*----------------------------------------------------------------------------*/
/*
  @brief    keytuple object (internal)

  This structure represents a FITS card (key, val, comment) in memory.
  A FITS header is a list of such structs. Although the struct is here
  made public for convenience, it is not supposed to be directly used.
  High-level FITS routines should do the job just fine, without need
  to know about this struct.
 */
/*----------------------------------------------------------------------------*/
typedef struct _keytuple_ {

    char    *   key;   /** Key: unique string in a list */
    char    *   val;   /** Value, always as a string */
    char    *   com;   /** Comment associated to key */
    char    *   lin;   /** Initial line in FITS header if applicable */
    int         typ;   /** Key type */

    /** Implemented as a doubly-linked list */
    struct _keytuple_ * next;
    struct _keytuple_ * prev;
} keytuple;

/*----------------------------------------------------------------------------*/
/**
  @enum        keytype
  @brief    Possible key types

  This enum stores all possible types for a FITS keyword. These determine
  the order of appearance in a header, they are a crucial point for
  DICB (ESO) compatibility. This classification is internal to this
  module.
 */
/*----------------------------------------------------------------------------*/
typedef enum _keytype_ {
    keytype_undef            =0,

    keytype_top                =1,

    /* Mandatory keywords */
    /* All FITS files */
    keytype_bitpix            =2,
    keytype_naxis            =3,

    keytype_naxis1            =11,
    keytype_naxis2            =12,
    keytype_naxis3            =13,
    keytype_naxis4            =14,
    keytype_naxisi            =20,
    /* Random groups only */
    keytype_group            =30,
    /* Extensions */
    keytype_pcount            =31,
    keytype_gcount            =32,
    /* Main header */
    keytype_extend            =33,
    /* Images */
    keytype_bscale            =34,
    keytype_bzero            =35,
    /* Tables */
    keytype_tfields            =36,
    keytype_tbcoli            =40,
    keytype_tformi            =41,

    /* Other primary keywords */
    keytype_primary            =100,

    /* HIERARCH ESO keywords ordered according to DICB */
    keytype_hierarch_dpr    =200,
    keytype_hierarch_obs    =201,
    keytype_hierarch_tpl    =202,
    keytype_hierarch_gen    =203,
    keytype_hierarch_tel    =204,
    keytype_hierarch_ins    =205,
    keytype_hierarch_det    =206,
    keytype_hierarch_log    =207,
    keytype_hierarch_pro    =208,
    /* Other HIERARCH keywords */
    keytype_hierarch        =300,

    /* HISTORY and COMMENT */
    keytype_history            =400,
    keytype_comment            =500,
    keytype_continue           =600,
    /* END */
    keytype_end                =1000
} keytype;

/*-----------------------------------------------------------------------------
                        Private to this module
 -----------------------------------------------------------------------------*/

static keytuple * keytuple_new(const char *, const char *, const char *, 
        const char *);
static void keytuple_del(keytuple *);
//static void keytuple_dmp(const keytuple *);
static keytype keytuple_type(const char *);
static int qfits_header_makeline(char *, const keytuple *, int);

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_header    FITS header handling
 *
 * This file contains definition and related methods for the FITS header
 * structure. This structure is meant to remain opaque to the user, who
 * only accesses it through the dedicated functions.
 *
 * The 'keytuple' type is strictly internal to this module.
 * It describes FITS cards as tuples (key,value,comment,line), where key
 * is always a non-NULL character string, value and comment are
 * allowed to be NULL. 'line' is a string containing the line as it
 * has been read from the input FITS file (raw). It is set to NULL if
 * the card is modified later. This allows in output two options:
 * either reconstruct the FITS lines by printing key = value / comment
 * in a FITS-compliant way, or output the lines as they were found in
 * input, except for the modified ones.
 *
 * The following functions are associated methods
 * to this data structure:
 *
 * - keytuple_new()      constructor
 * - keytuple_del()      destructor
 * - keytuple_dmp()      dumps a keytuple to stdout
 *
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                              Public functions
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    FITS header constructor
  @return    1 newly allocated (empty) qfits_header object.

  This is the main constructor for a qfits_header object. It returns
  an allocated linked-list handler with an empty card list.
 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_new(void)
{
    qfits_header    *    h;    
    h = qfits_malloc(sizeof(qfits_header));
    h->first = NULL;
    h->last  = NULL;
    h->n = 0;

    h->current = NULL;
    h->current_idx = -1;

    return h;
}

int qfits_header_n(const qfits_header* hdr) {
    if (!hdr) return -1;
    return hdr->n;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    FITS header default constructor.
  @return    1 newly allocated qfits_header object.

  This is a secondary constructor for a qfits_header object. It returns
  an allocated linked-list handler containing two cards: the first one
  (SIMPLE=T) and the last one (END).

 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_default(void)
{
    qfits_header    *    h;
    h = qfits_header_new();
    qfits_header_append(h, "SIMPLE", "T", "Fits format", NULL);
    qfits_header_append(h, "END", NULL, NULL, NULL);
    return h;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Add a new card to a FITS header
  @param    hdr qfits_header object to modify
  @param    key FITS key
  @param    val FITS value
  @param    com FITS comment
  @param    lin FITS original line if exists
  @return    void

  This function adds a new card into a header, at the one-before-last
  position, i.e. the entry just before the END entry if it is there.
  The key must always be a non-NULL string, all other input parameters
  are allowed to get NULL values.
 */
/*----------------------------------------------------------------------------*/
void qfits_header_add(
        qfits_header    *   hdr,
        const char      *   key,
        const char      *   val,
        const char      *   com,
        const char      *   lin)
{
    keytuple    *    k;
    keytuple    *    kbf;
    keytuple    *    first;
    keytuple    *    last;

    if (hdr==NULL || key==NULL) return;
    if (hdr->n<2) {
		fprintf(stderr, "Caution: qfits thinks it knows better than you: %s:%i key=\"%s\"\n", __FILE__, __LINE__, key);
		return;
	}

    first = (keytuple*)hdr->first;
    last  = (keytuple*)hdr->last;

    if (((keytype)first->typ != keytype_top) ||
        ((keytype)last->typ != keytype_end)) {
		fprintf(stderr, "Caution, qfits thinks it knows better than you: %s:%i\n", __FILE__, __LINE__);
		return;
	}
    
    /* Create new key tuple */
    k = keytuple_new(key, val, com, lin);

	/* Don't add duplicate SIMPLE, XTENSION or END entries. */
	if ((k->typ == keytype_top) || (k->typ == keytype_end)) {
		keytuple_del(k);
		return;
	}

    /* Find the last keytuple with same key type */
    /* NO, DO WHAT THE DOCUMENTATION SAYS WE'RE SUPPOSED TO DO.
     kbf = first;
     while (kbf!=NULL) {
     if ((k->typ>=kbf->typ) && (kbf->next) && (k->typ<kbf->next->typ)) break;
     kbf = kbf->next;
     }
     if (kbf==NULL) kbf = last->prev;
     */
    kbf = last->prev;

    /* Hook it into list */
    k->next = kbf->next;
    (kbf->next)->prev = k;
    kbf->next = k;
    k->prev = kbf;

    hdr->n ++;
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    add a new card to a FITS header
  @param    hdr     qfits_header object to modify
  @param    after    Key to specify insertion place
  @param    key     FITS key
  @param    val     FITS value
  @param    com     FITS comment
  @param    lin     FITS original line if exists
  @return    void

  Adds a new card to a FITS header, after the specified key. Nothing
  happens if the specified key is not found in the header. All fields
  can be NULL, except after and key.
 */
/*----------------------------------------------------------------------------*/
void qfits_header_add_after(
        qfits_header    *   hdr,
        const char      *   after,
        const char      *   key,
        const char      *   val,
        const char      *   com,
        const char      *   lin)
{
    keytuple    *   kreq;
    keytuple    *   k;
    char           exp_after[FITS_LINESZ+1];

    if (hdr==NULL || after==NULL || key==NULL) return;

    qfits_expand_keyword_r(after, exp_after);
    /* Locate where the entry is requested */
    kreq = (keytuple*)(hdr->first);
    while (kreq!=NULL) {
        if (!strcmp(kreq->key, exp_after)) break;
        kreq = kreq->next;
    }
    if (kreq==NULL) return;
    k = keytuple_new(key, val, com, lin);

    k->next = kreq->next;
    kreq->next->prev = k;
    kreq->next = k;
    k->prev = kreq;
    hdr->n ++;
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Append a new card to a FITS header.
  @param    hdr qfits_header object to modify
  @param    key FITS key
  @param    val FITS value
  @param    com FITS comment
  @param    lin FITS original line if exists
  @return    void

  Adds a new card in a FITS header as the last one. All fields can be
  NULL except key.
 */
/*----------------------------------------------------------------------------*/
void qfits_header_append(
        qfits_header    *   hdr,
        const char      *   key,
        const char      *   val,
        const char      *   com,
        const char      *   lin)
{
    keytuple    *    k;
    keytuple    *    last;

    if (hdr==NULL || key==NULL) return;

    k = keytuple_new(key, val, com, lin);
    if (hdr->n==0) {
        hdr->first = hdr->last = k;
        hdr->n = 1;
        return;
    }
    last  = (keytuple*)hdr->last;
    last->next = k;
    k->prev = last;
    hdr->last = k;
    hdr->n++;
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Delete a card in a FITS header.
  @param    hdr qfits_header to modify
  @param    key specifies which card to remove
  @return    void

  Removes a card from a FITS header. The first found card that matches
  the key is removed.
 */
/*----------------------------------------------------------------------------*/
void qfits_header_del(qfits_header * hdr, const char * key)
{
    keytuple    *   k;
    char            xkey[FITS_LINESZ];

    if (hdr==NULL || key==NULL) return;

    qfits_expand_keyword_r(key, xkey);
    k = (keytuple*)hdr->first;
    while (k!=NULL) {
        if (!strcmp(k->key, xkey)) break;
        k = k->next;
    }
    if (k==NULL)
        return;
    if(k == hdr->first) {
        hdr->first = k->next;
    } else {
        k->prev->next = k->next;
        k->next->prev = k->prev;
    }
    keytuple_del(k);
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Modifies a FITS card.
  @param    hdr qfits_header to modify
  @param    key FITS key
  @param    val FITS value
  @param    com FITS comment
  @return    void

  Finds the first card in the header matching 'key', and replaces its
  value and comment fields by the provided values. The initial FITS
  line is set to NULL in the card.
 */
/*----------------------------------------------------------------------------*/
void qfits_header_mod(
        qfits_header    *   hdr,
        const char      *   key,
        const char      *   val,
        const char      *   com)
{
    keytuple    *   k;
    char            xkey[FITS_LINESZ+1];

    if (hdr==NULL || key==NULL) return;

    qfits_expand_keyword_r(key, xkey);
    k = (keytuple*)hdr->first;
    while (k!=NULL) {
        if (!strcmp(k->key, xkey)) break;
        k=k->next;
    }
    if (k==NULL) return;
    
    if (k->val) qfits_free(k->val);
    if (k->com) qfits_free(k->com);
    if (k->lin) qfits_free(k->lin);
    k->val = NULL;
    k->com = NULL;
    k->lin = NULL;
    if (val) {
        if (strlen(val)>0) k->val = qfits_strdup(val);
    }
    if (com) {
        if (strlen(com)>0) k->com = qfits_strdup(com);
    }
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Sort a FITS header
  @param    hdr     Header to sort (modified)
  @return   -1 in error case, 0 otherwise
 */
/*----------------------------------------------------------------------------*/
int qfits_header_sort(qfits_header ** hdr) 
{
    qfits_header    *   sorted;
    keytuple        *   k;
    keytuple        *   kbf;
    keytuple        *   next;
    keytuple        *   last;

    /* Test entries */
    if (hdr == NULL) return -1;
    if (*hdr == NULL) return -1;
    if ((*hdr)->n < 2) return 0;
    
    /* Create the new FITS header */
    sorted = qfits_header_new();

    /* Move the first keytuple to the sorted empty header */
    k = (keytuple*)(*hdr)->first;
    next = k->next;
    sorted->first = sorted->last = k;
    k->next = k->prev = NULL;
    sorted->n = 1;
    
    /* Loop over the other tuples */
    while (next != NULL) {
        k = next;
        next = k->next;

        /* Find k's place in sorted */
        kbf = (keytuple*)sorted->first;
        while (kbf!=NULL) {
            if (k->typ < kbf->typ) break;
            kbf = kbf->next;
        }
        
        /* Hook k into sorted list */
        if (kbf == NULL) {
            /* k is last in sorted */
            last = sorted->last;
            sorted->last = k;
            k->next = NULL;
            k->prev = last;
            last->next = k;
        } else {
            /* k goes just before kbf */
            k->next = kbf;
            k->prev = kbf->prev;
            if (kbf->prev != NULL) (kbf->prev)->next = k;
            else sorted->first = k;
            kbf->prev = k;
        }
        (sorted->n) ++;
    }

    /* Replace the input header by the sorted one */
    (*hdr)->first = (*hdr)->last = NULL;
    qfits_header_destroy(*hdr);
    *hdr = sorted;
    
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Copy a FITS header
  @param    src    Header to replicate
  @return    Pointer to newly allocated qfits_header object.

  Makes a strict copy of all information contained in the source
  header. The returned header must be freed using qfits_header_destroy.
 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_copy(const qfits_header * src)
{
    qfits_header    *   fh_copy;
    keytuple        *   k;

    if (src==NULL) return NULL;

    fh_copy = qfits_header_new();
    k = (keytuple*)src->first;
    while (k!=NULL) {
        qfits_header_append(fh_copy, k->key, k->val, k->com, k->lin);
        k = k->next;
    }
    return fh_copy;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    qfits_header destructor
  @param    hdr qfits_header to deallocate
  @return    void

  Frees all memory associated to a given qfits_header object.
 */
/*----------------------------------------------------------------------------*/
void qfits_header_destroy(qfits_header * hdr)
{
    keytuple * k;
    keytuple * kn;

    if (hdr==NULL) return;

    k = (keytuple*)hdr->first;
    while (k!=NULL) {
        kn = k->next;
        keytuple_del(k);
        k = kn;
    }
    qfits_free(hdr);
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Return the value associated to a key, as a string
  @param    hdr qfits_header to parse
  @param    key key to find
  @return    pointer to statically allocated string

  Finds the value associated to the given key and return it as a
  string. The returned pointer is statically allocated, so do not
  modify its contents or try to free it.

  Returns NULL if no matching key is found or no value is attached.
 */
/*----------------------------------------------------------------------------*/
char * qfits_header_getstr(const qfits_header * hdr, const char * key)
{
    keytuple    *   k;
    char            xkey[FITS_LINESZ+1];

    if (hdr==NULL || key==NULL) return NULL;

    qfits_expand_keyword_r(key, xkey);
    k = (keytuple*)hdr->first;
    while (k!=NULL) {
        if (!strcmp(k->key, xkey)) break;
        k=k->next;
    }
    if (k==NULL) return NULL;
    return k->val;
}

int qfits_header_getstr_pretty(const qfits_header* hdr, const char* key, char* pretty, const char* defaultval) {
	char* val = qfits_header_getstr(hdr, key);
	if (!val) {
		if (defaultval)
			strcpy(pretty, defaultval);
		return -1;
	}
	qfits_pretty_string_r(val, pretty);
	return 0;
}

static keytuple* get_keytuple(qfits_header* hdr, int idx) {
    if (idx == 0) {
	    hdr->current_idx = 0;
	    hdr->current = hdr->first;
	    return hdr->current;
	} else if (idx == hdr->current_idx + 1) {
	    hdr->current = ((keytuple*) (hdr->current))->next;
	    hdr->current_idx++;
	    return hdr->current;
	} else {
        keytuple* k = (keytuple*)hdr->first;
	    int count=0;
	    while (count<idx) {
            k = k->next;
            count++;
        }
        return k;
	}
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Return the i-th key/val/com/line tuple in a header.
  @param    hdr        Header to consider
  @param    idx        Index of the requested card
  @param    key        Output key
  @param    val        Output value
  @param    com        Output comment
  @param    lin        Output initial line
  @return    int 0 if Ok, -1 if error occurred.

  This function is useful to browse a FITS header object card by card.
  By iterating on the number of cards (available in the 'n' field of
  the qfits_header struct), you can retrieve the FITS lines and their
  components one by one. Indexes run from 0 to n-1. You can pass NULL
  values for key, val, com or lin if you are not interested in a
  given field.

  @code
  int i;
  char key[FITS_LINESZ+1];
  char val[FITS_LINESZ+1];
  char com[FITS_LINESZ+1];
  char lin[FITS_LINESZ+1];

  for (i=0; i<hdr->n; i++) {
      qfits_header_getitem(hdr, i, key, val, com, lin);
    printf("card[%d] key[%s] val[%s] com[%s]\n", i, key, val, com);
  }
  @endcode

  This function has primarily been written to interface a qfits_header
  object to other languages (C++/Python). If you are working within a
  C program, you should use the other header manipulation routines
  available in this module.
 */
/*----------------------------------------------------------------------------*/
int qfits_header_getitem(
        const qfits_header  *   hdr,
        int                     idx,
        char                *   key,
        char                *   val,
        char                *   com,
        char                *   lin)
{
    keytuple    *   k;

    if (hdr==NULL) return -1;
    if (key==NULL && val==NULL && com==NULL && lin==NULL) return 0;
    if (idx<0 || idx>=hdr->n) return -1;

    k = get_keytuple((qfits_header*)hdr, idx);

    /* Fill return values */
    if (key!=NULL) strcpy(key, k->key);
    if (val!=NULL) {
        if (k->val!=NULL) strcpy(val, k->val);
        else val[0]=0;
    }
    if (com!=NULL) {
        if (k->com!=NULL) strcpy(com, k->com);
        else com[0]=0;
    }
    if (lin!=NULL) {
        if (k->lin!=NULL) strcpy(lin, k->lin);
        else lin[0]=0;
    }
    return 0;
}

int qfits_header_setitem(
                         qfits_header  *   hdr,
                         int                     idx,
                         char                *   key,
                         char                *   val,
                         char                *   com,
                         char                *   lin) {
    keytuple    *   k;

    if (!hdr) return -1;
    if (!key && !val && !com && !lin) return 0;
    if (idx<0 || idx>=hdr->n) return -1;

    k = get_keytuple(hdr, idx);

    // free existing strings as per keytuple_del
    if (k->key)
        qfits_free(k->key);
    if (k->val)
        qfits_free(k->val);
    if (k->com)
        qfits_free(k->com);
    if (k->lin)
        qfits_free(k->lin);

    // copy input strings.
    if (key)
        k->key = qfits_strdup(key);
    else
        k->key = NULL;

    if (val)
        k->val = qfits_strdup(val);
    else
        k->val = NULL;

    if (com)
        k->com = qfits_strdup(com);
    else
        k->com = NULL;

    // the rest of the code expects this to be exactly 80 chars.
    if (lin) {
        k->lin = qfits_malloc(80);
        memcpy(k->lin, lin, 80);
    } else
        k->lin = NULL;
        
    return 0;
}


/*----------------------------------------------------------------------------*/
/**
  @brief    Return the comment associated to a key, as a string
  @param    hdr qfits_header to parse
  @param    key key to find
  @return    pointer to statically allocated string

  Finds the comment associated to the given key and return it as a
  string. The returned pointer is statically allocated, so do not
  modify its contents or try to free it.

  Returns NULL if no matching key is found or no comment is attached.
 */
/*----------------------------------------------------------------------------*/
char * qfits_header_getcom(const qfits_header * hdr, const char * key)
{
    keytuple    *   k;
    char            xkey[FITS_LINESZ+1];

    if (hdr==NULL || key==NULL) return NULL;

    qfits_expand_keyword_r(key, xkey);
    k = (keytuple*)hdr->first;
    while (k!=NULL) {
        if (!strcmp(k->key, xkey)) break;
        k=k->next;
    }
    if (k==NULL) return NULL;
    return k->com;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Return the value associated to a key, as an int
  @param    hdr qfits_header to parse
  @param    key key to find
  @param    errval default value to return if nothing is found
  @return    int

  Finds the value associated to the given key and return it as an
  int. Returns errval if no matching key is found or no value is
  attached.
 */
/*----------------------------------------------------------------------------*/
int qfits_header_getint(
        const qfits_header  *   hdr, 
        const char          *   key, 
        int                     errval)
{
    char    *   c;
    int         d;

    if (hdr==NULL || key==NULL) return errval;

    c = qfits_header_getstr(hdr, key);
    if (c==NULL) return errval;
    if (sscanf(c, "%d", &d)!=1) return errval;
    return d;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Return the value associated to a key, as a double
  @param    hdr qfits_header to parse
  @param    key key to find
  @param    errval default value to return if nothing is found
  @return    double

  Finds the value associated to the given key and return it as a
  double. Returns errval if no matching key is found or no value is
  attached.
 */
/*----------------------------------------------------------------------------*/
double qfits_header_getdouble(
        const qfits_header  *   hdr, 
        const char          *   key, 
        double                  errval)
{
    char    *    c;
    char* endptr;
    double d;

    if (hdr==NULL || key==NULL) return errval;

    c = qfits_header_getstr(hdr, key);
    if (c==NULL) return errval;

    d = strtod(c, &endptr);
    if (endptr == c)
        return errval;
    return d;
    //return atof(c);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Return the value associated to a key, as a boolean (int).
  @param    hdr qfits_header to parse
  @param    key key to find
  @param    errval default value to return if nothing is found
  @return    int

  Finds the value associated to the given key and return it as a
  boolean. Returns errval if no matching key is found or no value is
  attached. A boolean is here understood as an int taking the value 0
  or 1. errval can be set to any other integer value to reflect that
  nothing was found.

  errval is returned if no matching key is found or no value is
  attached.

  A true value is any character string beginning with a 'y' (yes), a
  't' (true) or the digit '1'. A false value is any character string
  beginning with a 'n' (no), a 'f' (false) or the digit '0'.
 */
/*----------------------------------------------------------------------------*/
int qfits_header_getboolean(
        const qfits_header  *   hdr, 
        const char          *   key, 
        int                     errval)
{
    char    *    c;
    int            ret;

    if (hdr==NULL || key==NULL) return errval;

    c = qfits_header_getstr(hdr, key);
    if (c==NULL) return errval;
    if (strlen(c)<1) return errval;

    if (c[0]=='y' || c[0]=='Y' || c[0]=='1' || c[0]=='t' || c[0]=='T') {
        ret = 1;
    } else if (c[0]=='n' || c[0]=='N' || c[0]=='0' || c[0]=='f' || c[0]=='F') {
        ret = 0;
    } else {
        ret = errval;
    }
    return ret;
}

int qfits_header_write_line(const qfits_header* hdr, int line, char* result) {
    keytuple* k;
    int i;

    k = hdr->first;
    for (i=0; i<line; i++) {
        k = k->next;
        if (!k)
            return -1;
    }
    qfits_header_makeline(result, k, 1);
    return 0;
}


/*----------------------------------------------------------------------------*/
/**
  @brief    Dump a FITS header to an opened file.
  @param    hdr     FITS header to dump
  @param    out     Opened file pointer
  @return   int 0 if Ok, -1 otherwise
  Dumps a FITS header to an opened file pointer.
 */
/*----------------------------------------------------------------------------*/
int qfits_header_dump(
        const qfits_header  *   hdr,
        FILE                *   out)
{
    keytuple    *   k;
    char            line[81];
    int             n_out;    

    if (hdr==NULL) return -1;
    if (out==NULL) out=stdout;

    k = (keytuple*)hdr->first;
    n_out = 0;
    while (k!=NULL) {
        /* Make line from information in the node */
        qfits_header_makeline(line, k, 1);
        if ((fwrite(line, 1, 80, out))!=80) {
            fprintf(stderr, "error dumping FITS header");
            return -1;
        }
        n_out ++;
        k=k->next;
    }
    /* Blank-pad the output */
    memset(line, ' ', 80);
    while (n_out % 36) {
        fwrite(line, 1, 80, out);
        n_out++;
    }
    return 0;
}


int qfits_header_list(
        const qfits_header  *   hdr,
        FILE                *   out)
{
    keytuple    *   k;
    char            line[81];
    int             n_out;    

    if (hdr==NULL) return -1;
    if (out==NULL) out=stdout;

    k = (keytuple*)hdr->first;
    n_out = 0;
    while (k!=NULL) {
        /* Make line from information in the node */
        qfits_header_makeline(line, k, 1);
        if ((fwrite(line, 1, 80, out))!=80) {
            fprintf(stderr, "error dumping FITS header");
            return -1;
        }
        fprintf(out, "\n");
        n_out ++;
        k=k->next;
    }
    return 0;
}

/**@}*/

/*----------------------------------------------------------------------------*/
/**
  @brief    keytuple constructor
  @param    key        Key associated to key tuple (cannot be NULL).
  @param    val        Value associated to key tuple.
  @param    com        Comment associated to key tuple.
  @param    lin        Initial line read from FITS header (if applicable).
  @return    1 pointer to newly allocated keytuple.

  This function is a keytuple creator. NULL values and zero-length strings
  are valid parameters for all but the key field. The returned object must
  be deallocated using keytuple_del().

 */
/*----------------------------------------------------------------------------*/
static keytuple * keytuple_new(
        const char * key,
        const char * val,
        const char * com,
        const char * lin)
{
    char xkey[FITS_LINESZ+1];
    keytuple    *    k;

    if (key==NULL) return NULL;

    /* Allocate space for new structure */
    k = qfits_malloc(sizeof(keytuple));
    /* Hook a copy of the new key */
    qfits_expand_keyword_r(key, xkey);
    k->key = qfits_strdup(xkey);
    /* Hook a copy of the value if defined */
    k->val = NULL;
    if (val!=NULL) {
		// Why was this here?
        //if (strlen(val)>0)
		k->val = qfits_strdup(val);
    }
    /* Hook a copy of the comment if defined */
    k->com = NULL;
    if (com!=NULL) {
        if (strlen(com)>0) k->com = qfits_strdup(com);
    }
    /* Hook a copy of the initial line if defined */
    k->lin = NULL;
    if (lin!=NULL) {
        if (strlen(lin)>0) k->lin = qfits_strdup(lin);
    }
    k->next = NULL;
    k->prev = NULL;
    k->typ = keytuple_type(key);

    return k;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    keytuple type identification routine
  @param    key        String representing a FITS keyword.
  @return    A key type (see keytype enum)

  This function identifies the type of a FITS keyword when given the
  keyword as a string. Keywords are expected literally as they are
  found in a FITS header on disk.

 */
/*----------------------------------------------------------------------------*/
static keytype keytuple_type(const char * key)
{
    keytype kt;

    kt = keytype_undef;
    /* Assign type to key tuple */
    if (!strcmp(key, "SIMPLE") || !strcmp(key, "XTENSION")) kt = keytype_top;
    else if (!strcmp(key, "END"))                   kt = keytype_end;
    else if (!strcmp(key, "BITPIX"))                kt = keytype_bitpix;
    else if (!strcmp(key, "NAXIS"))                 kt = keytype_naxis;
    else if (!strcmp(key, "NAXIS1"))                kt = keytype_naxis1;
    else if (!strcmp(key, "NAXIS2"))                kt = keytype_naxis2;
    else if (!strcmp(key, "NAXIS3"))                kt = keytype_naxis3;
    else if (!strcmp(key, "NAXIS4"))                kt = keytype_naxis4;
    else if (!strncmp(key, "NAXIS", 5))             kt = keytype_naxisi;
    else if (!strcmp(key, "GROUP"))                 kt = keytype_group;
    else if (!strcmp(key, "PCOUNT"))                kt = keytype_pcount;
    else if (!strcmp(key, "GCOUNT"))                kt = keytype_gcount;
    else if (!strcmp(key, "EXTEND"))                kt = keytype_extend;
    else if (!strcmp(key, "BSCALE"))                kt = keytype_bscale;
    else if (!strcmp(key, "BZERO"))                 kt = keytype_bzero;
    else if (!strcmp(key, "TFIELDS"))               kt = keytype_tfields;
    else if (!strncmp(key, "TBCOL", 5))             kt = keytype_tbcoli;
    else if (!strncmp(key, "TFORM", 5))             kt = keytype_tformi;
    else if (!strncmp(key, "HIERARCH ESO DPR", 16)) kt = keytype_hierarch_dpr;
    else if (!strncmp(key, "HIERARCH ESO OBS", 16)) kt = keytype_hierarch_obs;
    else if (!strncmp(key, "HIERARCH ESO TPL", 16)) kt = keytype_hierarch_tpl;
    else if (!strncmp(key, "HIERARCH ESO GEN", 16)) kt = keytype_hierarch_gen;
    else if (!strncmp(key, "HIERARCH ESO TEL", 16)) kt = keytype_hierarch_tel;
    else if (!strncmp(key, "HIERARCH ESO INS", 16)) kt = keytype_hierarch_ins;
    else if (!strncmp(key, "HIERARCH ESO LOG", 16)) kt = keytype_hierarch_log;
    else if (!strncmp(key, "HIERARCH ESO PRO", 16)) kt = keytype_hierarch_pro;
    else if (!strncmp(key, "HIERARCH", 8))          kt = keytype_hierarch;
    else if (!strcmp(key, "HISTORY"))               kt = keytype_history;
    else if (!strcmp(key, "COMMENT"))               kt = keytype_comment;
    else if (!strcmp(key, "CONTINUE"))              kt = keytype_continue;
    else if ((int)strlen(key)<9)                    kt = keytype_primary;
    return kt;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Keytuple destructor.
  @param    k    Keytuple to deallocate.
  @return    void
  Keytuple destructor.
 */
/*----------------------------------------------------------------------------*/
static void keytuple_del(keytuple * k)
{
    if (k==NULL) return;
    if (k->key) qfits_free(k->key);
    if (k->val) qfits_free(k->val);
    if (k->com) qfits_free(k->com);
    if (k->lin) qfits_free(k->lin);
    qfits_free(k);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Keytuple dumper.
  @param    k    Keytuple to dump
  @return    void

  This function dumps a key tuple to stdout. It is meant for debugging
  purposes only.
 */
/*----------------------------------------------------------------------------*/
static void keytuple_dmp(const keytuple * k) {
    if (!k) return;
    printf("[%s]=[", k->key); 
    if (k->val) printf("%s", k->val);
    printf("]");
    if (k->com) printf("/[%s]", k->com);
    printf("\n");
    return;
}

void qfits_header_debug_dump(const qfits_header* hdr) {
	keytuple* k;
	if (hdr==NULL) return;
	k = (keytuple*)hdr->first;
	while (k) {
        keytuple_dmp(k);
		k=k->next;
	}
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Build a FITS line from the information contained in a card.
  @param    line pointer to allocated string to be filled
  @param    node pointer to card node in qfits_header linked-list
  @param    conservative flag to indicate conservative behaviour
  @return    int 0 if Ok, anything else otherwise

  Build a FITS line (80 chars) from the information contained in a
  node of a qfits_header linked-list. If the mode is set to
  conservative, the original FITS line will be used wherever present.
  If conservative is set to 0, a new line will be formatted.
 */
/*----------------------------------------------------------------------------*/
static int qfits_header_makeline(
        char            *   line,
        const keytuple  *   k,
        int                 conservative)
{
    char blankline[81];
    int     i;

    if (line==NULL || k==NULL) return -1;

    /* If a previous line information is there, use it as is */
    if (conservative) {
        if (k->lin != NULL) {
            memcpy(line, k->lin, 80);
            line[80]='\0';
            return 0;
        }
    }
    /* Got to build keyword from scratch */
    memset(blankline, 0, 81);
    qfits_card_build(blankline, k->key, k->val, k->com);
    memset(line, ' ', 80);
    i=0;
    while (blankline[i] != '\0') {
        line[i] = blankline[i];
        i++;
    }
    line[80]='\0';
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief	Find a matching key in a header.
  @param	hdr	qfits_header to parse
  @param	key	Key prefix to match
  @return	pointer to statically allocated string.

  This function finds the first keyword in the given header for 
  which the given 'key' is a prefix, and returns the full name
  of the matching key (NOT ITS VALUE). This is useful to locate
  any keyword starting with a given prefix. Careful with HIERARCH
  keywords, the shortFITS notation is not likely to be accepted here.

  Examples:

  @verbatim
  s = qfits_header_findmatch(hdr, "SIMP") returns "SIMPLE"
  s = qfits_header_findmatch(hdr, "HIERARCH ESO DET") returns
  the first detector keyword among the HIERACH keys.
  @endverbatim
 */
/*----------------------------------------------------------------------------*/
char * qfits_header_findmatch(const qfits_header * hdr, const char * key)
{
	keytuple    *   k;

	if (hdr==NULL || key==NULL) return NULL;

	k = (keytuple*)hdr->first;
	while (k!=NULL) {
		if (!strncmp(k->key, key, (int)strlen(key))) break;
		k=k->next;
	}
	if (k==NULL) return NULL;
	return k->key;
}
