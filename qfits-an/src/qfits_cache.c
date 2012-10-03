/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_cache.c,v 1.12 2006/12/04 15:34:48 yjung Exp $
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/*
 * $Author: yjung $
 * $Date: 2006/12/04 15:34:48 $
 * $Revision: 1.12 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <assert.h>

#include "qfits_cache.h"

#include "qfits_card.h"
#include "qfits_std.h"
#include "qfits_memory.h"
#include "qfits_thread.h"
#include "qfits_error.h"

static qfits_lock_t cache_lock = QFITS_LOCK_INIT;


/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/* Define this symbol to get debug symbols -- not recommended! */
#define QFITS_CACHE_DEBUG    0
#if QFITS_CACHE_DEBUG
#define qdebug( code ) { code }
#else
#define qdebug( code )
#endif

/*
 * Cache size: 
 * Maximum number of FITS file informations stored in the cache.
 */
#define QFITS_CACHESZ        128

/*
 * This static definition declares the maximum possible number of
 * extensions in a FITS file. It only has effects in the qfits_cache_add
 * function where a table is statically allocated for efficiency reasons.
 * If the number of extensions grows over this limit, change the value of
 * this constant. If the number of extensions is a priori unknown but can
 * grow much larger than a predictable value, the best solution is to
 * implement a dynamic memory allocation in qfits_cache_add.
 */
#define QFITS_MAX_EXTS            10192

/*-----------------------------------------------------------------------------
                                   New types
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*
  This structure stores all informations about a given FITS file.
  It is strictly internal to this module.
 */
/*----------------------------------------------------------------------------*/
typedef struct _qfits_cache_cell_ {
    char    *    name;    /* File name     */
    ino_t       inode; /* Inode */
    time_t        mtime;  /* Last modification date */
    int            filesize; /* File size in bytes */
    time_t        ctime;  /* Last modification date */

    int            exts;    /* # of extensions in file */

    int        *    ohdr;    /* Offsets to headers */
    int        *    shdr;    /* Header sizes */
    int        *    data;    /* Offsets to data */
    int        *    dsiz;    /* Data sizes */

    int         fsize; /* File size in blocks (2880 bytes) */
} qfits_cache_cell;

static qfits_cache_cell qfits_cache[QFITS_CACHESZ];
static int qfits_cache_last = -1;
static int qfits_cache_entries = 0;
static int qfits_cache_init = 0;

/*-----------------------------------------------------------------------------
                            Functions prototypes
 -----------------------------------------------------------------------------*/

static void qfits_cache_activate(void);
static int qfits_is_cached(const char * filename);
static int qfits_cache_add(const char * name);

#if QFITS_CACHE_DEBUG
static void qfits_cache_dump(void);
#endif

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_cache     FITS caching capabilities
 *
 * This modules implements a cache for FITS access routines.
 * The first time a FITS file is seen by the library, all corresponding
 * pointers are cached here. This speeds up multiple accesses to large
 * files by magnitudes.
 *
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                              Function codes
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Purge the qfits cache.
  @return    void

  This function is useful for programs running for a long period,
  to clean up the cache. Ideally in a daemon, it should be called
  by a timer at regular intervals. Notice that since the cache is
  fairly small, you should not need to care too much about this.
 */
/*----------------------------------------------------------------------------*/
void qfits_cache_purge_impl(void)
{
    int    i;

    qdebug(
        printf("qfits: purging cache...\n");
    );

    for (i=0; i<QFITS_CACHESZ; i++) {
        if (qfits_cache[i].name!=NULL) {
            qfits_free(qfits_cache[i].name);
            qfits_cache[i].name = NULL;
            qfits_free(qfits_cache[i].ohdr);
            qfits_free(qfits_cache[i].data);
            qfits_free(qfits_cache[i].shdr);
            qfits_free(qfits_cache[i].dsiz);
            qfits_cache_entries --;
        }
    }
    if (qfits_cache_entries!=0) {
        qdebug(
            printf("qfits: internal error in cache consistency\n");
        );
        exit(-1);
    }
    qfits_cache_last = -1;
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Query a FITS file offset from the cache.
  @param    filename    Name of the file to examine.
  @param    what        What should be queried (see below).
  @return    an integer offset, or -1 if an error occurred.

  This function queries the cache for FITS offset information. If the
  requested file name has never been seen before, it is completely parsed
  to extract all offset informations, which are then stored in the cache.
  The next query will get the informations from the cache, avoiding
  a complete re-parsing of the file. This is especially useful for large
  FITS files with lots of extensions, because querying the extensions
  is an expensive operation.

  This operation has side-effects: the cache is an automatically
  allocated structure in memory, that can only grow. Every request
  on a new FITS file will make it grow. The structure is pretty
  light-weight in memory, but nonetheless this is an issue for daemon-type
  programs which must run over long periods. The solution is to clean
  the cache using qfits_cache_purge() at regular intervals. This is left
  to the user of this library.

  To request information about a FITS file, you must pass an integer
  built from the following symbols:

  - @c QFITS_QUERY_N_EXT
  - @c QFITS_QUERY_HDR_START
  - @c QFITS_QUERY_DAT_START
  - @c QFITS_QUERY_HDR_SIZE
  - @c QFITS_QUERY_DAT_SIZE

  Querying the number of extensions present in a file is done
  simply with:

  @code
  next = qfits_query(filename, QFITS_QUERY_N_EXT);
  @endcode

  Querying the offset to the i-th extension header is done with:

  @code
  off = qfits_query(filename, QFITS_QUERY_HDR_START | i);
  @endcode

  i.e. you must OR (|) the extension number with the
  @c QFITS_QUERY_HDR_START symbol. Requesting offsets to extension data is
  done in the same way:

  @code
  off = qfits_query(filename, QFITS_QUERY_DAT_START | i);
  @endcode

  Notice that extension 0 is the main header and main data part
  of the FITS file.
 */
/*----------------------------------------------------------------------------*/
size_t qfits_query_impl_long(const char * filename, int what) {
    int    rank;
    int    which;
    size_t answer;

    qdebug(printf("qfits: cache req %s\n", filename););
    if ((rank=qfits_is_cached(filename))==-1) {
        rank = qfits_cache_add(filename);
    }
    if (rank==-1) {
        qfits_error("error adding file \"%s\" to cache", filename);
        return QFITS_QUERY_ERROR;
    }

    /* See what was requested */
    answer=QFITS_QUERY_ERROR;
    if (what & QFITS_QUERY_N_EXT) {
        answer = qfits_cache[rank].exts;
        qdebug(
            printf("qfits: query n_exts\n");
            printf("qfits: -> %d\n", answer);
        );
    } else if (what & QFITS_QUERY_HDR_START) {
        which = what & (~QFITS_QUERY_HDR_START);
        if (which>=0 && which<=qfits_cache[rank].exts) {
            answer = ((size_t)qfits_cache[rank].ohdr[which]) * FITS_BLOCK_SIZE;
        }
        qdebug(
            printf("qfits: query offset to header %d\n", which);
            printf("qfits: -> %d (%d bytes)\n", answer/2880, answer);
        );
    } else if (what & QFITS_QUERY_DAT_START) {
        which = what & (~QFITS_QUERY_DAT_START);
        if (which>=0 && which<=qfits_cache[rank].exts) {
            answer = ((size_t)qfits_cache[rank].data[which]) * FITS_BLOCK_SIZE;
        }
        qdebug(
            printf("qfits: query offset to data %d\n", which);
            printf("qfits: -> %d (%d bytes)\n", answer/2880, answer);
        );
    } else if (what & QFITS_QUERY_HDR_SIZE) {
        which = what & (~QFITS_QUERY_HDR_SIZE);
        if (which>=0 && which<=qfits_cache[rank].exts) {
            answer = ((size_t)qfits_cache[rank].shdr[which]) * FITS_BLOCK_SIZE;
        }
        qdebug(
            printf("qfits: query sizeof header %d\n", which);
            printf("qfits: -> %d (%d bytes)\n", answer/2880, answer);
        );
    } else if (what & QFITS_QUERY_DAT_SIZE) {
        which = what & (~QFITS_QUERY_DAT_SIZE);
        if (which>=0 && which<=qfits_cache[rank].exts) {
            answer = ((size_t)qfits_cache[rank].dsiz[which]) * FITS_BLOCK_SIZE;
        }
        qdebug(
            printf("qfits: query sizeof data %d\n", which);
            printf("qfits: -> %d (%d bytes)\n", answer/2880, answer);
        );
    }
    return answer;
}

int qfits_query_impl(const char * filename, int what) {
	int ires;
	size_t res = qfits_query_impl_long(filename, what);
	ires = (int)res;
	if (res != (size_t)res) {
		qfits_error("Warning: long return value %lli will get truncated to %i in qfits_query_impl()",
					(long long int)res, ires);
	}
	return ires;
}


/**@}*/

/**
   target_hdu: if != -1, the HDU you wish to know about; reading will stop after this
   HDU, and the cache cell will be incomplete.
 */
int qfits_parse_file(const char* filename,
		     qfits_cache_cell* qc,
		     int target_hdu) {
  struct stat sta;
  FILE    *    in = NULL;
  char        buf[FITS_BLOCK_SIZE];

  char    *    buf_c;
  int            n_blocks;
  int            found_it;
  int            xtend;
  int            naxis;
  char    *    read_val;
  int            last;
  int            end_of_file;
  int            data_bytes;
  int            skip_blocks;
  int         seeked;
  int            i;

  int rtn = -1;

  int *off_hdr = NULL;
  int *off_dat = NULL;
  int off_size = QFITS_MAX_EXTS;
  char getval_buf[FITS_LINESZ+1];

  /* Stat file to get its size */
  if (stat(filename, &sta)!=0) {
    qfits_error("cannot stat file \"%s\": %s", filename, strerror(errno));
    return -1;
  }

  /* Initialize cache cell */
  qc->exts=0;
  qc->inode= sta.st_ino;

  /* Open input file */
  if ((in=fopen(filename, "r"))==NULL) {
    qfits_error("cannot open file \"%s\": %s", filename, strerror(errno));
    goto cleanup;
  }

  /* Read first block in */
  if (fread(buf, 1, FITS_BLOCK_SIZE, in)!=FITS_BLOCK_SIZE) {
    qfits_error("error reading first block from \"%s\": %s\n", filename, strerror(errno));
    goto cleanup;
  }

  /* Identify FITS magic number */
  if (strncmp(buf, "SIMPLE  =", 9)) {
    qfits_error("file \"%s\" is not FITS (does not start with \"SIMPLE  =\")", filename);
    goto cleanup;
  }

  /*
   * Browse through file to identify primary HDU size and see if there
   * might be some extensions. The size of the primary data zone will
   * also be estimated from the gathering of the NAXIS?? values and
   * BITPIX.
   */

  /* Rewind input file, END card might be in first block */
  rewind(in);

  /* Initialize all counters */
  n_blocks = 0;
  found_it = 0;
  xtend = 0;
  naxis = 0;
  data_bytes = 1;

  /* Start looking for END card */
  while (found_it==0) {
    /* Read one FITS block */
    if (fread(buf, 1, FITS_BLOCK_SIZE, in)!=FITS_BLOCK_SIZE) {
      qfits_error("error reading file %s\n", filename);
      goto cleanup;
    }
    n_blocks ++;
    /* Browse through current block */
    buf_c = buf;
    for (i=0; i<FITS_NCARDS; i++) {
      /* Look for BITPIX keyword */
      if (strncmp(buf_c, "BITPIX ", 7) == 0) {
	read_val = qfits_getvalue_r(buf_c, getval_buf);
	data_bytes *= (int)atoi(read_val) / 8;
	if (data_bytes<0) data_bytes *= -1;
      }
      /* Look for NAXIS keyword */
      else if (strncmp(buf_c, "NAXIS", 5) == 0) {
	if (buf_c[5]==' ') {
	  /* NAXIS keyword */
	  read_val = qfits_getvalue_r(buf_c, getval_buf);
	  naxis = (int)atoi(read_val);
	} else {
	  /* NAXIS?? keyword (axis size) */
	  read_val = qfits_getvalue_r(buf_c, getval_buf);
	  data_bytes *= (int)atoi(read_val);
	}
      }
      /* Look for EXTEND keyword */
      else if (strncmp(buf_c, "EXTEND ", 7) == 0) {
	/* The EXTEND keyword is present: might be some extensions */
	read_val = qfits_getvalue_r(buf_c, getval_buf);
	if (read_val[0]=='T' || read_val[0]=='1') {
	  xtend=1;
	}
      }
      /* Look for END keyword */
      else if (strncmp(buf_c, "END ", 4) == 0) {
	found_it = 1;
      }
      buf_c += FITS_LINESZ;
    }
  }

  /* Set first HDU offsets */
  off_hdr = malloc(off_size * sizeof(int));
  off_dat = malloc(off_size * sizeof(int));
  assert(off_hdr);
  assert(off_dat);
  off_hdr[0] = 0;
  off_dat[0] = n_blocks;
    
  /* Last is the pointer to the last added extension, plus one. */
  last = 1;

  if (xtend) {
    /* Look for extensions */
    qdebug(
	   printf("qfits: searching for extensions in %s\n", filename);
	   );

    /*
     * Register all extension offsets
     */
    end_of_file = 0;
    while (end_of_file==0) {

      if ((last-1) == target_hdu) {
	printf("Early-quitting FITS file parsing at HDU %i (file \"%s\")\n",
	       target_hdu, filename);
	break;
      }

      /*
       * Skip the previous data section if pixels were declared
       */
      if (naxis>0) {
	/* Skip as many blocks as there are declared pixels */
	skip_blocks = data_bytes/FITS_BLOCK_SIZE;
	if ((data_bytes % FITS_BLOCK_SIZE)!=0) {
	  skip_blocks ++;
	}
	seeked = fseek(in, skip_blocks*FITS_BLOCK_SIZE, SEEK_CUR);
	if (seeked<0) {
	  qfits_error("error seeking file \"%s\"\n", filename);
	  goto cleanup;
	}
	/* Increase counter of current seen blocks. */
	n_blocks += skip_blocks;
      }

      /* Look for extension start */
      found_it=0;
      while ((found_it==0) && (end_of_file==0)) {
	if (fread(buf,1,FITS_BLOCK_SIZE,in)!=FITS_BLOCK_SIZE) {
	  /* Reached end of file */
	  end_of_file=1;
	  break;
	}
	n_blocks ++;
	/* Search for XTENSION at block top */
	if (strncmp(buf, "XTENSION=", 9) == 0) {
	  /* Got an extension */
	  found_it=1;
	  off_hdr[last] = n_blocks-1;
	}
      }
      if (end_of_file) break;

      /*
       * Look for extension END
       * Rewind one block backwards, END might be in same section as
       * XTENSION start.
       */
      if (fseek(in, -FITS_BLOCK_SIZE, SEEK_CUR)==-1) {
	qfits_error("error fseeking file \"%s\" backwards\n", filename);
	goto cleanup;
      }
      n_blocks--;
      found_it = 0;
      data_bytes = 1;
      naxis = 0;
      while ((found_it==0) && (end_of_file==0)) {
	if (fread(buf,1,FITS_BLOCK_SIZE,in)!=FITS_BLOCK_SIZE) {
	  qfits_error("XTENSION without END in %s\n", filename);
	  end_of_file=1;
	  break;
	}
	n_blocks++;

	/* Browse current block */
	buf_c = buf;
	for (i=0; i<FITS_NCARDS; i++) {
	  /* Look for BITPIX keyword */
	  if (strncmp(buf_c, "BITPIX ", 7) == 0) {
	    read_val = qfits_getvalue_r(buf_c, getval_buf);
	    data_bytes *= (int)atoi(read_val) / 8;
	    if (data_bytes<0) data_bytes *= -1;
	  } else
	    /* Look for NAXIS keyword */
	    if (strncmp(buf_c, "NAXIS", 5) == 0) {
	      if (buf_c[5]==' ') {
		/* NAXIS keyword */
		read_val = qfits_getvalue_r(buf_c, getval_buf);
		naxis = (int)atoi(read_val);
	      } else {
		/* NAXIS?? keyword (axis size) */
		read_val = qfits_getvalue_r(buf_c, getval_buf);
		data_bytes *= (int)atoi(read_val);
	      }
	    } else
	      /* Look for END keyword */
	      if (strncmp(buf_c, "END ", 4) == 0) {
		/* Got the END card */
		found_it=1;
		/* Update registered extension list */
		off_dat[last] = n_blocks;
		last ++;
		qc->exts ++;
		if (last >= off_size) {
		  off_size *= 2;
		  off_hdr = realloc(off_hdr, off_size * sizeof(int));
		  off_dat = realloc(off_dat, off_size * sizeof(int));
		  assert(off_hdr);
		  assert(off_dat);
		}
		break;
	      }
	  buf_c+=FITS_LINESZ;
	}
      }
    }
  }

  /* Close file */
  fclose(in);
  in = NULL;

  /* Allocate buffers in cache */
  qc->ohdr = qfits_malloc(last * sizeof(int));
  qc->data = qfits_malloc(last * sizeof(int));
  qc->shdr = qfits_malloc(last * sizeof(int));
  qc->dsiz = qfits_malloc(last * sizeof(int));
  /* Store retrieved pointers in the cache */
  for (i=0; i<last; i++) {
    /* Offsets to start */
    qc->ohdr[i] = off_hdr[i];
    qc->data[i] = off_dat[i];
    /* Sizes */
    qc->shdr[i] = off_dat[i] - off_hdr[i];
    if (i==last-1) {    
      qc->dsiz[i] = (sta.st_size/FITS_BLOCK_SIZE) - off_dat[i];
    } else {
      qc->dsiz[i] = off_hdr[i+1] - off_dat[i];
    }
  }
  qc->fsize = sta.st_size / FITS_BLOCK_SIZE;
  /* Add last modification date */
  qc->mtime = sta.st_mtime;
  qc->filesize  = sta.st_size;
  qc->ctime = sta.st_ctime;
  qc->name = qfits_strdup(filename);
  qfits_cache_entries ++;

  qdebug(
	 qfits_cache_dump();
	 );
  rtn = 0;

 cleanup:
  if (in)
    fclose(in);
  free(off_hdr);
  free(off_dat);
  if (rtn) {
    qfits_free(qc->name);
    qfits_free(qc->ohdr);
    qfits_free(qc->data);
    qfits_free(qc->shdr);
    qfits_free(qc->dsiz);
  }

  return rtn;
}



/*----------------------------------------------------------------------------*/
/**
  @brief    Add pointer information about a file into the qfits cache.
  @param    filename    Name of the file to examine.
  @return    index to the file information in the cache, or -1 if failure.

  This function picks a file name, and examines the corresponding FITS file
  to deduce all relevant pointers in the file (byte offsets). These byte
  offsets are later used to speed up header lookups. Example: requesting
  some keyword information in the header of the n-th extension will first
  fseek the file to the header start, then search from this position
  onwards. This means that the input FITS file is only parsed for extension
  positions once.

  What this function does is:

  - Open the file, read the first FITS block (@c FITS_BLOCK_SIZE bytes)
  - Check the file is FITS (must have SIMPLE  = at top)
  - Register start of first header at offset 0.
  - Look for END keyword, register start of first data section
    if NAXIS>0.
  - If the EXTEND=T line was found, continue looking for extensions.
  - For each consecutive extension, register extension header start
    and extension data start.
 */
/*----------------------------------------------------------------------------*/
static int qfits_cache_add(const char * filename)
{
    qfits_cache_cell * qc;

    /* Initialize cache if not done yet (done only once) */
    if (qfits_cache_init==0) {
        qfits_cache_init++;
        qfits_cache_activate();
    }

    /*
     * Prepare qfits cache for addition of a new entry
     */
    qfits_cache_last++;
    /* Rotate buffer if needed */
    if (qfits_cache_last >= QFITS_CACHESZ) {
        qfits_cache_last = 0;
    }
    /* Alias to current pointer in cache for easier reading */
    qc = &(qfits_cache[qfits_cache_last]);

    /* Clean cache cell if needed */
    if (qc->name!=NULL) {
        qfits_free(qc->name);
        qc->name = NULL;
        qfits_free(qc->ohdr);
        qfits_free(qc->data);
        qfits_free(qc->shdr);
        qfits_free(qc->dsiz);
	qc->ohdr = qc->shdr = qc->data = qc->dsiz = NULL;
        qfits_cache_entries --;
    }
    
    if (qfits_parse_file(filename, qc, -1)) {
      return -1;
    }

    /* Return index of the added file in the cache */
    return qfits_cache_last;
}

#if QFITS_CACHE_DEBUG
static void qfits_cache_dump(void)
{
    int i, j;

    printf("qfits: dumping cache...\n");

    printf("cache contains %d entries\n", qfits_cache_entries);
    for (i=0; i<QFITS_CACHESZ; i++) {
        if (qfits_cache[i].name!=NULL) {
            printf("qfits: -----> entry: %d\n", i);
            printf("qfits: name  %s\n", qfits_cache[i].name);
            printf("qfits: exts  %d\n", qfits_cache[i].exts);
            printf("qfits: size  %d\n", qfits_cache[i].fsize);
            printf("qfits: ohdr  %d\n"
                   "qfits: shdr  %d\n"
                   "qfits: data  %d\n"
                   "qfits: dsiz  %d\n",
                   qfits_cache[i].ohdr[0],
                   qfits_cache[i].shdr[0],
                   qfits_cache[i].data[0],
                   qfits_cache[i].dsiz[0]);
            if (qfits_cache[i].exts>0) {
                for (j=1; j<=qfits_cache[i].exts; j++) {
                    printf("qfits: %s [%d]\n", qfits_cache[i].name, j);
                    printf("qfits: ohdr  %d\n"
                           "qfits: shdr  %d\n"
                           "qfits: data  %d\n"
                           "qfits: dsiz  %d\n",
                           qfits_cache[i].ohdr[j],
                           qfits_cache[i].shdr[j],
                           qfits_cache[i].data[j],
                           qfits_cache[i].dsiz[j]);
                }
            }
        }
    }
    return;
}
#endif

/*----------------------------------------------------------------------------*/
/**
  @brief    initialize cache buffer with minimum size
 */
/*----------------------------------------------------------------------------*/
static void qfits_cache_activate(void)
{
    int i;
    qdebug(
        printf("qfits: activating cache...\n");
    );
    /* Set all slots to NULL */
    for (i=0; i<QFITS_CACHESZ; i++) {
        qfits_cache[i].name = NULL;
    }
    /* Register purge function with atexit */
    atexit(qfits_cache_purge);
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Find out if a file is in the cache already
  @param    filename    file name
  @return   int 1 if in the cache, 0 if not
 */
/*----------------------------------------------------------------------------*/
static int qfits_is_cached(const char * filename)
{
    int            i, n;
    struct stat sta;

    /* Stat input file */
    if (stat(filename, &sta)!=0) {
        return -1;
    }
    n=0;
    /* Loop over all cache entries */
    for (i=0; i<QFITS_CACHESZ; i++) {
        /* If entry is valid (name is not NULL) */
        if (qfits_cache[i].name!=NULL) {
            /* One more entry found */
            n++;
            /* If inode is the same */
            if ((qfits_cache[i].inode == sta.st_ino) &&
                (qfits_cache[i].mtime == sta.st_mtime) &&
                (qfits_cache[i].filesize  == sta.st_size) &&
                (qfits_cache[i].ctime == sta.st_ctime)) {
                /* This is the requested file */
                return i;
            }
        }
        /* Early exit: all entries have been browsed */
        if (n>=qfits_cache_entries) {
            return -1;
        }
    }
    return -1;
}








int qfits_query(const char * filename, int what) {
	int rtn;
	qfits_lock_lock(&cache_lock);
	rtn = qfits_query_impl(filename, what);
	qfits_lock_unlock(&cache_lock);
	return rtn;
}

size_t qfits_query_long(const char * filename, int what) {
	size_t rtn;
	qfits_lock_lock(&cache_lock);
	rtn = qfits_query_impl_long(filename, what);
	qfits_lock_unlock(&cache_lock);
	return rtn;
}

void qfits_cache_purge(void) {
	qfits_lock_lock(&cache_lock);
	qfits_cache_purge_impl();
	qfits_lock_unlock(&cache_lock);
}

/*
static int qfits_cache_add(const char * filename) {
	int rtn;
	qfits_lock_lock(&cache_lock);
	rtn = qfits_cache_add_impl(filename);
	qfits_lock_unlock(&cache_lock);
	return rtn;
}

#if QFITS_CACHE_DEBUG
static void qfits_cache_dump(void) {
	qfits_lock_lock(&cache_lock);
	qfits_cache_dump_impl();
	qfits_lock_unlock(&cache_lock);
}
#endif

static void qfits_cache_activate(void) {
	qfits_lock_lock(&cache_lock);
	qfits_cache_activate_impl();
	qfits_lock_unlock(&cache_lock);
}

static int qfits_is_cached(const char * filename) {
	int rtn;
	qfits_lock_lock(&cache_lock);
	rtn = qfits_is_cached_impl(filename);
	qfits_lock_unlock(&cache_lock);
	return rtn;
}
 */

