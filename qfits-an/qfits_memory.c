/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_memory.c,v 1.4 2006/08/24 15:01:56 yjung Exp $
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
 * $Date: 2006/08/24 15:01:56 $
 * $Revision: 1.4 $
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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <errno.h>
#include <assert.h>

#include "qfits_error.h"

/*-----------------------------------------------------------------------------
                                Defines
 -----------------------------------------------------------------------------*/

/*
  This symbol sets the debug level for the memory module. Debug levels are 
  defined as follows:

  0   no debugging.
  1   add tracing for memory leaks and diagnostics in qfits_memory_status 
  2   add lots of debug messages
*/
#ifndef QFITS_MEMORY_DEBUG
#define QFITS_MEMORY_DEBUG       0
#endif

/*
  This symbol defines the level of usage of the memory module. 

  0   Use the memory system calls.
  1   Use the memory system calls, but exit if they are not succesfull
  2   Fully use the memory functions
*/
#ifndef QFITS_MEMORY_MODE
#define QFITS_MEMORY_MODE        0
#endif

/* Initial number of entries in memory table */
/* If this number is big, the size of the memory table can become
   problematic.
 */
#ifndef QFITS_MEMORY_MAXPTRS
#define QFITS_MEMORY_MAXPTRS     200003
#endif

/* Identify true RAM memory */
#define MEMTYPE_RAM         'R'
/* Identify swap memory */
#define MEMTYPE_SWAP        'S'
/* Identify memory-mapped file */
#define MEMTYPE_MMAP        'M'

/* Minimal page size in bytes */
#define MEMPAGESZ           2048

/* Size of temporary dir name */
#define TMPDIRNAMESZ        1024

/* Size of temporary file names */
#define TMPFILENAMESZ       1024

/* Size of source file names */
#define SRCFILENAMESZ       64

/* Size of mapped file names */
#define MAPFILENAMESZ       256

/*-----------------------------------------------------------------------------
                                Macros
 -----------------------------------------------------------------------------*/

/* Debug */
#if (QFITS_MEMORY_DEBUG>=2)
#define qfits_mem_debug( code ) { code }
#else
#define qfits_mem_debug( code )
#endif

/* A very simple hash */
#define PTR_HASH(ptr) (((unsigned long int) ptr) % QFITS_MEMORY_MAXPTRS)

/*-----------------------------------------------------------------------------
                        Private variables
 -----------------------------------------------------------------------------*/

/* Initialization flag */
static int  qfits_memory_initialized=0;

/* Path to temporary directory */
static char qfits_memory_tmpdirname[TMPDIRNAMESZ] = ".";

/*----------------------------------------------------------------------------*/
/*
  This table holds a list pointer cells (all the ones allocated so far).
  It is strictly internal to this source file.
 */
/*----------------------------------------------------------------------------*/
static struct {
    /* Number of active cells */
    int                 ncells;
    /* Total allocated memory in bytes */
    size_t              alloc_total;
    /* Total allocated RAM in bytes */
    size_t              alloc_ram;
    /* Total allocated VM in bytes */
    size_t              alloc_swap;
    /* Peak allocation ever seen for diagnostics */
    size_t              alloc_max;
    /* Peak number of pointers ever seen for diagnostics */
    int                 max_cells;

    /* Current number of swap files */
    int                 nswapfiles;
    /* Registration counter for swap files */
    int                 file_reg;

    /* Current number of memory-mapped files */
    int                 n_mm_files;
    /* Current number of mappings derived from files */
    int                 n_mm_mappings;

#ifdef __linux__
    /* Page size in bytes (Linux only) */
    int                 pagesize;
    /* Value found for RLIMIT_DATA (Linux only) */
    int                 rlimit_data;
#endif
} qfits_memory_table;

/* Various infos about the pointers */
/* List of pointers (outside of cells for efficiency reason) */
static void *   qfits_memory_p_val[QFITS_MEMORY_MAXPTRS];
/* Pointed size in bytes */
static size_t   qfits_memory_p_size[QFITS_MEMORY_MAXPTRS];
#if (QFITS_MEMORY_DEBUG>=1)
/* Name of the source file where the alloc was requested */
static char *   qfits_memory_p_filename[QFITS_MEMORY_MAXPTRS];
/* Line number where the alloc was requested */
static int      qfits_memory_p_lineno[QFITS_MEMORY_MAXPTRS];
#endif
/* Memory type: RAM, swap, or mapped file */
static char     qfits_memory_p_memtype[QFITS_MEMORY_MAXPTRS];
/* Swap memory only */
/* Swap file ID */
static int      qfits_memory_p_swapfileid[QFITS_MEMORY_MAXPTRS];
/* Swap file descriptor */
static int      qfits_memory_p_swapfd[QFITS_MEMORY_MAXPTRS];
/* Mapped files only */
/* Name of mapped file */
static char     qfits_memory_p_mm_filename[QFITS_MEMORY_MAXPTRS][MAPFILENAMESZ];
/* Hash of mapped file name for quick search */
static unsigned qfits_memory_p_mm_hash[QFITS_MEMORY_MAXPTRS];
/* Reference counter for this pointer */
static int      qfits_memory_p_mm_refcount[QFITS_MEMORY_MAXPTRS];

/*-----------------------------------------------------------------------------
                    Private function prototypes 
 -----------------------------------------------------------------------------*/

static unsigned qfits_memory_hash(const char *);
static void qfits_memory_init(void);
static void qfits_memory_cleanup(void);
static int qfits_memory_addcell(void*, size_t, const char*, int, char, int, 
        int, const char*);
static int qfits_memory_remcell(int);
static void qfits_memory_dumpcell(int, FILE*);
static char * qfits_memory_tmpfilename(int);
static char * strdup_(const char * str);
void qfits_memory_status_(const char *, int);

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_memory     POSIX-compatible extended memory handling
 *
 * qfits_memory is a small and efficient module offering memory extension 
 * capabitilies to ANSI C programs running on POSIX-compliant systems. It
 * offers several useful features such as memory leak detection, protection for 
 * free on NULL or unallocated pointers, and virtually unlimited memory space.
 * qfits_memory requires the @c mmap() system call to be implemented in the 
 * local C library to function. This module has been tested on a number of 
 * current Unix * flavours and is reported to work fine.
 * The current limitation is the limited number of pointers it can handle at
 * the same time.
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                            Function codes
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Allocate memory.
  @param    size        Size (in bytes) to allocate.
  @param    filename    Name of the file where the alloc took place.
  @param    lineno      Line number in the file.
  @return   1 newly allocated pointer.

  This function is a replacement call for malloc. It should never be called
  directly but through a macro instead, as:

  @code
  qfits_memory_malloc(size, __FILE__, __LINE__)
  @endcode
 */
/*----------------------------------------------------------------------------*/
void * qfits_memory_malloc(
        size_t          size, 
        const char  *   filename, 
        int             lineno)
{
    void    *   ptr;
    char    *   fname;
    int         swapfileid;
    int         swapfd;
    char        wbuf[MEMPAGESZ];
    int         nbufs;
    int         memtype;
    int         i;
#ifdef __linux__
    int         p;
#endif

    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if (QFITS_MEMORY_MODE == 0) return malloc(size);
    else if (QFITS_MEMORY_MODE == 1) {
        ptr = malloc(size);
        if (ptr == NULL) exit(1);
        else return ptr;
    }
    
    /* Initialize table if needed */
    if (qfits_memory_initialized==0) {
        qfits_memory_init();
        qfits_memory_initialized++;
    }

    /* Protect the call */
    if (size==0) {
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: malloc called with 0 size - %s (%d)\n",
                    filename, lineno);
        );
        return NULL;
    }

    /* Try to allocate in memory */
#ifdef __linux__
    /* Linux does not honor the RLIMIT_DATA limit.
     * The only way to limit the amount of memory taken by
     * a process is to set RLIMIT_AS, which unfortunately also
     * limits down the maximal amount of memory addressable with
     * mmap() calls, making on-the-fly swap space creation useless
     * in this module. To avoid this, the RLIMIT_DATA value
     * is honored here with this test.
     */
    ptr = NULL;
    if (qfits_memory_table.rlimit_data<1) {
        /* No limit set on RLIMIT_DATA: proceed with malloc */
        ptr = malloc(size);
    } else if (qfits_memory_table.alloc_total+size <= 
            (size_t)qfits_memory_table.rlimit_data) {
        /* Next allocation will still be within limits: proceed */
        ptr = malloc(size);
    }
#else
    ptr = malloc(size);
#endif
    if (ptr==NULL) {
        /* No more RAM available: try to allocate private swap */
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: hit a NULL pointer -- swapping\n");
        );

        /* Create swap file with rights: rw-rw-rw- */
        swapfileid = ++ qfits_memory_table.file_reg;
        fname = qfits_memory_tmpfilename(swapfileid);
        swapfd = open(fname, O_RDWR | O_CREAT);
        if (swapfd==-1) {
            fprintf(stderr, "qfits_mem: cannot create swap file\n");
            exit(-1);
        }
        fchmod(swapfd, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);

        /* Compute number of passes to insert buffer */
        nbufs = size / MEMPAGESZ;
        if (size % MEMPAGESZ != 0) nbufs ++;

        /* Dump empty buffers into file */
        memset(wbuf, 0, MEMPAGESZ);
        for (i=0; i<nbufs; i++) {
            if (write(swapfd, wbuf, MEMPAGESZ)==-1) {
                perror("write");
                fprintf(stderr,
                        "qfits_mem: fatal error: cannot create swapfile\n");
                close(swapfd);
                remove(fname);
                exit(-1);
            }
        }

        /* mmap() the swap file */
        ptr = (void*)mmap(0,
                          nbufs * MEMPAGESZ,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE,
                          swapfd,
                          0);
        if ((char*)ptr == (char*)-1) {
            perror("mmap");
            fprintf(stderr,
                    "qfits_mem: fatal error: mmap failed for swap file\n");
            close(swapfd);
            remove(fname);
            exit(-1);
        }

        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: swap [%s] created for %ld bytes\n", 
                fname, (long)size);
        );

        memtype = MEMTYPE_SWAP;
        qfits_memory_table.alloc_swap += size;
        qfits_memory_table.nswapfiles ++;
    } else {
        /* Memory allocation succeeded */
#ifdef __linux__
        /*
         * On Linux, the returned pointer might not be honored later.
         * To make sure the returned memory is actually usable, it has to
         * be touched. The following will touch one byte every 'pagesize'
         * bytes to make sure all blocks are visited and properly allocated
         * by the OS.
         */
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: touching memory (Linux)\n");
        );
        for (p=0; p<(int)size; p+=qfits_memory_table.pagesize) 
            ((char*)ptr)[p] = 0;
#endif
        swapfd = -1;
        swapfileid = -1;
        memtype = MEMTYPE_RAM;
        qfits_memory_table.alloc_ram   += size;
    }
    
    /* Print out message in debug mode */
    qfits_mem_debug(
        fprintf(stderr, "qfits_mem: %p alloc(%ld) in %s (%d)\n",
            ptr, (long)size, filename, lineno);
    );

    /* Add cell into general table */
    qfits_memory_addcell(  ptr,
                            size,
                            filename,
                            lineno,
                            memtype,
                            swapfileid,
                            swapfd,
                            NULL);
    /* Adjust size */
    qfits_memory_table.alloc_total += size;
    /* Remember biggest allocated block */
    if (qfits_memory_table.alloc_total > qfits_memory_table.alloc_max)
        qfits_memory_table.alloc_max = qfits_memory_table.alloc_total;

    /* Insert memory stamp */
    return (void*)ptr;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Allocate memory.
  @param    nmemb       Number of elements to allocate.
  @param    size        Size (in bytes) of each element.
  @param    filename    Name of the file where the alloc took place.
  @param    lineno      Line number in the file.
  @return   1 newly allocated pointer.

  This function is a replacement call for calloc. It should never be called
  directly but through a macro instead, as:

  @code
  qfits_memory_calloc(nmemb, size, __FILE__, __LINE__)
  @endcode
 */
/*----------------------------------------------------------------------------*/
void * qfits_memory_calloc(
        size_t          nmemb, 
        size_t          size, 
        const char  *   filename, 
        int             lineno)
{
    void    *   ptr;

    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if (QFITS_MEMORY_MODE == 0) return calloc(nmemb, size);
    else if (QFITS_MEMORY_MODE == 1) {
        ptr = calloc(nmemb, size);
        if (ptr == NULL) exit(1);
        else return ptr;
    }
    
    ptr = qfits_memory_malloc(nmemb * size, filename, lineno);
    return memset(ptr, 0, nmemb * size);
}


// copied from ioutils.c

static void get_mmap_size(size_t start, size_t size, off_t* mapstart, size_t* mapsize, int* pgap) {
	int ps = getpagesize();
	int gap = start % ps;
	// start must be a multiple of pagesize.
	*mapstart = start - gap;
	*mapsize  = size  + gap;
	*pgap = gap;
}

void qfits_memory_fdealloc2(
        void        *   ptr, 
		size_t len,
        const char  *   filename, 
        int             lineno) {
	if (munmap(ptr, len)) {
		qfits_error("qfits_memory_fdealloc2(%s:%i): Failed to munmap(): %s\n",
					filename, lineno, strerror(errno));
	}
}


void* qfits_memory_falloc2(
	const char* name,
	size_t      offs,
	size_t      size,
	char** freeaddr,
	size_t* freesize,
	const char  *   srcname,
	int             srclin) {

    char        *   ptr;
    struct stat     sta;
    int             fd;
	int eno;
	size_t maplen;
	off_t mapstart;
	int mapoff;

	/* Check file's existence and compute its size */
	if (stat(name, &sta)==-1) {
		qfits_warning("qfits_memory_falloc2(%s:%i): cannot stat file \"%s\"\n",
					  srcname, srclin, name);
		if (QFITS_MEMORY_MODE == 0) return NULL;
		else exit(1);
	}
	/* Check offset request does not go past end of file */
	if ((offs + size) > (size_t)sta.st_size) {
		qfits_warning("qfits_memory_falloc2(%s:%i): offset request exceeds file size (%zu + %zu = %zu > %zu) for file \"%s\"\n",
			      srcname, srclin, offs, size, (offs + size), (size_t)sta.st_size, name);
		if (QFITS_MEMORY_MODE == 0) return NULL;
		else exit(1);
	}
	/* Open file */
	if ((fd=open(name, O_RDONLY))==-1) {
		qfits_warning("qfits_memory_falloc2(%s:%i): failed to open file \"%s\": %s\n",
					  srcname, srclin, name, strerror(errno));
		if (QFITS_MEMORY_MODE == 0) return NULL;
		else exit(1);
	}

	/* Memory-map input file */
	// mmap requires page-aligned offsets.
	get_mmap_size(offs, size, &mapstart, &maplen, &mapoff);
	ptr = (char*)mmap(0, maplen, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd,
					  mapstart);
	eno = errno;
        
	/* Close file */
	close(fd);
	if (ptr == MAP_FAILED || ptr==NULL) {
		qfits_warning("qfits_memory_falloc2(%s:%i): failed to mmap file \"%s\": %s\n",
					  srcname, srclin, name, strerror(eno));
		if (QFITS_MEMORY_MODE == 0) return NULL;
		else exit(1);
	}
	if (freeaddr) {
		*freeaddr = ptr;
	}
	if (freesize) {
		*freesize = maplen;
	}
	return ptr + mapoff;
}



/*----------------------------------------------------------------------------*/
/**
  @brief    Map a file's contents to memory as a char pointer.
  @param    name        Name of the file to map
  @param    offs        Offset to the first mapped byte in file.
  @param    size        Returned size of the mapped file in bytes.
  @param    srcname     Name of the source file making the call.
  @param    srclin      Line # where the call was made.
  @return   A pointer to char, to be freed using qfits_memory_fdealloc().

  This function takes in input the name of a file. It tries to map the file 
  into memory and if it succeeds, returns the file's contents as a char pointer.
  It also modifies the input size variable to be the size of the mapped file in
  bytes. This function is normally never directly called but through the 
  falloc() macro.

  The offset indicates the starting point for the mapping, i.e. if you are not 
  interested in mapping the whole file but only from a given place.

  The returned pointer ptr must be deallocated with qfits_memory_fdealloc(ptr)
 */
/*----------------------------------------------------------------------------*/
char * qfits_memory_falloc(
						   const char        *   name,
        size_t          offs,
        size_t      *   size,
        const char  *   srcname,
        int             srclin)
{
    unsigned        mm_hash;
    char        *   ptr;
    struct stat     sta;
    int             fd;
    int             nptrs;
    int             i;
	int eno;

    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if ((QFITS_MEMORY_MODE == 0) || (QFITS_MEMORY_MODE == 1)) {

        if (size!=NULL) *size = 0;

        /* Check file's existence and compute its size */
        if (stat(name, &sta)==-1) {
			qfits_warning("qfits_memory_falloc(%s:%i): cannot stat file \"%s\"\n",
						  srcname, srclin, name);
            if (QFITS_MEMORY_MODE == 0) return NULL;
            else exit(1);
        }
        /* Check offset request does not go past end of file */
        if (offs>=(size_t)sta.st_size) {
			qfits_warning("qfits_memory_falloc(%s:%i): offset request exceeds file size (%zu > %zu) for file \"%s\"\n",
                                      srcname, srclin, offs, (size_t)sta.st_size, name);
            if (QFITS_MEMORY_MODE == 0) return NULL;
            else exit(1);
        }

        /* Open file */
        if ((fd=open(name, O_RDONLY))==-1) {
			qfits_warning("qfits_memory_falloc(%s:%i): failed to open file \"%s\": %s\n",
						  srcname, srclin, name, strerror(errno));
            if (QFITS_MEMORY_MODE == 0) return NULL;
            else exit(1);
        }

        /* Memory-map input file */
        ptr = (char*)mmap(0, sta.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
		eno = errno;
        
        /* Close file */
        close(fd);
        if (ptr == MAP_FAILED || ptr==NULL) {
			qfits_warning("qfits_memory_falloc(%s:%i): failed to mmap file \"%s\": %s\n",
						  srcname, srclin, name, strerror(eno));
            if (QFITS_MEMORY_MODE == 0) return NULL;
            else exit(1);
        }

        qfits_mem_debug(
            fprintf(stderr,
                    "qfits_mem: falloc mmap succeeded for [%s] - %s (%d)\n",
                    name, srcname, srclin);
        );

        if (size!=NULL) (*size) = sta.st_size;
        
        return ptr + offs;
    }

    /* Protect the call */
    if (size!=NULL) *size = 0;

    /* Initialize table if needed */
    if (qfits_memory_initialized==0) {
        qfits_memory_init();
        qfits_memory_initialized++;
    }

    if (qfits_memory_table.ncells>0) {
        /* Check if file has already been mapped */
        /* Compute hash for this name */
        mm_hash = qfits_memory_hash(name);
        /* Loop over all memory cells */
        nptrs=0;
        for (i=0; i<QFITS_MEMORY_MAXPTRS; i++) {
            if (qfits_memory_p_val[i]!=NULL)
                nptrs++;
            if ((qfits_memory_p_val[i]!=NULL) &&
                (qfits_memory_p_mm_filename[i] != NULL) &&
                (qfits_memory_p_mm_hash[i] == mm_hash)) {
                if (!strncmp(qfits_memory_p_mm_filename[i], name,
                             MAPFILENAMESZ)) {
                    /* File already mapped */
                    /* Check offset consistency wrt file size */
                    if (offs >= qfits_memory_p_size[i]) {
                        qfits_mem_debug(
                            fprintf(stderr,
                                "qfits_mem: falloc offset larger than file sz");
                        );
                        return NULL;
                    }
                    /* Increase reference counter */
                    qfits_memory_p_mm_refcount[i] ++;
                    qfits_mem_debug(
                        fprintf(stderr,
                                "qfits_mem: incref on %s (%d mappings)\n",
                                name,
                                qfits_memory_p_mm_refcount[i]);
                    );
                    /* Increase number of mappings */
                    qfits_memory_table.n_mm_mappings ++;
                    /* Build up return pointer */
                    ptr = (char*)qfits_memory_p_val[i] + offs;
                    /* Available size is filesize minus offset */
                    if (size!=NULL) {
                        *size = qfits_memory_p_size[i] - offs;
                    }
                    /* Return constructed pointer as void * */
                    return (void*)ptr;
                }
            }
            if (nptrs>=qfits_memory_table.ncells) break;
        }
    }

    /* First mapping attempt for this file */
    /* Check file's existence and compute its size */
    if (stat(name, &sta)==-1) {
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: cannot stat file %s - %s (%d)\n",
                    name, srcname, srclin);
        );
        return NULL;
    }
    /* Check offset request does not go past end of file */
    if (offs>=(size_t)sta.st_size) {
        qfits_mem_debug(
            fprintf(stderr,
                "qfits_mem: falloc offsets larger than file size");
        );
        return NULL;
    }

    /* Open file */
    if ((fd=open(name, O_RDONLY))==-1) {
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: cannot open file %s - %s (%d)\n",
                    name, srcname, srclin);
        );
        return NULL;
    }

    /* Memory-map input file */
    ptr = (char*)mmap(0, sta.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE,fd,0);
    
    /* Close file */
    close(fd);
    if (ptr == (char*)-1 || ptr==NULL) {
        qfits_mem_debug(
            perror("mmap");
            fprintf(stderr, "qfits_mem: falloc cannot mmap file %s", name);
        );
        return NULL;
    }

    qfits_memory_table.n_mm_files ++;
    qfits_memory_table.n_mm_mappings ++;
    qfits_mem_debug(
        fprintf(stderr,
                "qfits_mem: falloc mmap succeeded for [%s] - %s (%d)\n",
                name, srcname, srclin);
    );

    /* Add cell into general table */
    (void) qfits_memory_addcell((void*)ptr, sta.st_size, srcname, srclin, 
                           MEMTYPE_MMAP, -1, -1, name);

    if (size!=NULL) (*size) = sta.st_size;
    
    return ptr + offs;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Free memory that has been allocated with falloc
  @param    ptr         Pointer to free.
  @param    offs        Offset to the first mapped byte in file.
  @param    size        size to unmap
  @param    filename    Name of the file where the dealloc took place.
  @param    lineno      Line number in the file.
  @return   void
 */
/*----------------------------------------------------------------------------*/
void qfits_memory_fdealloc(
        void        *   ptr, 
        size_t          offs,
        size_t          size, 
        const char  *   filename, 
        int             lineno)
{
    int     i;
    int     pos;
    char *  swapname;
    int     nptrs;
    int     ii;

    /* Do nothing for a NULL pointer */
    if (ptr==NULL) {
        /* Output a warning */
        fprintf(stderr, "qfits_mem: free requested on NULL ptr -- %s (%d)\n",
                filename, lineno);
        return;
    }
    
    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if ((QFITS_MEMORY_MODE == 0) || (QFITS_MEMORY_MODE == 1)) {
        munmap((char*)(ptr)-offs, size);
        return;
    }
    
    /* Locate pointer in main table */
    nptrs = 0;
    pos = -1;
    i = PTR_HASH(ptr);
    for (ii=0; ii<QFITS_MEMORY_MAXPTRS; ii++) {
        if (++i == QFITS_MEMORY_MAXPTRS) i = 0;
        if (qfits_memory_p_val[i] == NULL) continue;
        nptrs++;
        if (qfits_memory_p_val[i] == ptr) {
            pos=i;
            break;
        }
        if (qfits_memory_p_memtype[i]==MEMTYPE_MMAP) {
            if (((char*)qfits_memory_p_val[i]<=(char*)ptr) &&
                (((char*)qfits_memory_p_val[i] + 
                  qfits_memory_p_size[i]) >= (char*)ptr)) {
                pos = i;
                break;
            }
        }
        if (nptrs>=qfits_memory_table.ncells) break;
    }
    if (pos==-1) {
        fprintf(stderr,
                "qfits_mem: %s (%d) free req. on unallocated pointer (%p)\n",
                filename, lineno, ptr);
        /* Pointer sent to system's free() function, maybe it should not? */
        free(ptr);
        return;
    }

    /* Deallocate pointer */
    switch (qfits_memory_p_memtype[pos]) {
        case MEMTYPE_RAM:
            /* --- RAM pointer */
            /* Free normal memory pointer */
            free(ptr);
            qfits_memory_table.alloc_ram -= qfits_memory_p_size[pos];
            break;
        case MEMTYPE_SWAP:
            /* --- SWAP pointer */
            swapname = qfits_memory_tmpfilename(qfits_memory_p_swapfileid[pos]);
            qfits_mem_debug(
                    fprintf(stderr, "qfits_mem: deallocating swap file [%s]\n", 
                        swapname);
            );
            /* Munmap file */
            if (munmap(ptr, qfits_memory_p_size[pos])!=0) {
                qfits_mem_debug( perror("munmap"); );
            }
            /* Close swap file */
            if (close(qfits_memory_p_swapfd[pos])==-1) {
                qfits_mem_debug( perror("close"); );
            }
            /* Remove swap file */
            if (remove(swapname)!=0) {
                qfits_mem_debug( perror("remove"); );
            }
            qfits_memory_table.alloc_swap -= qfits_memory_p_size[pos];
            qfits_memory_table.nswapfiles --;
            break;
        case MEMTYPE_MMAP:
            /* --- MEMORY-MAPPED pointer */
            /* Decrease reference count */
            qfits_memory_p_mm_refcount[pos] --;
            /* Decrease total number of mappings */
            qfits_memory_table.n_mm_mappings --;
            /* Non-null ref count means the file stays mapped */
            if (qfits_memory_p_mm_refcount[pos]>0) {
                qfits_mem_debug(
                    fprintf(stderr, "qfits_mem: decref on %s (%d mappings)\n",
                            qfits_memory_p_mm_filename[pos],
                            qfits_memory_p_mm_refcount[pos]);
                );
                return;
            }
            /* Ref count reached zero: unmap the file */
            qfits_mem_debug(
                    fprintf(stderr,
                        "qfits_mem: unmapping file %s\n",
                        qfits_memory_p_mm_filename[pos]);
            );
            munmap((char*)qfits_memory_p_val[pos],
                    qfits_memory_p_size[pos]);
            /* Decrease total number of mapped files */
            qfits_memory_table.n_mm_files --;
            break;
        default:
            qfits_mem_debug(
                    fprintf(stderr, "qfits_mem: unknown memory cell type???");
            );
            break;
    }

    if (qfits_memory_p_memtype[pos]!=MEMTYPE_MMAP) {
        /* Adjust allocated totals */
        qfits_memory_table.alloc_total -= qfits_memory_p_size[pos];

        /* Print out message in debug mode */
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: free(%p) %ld bytes in %s (%d)\n",
                    ptr,
                    (long)qfits_memory_p_size[pos],
                    filename,
                    lineno);
        );
    }
    /* Remove cell from main table */
    qfits_memory_remcell(pos);
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Free memory.
  @param    ptr         Pointer to free.
  @param    filename    Name of the file where the dealloc took place.
  @param    lineno      Line number in the file.
  @return   void

  Free the memory associated to a given pointer. Prints out a warning on stderr
  if the requested pointer is NULL or cannot be found in the extended memory 
  table.
 */
/*----------------------------------------------------------------------------*/
void qfits_memory_free(
        void        *   ptr, 
        const char  *   filename, 
        int             lineno)
{
    int     i;
    int     pos;
    char *  swapname;
    int     nptrs;
    int     ii;

    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if ((QFITS_MEMORY_MODE == 0) || (QFITS_MEMORY_MODE == 1)) {
        free(ptr);
        return;
    }
    
    /* Do nothing for a NULL pointer */
    if (ptr==NULL) {
        /* Output a warning */
        fprintf(stderr, "qfits_mem: free requested on NULL ptr -- %s (%d)\n",
                filename, lineno);
        return;
    }

    /* Locate pointer in main table */
    nptrs = 0;
    pos = -1;
    i = PTR_HASH(ptr);
    for (ii=0; ii<QFITS_MEMORY_MAXPTRS; ii++) {
        if (++i == QFITS_MEMORY_MAXPTRS) i = 0;
        if (qfits_memory_p_val[i] == NULL) continue;
        nptrs++;
        if (qfits_memory_p_val[i] == ptr) {
            pos=i;
            break;
        }
        if (qfits_memory_p_memtype[i]==MEMTYPE_MMAP) {
            if (((char*)qfits_memory_p_val[i]<=(char*)ptr) &&
                (((char*)qfits_memory_p_val[i] + 
                  qfits_memory_p_size[i]) >= (char*)ptr)) {
                pos = i;
                break;
            }
        }
        if (nptrs>=qfits_memory_table.ncells) break;
    }
    if (pos==-1) {
        fprintf(stderr,
                "qfits_mem: %s (%d) free requested on unallocated ptr (%p)\n",
                filename, lineno, ptr);
        /* Pointer sent to system's free() function, maybe it should not? */
        free(ptr);
        return;
    }

    /* Deallocate pointer */
    switch (qfits_memory_p_memtype[pos]) {
        case MEMTYPE_RAM:
            /* --- RAM pointer */
            /* Free normal memory pointer */
            free(ptr);
            qfits_memory_table.alloc_ram -= qfits_memory_p_size[pos];
            break;
        case MEMTYPE_SWAP:
            /* --- SWAP pointer */
            swapname = qfits_memory_tmpfilename(qfits_memory_p_swapfileid[pos]);
            qfits_mem_debug(
                    fprintf(stderr, "qfits_mem: deallocating swap file [%s]\n", 
                        swapname);
            );
            /* Munmap file */
            if (munmap(ptr, qfits_memory_p_size[pos])!=0) {
                qfits_mem_debug( perror("munmap"); );
            }
            /* Close swap file */
            if (close(qfits_memory_p_swapfd[pos])==-1) {
                qfits_mem_debug( perror("close"); );
            }
            /* Remove swap file */
            if (remove(swapname)!=0) {
                qfits_mem_debug( perror("remove"); );
            }
            qfits_memory_table.alloc_swap -= qfits_memory_p_size[pos];
            qfits_memory_table.nswapfiles --;
            break;
        case MEMTYPE_MMAP:
            /* --- MEMORY-MAPPED pointer */
            /* Decrease reference count */
            qfits_memory_p_mm_refcount[pos] --;
            /* Decrease total number of mappings */
            qfits_memory_table.n_mm_mappings --;
            /* Non-null ref count means the file stays mapped */
            if (qfits_memory_p_mm_refcount[pos]>0) {
                qfits_mem_debug(
                    fprintf(stderr, "qfits_mem: decref on %s (%d mappings)\n",
                            qfits_memory_p_mm_filename[pos],
                            qfits_memory_p_mm_refcount[pos]);
                );
                return;
            }
            /* Ref count reached zero: unmap the file */
            qfits_mem_debug(
                    fprintf(stderr,
                        "qfits_mem: unmapping file %s\n",
                        qfits_memory_p_mm_filename[pos]);
            );
            munmap((char*)qfits_memory_p_val[pos],
                    qfits_memory_p_size[pos]);
            /* Decrease total number of mapped files */
            qfits_memory_table.n_mm_files --;
            break;
        default:
            qfits_mem_debug(
                    fprintf(stderr, "qfits_mem: unknown memory cell type???");
            );
            break;
    }

    if (qfits_memory_p_memtype[pos]!=MEMTYPE_MMAP) {
        /* Adjust allocated totals */
        qfits_memory_table.alloc_total -= qfits_memory_p_size[pos];

        /* Print out message in debug mode */
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: free(%p) %ld bytes in %s (%d)\n",
                    ptr,
                    (long)qfits_memory_p_size[pos],
                    filename,
                    lineno);
        );
    }
    /* Remove cell from main table */
    qfits_memory_remcell(pos);
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Re-Allocate memory.
  @param    ptr         Pointer to free.
  @param    size        Size (in bytes) to allocate.
  @param    filename    Name of the file where the alloc took place.
  @param    lineno      Line number in the file.
  @return   1 newly allocated pointer.

  This function is a replacement call for realloc. It should never be called
  directly but through a macro instead, as:

  @code
  qfits_memory_realloc(nmemb, size, __FILE__, __LINE__)
  @endcode
 */
/*----------------------------------------------------------------------------*/
void * qfits_memory_realloc(
        void        *   ptr, 
        size_t          size, 
        const char  *   filename, 
        int             lineno)
{
    void    *   ptr2;
    size_t      small_sz;
    size_t      ptr_sz;
    int         pos = -1;
    int         i;
    
    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if (QFITS_MEMORY_MODE == 0) return realloc(ptr, size);
    else if (QFITS_MEMORY_MODE == 1) {
        ptr2 = realloc(ptr, size);
        if (ptr2 == NULL) exit(1);
        else return ptr2;
    }

    if (ptr == NULL) return qfits_memory_malloc(size, filename, lineno);

    /* Get the pointer size */
    for (i=0; i<QFITS_MEMORY_MAXPTRS; i++) {
        if (qfits_memory_p_val[i] == NULL) continue;
        if (qfits_memory_p_val[i] == ptr) {
            pos = i;
            break;
        }
    }
    if (pos==-1) {
        fprintf(stderr,
            "qfits_mem: %s (%d) realloc requested on unallocated ptr (%p)\n",
            filename, lineno, ptr);
        /* Pointer sent to system's realloc() function, maybe it should not? */
        return realloc(ptr, size);
    }
    ptr_sz = qfits_memory_p_size[pos];
    
    /* Compute the smaller size */
    small_sz = size < ptr_sz ? size : ptr_sz;
    
    /* Allocate the new pointer */
    ptr2 = qfits_memory_malloc(size, filename, lineno);
    
    /* Copy the common data */
    memcpy(ptr2, ptr, small_sz);

    /* Free the passed ptr */
    qfits_memory_free(ptr, filename, lineno);
    
    /* Return  */
    return ptr2;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Duplicate a string using calloc.
  @param    s       String to duplicate.
  @param    filename    Name of the file where the call took place.
  @param    lineno      Line number in the file.
  @return   1 newly allocated character string.

  This function calls in turn calloc to perform the allocation. It should
  never be called directly but only through a macro, like:

  @code
  qfits_memory_strdup(s, __FILE__, __LINE__)
  @endcode

  This function calls qfits_memory_malloc() to do the allocation.
 */
/*----------------------------------------------------------------------------*/
char * qfits_memory_strdup(
        const char  *   s, 
        const char  *   filename, 
        int             lineno)
{
    char    *   t;
    
    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if (QFITS_MEMORY_MODE == 0) return strdup_(s);
    else if (QFITS_MEMORY_MODE == 1) {
        t = strdup_(s);
        if (t == NULL) exit(1);
        else return t;
    }

    if (s==NULL) return NULL;
    t = qfits_memory_malloc(1+strlen(s), filename, lineno);
    return strcpy(t, s);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Display memory status information.
  @return   void

  This function is meant for debugging purposes, but it is recommended to
  call it at the end of every executable making use of the extended memory
  features. 
 */
/*----------------------------------------------------------------------------*/
void qfits_memory_status(void)
{
    int     i;

    /* If QFITS_MEMORY_MODE is 0 or 1, do not use the qfits_memory model  */
    if ((QFITS_MEMORY_MODE == 0) || (QFITS_MEMORY_MODE == 1)) return;
    
#if (QFITS_MEMORY_DEBUG>=1)
    fprintf(stderr, "#----- memory diagnostics -----\n");

    fprintf(stderr,
            "#- Peak memory usage\n"
            "ALL_maxalloc_kb     %ld\n"
            "ALL_maxpointers     %d\n",
            (long)(qfits_memory_table.alloc_max/1024),
            qfits_memory_table.max_cells);
    fprintf(stderr,
            "#- Local implementation\n"
            "TAB_ptrs            %d\n"
            "TAB_size            %u bytes\n",
            QFITS_MEMORY_MAXPTRS,
            (unsigned)sizeof(qfits_memory_table));
#ifdef __linux__
    fprintf(stderr,
            "#- Linux specific\n"
            "LINUX_pagesize      %d bytes\n"
            "LINUX_RLIMIT_DATA   %d kb\n",
            qfits_memory_table.pagesize,
            qfits_memory_table.rlimit_data);
#endif
#endif

    if (qfits_memory_table.ncells<1) return;
    fprintf(stderr, "#----- memory diagnostics -----\n");

    fprintf(stderr,
            "#- ALL status\n"
            "ALL_npointers       %d\n"
            "ALL_size            %ld\n"
            "ALL_maxalloc_kb     %ld\n"
            "ALL_maxpointers     %d\n",
            qfits_memory_table.ncells,
            (long)qfits_memory_table.alloc_total,
            (long)(qfits_memory_table.alloc_max/1024),
            qfits_memory_table.max_cells);

    if (qfits_memory_table.alloc_ram > 0) {
        fprintf(stderr, 
                "#- RAM status\n"
                "RAM_alloc           %ld\n",
                (long)qfits_memory_table.alloc_ram);
    }
    if (qfits_memory_table.alloc_swap > 0) {
        fprintf(stderr,
                "#- SWP status\n"
                "SWP_alloc           %ld\n"
                "SWP_files           %d\n",
                (long)qfits_memory_table.alloc_swap,
                qfits_memory_table.nswapfiles);
    }

    if (qfits_memory_table.n_mm_files>0) {
        fprintf(stderr,
                "#- MAP status\n"
                "MAP_files           %d\n"
                "MAP_mappings        %d\n",
                qfits_memory_table.n_mm_files,
                qfits_memory_table.n_mm_mappings);
    }

    fprintf(stderr, "#- pointer details\n");
    for (i=0; i<QFITS_MEMORY_MAXPTRS; i++) {
        qfits_memory_dumpcell(i, stderr);
    }
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Tell if there is still some memory allocated
  @return   1 if the memory table is tempty, 0 if no, 
            -1 if the memory model is off
 */
/*----------------------------------------------------------------------------*/
int qfits_memory_is_empty(void)
{
    if ((QFITS_MEMORY_MODE == 0) || (QFITS_MEMORY_MODE == 1)) return -1;
    if (qfits_memory_table.ncells<1) return 1;
    else return 0;
}

/**@}*/

/*
 * posted to comp.sys.next.programmer:
 * 
 *   
 * From: moser@ifor.math.ethz.ch (Dominik Moser,CLV A4,2 40 19,720 49 89)
 * Subject: Re: Compile problems (pgp 2.6.3i)
 * Date: 10 Jul 1996 06:50:42 GMT
 * Organization: Swiss Federal Institute of Technology (ETHZ)
 * References: <4rrhvj$6fr@bagan.srce.hr>
 * Message-ID: <4rvjs2$6oh@elna.ethz.ch>
 * 
 * Most systems don't have this (yet)
 
 */
static char * strdup_(const char * str)
{
    char    *   p;

    if ((p = malloc(strlen(str)+1)) == NULL)
    return((char *) NULL);

    (void) strcpy(p, str);

    return(p);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Hash a string to an unsigned value.
  @param    key     String to hash
  @return   1 unsigned value as a hash for the given string.
  
  This hash function has been taken from an Article in Dr Dobbs Journal. This 
  is normally a collision-free function, distributing keys evenly. The key is 
  stored anyway in the struct so that collision can be avoided by comparing the
  key itself in last resort.
 */
/*----------------------------------------------------------------------------*/
static unsigned qfits_memory_hash(const char * key)
{
    int         len;
    unsigned    hash;
    int         i;

    len = strlen(key);
    for (hash=0, i=0; i<len; i++) {
        hash += (unsigned)key[i];
        hash += (hash<<10);
        hash ^= (hash>>6);
    }
    hash += (hash <<3);
    hash ^= (hash >>11);
    hash += (hash <<15);
    return hash;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Initialize extended memory features.
  @return   void

  This function is implicitly called by the first malloc() or calloc() or 
  strdup_() execution. It allocates a minimal number of memory cells into
  the global extended memory table. It also install atexit routines the first 
  time it is called, and increases the number of possible descriptors to the 
  maximum.
 */
/*----------------------------------------------------------------------------*/
static void qfits_memory_init(void)
{
    struct rlimit rlim;

    qfits_mem_debug(
        fprintf(stderr,
                "qfits_mem: initializing main table size=%d ptrs (%ld bytes)\n",
                QFITS_MEMORY_MAXPTRS,
                (long)sizeof(qfits_memory_table));
    );
    /* Initialize memory table */
    memset(&qfits_memory_table, 0, sizeof(qfits_memory_table));

    /* Install cleanup routine at exit */
    atexit(qfits_memory_cleanup);
        
    /* Increase number of descriptors to maximum */
    getrlimit(RLIMIT_NOFILE, &rlim);
    qfits_mem_debug(
        fprintf(stderr, "qfits_mem: increasing from %ld to %ld file handles\n",
                (long)rlim.rlim_cur,
                (long)rlim.rlim_max);
    );
    rlim.rlim_cur = rlim.rlim_max;
    setrlimit(RLIMIT_NOFILE, &rlim);

#ifdef __linux__
    /* Get RLIMIT_DATA on Linux */
    getrlimit(RLIMIT_DATA, &rlim);
    qfits_memory_table.rlimit_data = rlim.rlim_cur;
    qfits_mem_debug(
        fprintf(stderr, "qfits_mem: got RLIMIT_DATA=%d\n",
                qfits_memory_table.rlimit_data);
    );
    /* Get page size on Linux */
    qfits_memory_table.pagesize = getpagesize();

#endif
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Removes all swap files.
  @return   void

  This routine will delete all swap files from the temporary area.
 */
/*----------------------------------------------------------------------------*/
static void qfits_memory_cleanup(void)
{
    int     reg;

    if (qfits_memory_table.file_reg>0) {
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: cleaning up swap files... ");
        );
        /*
         * Call remove() on all possible VM files. If the file exists, it
         * is effectively removed. It it does not, ignore the error.
         * This is not the cleanest way of doing it, but this function is
         * meant to be called also in cases of emergency (e.g. segfault),
         * so it should not rely on a correct memory table.
         */
        for (reg=0; reg<qfits_memory_table.file_reg; reg++) {
            remove(qfits_memory_tmpfilename(reg+1));
        }
        qfits_mem_debug(
            fprintf(stderr, "qfits_mem: done cleaning swap files\n");
        );
    }
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Add allocation cell to qfits_memory_table.
  @param    pointer         Pointer value.
  @param    size            Pointer size.
  @param    filename        Name of the C source file where alloc was done.
  @param    lineno          Line # where the allocation took place.
  @param    memtype         Memory type: RAM or SWAP.
  @param    swapfileid      Associated swap file ID (if any).
  @param    swapfd          Associated swap file descriptor (if any).
  @param    mm_filename     Mapped file name (if any)
  @return   the index in the qfits_memory_table of the added cell

  Add a memory cell in the xtended memory table to register that a new 
  allocation took place. 
  This call is not protected against illegal parameter values, so make sure 
  the passed values are correct!
 */
/*----------------------------------------------------------------------------*/
static int qfits_memory_addcell(
        void        *   pointer,
        size_t          size,
        const char  *   filename,
        int             lineno,
        char            memtype,
        int             swapfileid,
        int             swapfd,
        const char        *   mm_filename)
{
    int pos, ii;
    
    /* Check there is still some space left */
    if (qfits_memory_table.ncells >= QFITS_MEMORY_MAXPTRS) {
        fprintf(stderr, "fatal qfits_memory error: reached max pointers (%d)\n",
                QFITS_MEMORY_MAXPTRS);
        exit(-1);
    }
    /* Find an available slot */
    pos = PTR_HASH(pointer);
    for (ii = 0; ii<QFITS_MEMORY_MAXPTRS; ii++) {
        if (++pos == QFITS_MEMORY_MAXPTRS) pos = 0;
        if (qfits_memory_p_val[pos] == NULL) break;
    }
    qfits_mem_debug(
            fprintf(stderr, "qfits_mem: freecell found at pos %d\n", pos);
            );
    
    /* Store information */
    qfits_memory_p_val[pos] = pointer;
    qfits_memory_p_size[pos] = size;

    /* Filename and line number */
#if (QFITS_MEMORY_DEBUG>=1)
    qfits_memory_p_filename[pos] = filename;
    qfits_memory_p_lineno[pos] = lineno;
#endif
    
    qfits_memory_p_memtype[pos] = memtype;
    qfits_memory_p_swapfileid[pos] = swapfileid;
    qfits_memory_p_swapfd[pos] = swapfd;
    
    if (mm_filename!=NULL) {
        strncpy(qfits_memory_p_mm_filename[pos], mm_filename,MAPFILENAMESZ);
        qfits_memory_p_mm_hash[pos] = qfits_memory_hash(mm_filename);
        qfits_memory_p_mm_refcount[pos] = 1;
    } else {
        qfits_memory_p_mm_filename[pos][0] = 0;
        qfits_memory_p_mm_hash[pos] = 0;
        qfits_memory_p_mm_refcount[pos] = 0;
    }
    qfits_memory_table.ncells ++;
    if (qfits_memory_table.ncells > qfits_memory_table.max_cells)
        qfits_memory_table.max_cells = qfits_memory_table.ncells;
    return pos;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Remove a memory cell from the xtended memory table.
  @param    pos     Position of the pointer in the table.
  @return   int 0 if Ok, -1 if error occurred.

  Remove the specified cell in qfits_memory_table. 
  This call is not protected against illegal parameter values, so make sure 
  the passed values are correct!
 */
/*----------------------------------------------------------------------------*/
static int qfits_memory_remcell(int pos)
{
    qfits_mem_debug(
        fprintf(stderr, "qfits_mem: removing cell from pos %d (cached)\n", pos);
    );
    /* Set pointer to NULL */
    qfits_memory_p_val[pos] = NULL;
    /* Decrement number of allocated pointers */
    qfits_memory_table.ncells --;
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Dump a memory cell to an open file pointer.
  @param    cell    Cell to dump.
  @param    out     Open file pointer to dump to.
  @return   void

  This function is meant for debugging purposes only. It takes in input a
  pointer to a memory cell and dumps it to the requested file pointer (it
  is Ok to provide stdout or stderr as file pointers). If the passed
  position is invalid or the table pointer is NULL, this function returns
  immediately.
 */
/*----------------------------------------------------------------------------*/
static void qfits_memory_dumpcell(
        int         pos, 
        FILE    *   out)
{
    if (pos<0 || pos>=QFITS_MEMORY_MAXPTRS) return;
    if (qfits_memory_p_val[pos]==NULL) return;

    if (qfits_memory_p_memtype[pos] == MEMTYPE_MMAP) {
#if (QFITS_MEMORY_DEBUG>=1)
        fprintf(out,
            "M(%p) - %s (%d) maps [%s] for %ld bytes",
            qfits_memory_p_val[pos],
            qfits_memory_p_filename[pos],
            qfits_memory_p_lineno[pos],
            qfits_memory_p_mm_filename[pos],
            (long)qfits_memory_p_size[pos]);
#else
        fprintf(out,
            "M(%p) maps [%s] for %ld bytes",
            qfits_memory_p_val[pos],
            qfits_memory_p_mm_filename[pos],
            (long)qfits_memory_p_size[pos]);
#endif
    } else {
#if (QFITS_MEMORY_DEBUG>=1)
        fprintf(out, "%c(%p) - %s (%d) for %ld bytes",
            qfits_memory_p_memtype[pos],
            qfits_memory_p_val[pos],
            qfits_memory_p_filename[pos],
            qfits_memory_p_lineno[pos],
            (long)qfits_memory_p_size[pos]);
#else
        fprintf(out, "%c(%p) for %ld bytes",
            qfits_memory_p_memtype[pos],
            qfits_memory_p_val[pos],
            (long)qfits_memory_p_size[pos]);
#endif
    }
    if (qfits_memory_p_memtype[pos]==MEMTYPE_SWAP) {
        fprintf(out, " swf[%s][%d]",
                qfits_memory_tmpfilename(qfits_memory_p_swapfileid[pos]),
                qfits_memory_p_swapfd[pos]);
    }
    fprintf(out, "\n");
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Compute filename associated to a temporary file ID.
  @param    reg     Registration number of temporary file name.
  @return   pointer to statically allocated char string.

  This function computes the valid file name associated to a temporary file
  ID. It computes the result, stores it in an internal static string and
  returns a pointer to it.
 */
/*----------------------------------------------------------------------------*/
static char * qfits_memory_tmpfilename(int reg)
{
    static char qfits_mem_tmpfilename[TMPFILENAMESZ];
    /* Create file name using tmp directory as a base */
    sprintf(qfits_mem_tmpfilename, "%s/vmswap_%05ld_%05x", 
            qfits_memory_tmpdirname, (long)getpid(), reg);
    return qfits_mem_tmpfilename;
}
