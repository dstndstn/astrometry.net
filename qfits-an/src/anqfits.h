/*
 This file was added by the Astrometry.net team.
 Copyright 2007,2010 Dustin Lang.
 */

#ifndef ANQFITS_H
#define ANQFITS_H

#include "qfits_header.h"
#include "qfits_table.h"
#include "qfits_keywords.h"

// Everything we know about a FITS extension.
struct anqfits_ext_t {
	//bool inited;
	// Offsets to header, in FITS blocks
	// --> int works for ~12 TB files.
	int hdr_start;
	// Header size
	int hdr_size;
	// Offsets to data
	int data_start;
	// Data size
	int data_size;
	qfits_header* hdr;
	bool table_read;
	qfits_table* table;
};
typedef struct anqfits_ext_t anqfits_ext_t;

struct anqfits_t {
    char* filename;
    //ino_t       inode ; /* Inode */
    //time_t        mtime;  /* Last modification date */
    //int filesize; /* File size in bytes */
    //time_t        ctime;  /* Last modification date */

    int Nexts;    // # of extensions in file

	anqfits_ext* exts;

    off_t filesize ; // File size in FITS blocks (2880 bytes)
};
typedef struct anqfits_t anqfits_t;

anqfits_t* anqfits_open(const char* filename);

void anqfits_close(anqfits_t* qf);

int anqfits_n_ext(const anqfits_t* qf);

// In FITS blocks.
int anqfits_header_start(const anqfits_t* qf, int ext);

// In FITS blocks.
int anqfits_header_size(const anqfits_t* qf, int ext);

// In FITS blocks.
int anqfits_data_start(const anqfits_t* qf, int ext);

// In FITS blocks.
int anqfits_data_size(const anqfits_t* qf, int ext);

qfits_header* anqfits_get_header(const anqfits_t* qf, int ext);

qfits_table* anqfits_get_table(const anqfits_t* qf, int ext);

/*
 Deprecated // ?
 int anqfits_is_table_2(const anqfits_t* qf, int ext);
 Deprecated // ?
 char* anqfits_query_ext_2(const anqfits_t* qf, int ext, const char* keyword);
 */

#endif

