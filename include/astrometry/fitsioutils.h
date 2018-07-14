/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef FITSIO_UTILS_H
#define FITSIO_UTILS_H

#include <stdint.h>

#include "astrometry/qfits_header.h"
#include "astrometry/qfits_image.h"
#include "astrometry/qfits_table.h"
#include "astrometry/keywords.h"
#include "astrometry/an-bool.h"

void fits_use_error_system(void);

Malloc
char* fits_to_string(const qfits_header* hdr, int* size);

int fits_write_float_image(const float* img, int nx, int ny, const char* fn);

int fits_write_u8_image(const uint8_t* img, int nx, int ny, const char* fn);

int fits_write_i16_image(const int16_t* img, int nx, int ny, const char* fn);

/** Creates a FITS header for the image described in "qd". */
qfits_header* fits_get_header_for_image(const qfitsdumper* qd, int W,
                                        qfits_header* addtoheader);

qfits_header* fits_get_header_for_image2(int W, int H, int bitpix,
                                         qfits_header* addtoheader);

qfits_header* fits_get_header_for_image3(int W, int H, int bitpix, int planes,
                                         qfits_header* addtoheader);

/* Writes the FITS header to the given filename, then pads and closes the file.
 */
int fits_write_header(const qfits_header* hdr, const char* filename);

/** Writes the given FITS header and image.

 If "hdr" is null, a standard image header will be written; "W" must be the image width.

 Of "hdr" is non-null, "W" is ignored.
 */
int fits_write_header_and_image(const qfits_header* hdr, const qfitsdumper* qd, int W);


double fits_get_double_val(const qfits_table* table, int column,
                           const void* rowdata);

/*
 Returns 1 if the given keyword is one of the required keywords in a BINTABLE
 specification.
 */
int fits_is_table_header(const char* keyword);

/*
 Returns 1 if the given keyword is one of the required keywords in the
 primary header.
 */
int fits_is_primary_header(const char* key);

/*
 Copies headers that aren't part of the BINTABLE specification from "src"
 to "dest".
 */
void fits_copy_non_table_headers(qfits_header* dest, const qfits_header* src);

/*
 Retrieves the value of the header card "key" as a string,
 returning a newly-allocated string which should be free()'d.
 It will be "prettied" via qfits_pretty_string_r.
 */
char* fits_get_dupstring(const qfits_header* hdr, const char* key);

char* fits_get_long_string(const qfits_header* hdr, const char* key);

void
ATTRIB_FORMAT(printf,4,5)
    fits_header_addf(qfits_header* hdr, const char* key, const char* comment,
                     const char* format, ...);

void
ATTRIB_FORMAT(printf,4,5)
    fits_header_addf_longstring(qfits_header* hdr, const char* key, const char* comment,
                                const char* format, ...);

void fits_header_add_longstring_boilerplate(qfits_header* hdr);

void
ATTRIB_FORMAT(printf,4,5)
    fits_header_modf(qfits_header* hdr, const char* key, const char* comment,
                     const char* format, ...);

void fits_header_add_int(qfits_header* hdr, const char* key, int val,
                         const char* comment);

void fits_header_add_double(qfits_header* hdr, const char* key, double val,
                            const char* comment);

// Add if it doesn't exist, mod if it does.
void fits_header_set_double(qfits_header* hdr, const char* key, double val,
                            const char* comment);
void fits_header_set_int(qfits_header* hdr, const char* key, int val,
                         const char* comment);


void fits_header_mod_int(qfits_header* hdr, const char* key, int val,
                         const char* comment);

void fits_header_mod_double(qfits_header* hdr, const char* key, double val,
                            const char* comment);

int fits_update_value(qfits_header* hdr, const char* key, const char* newvalue);

qfits_table* fits_copy_table(qfits_table* tbl);

int an_fits_copy_header(const qfits_header* src, qfits_header* dest, char* key);

int fits_copy_all_headers(const qfits_header* src, qfits_header* dest, char* targetkey);

int fits_append_all_headers(const qfits_header* src, qfits_header* dest, char* targetkey);

int fits_add_args(qfits_header* src, char** args, int argc);

int 
ATTRIB_FORMAT(printf,2,3)
    fits_add_long_comment(qfits_header* dst, const char* format, ...);

int 
ATTRIB_FORMAT(printf,2,3)
    fits_append_long_comment(qfits_header* dst, const char* format, ...);

int 
ATTRIB_FORMAT(printf,2,3)
    fits_add_long_history(qfits_header* dst, const char* format, ...);

// how many FITS blocks are required to hold 'size' bytes?
int fits_blocks_needed(int size);

size_t fits_bytes_needed(size_t size);

int fits_pad_file_with(FILE* fid, char pad);

int fits_pad_file(FILE* fid);

int fits_pad_file_name(char* filename);

void fits_fill_endian_string(char* str);

char* fits_get_endian_string(void);

int fits_check_endian(const qfits_header* header);

int fits_check_uint_size(const qfits_header* header);

int fits_check_double_size(const qfits_header* header);

void fits_add_endian(qfits_header* header);

void fits_add_reverse_endian(qfits_header* header);
void fits_mod_reverse_endian(qfits_header* header);

void fits_add_uint_size(qfits_header* header);

void fits_add_double_size(qfits_header* header);

int fits_find_column(const qfits_table* table, const char* colname);

int fits_find_table_column(const char* fn, const char* colname,
                           off_t* start, off_t* size, int* extension);

qfits_table* fits_get_table_column(const char* fn, const char* colname, int* pcol);

int fits_add_column(qfits_table* table, int column, tfits_type type,
                    int ncopies, const char* units, const char* label);

int fits_offset_of_column(qfits_table* table, int colnum);

// write single column fields:
int fits_write_data_A(FILE* fid, char value);
int fits_write_data_B(FILE* fid, uint8_t value);
int fits_write_data_D(FILE* fid, double value, anbool flip);
int fits_write_data_E(FILE* fid, float value, anbool flip);
int fits_write_data_I(FILE* fid, int16_t value, anbool flip);
int fits_write_data_J(FILE* fid, int32_t value, anbool flip);
int fits_write_data_K(FILE* fid, int64_t value, anbool flip);
int fits_write_data_L(FILE* fid, char value);
int fits_write_data_X(FILE* fid, unsigned char value);

int fits_write_data(FILE* fid, void* pvalue, tfits_type type, anbool flip);

// Writes one cell of a FITS table (which may be an array or scalar)
// that has already been converted to FITS format "type".
// If "vvalue" is NULL, just skips past that number of bytes.
int fits_write_data_array(FILE* fid, const void* vvalue, tfits_type type,
                          int N, anbool flip);

#endif
