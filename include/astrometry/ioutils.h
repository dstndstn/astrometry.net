/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef IOUTILS_H
#define IOUTILS_H

#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include "astrometry/an-bool.h"
#include "astrometry/bl.h"
#include "astrometry/keywords.h"

extern uint32_t ENDIAN_DETECTOR;

void QSORT_R(void* base, size_t nmembers, size_t member_size,
             void* token, int (*compar)(void *, const void *, const void *));

/**
   You should define the "comparison" function like this:

   static int QSORT_COMPARISON_FUNCTION(my_comparison, void* token, const void* v1, const void* v2) {
 */
#define QSORT_COMPARISON_FUNCTION(func, thunk, v1, v2) func(thunk, v1, v2)

int copy_file(const char* infn, const char* outfn);

int pad_fid(FILE* fid, size_t len, char pad);
int pad_file(char* filename, size_t len, char pad);

/*
 The "basename" function may overwrite its arg and may return a pointer
 to static memory... both undesirable.  This replacement returns a newly-
 allocated string containing the result.
 */
Malloc
char* basename_safe(const char* path);

Malloc
char* dirname_safe(const char* path);

// Returns (system + user) CPU time, in seconds.
float get_cpu_usage(void);

//int log_resource_usage(int loglevel);

/*
 Searches for the given "filename" in the given set of directories.
 Returns a newly allocated string "dir/filename", or NULL if none of
 the paths exists and is readable.
 */
char* find_file_in_dirs(const char** dirs, int ndirs, const char* filename, anbool allow_absolute);

// Are strings s1 and s2 equal?
anbool streq(const char* s1, const char* s2);
anbool strcaseeq(const char* s1, const char* s2);

/*
 Copy data from "fin" to "fout", starting at offset "offset"
 (from the beginning of "fin"), and length "length".
 Returns 0 on success.
 */
int pipe_file_offset(FILE* fin, off_t offset, off_t length, FILE* fout);

int write_file(const char* fn, const char* data, int len);

/*
 It's not really _safe_ as such, it just prints an error message if it fails...
 */
void
ATTRIB_FORMAT(printf,2,3)
    asprintf_safe(char** strp, const char* format, ...);

int run_command_get_outputs(const char* cmd, sl** outlines, sl** errlines);

void get_mmap_size(size_t start, size_t size, off_t* mapstart, size_t* mapsize, int* pgap);

// If "dir" is NULL, create temp file in $TMP, or /tmp if not set.
char* create_temp_file(const char* fn, const char* dir);

// If "dir" is NULL, create temp file in $TMP, or /tmp if not set.
char* create_temp_dir(const char* name, const char* dir);

char* shell_escape(const char* str);

int mkdir_p(const char* path);

// Returns 0 on error.
time_t file_get_last_modified_time(const char* fn);

anbool file_exists(const char* fn);

anbool file_readable(const char* fn);

anbool file_executable(const char* fn);

anbool path_is_dir(const char* path);

void* file_get_contents(const char* fn, size_t* len, anbool addzero);

char* file_get_contents_offset(const char* fn, int offset, int length);

sl* fid_add_lines(FILE* fid, anbool include_newlines, sl* list);

sl* file_get_lines(const char* fn, anbool include_newlines);

sl* fid_get_lines(FILE* fid, anbool include_newlines);

sl* dir_get_contents(const char* path, sl* result, anbool filesonly, anbool recursive);

int file_get_last_modified_string(const char* fn, const char* timeformat,
                                  anbool utc, char* output, size_t outsize);

/**
 Splits the given "str" into words, so that the first line is at most
 "firstlinew" long.  Subsequent lines have length <= "linew".  If
 "lst" is non-NULL, the words are added into it.  Otherwise a new sl
 is allocated.
 */
sl* split_long_string(const char* str, int firstlinew, int linew, sl* lst);

// Split a string on the first instance of "splitstr".
// Places the addresses of (newly-allocated) copies of the first and seconds parts of the string.
// Returns 1 if the string is found.
int split_string_once(const char* str, const char* splitstr, char** first, char** second);

/**
 If "cmdline" starts with "keyword", returns 1 and places the address of
 the start of the next word in "p_next_word".
 */
int is_word(const char* cmdline, const char* keyword, char** p_next_word);

int starts_with(const char* str, const char* prefix);

int ends_with(const char* str, const char* prefix);

char* strdup_safe(const char* str);

void add_sigbus_mmap_warning(void);
void reset_sigbus_mmap_warning(void);

int write_u8(FILE* fout, unsigned char val);
int write_u16(FILE* fout, unsigned int val);
int write_u32(FILE* fout, unsigned int val);
int write_uints(FILE* fout, unsigned int* val, int n);
int write_double(FILE* fout, double val);
int write_float(FILE* fout, float val);
int write_fixed_length_string(FILE* fout, char* s, int length);
int write_string(FILE* fout, char* s);

int write_u32_portable(FILE* fout, unsigned int val);
int write_u32s_portable(FILE* fout, unsigned int* val, int n);

int read_u8(FILE* fin, unsigned char* val);
int read_u16(FILE* fout, unsigned int* val);
int read_u32(FILE* fin, unsigned int* val);
int read_double(FILE* fin, double* val);
int read_fixed_length_string(FILE* fin, char* s, int length);
char* read_string(FILE* fin);
char* read_string_terminated(FILE* fin, const char* terminators, int nterminators,
                             anbool include_terminator);

int read_u32_portable(FILE* fin, unsigned int* val);
int read_u32s_portable(FILE* fin, unsigned int* val, int n);

struct buffered_read_data {
    void* buffer;
    int blocksize;
    int elementsize;
    int ntotal;
    int nbuff;
    int off;
    int buffind;
    int (*refill_buffer)(void* userdata, void* buffer, unsigned int offs, unsigned int nelems);
    void* userdata;
};
typedef struct buffered_read_data bread_t;

bread_t* buffered_read_new(int elementsize, int Nbuffer, int Ntotal,
                           int (*refill_buffer)(void* userdata, void* buffer, unsigned int offs, unsigned int nelems),
                           void* userdata);

void* buffered_read(bread_t* buff);

void buffered_read_pushback(bread_t* br);

void buffered_read_reset(bread_t* br);

void buffered_read_free(bread_t* br);

void buffered_read_resize(bread_t* br, int newsize);

#endif
