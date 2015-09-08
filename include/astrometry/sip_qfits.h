/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef SIP_QFITS_H
#define SIP_QFITS_H

#include "astrometry/qfits_header.h"
#include "astrometry/sip.h"

int sip_get_image_size(const qfits_header* hdr, int* pW, int* pH);

sip_t* sip_read_tan_or_sip_header_file_ext(const char* fn, int ext, sip_t* dest, anbool forcetan);

qfits_header* sip_create_header(const sip_t* sip);

qfits_header* tan_create_header(const tan_t* tan);

void sip_add_to_header(qfits_header* hdr, const sip_t* sip);

void tan_add_to_header(qfits_header* hdr, const tan_t* tan);

sip_t* sip_read_header_file(const char* fn, sip_t* dest);

sip_t* sip_read_header_file_ext(const char* fn, int ext, sip_t* dest);

sip_t* sip_read_header_file_ext_only(const char* fn, int ext, sip_t* dest);

tan_t* tan_read_header_file(const char* fn, tan_t* dest);

tan_t* tan_read_header_file_ext(const char* fn, int ext, tan_t* dest);

tan_t* tan_read_header_file_ext_only(const char* fn, int ext, tan_t* dest);

sip_t* sip_read_header(const qfits_header* hdr, sip_t* dest);

tan_t* tan_read_header(const qfits_header* hdr, tan_t* dest);


sip_t* sip_from_string(const char* str, int len, sip_t* dest);


int tan_write_to(const tan_t* tan, FILE* fid);

int sip_write_to(const sip_t* sip, FILE* fid);

int sip_write_to_file(const sip_t* sip, const char* fn);

int tan_write_to_file(const tan_t* tan, const char* fn);

#endif
