/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Keir Mierle, David W. Hogg, Sam Roweis and Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef SIP_QFITS_H
#define SIP_QFITS_H

#include "qfits.h"
#include "sip.h"

qfits_header* sip_create_header(const sip_t* sip);

qfits_header* tan_create_header(const tan_t* tan);

void sip_add_to_header(qfits_header* hdr, const sip_t* sip);

void tan_add_to_header(qfits_header* hdr, const tan_t* tan);

sip_t* sip_read_header_file(const char* fn, sip_t* dest);

tan_t* tan_read_header_file(const char* fn, tan_t* dest);

sip_t* sip_read_header(const qfits_header* hdr, sip_t* dest);

tan_t* tan_read_header(const qfits_header* hdr, tan_t* dest);


int sip_write_to_file(const sip_t* sip, const char* fn);

int tan_write_to_file(const tan_t* tan, const char* fn);

#endif
