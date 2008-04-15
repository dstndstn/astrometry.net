/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef FILEUTIL_H
#define FILEUTIL_H

#include <stdio.h>
#include "starutil.h"
#include "xylist.h"

#define COMMENT_CHAR 35 // #
#define FOPEN_ERR -301
#define READ_FAIL -1

void fopenout(char* fn, FILE** pfid);

#define free_fn(n) {free(n);}

#define mk_catfn(s)    mk_filename(s,".objs.fits")
#define mk_idfn(s)    mk_filename(s,".id.fits")
#define mk_streefn(s)  mk_filename(s,".skdt.fits")
#define mk_ctreefn(s)  mk_filename(s,".ckdt.fits")
#define mk_quadfn(s)   mk_filename(s,".quad.fits")
#define mk_codefn(s)   mk_filename(s,".code.fits")
#define mk_qidxfn(s)   mk_filename(s,".qidx.fits")
#define mk_fieldfn(s)  mk_filename(s,".fits")
#define mk_field0fn(s) mk_filename(s,".xyl0")
#define mk_idlistfn(s) mk_filename(s,".ids0")
#define mk_rdlsfn(s)  mk_filename(s,".rdls")

char *mk_filename(const char *basename, const char *extension);

#endif
