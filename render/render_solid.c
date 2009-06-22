/*
   This file is part of the Astrometry.net suite.
   Copyright 2007 Keir Mierle and Dustin Lang.

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
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

#include "tilerender.h"
#include "render_solid.h"

int render_solid(cairo_t* cairo, render_args_t* args) {
	double rgba[] = { 0,0,0,1 };
	get_first_rgba_arg_of_type(args, "solid_rgba ", rgba);
	cairo_set_source_rgba(cairo, rgba[0], rgba[1], rgba[2], rgba[3]);
	cairo_paint(cairo);
    return 0;
}

