/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

#include "bl-nl.ph"

InlineDefine
number NLF(get)(nl* list, int n) {
	number* ptr = (number*)bl_access(list, n);
	return *ptr;
}

InlineDefine
number NLF(get_const)(const nl* list, int n) {
	number* ptr = (number*)bl_access_const(list, n);
	return *ptr;
}

InlineDefine
int NLF(size)(const nl* list) {
    return bl_size(list);
}

InlineDefine
number* NLF(access)(nl* list, int j) {
    return (number*)bl_access(list, j);
}

