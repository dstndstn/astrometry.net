/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include "astrometry/bl-nl.ph"

InlineDefine
number NLF(get)(nl* list, size_t n) {
	number* ptr = (number*)bl_access(list, n);
	return *ptr;
}

InlineDefine
number NLF(get_const)(const nl* list, size_t n) {
	number* ptr = (number*)bl_access_const(list, n);
	return *ptr;
}

InlineDefine
size_t NLF(size)(const nl* list) {
    return bl_size(list);
}

InlineDefine
number* NLF(access)(nl* list, size_t j) {
    return (number*)bl_access(list, j);
}

