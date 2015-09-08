# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

CAIRO_INC ?= $(shell pkg-config --cflags cairo 2>/dev/null)
CAIRO_LIB ?= $(shell pkg-config --libs cairo 2>/dev/null)
