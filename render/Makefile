# This file is part of the Astrometry.net suite.
# Copyright 2006, Dustin Lang, Keir Mierle and Sam Roweis.
#
# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.
#
# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

BASEDIR := ..
COMMON := $(BASEDIR)/util

.PHONY: all
all:

# Reset CFLAGS, unless the user set a value in the environment.
CFLAGS ?= 
CFLAGS := $(CFLAGS)

include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.libkd
include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.netpbm
include $(COMMON)/makefile.png
include $(COMMON)/makefile.jpeg

LDFLAGS := $(LDFLAGS_DEF)
LDFLAGS += -lm

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(QFITS_INC)
CFLAGS += $(LIBKD_INC)
CFLAGS += $(ANUTILS_INC)
CFLAGS += $(ANFILES_INC)
CFLAGS  += $(CAIRO_INC)
CFLAGS  += $(NETPBM_INC)
CFLAGS  += $(PNG_INC)
CFLAGS  += $(JPEG_INC)

LDFLAGS += $(CAIRO_LIB)

MISC_EXECS := ppmnormrgb whitebalance zoomout

USNOBMAP_EXECS := tilerender load-db image-pyramid render-mercator \
	make-merctree 

USNOBMAP_MAIN_OBJS := $(addsuffix .o,$(USNOBMAP_EXECS))
MISC_MAIN_OBJS := $(addsuffix .o,$(MISC_EXECS))

$(ANUTILS_LIB): $(REMAKE_ANUTILS)

$(LIBKD_LIB): $(REMAKE_LIBKD)

all: $(USNOBMAP_EXECS) $(MISC_EXECS)

$(COMMON)/cairoutils.o:
	$(MAKE) -C $(COMMON) cairoutils.o

load-db: load-db.o $(COMMON)/cairoutils.o $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(NETPBM_LIB) $(JPEG_LIB) $(PNG_LIB)

image-pyramid: image-pyramid.o $(COMMON)/cairoutils.o $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(NETPBM_LIB) $(JPEG_LIB) $(PNG_LIB)

TILERENDER_OBJS := tilerender.o mercrender.o merctree.o merc.o \
		render_tycho.o \
		render_gridlines.o \
		render_usnob.o \
		render_rdls.o \
		render_boundary.o \
		render_constellation.o \
		render_messier.o \
		render_solid.o \
		render_images.o \
		render_cairo.o \
		render_skdt.o \
		render_quads.o

MISC_OBJS := merctree.o pnmutils.o

tilerender: $(TILERENDER_OBJS) \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) \
		$(COMMON)/cairoutils.o
	$(MAKE) -C $(COMMON) cairoutils.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(NETPBM_LIB) $(JPEG_LIB) $(PNG_LIB) -lz

render-mercator: render-mercator.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

make-merctree: make-merctree.o merctree.o merc.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

zoomout: zoomout.o pnmutils.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ANUTILS_LIB) $(NETPBM_LIB)

ppmnormrgb: ppmnormrgb.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(NETPBM_LIB)

whitebalance: whitebalance.o pnmutils.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(NETPBM_LIB) -lm

.PHONY: tags
tags:
	etags `find . -name "*.h" -o -name "*.c"`

ALL_OBJS := $(USNOBMAP_MAIN_OBJS) $(MISC_MAIN_OBJS) \
	$(TILERENDER_OBJS) $(MISC_OBJS)

DEP_OBJ := $(ALL_OBJS)
DEP_PREREQS := $(QFITS_LIB)

CFLAGS += $(NETPBM_INC)
include $(COMMON)/makefile.deps

.PHONY: allclean realclean clean

allclean: clean
	rm -f *.o *~ *.dep 

realclean: allclean

clean:
	rm -f $(USNOBMAP_EXECS) $(MISC_EXECS) $(ALL_OBJS) \
		$(DEPS) deps


