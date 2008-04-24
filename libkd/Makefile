#  This file is part of libkd.
#  Copyright 2006, 2007 Dustin Lang and Keir Mierle.
#
#  libkd is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 2.
#
#  libkd is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with libkd; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

BASEDIR := ..
COMMON := $(BASEDIR)/util

all:

include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils
include $(COMMON)/makefile.qfits

INTERNALS := \
	kdint_ddd.o \
	kdint_fff.o \
	kdint_ddu.o \
	kdint_duu.o \
	kdint_dds.o \
	kdint_dss.o

KD := kdtree.o kdtree_dim.o kdtree_mem.o
KD_FITS := kdtree_fits_io.o

# These are #included by other source files.
INTERNAL_SOURCES := kdtree_internal.c kdtree_internal_fits.c

CFLAGS := $(CFLAGS_DEF)
CFLAGS += $(QFITS_INC)
CFLAGS += $(ANUTILS_INC)

# Track memory usage?
#CFLAGS += -DKDTREE_MEM_TRACK

LDFLAGS := $(LDFLAGS_DEF)
LDFLAGS += -lm

LIBKD := libkd.a

LIBKD_NOIO := libkd-noio.a

INTERNALS_NOIO := \
	kdint_ddd_noio.o \
	kdint_fff_noio.o \
	kdint_ddu_noio.o \
	kdint_duu_noio.o \
	kdint_dds_noio.o \
	kdint_dss_noio.o

DEP_OBJ := $(KD) $(KD_FITS) $(INTERNALS) $(INTERNALS_NOIO)

%_noio.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS) -DKDTREE_NO_FITS

all: $(REMAKE_ANUTILS) $(LIBKD) checktree fix-bb
#$(LIBKD_NOIO) 

$(LIBKD): $(KD) $(KD_FITS) $(INTERNALS)
	-rm -f $@
	$(AR) rc $@ $^
	$(RANLIB) $@

$(LIBKD_NOIO): $(KD) $(INTERNALS_NOIO)
	-rm -f $@
	$(AR) rc $@ $^
	$(RANLIB) $@

checktree: checktree.o $(LIBKD) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

fix-bb: fix-bb.o $(LIBKD) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

LIBKD_INSTALL := fix-bb checktree

install: $(LIBKD_INSTALL)
	@echo Installing in directory $(INSTALL_DIR)
	mkdir -p $(INSTALL_DIR)/bin
	@for x in $(LIBKD_INSTALL); do \
		echo cp $$x $(INSTALL_DIR)/bin; \
		cp $$x $(INSTALL_DIR)/bin; \
	done

### TESTS are great

ALL_TEST_FILES = test_libkd test_libkd_io
ALL_TEST_EXTRA_OBJS =
ALL_TEST_LIBS = $(LIBKD) $(ANUTILS_LIB) $(QFITS_LIB)

include $(COMMON)/makefile.tests

test_libkd: $(LIBKD_NOIO) $(ANUTILS_LIB)

test_libkd_io: $(LIBKD) $(ANUTILS_LIB) $(QFITS_LIB)

DEP_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)

### /TESTS

.PHONY: realclean clean
clean:
	-rm $(LIBKD) $(KD) $(KD_FITS) deps $(DEPS) \
		checktree checktree.o \
		fix-bb fix-bb.o \
		$(INTERNALS) $(INTERNALS_NOIO) $(LIBKD_NOIO)

realclean: clean
	-rm *~ *.dep

.PHONY: tags
tags:
	etags `find . -name "*.c" -o -name "*.h"`

DEP_PREREQS := $(QFITS_LIB)

include $(COMMON)/makefile.deps

# Borrowed from makefile.deps
%_noio.dep : %.c
	$(CC) $(DEP_ARGS) $< -MT $(addsuffix .o, $(basename $@)) $(DEP_OUT) $@


cflags:
	echo $(CFLAGS)
