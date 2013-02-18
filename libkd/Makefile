#  This file is part of libkd.
#  Copyright 2006-2008 Dustin Lang and Keir Mierle.
#  Copyright 2012 Dustin Lang.
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
.PHONY: all

include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils
include $(COMMON)/makefile.anfiles
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

DT := dualtree.o dualtree_rangesearch.o dualtree_nearestneighbour.o

INSTALL_H := kdtree.h kdtree_fits_io.h dualtree.h \
	dualtree_nearestneighbour.h dualtree_rangesearch.h

# These are #included by other source files.
INTERNAL_SOURCES := kdtree_internal.c kdtree_internal_fits.c

CFLAGS := $(CFLAGS_DEF)
CFLAGS += $(QFITS_INC)
CFLAGS += $(ANUTILS_INC)

# Track memory usage?
#CFLAGS += -DKDTREE_MEM_TRACK

LDFLAGS := $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += -lm

LIBKD := libkd.a

LIBKD_NOIO := libkd-noio.a

INTERNALS_NOIO := \
	kdint_ddd_noio.o \
	kdint_fff_noio.o \
	kdint_ddu_noio.o \
	kdint_duu_noio.o \
	kdint_dds_noio.o \
	kdint_dss_noio.o

DEP_OBJ := $(KD) $(KD_FITS) $(INTERNALS) $(INTERNALS_NOIO) $(DT)

%_noio.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS) -DKDTREE_NO_FITS

all: $(LIBKD) checktree fix-bb

$(LIBKD): $(KD) $(KD_FITS) $(INTERNALS) $(DT)
	-rm -f $@
	$(AR) rc $@ $^
	$(RANLIB) $@

$(LIBKD_NOIO): $(KD) $(INTERNALS_NOIO) $(DT)
	-rm -f $@
	$(AR) rc $@ $^
	$(RANLIB) $@

pyspherematch: spherematch_c.so
	@echo pyspherematch

PYSPHEREMATCH_OBJ := pyspherematch.o

spherematch_c.so: pyspherematch.c setup.py $(LIBKD) $(QFITS_LIB) $(ANUTILS_LIB)
	python setup.py build --force --build-base build --build-platlib build/lib
	cp build/lib/spherematch_c.so $@

# pyspherematch.c includes Python.h
ifeq ($(MAKECMDGOALS),spherematch_c.so)
  DEP_OBJ += $(PYSPHEREMATCH_OBJ)
endif
ifeq ($(MAKECMDGOALS),pyspherematch)
  DEP_OBJ += $(PYSPHEREMATCH_OBJ)
endif

checktree: checktree.o $(LIBKD) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

fix-bb: fix-bb.o $(LIBKD) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

DEP_OBJ += fix-bb.o checktree.o

LIBKD_INSTALL := fix-bb checktree

.PHONY: install
install: $(LIBKD_INSTALL) $(LIBKD)
	@echo Installing in directory '$(INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/bin'
	@for x in $(LIBKD_INSTALL); do \
		echo cp $$x '$(INSTALL_DIR)/bin'; \
		cp $$x '$(INSTALL_DIR)/bin'; \
	done
	mkdir -p '$(INSTALL_DIR)/include'
	@for x in $(INSTALL_H); do \
		echo cp $$x '$(INSTALL_DIR)/include'; \
		cp $$x '$(INSTALL_DIR)/include'; \
	done
	mkdir -p '$(INSTALL_DIR)/lib'
	@for x in $(LIBKD); do \
		echo cp $$x '$(INSTALL_DIR)/lib'; \
		cp $$x '$(INSTALL_DIR)/lib'; \
	done
	# Try building and installing python spherematch module
	-$(MAKE) install-spherematch

.PHONY: install-spherematch
install-spherematch: spherematch_c.so
	mkdir -p '$(INSTALL_DIR)/python/astrometry/libkd'
	@for x in spherematch_c.so spherematch.py __init__.py; do \
		echo cp $$x '$(INSTALL_DIR)/python/astrometry/libkd'; \
		cp $$x '$(INSTALL_DIR)/python/astrometry/libkd'; \
	done


### TESTS are great

ALL_TEST_FILES = test_libkd test_libkd_io test_dualtree_nn
ALL_TEST_EXTRA_OBJS =
ALL_TEST_LIBS = $(LIBKD) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

include $(COMMON)/makefile.tests

test_libkd: $(LIBKD_NOIO) $(ANUTILS_LIB)

test_libkd_io: $(LIBKD) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_dualtree_nn: $(LIBKD) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

DEP_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)

### /TESTS

.PHONY: clean
clean:
	-rm -f $(LIBKD) $(KD) $(KD_FITS) deps $(DEPS) \
		checktree checktree.o \
		fix-bb fix-bb.o \
		$(INTERNALS) $(INTERNALS_NOIO) $(LIBKD_NOIO) $(DT) \
		$(ALL_TESTS_CLEAN) \
		$(PYSPHEREMATCH_OBJ) spherematch_c.so *~ *.dep deps
	-rm -Rf build

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
