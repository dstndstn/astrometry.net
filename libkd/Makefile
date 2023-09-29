# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE

BASEDIR := ..
COMMON := $(BASEDIR)/util

all:
.PHONY: all

include $(COMMON)/makefile.common
include $(COMMON)/makefile.libkd

SO=$(SHAREDLIB_SUFFIX)

INTERNALS := \
	kdint_ddd.o \
	kdint_fff.o \
	kdint_lll.o \
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

INC := $(QFITS_INC)
INC += $(ANUTILS_INC)
INC += $(LIBKD_INC)

CFLAGS := $(CFLAGS_DEF)
CFLAGS += $(ANUTILS_CFLAGS)
CFLAGS += $(INC)

# Track memory usage?
#CFLAGS += -DKDTREE_MEM_TRACK

LDFLAGS := $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(LIBKD_LIB)
LDLIBS += $(QFITS_LIB)
LDLIBS += $(ANUTILS_LIB)

LIBKD := libkd.a

SLIB := $(LIBKD)
SLIB += $(ANUTILS_SLIB)

LIBKD_NOIO := libkd-noio.a

INTERNALS_NOIO := \
	kdint_ddd_noio.o \
	kdint_fff_noio.o \
	kdint_lll_noio.o \
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

pyspherematch: spherematch_c$(PYTHON_SO_EXT)
	@echo pyspherematch
py: pyspherematch
.PHONY: py
.PHONY: pyspherematch

PYSPHEREMATCH_OBJ := pyspherematch.o

spherematch_c$(PYTHON_SO_EXT): pyspherematch.c setup.py $(SLIB)
	LDFLAGS="$(LDFLAGS)" LDLIBS="$(LDLIBS)" \
	SLIB="$(SLIB)" \
	INC="$(INC)" \
	CFLAGS="$(CFLAGS)" \
	$(PYTHON) setup.py build_ext --inplace --force --build-temp .

# pyspherematch.c includes Python.h -- so have to make sure to add Python.h include path
# (otherwise, get an obscure message from numpy about needing a python built with unicode)
ifeq ($(MAKECMDGOALS),spherematch_c$(PYTHON_SO_EXT))
  CFLAGS += $(shell $(PYTHON_CONFIG) --cflags)
  DEP_OBJ += $(PYSPHEREMATCH_OBJ)
endif
ifeq ($(MAKECMDGOALS),pyspherematch)
  CFLAGS += $(shell $(PYTHON_CONFIG) --cflags)
  DEP_OBJ += $(PYSPHEREMATCH_OBJ)
endif

checktree: checktree.o $(SLIB)

fix-bb: fix-bb.o $(SLIB)

demo: demo.o $(SLIB)

DEP_OBJ += fix-bb.o checktree.o

PY_INSTALL_DIR := $(PY_BASE_INSTALL_DIR)/libkd

.PHONY: install
install: $(LIBKD)
	@echo Installing in directory '$(INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/bin'
	mkdir -p '$(INCLUDE_INSTALL_DIR)'
	@for x in $(INSTALL_H); do \
		echo cp '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)'; \
		cp '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)'; \
	done
	mkdir -p '$(LIB_INSTALL_DIR)'
	@for x in $(LIBKD); do \
		echo cp $$x '$(LIB_INSTALL_DIR)'; \
		cp $$x '$(LIB_INSTALL_DIR)'; \
	done
	-$(MAKE) install-spherematch

.PHONY: install-spherematch
install-spherematch: spherematch_c$(PYTHON_SO_EXT)
	mkdir -p '$(PY_INSTALL_DIR)'
	@for x in spherematch_c$(PYTHON_SO_EXT) spherematch.py __init__.py; do \
		echo cp $$x '$(PY_INSTALL_DIR)'; \
		cp $$x '$(PY_INSTALL_DIR)'; \
	done


### TESTS are great

ALL_TEST_FILES = test_libkd test_libkd_io test_dualtree_nn
ALL_TEST_EXTRA_OBJS =
ALL_TEST_LIBS = $(SLIB)

include $(COMMON)/makefile.tests

test_libkd: $(SLIB)
test_libkd_io: $(SLIB)
test_dualtree_nn: $(SLIB)

DEP_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)

### /TESTS

.PHONY: clean
clean:
	-rm -f $(LIBKD) $(KD) $(KD_FITS) deps $(DEPS) \
		checktree checktree.o \
		fix-bb fix-bb.o \
		$(INTERNALS) $(INTERNALS_NOIO) $(LIBKD_NOIO) $(DT) \
		$(ALL_TESTS_CLEAN) \
		$(PYSPHEREMATCH_OBJ) spherematch_c$(PYTHON_SO_EXT) *~ *.dep deps
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
