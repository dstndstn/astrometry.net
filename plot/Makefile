# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

BASEDIR := ..
COMMON := $(BASEDIR)/util

CATALOGS := $(BASEDIR)/catalogs
CATS_INC := -I$(CATALOGS)
CATS_SLIB := $(CATALOGS)/libcatalogs.a
CATS_LIB :=

.PHONY: all
all:

# Detect GSL -- this minimum version was chosen to match the version in gsl-an.
# Earlier versions would probably work fine.
SYSTEM_GSL ?= $(shell (pkg-config --atleast-version=1.14 gsl && echo "yes") || echo "no")
# Make this variable visible to recursive "make" calls
export SYSTEM_GSL

include $(COMMON)/makefile.common
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.cfitsio
include $(COMMON)/makefile.png
include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg
include $(COMMON)/makefile.netpbm

ifneq ($(MAKECMDGOALS),clean)
    include $(COMMON)/makefile.os-features
endif

$(COMMON)/makefile.os-features:
	$(MAKE) -C $(COMMON) makefile.os-features

ifeq ($(HAVE_NETPBM),yes)
else
NETPBM_INC :=
NETPBM_LIB :=
endif

SO=$(SHAREDLIB_SUFFIX)

LDFLAGS := $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(ANFILES_LIB)

SLIB += $(CATS_SLIB)
SLIB += $(ANFILES_SLIB)

CAIRO_SLIB := $(COMMON)/cairoutils.o
CAIRO_SLIB += $(SLIB)

CAIRO_LIBS := $(LDLIBS)
CAIRO_LIBS += $(CAIRO_LIB)
CAIRO_LIBS += $(PNG_LIB)
CAIRO_LIBS += $(NETPBM_LIB)
CAIRO_LIBS += $(JPEG_LIB)

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(CATS_INC)
CFLAGS += $(ANFILES_CFLAGS)
CFLAGS += $(ANFILES_INC)
# boilerplate.h
CFLAGS += -I$(ANUTILS_DIR)
# wcs-resample.h
CFLAGS += -I.
CFLAGS += $(CFITS_INC)

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

ALL_WCSLIB_TARGETS :=

OTHER_OBJS := plot-constellations.o plotquad.o plotxy.o

ANLIBS := $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

# see also setup.py dependency list...
PLOTSTUFF := plotstuff.o plotfill.o plotxy.o plotimage.o \
		plotannotations.o plotgrid.o plotoutline.o plotindex.o plotradec.o \
		plothealpix.o plotmatch.o

CAIROEXECS := plotquad plotxy plot-constellations

INSTALL_CAIRO_EXECS := $(CAIROEXECS)

PLOT_INSTALL_H := plotannotations.h plotfill.h plotgrid.h plotimage.h \
	plotoutline.h plotstuff.h plotxy.h

ALL_OBJ := $(OTHER_OBJS)
ALL_EXECS :=

ALL_OBJ += $(PLOTSTUFF)

cairo: $(CAIROEXECS)
.PHONY: cairo

PY_INSTALL_DIR := $(PY_BASE_INSTALL_DIR)/plot
LINK_DIR := $(PY_BASE_LINK_DIR)/plot

PYTHON_EXECS := plotann.py
PYTHON_INSTALL := $(PYTHON_EXECS) __init__.py

PYTHON_EXTRA_INSTALL := plotstuff.py plotstuff_c.py _plotstuff_c$(PYTHON_SO_EXT)

install-extra: $(INSTALL_CAIRO_EXECS)
	@echo Installing in directory '$(INSTALL_DIR)'
	$(MKDIR) '$(INSTALL_DIR)/bin'
	@for x in $(INSTALL_CAIRO_EXECS); do \
		echo $(CP) $$x '$(INSTALL_DIR)/bin'; \
		$(CP) $$x '$(INSTALL_DIR)/bin'; \
	done
	$(MKDIR) '$(INCLUDE_INSTALL_DIR)'
	@for x in $(PLOT_INSTALL_H); do \
		echo $(CP) '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)'; \
		$(CP) '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)'; \
	done
	$(MAKE) $(PYTHON_EXTRA_INSTALL)
	$(MKDIR) '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_EXTRA_INSTALL); do \
		echo $(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@for x in $(PYTHON_INSTALL); do \
		echo $(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@echo Making symlinks in directory '$(BIN_INSTALL_DIR)'
	$(MKDIR) '$(BIN_INSTALL_DIR)'
	@for x in $(PYTHON_EXECS); do \
		echo ln -f -s '$(LINK_DIR)/'$$x '$(BIN_INSTALL_DIR)/'$$x; \
		ln -f -s '$(LINK_DIR)/'$$x '$(BIN_INSTALL_DIR)/'$$x; \
	done

$(COMMON)/cairoutils.o:
	$(MAKE) -C $(COMMON) cairoutils.o

plotquad.o: plotquad.c
	$(CC) -o $@ -c $< $(CPPFLAGS) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC)
plotquad: plotquad.o $(CAIRO_SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CAIRO_LIBS)

plot-constellations.o: plot-constellations.c
	$(CC) -o $@ -c $< $(CPPFLAGS) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) $(CATS_INC)
plot-constellations: plot-constellations.o $(CAIRO_SLIB) $(CATS_SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(CATS_LIB) $(CAIRO_LIBS)

plotstuff.o: plotstuff.c
	$(CC) -o $@ -c $< $(CPPFLAGS) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC)
plotstuff: plotstuff-main.o $(PLOTSTUFF) $(CAIRO_SLIB) $(CATS_SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(CATS_LIB) $(CAIRO_LIBS) 
ALL_OBJ += plotstuff-main.o

plotxy.o: plotxy.c
	$(CC) -o $@ -c $< $(CPPFLAGS) $(CFLAGS) $(CAIRO_INC)
plotxy-main.o: plotxy-main.c
	$(CC) -o $@ -c $< $(CFLAGS) $(CPPFLAGS) $(CAIRO_INC) $(NETPBM_INC)
plotxy: plotxy-main.o $(PLOTSTUFF) $(CATS_SLIB) $(CAIRO_SLIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIBS)
ALL_OBJ += plotxy-main.o

PLOTSTUFF_SRCS = $(subst .o,.c,$(PLOTSTUFF))
PLOTSTUFF_HDRS = $(addprefix $(INCLUDE_DIR)/,$(subst .o,.h,$(PLOTSTUFF)))

py: pyplotstuff
.PHONY: py

pyplotstuff: _plotstuff_c$(PYTHON_SO_EXT)
.PHONY: pyplotstuff

_plotstuff_c$(PYTHON_SO_EXT): plotstuff.i $(PLOTSTUFF) $(CATS_LIB) $(CAIRO_SLIB)
	LDFLAGS="$(LDFLAGS)" LDLIBS="$(LDLIBS) $(CAIRO_LIBS)" \
	SLIB="$(CAIRO_SLIB)" \
	INC="$(ANFILES_INC) $(CAIRO_INC) $(NETPBM_INC)" \
	CFLAGS_SWIG="$(ANFILES_CFLAGS)" \
	CFLAGS="$(CFLAGS)" \
	$(PYTHON) setup.py build_ext -v --inplace --build-temp .

test_plotstuff-main.c: test_plotstuff.c
	$(AN_SHELL) $(MAKE_TESTS) $^ > $@
test_plotstuff: test_plotstuff-main.o test_plotstuff.o $(COMMON)/cutest.o $(PLOTSTUFF) $(CATS_SLIB) $(CAIRO_SLIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIBS)

#######################################

# Add the basename of your test sources here...
ALL_TEST_FILES = test_plotstuff

$(ALL_TEST_FILES): $(SLIB)

ALL_TEST_EXTRA_OBJS := $(PLOTSTUFF)
ALL_TEST_EXTRA_LDFLAGS := $(CAIRO_LIBS) -lm
ALL_TEST_LIBS := $(CAIRO_SLIB)

# Add the dependencies here...
#test_multiindex2: test_multiindex2.o $(SLIB)

tests: $(ALL_TEST_FILES)
.PHONY: tests

include $(COMMON)/makefile.tests

ALL_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)
ALL_OBJ += $(ALL_TEST_EXTRA_OBJS) test.o

GENERATED_FILES += $(ALL_TEST_FILES_MAIN_C)

ALL_EXECS += $(ALL_TEST_FILES) test

# END OF KEIR'S SUPER SIMPLE TESTING FRAMEWORK
######################################################

DEP_OBJ := $(ALL_OBJ)
DEP_PREREQS := $(QFITS_LIB)

CFLAGS += $(CAIRO_INC)
include $(COMMON)/makefile.deps

.PHONY: clean

clean:
	rm -f $(DEPS) $(ALL_OBJ) \
		$(NODEP_OBJS) plot-constellations plotquad plotxy \
		$(ALL_EXECS) $(GENERATED_FILES) $(ALL_TESTS_CLEAN) \
		plotstuff _plotstuff_c$(PYTHON_SO_EXT) plotstuff_c.py plotstuff_wrap.c *.o *~ *.dep deps
