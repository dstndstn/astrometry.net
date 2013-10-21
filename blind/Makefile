# This file is part of the Astrometry.net suite.
# Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
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

CATALOGS := $(BASEDIR)/catalogs
CATS_INC := -I$(CATALOGS)
CATS_SLIB := $(CATALOGS)/libcatalogs.a
CATS_LIB :=

.PHONY: all
all:

include $(COMMON)/makefile.common
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.cfitsio
include $(COMMON)/makefile.png
include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg
include $(COMMON)/makefile.zlib
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

ENGINE_LIB := libastrometry.a
ENGINE_SO := libastrometry.so

LDFLAGS := $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(ANFILES_LIB)

SLIB := $(ENGINE_LIB)
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
CFLAGS += $(CFITS_INC)
CFLAGS += $(ZLIB_INC)

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

ALL_WCSLIB_TARGETS :=

QFITS_UTILS := 
#resort-xylist

CFITS_UTILS := tablist modhead fitscopy tabmerge liststruc \
		listhead imcopy imarith imstat
# fitstomatlab 

FITS_UTILS := $(QFITS_UTILS) $(CFITS_UTILS)

MISC_EXECS := project-usnob diffractionFlag_check \
	hpgrid read_nomad usnob-scamp-catalog

UTIL_OBJS := 

OTHER_OBJS := usnob.o usnob-fits.o catalog.o codefile.o verify.o \
	solver.o matchfile.o matchobj.o solvedclient.o solvedfile.o pnpoly.o \
	tweak.o blind-main.o \
	plot-constellations.o quadcenters.o startree2rdls.o \
	nomad.o nomad-fits.o blindutils.o \
	engine-main.o engine.o plotquad.o plotxy.o tweak2.o

NOT_INSTALLED_PIPELINE := blind agreeable certifiable usnobtofits \
	startree hpquads codetree unpermute-quads unpermute-stars \
	solvedserver printsolved mergesolved subwcs \
	2masstofits nomadtofits \
	augment-xylist merge-index index-to-table setsolved \
	startree2 uniformize-catalog \
	local-index index-info control-program

PIPELINE := wcs-grab solve-field

MAIN_PROGS := image2xy new-wcs fits-guess-scale fit-wcs
# hpquads

SIMPLE_PROGS := nomadtofits usnobtofits wcs-grab get-wcs
# hpowned

PROGS := astrometry-engine build-astrometry-index \
	$(MAIN_PROGS) $(SIMPLE_PROGS)

PROSPECTUS := quadidx codeprojections quadscales quadsperstar \
	quadcenters startree2rdls

ANLIBS := $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

INSTALL_LIB := $(ENGINE_LIB) $(ENGINE_SO)

ENGINE_OBJS := \
		engine.o blindutils.o blind.o solver.o quad-utils.o \
		matchfile.o matchobj.o solvedclient.o solvedfile.o tweak2.o \
		verify.o tweak.o

# These are required by solve-field and friends
ENGINE_OBJS += new-wcs.o fits-guess-scale.o cut-table.o \
	resort-xylist.o

# Conversion routines
ENGINE_OBJS += 2mass-fits.o 2mass.o nomad-fits.o nomad.o ucac3-fits.o ucac3.o \
	ucac4-fits.o ucac4.o usnob-fits.o usnob.o

BUILD_INDEX_OBJS := build-index.o uniformize-catalog.o startree2.o hpquads.o \
	quad-builder.o quad-utils.o codefile.o codetree.o unpermute-stars.o \
	unpermute-quads.o merge-index.o 
ENGINE_OBJS += $(BUILD_INDEX_OBJS)

#augment-xylist.o

# see also setup.py dependency list...
PLOTSTUFF := plotstuff.o plotfill.o plotxy.o plotimage.o \
		plotannotations.o plotgrid.o plotoutline.o plotindex.o plotradec.o \
		plothealpix.o plotmatch.o matchfile.o matchobj.o

$(ENGINE_LIB): $(ENGINE_OBJS)
	-rm -f $@
	$(AR) rc $@ $(ENGINE_OBJS)
	$(RANLIB) $@

$(ENGINE_SO): $(ENGINE_OBJS) $(SLIB)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^ $(LDLIBS)

# old and miscellaneous executables that aren't part of the pipeline.
OLDEXECS := plotquads rawstartree checkquads

CAIROEXECS := plotquad plotxy plot-constellations plotstuff 
# whynot

OLDEXECS_OBJS := catalog.o verify.o matchfile.o \
	$(UTIL_OBJS)

PIPELINE_MAIN_OBJ := $(addsuffix .o,$(PIPELINE))
PROSPECTUS_MAIN_OBJ := $(addsuffix .o,$(PROSPECTUS))
FITS_UTILS_MAIN_OBJ := $(addsuffix .o,$(FITS_UTILS))

INSTALL_EXECS := $(FITS_UTILS) $(PIPELINE) $(PROGS)
INSTALL_CAIRO_EXECS := $(CAIROEXECS)

INSTALL_H := 2mass-fits.h 2mass.h allquads.h augment-xylist.h axyfile.h \
	engine.h blind.h blindutils.h build-index.h catalog.h \
	codefile.h codetree.h fits-guess-scale.h hpquads.h \
	image2xy-files.h matchfile.h matchobj.h merge-index.h \
	new-wcs.h nomad-fits.h nomad.h quad-builder.h quad-utils.h \
	resort-xylist.h solvedclient.h \
	solvedfile.h solver.h tweak.h uniformize-catalog.h \
	unpermute-quads.h unpermute-stars.h usnob-fits.h usnob.h verify.h \
	tweak2.h

PLOT_INSTALL_H := plotannotations.h plotfill.h plotgrid.h plotimage.h \
	plotoutline.h plotstuff.h plotxy.h

ALL_OBJ := $(UTIL_OBJS) $(KDTREE_OBJS) $(QFITS_OBJ) \
	$(PIPELINE_MAIN_OBJ) $(PROSPECTUS_MAIN_OBJ) $(FITS_UTILS_MAIN_OBJ) \
	$(OTHER_OBJS)
ALL_EXECS :=

ALL_OBJ += $(PLOTSTUFF)

all: $(QFITS_SLIB) $(LIBKD_LIB_FILE) \
	$(ANUTILS_LIB_FILE) $(ANFILES_LIB_FILE) \
	$(PROGS) $(PIPELINE) $(FITS_UTILS) fitsverify \
	$(ENGINE_LIB) $(ENGINE_SO)

$(MAIN_PROGS): %: %-main.o $(SLIB)
ALL_OBJ += $(addsuffix -main.o,$(MAIN_PROGS))

$(SIMPLE_PROGS): %: %.o $(SLIB)
ALL_OBJ += $(addsuffix .o,$(SIMPLE_PROGS))

cairo: $(CAIROEXECS)
.PHONY: cairo

.PHONY: pipeline
pipeline: $(PIPELINE)

.PHONY: prospectus_progs
prospectus_progs: $(PROSPECTUS)

.PHONY: fitsutils
fitsutils: $(FITS_UTILS)

simplexy: image2xy
.PHONY: simplexy

PY_INSTALL_DIR := $(PY_BASE_INSTALL_DIR)/blind
LINK_DIR := $(PY_BASE_LINK_DIR)/blind

PYTHON_EXECS := plotann.py
PYTHON_INSTALL := $(PYTHON_EXECS) __init__.py

install: $(INSTALL_EXECS) $(INSTALL_LIB)
	@echo Installing in directory '$(INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/bin'
	@for x in $(INSTALL_EXECS); do \
		echo cp $$x '$(INSTALL_DIR)/bin'; \
		cp $$x '$(INSTALL_DIR)/bin'; \
	done
	mkdir -p '$(ETC_INSTALL_DIR)'
	python -c "import os; print open('../etc/astrometry.cfg-dist').read().replace('DATA_INSTALL_DIR', '$(DATA_FINAL_DIR)')" > '$(ETC_INSTALL_DIR)/astrometry.cfg'
	mkdir -p '$(INCLUDE_INSTALL_DIR)'
	@for x in $(INSTALL_H); do \
		echo cp $$x '$(INCLUDE_INSTALL_DIR)'; \
		cp $$x '$(INCLUDE_INSTALL_DIR)'; \
	done
	mkdir -p '$(INSTALL_DIR)/lib'
	@for x in $(INSTALL_LIB); do \
		echo cp $$x '$(INSTALL_DIR)/lib'; \
		cp $$x '$(INSTALL_DIR)/lib'; \
	done
	mkdir -p '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo cp $$x '$(PY_INSTALL_DIR)/'$$x; \
		cp $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@echo Making symlinks in directory '$(BIN_INSTALL_DIR)'
	mkdir -p '$(BIN_INSTALL_DIR)'
	@for x in $(PYTHON_EXECS); do \
		echo ln -f -s '$(PY_INSTALL_DIR)/'$$x '$(BIN_INSTALL_DIR)/'$$x; \
		ln -f -s '$(LINK_DIR)/'$$x '$(BIN_INSTALL_DIR)/'$$x; \
	done


PYTHON_EXTRA_INSTALL := plotstuff.py plotstuff_c.py _plotstuff_c.so

install-extra: $(INSTALL_CAIRO_EXECS)
	@echo Installing in directory '$(INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/bin'
	@for x in $(INSTALL_CAIRO_EXECS); do \
		echo cp $$x '$(INSTALL_DIR)/bin'; \
		cp $$x '$(INSTALL_DIR)/bin'; \
	done
	mkdir -p '$(INCLUDE_INSTALL_DIR)'
	@for x in $(PLOT_INSTALL_H); do \
		echo cp $$x '$(INCLUDE_INSTALL_DIR)'; \
		cp $$x '$(INCLUDE_INSTALL_DIR)'; \
	done
	$(MAKE) $(PYTHON_EXTRA_INSTALL)
	mkdir -p '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_EXTRA_INSTALL); do \
		echo cp $$x '$(PY_INSTALL_DIR)/'$$x; \
		cp $$x '$(PY_INSTALL_DIR)/'$$x; \
	done

test-solver: test-solver.o solver_test.o $(SLIB)
test-solver-2: test-solver-2.o solver_test_2.o $(SLIB)

NODEP_OBJS += solver_test.o solver_test_2.o
ALL_OBJ += test-solver.o test-solver-2.o

CFLAGS_DEBUG = $(subst -DNDEBUG,,$(CFLAGS))

test-solver.o: test-solver.c
	$(CC) $(CFLAGS_DEBUG) -o $@ -c $<
test-solver-2.o: test-solver-2.c
	$(CC) $(CFLAGS_DEBUG) -o $@ -c $<

solver_test.o: solver.c
	$(CC) $(CFLAGS_DEBUG) -DTESTING=1 -DTESTING_TRYALLCODES=1 -o $@ -c $<
solver_test_2.o: solver.c
	$(CC) $(CFLAGS_DEBUG) -DTESTING=1 -DTESTING_TRYPERMUTATIONS=1 -o $@ -c $<


#######################################

# Add the basename of your test sources here...
ALL_TEST_FILES = test_2mass \
	test_usnob test_nomad test_matchfile test_blindutils \
	test_resort-xylist test_tweak \
	test_multiindex2 test_codefile
$(ALL_TEST_FILES): $(SLIB)

ALL_TEST_EXTRA_OBJS :=
ALL_TEST_EXTRA_LDFLAGS := -lm
ALL_TEST_LIBS := $(SLIB)

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

demo_dsmooth: demo_dsmooth.o $(CAIRO_SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(CAIRO_LIBS)

control-program: control-program.o $(SLIB)
ALL_OBJ += control-program.o

image2xy: image2xy-main.o image2xy-files.o $(CFITS_SLIB) $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)

fit-wcs: fit-wcs-main.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS)

ALL_OBJ += image2xy-files.o

wcs-pv2sip: wcs-pv2sip.o tweak.o $(SLIB)
ALL_OBJ += wcs-pv2sip.o

hpgrid: hpgrid.o $(SLIB)

$(CFITS_UTILS) :: %: %.o $(CFITS_SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)

fitsverify: ftverify.c fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c $(CFITS_SLIB)
	$(CC) -DSTANDALONE -trigraphs $(CFITS_INC) -o $@ $^ $(CFITS_LIB) -lm

$(OLDEXECS) :: %: %.o $(OLDEXECS_OBJS) $(SLIB)

ucac3tofits: ucac3tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac3tofits.o

ucac4tofits: ucac4tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac4tofits.o

2masstofits: 2masstofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) $(ZLIB_LIB)

build-astrometry-index: build-index-main.o $(SLIB) #$(BUILD_INDEX_OBJS) $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS)
ALL_OBJ += build-index-main.o

astrometry-engine: engine-main.o $(SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)

solve-field: solve-field.o augment-xylist.o image2xy-files.o $(SLIB) $(CFITS_SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)
ALL_OBJ += solve-field.o image2xy-files.o

augment-xylist: augment-xylist-main.o augment-xylist.o image2xy-files.o \
		$(SLIB) $(CFITS_SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)
ALL_OBJ += augment-xylist-main.o augment-xylist.o

$(COMMON)/cairoutils.o:
	$(MAKE) -C $(COMMON) cairoutils.o

plotquad.o: plotquad.c
	$(CC) -o $@ -c $< $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC)
plotquad: plotquad.o $(CAIRO_SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CAIRO_LIBS)

plot-constellations.o: plot-constellations.c
	$(CC) -o $@ -c $< $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) $(CATS_INC)
plot-constellations: plot-constellations.o $(CAIRO_SLIB) $(CATS_SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(CATS_LIB) $(CAIRO_LIBS)

plotstuff.o: plotstuff.c
	$(CC) -o $@ -c $< $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC)
plotstuff: plotstuff-main.o $(PLOTSTUFF) $(CAIRO_SLIB) $(CATS_SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(CATS_LIB) $(CAIRO_LIBS) 
ALL_OBJ += plotstuff-main.o

plotxy.o: plotxy.c
	$(CC) -o $@ -c $< $(CFLAGS) $(CAIRO_INC) 
plotxy-main.o: plotxy-main.c
	$(CC) -o $@ -c $< $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC)
plotxy: plotxy-main.o $(PLOTSTUFF) $(CATS_SLIB) $(CAIRO_SLIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIBS)
ALL_OBJ += plotxy-main.o

PLOTSTUFF_SRCS = $(subst .o,.c,$(PLOTSTUFF))
PLOTSTUFF_HDRS = $(subst .o,.h,$(PLOTSTUFF))

py: pyplotstuff
.PHONY: py

pyplotstuff: _plotstuff_c.so 
.PHONY: pyplotstuff

plotstuff_wrap.c: plotstuff.i $(PLOTSTUFF_SRCS) $(PLOTSTUFF_HDRS) \
		$(ANUTILS_DIR)/sip.h $(ANUTILS_DIR)/anwcs.h
	swig -python $(ANFILES_INC) $<

_plotstuff_c.so: plotstuff_wrap.c $(PLOTSTUFF) $(CATS_LIB) $(CAIRO_SLIB)
	LDFLAGS="$(LDFLAGS)" LDLIBS="$(LDLIBS) $(CAIRO_LIBS)" \
	SLIB="$(CAIRO_SLIB)" \
	INC="$(ANFILES_INC) $(CAIRO_INC) $(NETPBM_INC)" \
	CFLAGS="$(ANFILES_CFLAGS)" \
	python setup.py build_ext -v --inplace --build-temp .

test_plotstuff-main.c: test_plotstuff.c
	$(AN_SHELL) $(MAKE_TESTS) $^ > $@
test_plotstuff: test_plotstuff-main.o test_plotstuff.o $(COMMON)/cutest.o $(PLOTSTUFF) $(CATS_SLIB) $(CAIRO_SLIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIBS)

DEP_OBJ := $(ALL_OBJ)
DEP_PREREQS := $(QFITS_LIB)

CFLAGS += $(CAIRO_INC)
include $(COMMON)/makefile.deps

.PHONY: clean

clean:
	rm -f $(EXECS) $(EXTRA_EXECS) $(SOLVER_EXECS) $(MISC_EXECS) $(PROGS) \
		$(PIPELINE) $(PROSPECTUS) $(DEPS) deps $(FITS_UTILS) $(ALL_OBJ) \
		$(NODEP_OBJS) plot-constellations fitsverify plotquad plotxy \
		$(ALL_EXECS) $(GENERATED_FILES) $(ALL_TESTS_CLEAN) \
		$(ENGINE_LIB) $(ENGINE_SO) _plotstuff_c.so *.o *~ *.dep deps \
		plotstuff



