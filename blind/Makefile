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
CATS_LIB := $(CATALOGS)/libcatalogs.a

.PHONY: all
all:

include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.cfitsio
include $(COMMON)/makefile.libkd
include $(COMMON)/makefile.gsl
include $(COMMON)/makefile.png
include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg
include $(COMMON)/makefile.zlib
include $(COMMON)/makefile.wcslib

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

LDFLAGS := $(LDFLAGS_DEF)
LDFLAGS += -fPIC

LDLIBS := $(LDLIBS_DEF)
LDLIBS += -lm

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(ANUTILS_INC)
CFLAGS += $(ANFILES_INC)
CFLAGS += $(LIBKD_INC)
CFLAGS += $(GSL_INC)
CFLAGS += $(QFITS_INC)
CFLAGS += $(CFITS_INC)
CFLAGS += $(ZLIB_INC)
CFLAGS += $(WCSLIB_INC)

CFLAGS += $(CATS_INC)

ALL_WCSLIB_TARGETS :=

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

QFITS_UTILS := resort-xylist

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
#add-text.o 

2MASSOBJ := 2masstofits.o 2mass-fits.o 2mass.o

NOT_INSTALLED_PIPELINE := blind agreeable certifiable usnobtofits hpowned \
	startree hpquads codetree unpermute-quads unpermute-stars \
	solvedserver printsolved mergesolved subwcs \
	2masstofits nomadtofits \
	augment-xylist merge-index index-to-table setsolved \
	startree2 uniformize-catalog \
	local-index index-info control-program

PIPELINE := wcs-grab fits-guess-scale new-wcs get-wcs \
	solve-field query-starkd

PROGS := astrometry-engine image2xy fit-wcs \
	build-astrometry-index 

PROSPECTUS := quadidx codeprojections quadscales quadsperstar \
	quadcenters startree2rdls

ANLIBS := $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

ENGINE_LIB := libastrometry.a
ENGINE_SO := libastrometry.so

INSTALL_LIB := $(ENGINE_LIB) $(ENGINE_SO)

ENGINE_OBJS := \
		engine.o blindutils.o blind.o solver.o quad-utils.o \
		matchfile.o matchobj.o solvedclient.o solvedfile.o tweak2.o \
		verify.o tweak.o 

# see also setup.py dependency list...
PLOTSTUFF := plotstuff.o plotfill.o plotxy.o plotimage.o \
		plotannotations.o plotgrid.o plotoutline.o plotindex.o plotradec.o \
		plothealpix.o plotmatch.o matchfile.o matchobj.o

$(ENGINE_LIB): $(ENGINE_OBJS)
	-rm -f $@
	$(AR) rc $@ $(ENGINE_OBJS)
	$(RANLIB) $@

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

$(ENGINE_SO): $(ENGINE_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB) $(GSL_LIB)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^ $(LDLIBS)

# old and miscellaneous executables that aren't part of the pipeline.
OLDEXECS := plotquads rawstartree checkquads

CAIROEXECS := plotquad plotxy plot-constellations plotstuff \
	whynot

OLDEXECS_OBJS := catalog.o verify.o matchfile.o \
	$(UTIL_OBJS)

PIPELINE_MAIN_OBJ := $(addsuffix .o,$(PIPELINE))
PROSPECTUS_MAIN_OBJ := $(addsuffix .o,$(PROSPECTUS))
FITS_UTILS_MAIN_OBJ := $(addsuffix .o,$(FITS_UTILS))

IMAGE2XY := image2xy

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
	$(OTHER_OBJS) $(2MASSOBJ)
ALL_EXECS :=

ALL_OBJ += $(PLOTSTUFF)

all: $(QFITS_LIB) $(LIBKD_LIB) \
	$(ANUTILS_LIB) $(ANFILES_LIB) \
	$(PROGS) $(PIPELINE) $(PROSPECTUS) $(FITS_UTILS) fitsverify \
	$(ENGINE_LIB) $(ENGINE_SO)

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
	mkdir -p '$(CONF_BASE_DIR)'
	python -c "import os; print open('../etc/astrometry.cfg-dist').read().replace('INSTALL_DIR', '$(CONF_BASE_DIR)')" > '$(CONF_BASE_DIR)/astrometry.cfg'
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
	mkdir -p '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_EXTRA_INSTALL); do \
		echo cp $$x '$(PY_INSTALL_DIR)/'$$x; \
		cp $$x '$(PY_INSTALL_DIR)/'$$x; \
	done

test-solver: test-solver.o solver_test.o verify.o matchfile.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)
test-solver-2: test-solver-2.o solver_test_2.o verify.o matchfile.o quad-utils.o codefile.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)

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

ALL_TEST_EXTRA_OBJS :=
ALL_TEST_EXTRA_LDFLAGS := -lm

# Add the dependencies here...
test_multiindex2: test_multiindex2.o solver.o verify.o \
	tweak2.o tweak.o matchfile.o matchobj.o quad-utils.o \
	$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)
ALL_TEST_EXTRA_OBJS += solver.o verify.o tweak2.o tweak.o quad-utils.o

#../util/cutest.o \

tests: $(ALL_TEST_FILES)
.PHONY: tests

test_codefile: codefile.o quad-utils.o $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += codefile.o

test_2mass: 2mass.o 2mass-fits.o $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += 2mass-fits.o 2mass.o

test_usnob: usnob-fits.o usnob.o $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += usnob-fits.o usnob.o

test_nomad: nomad-fits.o nomad.o $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += nomad-fits.o nomad.o

test_matchfile: matchfile.o matchobj.o $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += matchfile.o matchobj.o

test_blindutils: blindutils.o $(ANUTILS_LIB) $(GSL_LIB)
ALL_TEST_EXTRA_OBJS += blindutils.o

test_resort-xylist: resort-xylist.o $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += resort-xylist.o

test_tweak: test_tweak.o tweak.o $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += tweak.o

ALL_TEST_LIBS += $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) 

include $(COMMON)/makefile.tests

ALL_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)
ALL_OBJ += $(ALL_TEST_EXTRA_OBJS) test.o

GENERATED_FILES += $(ALL_TEST_FILES_MAIN_C)

ALL_EXECS += $(ALL_TEST_FILES) test

# END OF KEIR'S SUPER SIMPLE TESTING FRAMEWORK
######################################################

demo_dsmooth: demo_dsmooth.o dsmooth.o \
	$(COMMON)/cairoutils.o $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) -pg $(LDLIBS)

control-program: control-program.o \
		$(ENGINE_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)
ALL_OBJ += control-program.o

# qfits is only needed for writing debugging images...
image2xy: image2xy-main.o image2xy-files.o $(SIMPLEXY_OBJ) $(CFITS_DEP) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)

ALL_OBJ += image2xy-main.o image2xy-files.o

new-wcs: new-wcs-main.o new-wcs.o $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

ALL_OBJ += new-wcs-main.o

rdls2hpls: rdls2hpls.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

rdlstohealpix: rdlstohealpix.o \
		$(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

certifiable: certifiable.o matchfile.o matchobj.o solvedfile.o \
		$(UTILS_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

fits-guess-scale: fits-guess-scale-main.o fits-guess-scale.o \
		$(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

ALL_OBJ += fits-guess-scale-main.o

get-wcs: get-wcs.o \
		$(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)


fit-wcs: fit-wcs-main.o tweak2.o tweak.o verify.o matchobj.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) $(WCS_LIB_DEP)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCS_LIB) $(LDLIBS)
ALL_OBJ += fit-wcs-main.o

wcs-pv2sip: wcs-pv2sip.o tweak.o $(ANLIBS)
ALL_OBJ += wcs-pv2sip.o

hpgrid: hpgrid.o $(ANUTILS_LIB) $(GSL_LIB)

$(CFITS_UTILS) :: %: %.o $(CFITS_DEP)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)

fitsverify: ftverify.c fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c $(CFITS_DEP)
	$(CC) -DSTANDALONE -trigraphs $(CFITS_INC) -o $@ $^ $(CFITS_LIB) -lm

$(OLDEXECS) :: %: %.o $(OLDEXECS_OBJS) \
		$(ANUTILS_LIB) $(GSL_LIB) $(LIBKD_LIB) $(QFITS_LIB)

catalog_analysis: catalog_analysis.o pnpoly.o $(ANUTILS_LIB) $(GSL_LIB)

nomadtofits: nomadtofits.o nomad-fits.o nomad.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

ucac3tofits: ucac3tofits.o ucac3-fits.o ucac3.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) -lbz2 $(LDLIBS)
ALL_OBJ += ucac3tofits.o ucac3-fits.o ucac3.o

ucac4tofits: ucac4tofits.o ucac4-fits.o ucac4.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) -lbz2 $(LDLIBS)
ALL_OBJ += ucac4tofits.o ucac4-fits.o ucac4.o

read_nomad: read_nomad.o nomad.o

2masstofits: $(2MASSOBJ) $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(ZLIB_LIB) $(LDLIBS)

diffractionFlag_check: diffractionFlag_check.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

project-usnob: project-usnob.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

usnobtofits: usnobtofits.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

hpowned: hpowned.o $(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

startree: startree.o catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

startree2: startree2-main.o startree2.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += startree2-main.o startree2.o

uniformize-catalog: uniformize-catalog-main.o uniformize-catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += uniformize-catalog-main.o uniformize-catalog.o

build-astrometry-index: build-index-main.o build-index.o uniformize-catalog.o startree2.o hpquads.o \
		quad-builder.o quad-utils.o codefile.o codetree.o unpermute-stars.o \
		unpermute-quads.o merge-index.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)

ALL_OBJ += build-index-main.o build-index.o uniformize-catalog.o quad-utils.o

hpquads: hpquads-main.o hpquads.o quad-builder.o quad-utils.o codefile.o pnpoly.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += hpquads-main.o quad-builder.o hpquads.o

allquads: allquads-main.o allquads.o quad-builder.o quad-utils.o codefile.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

codetree: codetree-main.o codetree.o codefile.o quad-utils.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += codetree-main.o

unpermute-stars: unpermute-stars-main.o unpermute-stars.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += unpermute-stars-main.o

unpermute-quads: unpermute-quads-main.o unpermute-quads.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += unpermute-quads-main.o

merge-index: merge-index-main.o merge-index.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
ALL_OBJ += merge-index-main.o

local-index: local-index.o build-index.o codetree.o startree2.o codetree.o \
		hpquads.o uniformize-catalog.o merge-index.o unpermute-quads.o unpermute-stars.o \
		quad-builder.o quad-utils.o codefile.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCSLIB_LIB) $(LDLIBS)
ALL_OBJ += local-index.o

index-to-table: index-to-table.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

threadtest: threadtest.o $(ENGINE_LIB) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)

astrometry-engine: engine-main.o $(ENGINE_LIB) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)


### TWEAK_DEBUG_PLOTS
#backend: backend-main.o $(BACKEND_LIB) \
#		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB) \
#		$(PLOTSTUFF) $(COMMON)/cairoutils.o
#	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB)

verify: verify-main.o verify.o verify2.o matchfile.o matchobj.o $(PLOTSTUFF) $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

keirthing: keirthing.o tweak.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)

dstnthing: dstnthing.o tweak2.o tweak.o matchfile.o matchobj.o verify.o $(PLOTSTUFF) \
		$(COMMON)/cairoutils.o $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB) $(LDLIBS)
ALL_OBJ += dstnthing.o

ALL_OBJ += verify-main.o

solve-field: solve-field.o augment-xylist.o blindutils.o matchfile.o matchobj.o \
		image2xy-files.o resort-xylist.o cut-table.o \
		fits-guess-scale.o new-wcs.o \
		$(CFITS_DEP) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCSLIB_LIB) $(CFITS_LIB) $(LDLIBS)

ALL_OBJ += cut-table.o

augment-xylist: augment-xylist-main.o augment-xylist.o cut-table.o fits-guess-scale.o \
		image2xy-files.o resort-xylist.o blindutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(CFITS_DEP)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)

ALL_OBJ += augment-xylist-main.o

blind: blind-main.o blind.o solver.o quad-utils.o matchfile.o matchobj.o \
		solvedclient.o solvedfile.o tweak2.o verify.o \
		tweak.o \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@.tmp $^ $(LDLIBS)
	mv $@.tmp $@

tweak: tweak-main.o verify.o matchfile.o \
		$(COMMON)/cairoutils.o $(PLOTSTUFF) \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB) $(LDLIBS)

whynot: whynot.o matchfile.o matchobj.o codefile.o solvedclient.o \
		solvedfile.o verify.o quad-utils.o \
		$(PLOTSTUFF) $(COMMON)/cairoutils.o $(CATS_LIB) \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB) 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB) $(LDLIBS)
ALL_OBJ += whynot.o

quadcenters: quadcenters.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

startree2rdls: startree2rdls.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

agreeable: agreeable.o matchfile.o matchobj.o solvedfile.o solvedclient.o verify.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

index-info: index-info.o $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

subwcs: subwcs.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) \
		$(QFITS_LIB) $(GSL_LIB)
ALL_OBJS += subwcs.o

wcs-grab: wcs-grab.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) \
		$(QFITS_LIB)

printsolved: printsolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) 

setsolved: setsolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) 

mergesolved: mergesolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

solvedserver: solvedserver.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) \
		$(QFITS_LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

plotquad.o: plotquad.c 
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c $< -o $@

$(COMMON)/cairoutils.o:
	$(MAKE) -C $(COMMON) cairoutils.o

onepixel: onepixel.o $(COMMON)/cairoutils.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(LDLIBS)

plotquad: plotquad.o matchfile.o matchobj.o $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(LDLIBS)

plot-xy-and-quad: plot-xy-and-quad.o matchfile.o $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(LDLIBS)

plot-constellations.o: plot-constellations.c
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) $(CATS_INC) -c -o $@ $<

plot-constellations: plot-constellations.o $(CATS_LIB) $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(CAIRO_LIB) $(NETPBM_LIB) $(PNG_LIB) $(JPEG_LIB) $(LDLIBS)

catalog-dists: catalog-dists.o 2mass-fits.o usnob-fits.o usnob.o \
		$(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

add-text: add-text.o $(COMMON)/cairoutils.o $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(CAIRO_LIB) $(NETPBM_LIB) $(PNG_LIB) $(JPEG_LIB) $(LDLIBS)

plotstuff.o: plotstuff.c
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c -o $@ $<

plotstuff: plotstuff-main.o $(PLOTSTUFF) $(CATS_LIB) $(COMMON)/cairoutils.o $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB) $(LDLIBS)
ALL_OBJ += plotstuff-main.o

PLOTSTUFF_SRCS = $(subst .o,.c,$(PLOTSTUFF))
PLOTSTUFF_HDRS = $(subst .o,.h,$(PLOTSTUFF))

py: pyplotstuff
.PHONY: py

pyplotstuff: _plotstuff_c.so 
.PHONY: pyplotstuff

plotstuff_wrap.c: plotstuff.i $(PLOTSTUFF_SRCS) $(PLOTSTUFF_HDRS) $(ANUTILS_DIR)/sip.h $(ANUTILS_DIR)/anwcs.h
	swig -python $(ANUTILS_INC) $(QFITS_INC) $<

_plotstuff_c.so: plotstuff_wrap.c $(PLOTSTUFF) $(CATS_LIB) \
		$(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	python setup.py -v build_ext -v --inplace

plotxy.o: plotxy.c
	$(CC) $(CFLAGS) $(CAIRO_INC) -c -o $@ $<

plotxy-main.o: plotxy-main.c
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c $< -o $@

plotxy: plotxy-main.o $(PLOTSTUFF) $(CATS_LIB) $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB) $(LDLIBS)
ALL_OBJ += plotxy-main.o

test_plotstuff: test_plotstuff-main.o test_plotstuff.o $(COMMON)/cutest.o $(PLOTSTUFF) $(COMMON)/cairoutils.o $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(WCSLIB_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(LDLIBS)
test_plotstuff-main.c: test_plotstuff.c
	$(AN_SHELL) $(MAKE_TESTS) $^ > $@

query-starkd: query-starkd.o $(ANLIBS)
ALL_OBJ += query-starkd.o

quadidx: quadidx.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

codeprojections: codeprojections.o $(UTIL_OBJS) $(ANLIBS)

quadscales: quadscales.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

index-stats: index-stats.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

quadsperstar: quadsperstar.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) \
		$(QFITS_LIB)

resort-xylist: resort-xylist-main.o resort-xylist.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

ALL_OBJ += resort-xylist-main.o

bgsubtract: bgsubtract.o ctmf.o $(COMMON)/cairoutils.o $(UTIL_OBJS) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) $(LDLIBS)

usnob-scamp-catalog: usnob-scamp-catalog.o usnob-fits.o usnob.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

create-scamp-catalog: create-scamp-catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB)

DEP_OBJ := $(ALL_OBJ)
DEP_PREREQS := $(QFITS_LIB)

CFLAGS += $(CAIRO_INC)
include $(COMMON)/makefile.deps

.PHONY: clean

clean:
	rm -f $(EXECS) $(EXTRA_EXECS) $(SOLVER_EXECS) $(MISC_EXECS) $(PROGS) \
		$(PIPELINE) $(PROSPECTUS) $(DEPS) deps $(FITS_UTILS) $(ALL_OBJ) $(IMAGE2XY) \
		$(NODEP_OBJS) plot-constellations fitsverify plotquad plotxy \
		$(ALL_EXECS) $(GENERATED_FILES) $(ALL_TESTS_CLEAN) \
		$(ENGINE_LIB) $(ENGINE_SO) _plotstuff_c.so *.o *~ *.dep deps
	rm -Rf build



