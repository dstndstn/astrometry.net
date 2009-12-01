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
DATA := $(BASEDIR)/data

.PHONY: all
all:

include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.cfitsio
include $(COMMON)/makefile.libkd
include $(COMMON)/makefile.gsl
include $(COMMON)/makefile.netpbm
include $(COMMON)/makefile.png
include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg

LDFLAGS := $(LDFLAGS_DEF)
LDFLAGS += -lm
LDFLAGS += -fPIC

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(QFITS_INC)
CFLAGS += $(CFITS_INC)
CFLAGS += $(LIBKD_INC)
CFLAGS += $(ANUTILS_INC)
CFLAGS += $(ANFILES_INC)
CFLAGS += $(GSL_INC)

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

QFITS_UTILS := fitsgetext subtable tabsort column-merge add-healpix-column \
		resort-xylist fits-flip-endian fits-dedup wcs-resample

CFITS_UTILS := tablist modhead fitscopy tabmerge fitstomatlab liststruc \
		listhead imcopy imarith imstat

FITS_UTILS := $(QFITS_UTILS) $(CFITS_UTILS)

MISC_EXECS := project-usnob diffractionFlag_check \
	hpgrid read_nomad usnob-scamp-catalog

UTIL_OBJS := 

OTHER_OBJS := usnob.o usnob-fits.o catalog.o \
	codefile.o histogram.o verify.o \
	solver.o matchfile.o solvedclient.o solvedfile.o intmap.o pnpoly.o \
	blind_wcs.o tweak.o blind-main.o \
	plot-constellations.o quadcenters.o startree2rdls.o \
	nomad.o nomad-fits.o \
	noisesim.o noise.o blindutils.o \
	backend-main.o backend.o plotquad.o plotxy.o add-text.o

2MASSOBJ := 2masstofits.o 2mass-fits.o 2mass.o

PIPELINE := usnobtofits build-an-catalog hpowned cut-an \
	startree hpquads codetree unpermute-quads unpermute-stars blind \
	solvedserver printsolved xylist2fits mergesolved rdlsinfo xylsinfo \
	wcsinfo wcs-grab fits-guess-scale agreeable new-wcs get-wcs \
	augment-xylist solve-field plotcat wcs-xy2rd wcs-rd2xy \
	certifiable get-healpix 2masstofits nomadtofits \
	merge-index index-to-table endian-flip-index setsolved

PROGS := backend image2xy control-program

PROSPECTUS := quadidx codeprojections quadscales quadsperstar \
	whynot quadcenters startree2rdls

ANLIBS := $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

BACKEND_LIB := libbackend.a

BACKEND_SO := libbackend.so

BACKEND_OBJS := \
		backend.o blindutils.o blind.o solver.o \
		matchfile.o solvedclient.o solvedfile.o verify.o \
		blind_wcs.o tweak.o

$(BACKEND_LIB): $(BACKEND_OBJS)
	-rm -f $@
	$(AR) rc $@ $(BACKEND_OBJS)
	$(RANLIB) $@

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

$(BACKEND_SO): $(BACKEND_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB) $(GSL_LIB)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

# old and miscellaneous executables that aren't part of the pipeline.
OLDEXECS := plotquads rawstartree checkquads

CAIROEXECS := plotquad plotxy plot-constellations add-text plotstuff

OLDEXECS_OBJS := catalog.o histogram.o verify.o intmap.o matchfile.o \
	$(UTIL_OBJS)

PIPELINE_MAIN_OBJ := $(addsuffix .o,$(PIPELINE))
PROSPECTUS_MAIN_OBJ := $(addsuffix .o,$(PROSPECTUS))
FITS_UTILS_MAIN_OBJ := $(addsuffix .o,$(FITS_UTILS))

SIMPLEXY_OBJ := dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o dobjects.o \
	dpeaks.o dselip.o dsigma.o dsmooth.o image2xy.o simplexy.o radix.o ctmf.o

IMAGE2XY := image2xy

INSTALL_EXECS := $(FITS_UTILS) $(PIPELINE) $(PROGS)
INSTALL_CAIRO_EXECS := $(CAIROEXECS)

ALL_OBJ := $(UTIL_OBJS) $(KDTREE_OBJS) $(QFITS_OBJ) \
	$(PIPELINE_MAIN_OBJ) $(PROSPECTUS_MAIN_OBJ) $(FITS_UTILS_MAIN_OBJ) \
	$(OTHER_OBJS) $(2MASSOBJ) $(SIMPLEXY_OBJ)
ALL_EXECS :=

all: $(QFITS_LIB) $(CFITS_LIB) $(LIBKD_LIB) \
	$(REMAKE_LIBKD) $(REMAKE_ANUTILS) $(REMAKE_ANFILES) \
	$(PROGS) $(PIPELINE) $(PROSPECTUS) $(FITS_UTILS) fitsverify \
	$(BACKEND_LIB) $(BACKEND_SO)

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

install: $(INSTALL_EXECS)
	@echo Installing in directory $(INSTALL_DIR)
	mkdir -p $(INSTALL_DIR)/bin
	@for x in $(INSTALL_EXECS); do \
		echo cp $$x $(INSTALL_DIR)/bin; \
		cp $$x $(INSTALL_DIR)/bin; \
	done
	mkdir -p $(INSTALL_DIR)/etc
	sed "s+add_path /usr/local/astrometry/+add_path $(INSTALL_DIR)/+" < ../etc/backend.cfg-dist > $(INSTALL_DIR)/etc/backend.cfg

install-extra: $(INSTALL_CAIRO_EXECS)
	@echo Installing in directory $(INSTALL_DIR)
	mkdir -p $(INSTALL_DIR)/bin
	@for x in $(INSTALL_CAIRO_EXECS); do \
		echo cp $$x $(INSTALL_DIR)/bin; \
		cp $$x $(INSTALL_DIR)/bin; \
	done

test-solver: test-solver.o solver_test.o blind_wcs.o verify.o \
		matchfile.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)

NODEP_OBJS += solver_test.o
ALL_OBJ += test-solver.o

solver_test.o: solver.c
	$(CC) $(CFLAGS) -DTESTING=1 -o $@ -c $<

#######################################

# Add the basename of your test sources here...
ALL_TEST_FILES = test_dfind test_ctmf test_dsmooth test_2mass \
	test_usnob test_nomad test_matchfile test_blindutils \
	test_dcen3x3 test_resort-xylist test_simplexy

ALL_TEST_EXTRA_OBJS :=
ALL_TEST_LIBS :=

# Add the dependencies here...
TEST_DFIND_OBJS := dfind.o
ALL_TEST_EXTRA_OBJS += $(TEST_DFIND_OBJS)
test_dfind: $(TEST_DFIND_OBJS) $(ANUTILS_LIB)

TEST_CTMF_OBJS := ctmf.o
ALL_TEST_EXTRA_OBJS += $(TEST_CTMF_OBJS)
test_ctmf: $(TEST_CTMF_OBJS)

TEST_DSMOOTH_OBJS := dsmooth.o
ALL_TEST_EXTRA_OBJS += $(TEST_DSMOOTH_OBJS)
test_dsmooth: $(TEST_DSMOOTH_OBJS)

test_2mass: 2mass.o 2mass-fits.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += 2mass-fits.o 2mass.o

test_usnob: usnob-fits.o usnob.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += usnob-fits.o usnob.o

test_nomad: nomad-fits.o nomad.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += nomad-fits.o nomad.o

test_matchfile: matchfile.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += matchfile.o

test_blindutils: blindutils.o $(ANUTILS_LIB)
ALL_TEST_EXTRA_OBJS += blindutils.o

test_dcen3x3: dcen3x3.o
ALL_TEST_EXTRA_OBJS += dcen3x3.o

test_simplexy: $(SIMPLEXY_OBJ) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += $(SIMPLEXY_OBJ)

test_resort-xylist: resort-xylist.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += resort-xylist.o

ALL_TEST_LIBS += $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

include $(COMMON)/makefile.tests

ALL_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)
ALL_OBJ += $(ALL_TEST_EXTRA_OBJS) test.o

GENERATED_FILES += $(ALL_TEST_FILES_MAIN_C)

ALL_EXECS += $(ALL_TEST_FILES) test

# END OF KEIR'S SUPER SIMPLE TESTING FRAMEWORK
######################################################

demo_dsmooth: demo_dsmooth.o dsmooth.o \
	$(COMMON)/cairoutils.o $(ANUTILS_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) -pg

control-program: control-program.o \
		$(BACKEND_OBJS) $(SIMPLEXY_OBJ) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
ALL_OBJ += control-program.o

# qfits is only needed for writing debugging images...
image2xy: image2xy-main.o image2xy-files.o $(SIMPLEXY_OBJ) $(CFITS_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += image2xy-main.o image2xy-files.o

new-wcs: new-wcs-main.o new-wcs.o $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += new-wcs-main.o

rdls2hpls: rdls2hpls.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

rdlstohealpix: rdlstohealpix.o \
		$(ANUTILS_LIB) $(QFITS_LIB)

xylist2fits: xylist2fits.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

certifiable: certifiable.o histogram.o matchfile.o solvedfile.o \
		$(UTILS_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

fits-guess-scale: fits-guess-scale-main.o fits-guess-scale.o \
		$(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += fits-guess-scale-main.o

get-wcs: get-wcs.o \
		$(ANUTILS_LIB) $(QFITS_LIB)

wcs-xy2rd: wcs-xy2rd-main.o wcs-xy2rd.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += wcs-xy2rd-main.o

wcs-rd2xy: wcs-rd2xy-main.o wcs-rd2xy.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += wcs-rd2xy-main.o

wcs-pv2sip: wcs-pv2sip.o tweak.o \
		$(ANLIBS) $(GSL_LIB)
ALL_OBJ += wcs-pv2sip.o

noisesim: noisesim.o noise.o $(ANUTILS_LIB)

noisesim2: noisesim2.o noise.o $(ANUTILS_LIB)

hpgrid: hpgrid.o $(ANUTILS_LIB)

get-healpix: get-healpix.o $(ANUTILS_LIB)

$(CFITS_UTILS) :: %: %.o $(CFITS_LIB)

fitsverify: ftverify.c fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c $(CFITS_LIB)
	$(CC) -DSTANDALONE -trigraphs $(CFITS_INC) -o $@ $^ -lm

$(OLDEXECS) :: %: %.o $(OLDEXECS_OBJS) \
		$(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

catalog_analysis: catalog_analysis.o pnpoly.o $(ANUTILS_LIB)

nomadtofits: nomadtofits.o nomad-fits.o nomad.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

read_nomad: read_nomad.o nomad.o

2masstofits: $(2MASSOBJ) $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lz

diffractionFlag_check: diffractionFlag_check.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

project-usnob: project-usnob.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

usnobtofits: usnobtofits.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

hpowned: hpowned.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

build-an-catalog: build-an-catalog.o usnob-fits.o usnob.o \
		2mass-fits.o 2mass.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

cut-an: cut-an.o catalog.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

startree: startree.o catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

startree2: startree2-main.o startree2.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += startree2-main.o startree2.o

hpquads: hpquads.o quad-builder.o quad-utils.o codefile.o pnpoly.o histogram.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

allquads: allquads-main.o allquads.o quad-builder.o quad-utils.o codefile.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

codetree: codetree.o codefile.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

unpermute-stars: unpermute-stars-main.o unpermute-stars.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += unpermute-stars-main.o

unpermute-quads: unpermute-quads-main.o unpermute-quads.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += unpermute-quads-main.o

merge-index: merge-index-main.o merge-index.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += merge-index-main.o

local-index: local-index.o merge-index.o unpermute-quads.o unpermute-stars.o \
		allquads.o quad-builder.o quad-utils.o codefile.o wcs-xy2rd.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += local-index.o

index-to-table: index-to-table.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

endian-flip-index: endian-flip-index.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

threadtest: threadtest.o $(BACKEND_LIB) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)

backend: backend-main.o $(BACKEND_LIB) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)

verify: verify-main.o verify.o verify2.o matchfile.o $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)


ALL_OBJ += verify-main.o

solve-field: solve-field.o augment-xylist.o blindutils.o matchfile.o \
		image2xy-files.o resort-xylist.o tabsort.o \
		fits-guess-scale.o wcs-rd2xy.o new-wcs.o \
		scamp.o \
		$(SIMPLEXY_OBJ) $(CFITS_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += scamp.o

augment-xylist: augment-xylist-main.o augment-xylist.o tabsort.o fits-guess-scale.o \
		image2xy-files.o resort-xylist.o blindutils.o \
		$(SIMPLEXY_OBJ) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(CFITS_LIB)

ALL_OBJ += augment-xylist-main.o

blind: blind-main.o blind.o solver.o matchfile.o \
		solvedclient.o solvedfile.o verify.o \
		intmap.o blind_wcs.o histogram.o tweak.o \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@.tmp $^
	mv $@.tmp $@

whynot: whynot.o matchfile.o codefile.o \
		solvedclient.o solvedfile.o \
		verify.o intmap.o blind_wcs.o \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)

quadcenters: quadcenters.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

startree2rdls: startree2rdls.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

agreeable: agreeable.o matchfile.o solvedfile.o solvedclient.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

rdlsinfo: rdlsinfo.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

xylsinfo: xylsinfo.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

wcsinfo: wcsinfo.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) \
		$(QFITS_LIB)

wcs-grab: wcs-grab.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) \
		$(QFITS_LIB)

printsolved: printsolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB) 

setsolved: setsolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB) 

mergesolved: mergesolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

solvedserver: solvedserver.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) \
		$(QFITS_LIB)
	$(CC) $(CFLAGS) -o $@ $^ -lm

plotcat: plotcat.o 2mass-fits.o 2mass.o nomad-fits.o nomad.o catalog.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

plotquad.o: plotquad.c 
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c $< -o $@

$(COMMON)/cairoutils.o:
	$(MAKE) -C $(COMMON) cairoutils.o

onepixel: onepixel.o $(COMMON)/cairoutils.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

plotquad: plotquad.o matchfile.o $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

plot-xy-and-quad: plot-xy-and-quad.o matchfile.o $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

plot-constellations.o: plot-constellations.c
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c -o $@ $<

plot-constellations: plot-constellations.o $(COMMON)/cairoutils.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(CAIRO_LIB) $(NETPBM_LIB) $(PNG_LIB) $(JPEG_LIB)

catalog-dists: catalog-dists.o 2mass-fits.o usnob-fits.o usnob.o \
		$(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

add-text: add-text.o $(COMMON)/cairoutils.o $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(CAIRO_LIB) $(NETPBM_LIB) $(PNG_LIB) $(JPEG_LIB)

plotstuff.o: plotstuff.c
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c -o $@ $<

PLOTSTUFF := plotstuff.o plotxy.o plotimage.o plotannotations.o
ALL_OBJ += $(PLOTSTUFF)

plotstuff: plotstuff-main.o $(PLOTSTUFF) $(COMMON)/cairoutils.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)
ALL_OBJ += plotstuff-main.o

plotxy.o: plotxy.c
	$(CC) $(CFLAGS) $(CAIRO_INC) -c -o $@ $<

plotxy: plotxy-main.o $(PLOTSTUFF) $(COMMON)/cairoutils.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)
ALL_OBJ += plotxy-main.o

quadidx: quadidx.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

codeprojections: codeprojections.o $(UTIL_OBJS) $(ANLIBS)

quadscales: quadscales.o histogram.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

quadsperstar: quadsperstar.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) \
		$(QFITS_LIB)

fitsgetext: fitsgetext.o $(ANUTILS_LIB) $(QFITS_LIB)

fits-dedup: fits-dedup.o $(ANUTILS_LIB) $(QFITS_LIB)

wcs-resample: wcs-resample.o $(ANUTILS_LIB) $(QFITS_LIB)

fits-flip-endian: fits-flip-endian.o $(ANUTILS_LIB) $(QFITS_LIB)

subtable: subtable.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

tabsort: tabsort-main.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += tabsort-main.o

column-merge: column-merge.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

add-healpix-column: add-healpix-column.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

resort-xylist: resort-xylist-main.o resort-xylist.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += resort-xylist-main.o

bgsubtract: bgsubtract.o ctmf.o $(COMMON)/cairoutils.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

usnob-scamp-catalog: usnob-scamp-catalog.o usnob-fits.o usnob.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

create-scamp-catalog: create-scamp-catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

DEP_OBJ := $(ALL_OBJ)
DEP_PREREQS := $(QFITS_LIB)

CFLAGS += $(CFITSIO_CFLAGS)
CFLAGS += $(CAIRO_INC)
include $(COMMON)/makefile.deps

.PHONY: allclean realclean clean

allclean: clean
	rm -f *.o *~ *.dep 

realclean: allclean

clean:
	rm -f $(EXECS) $(EXTRA_EXECS) $(SOLVER_EXECS) $(MISC_EXECS) $(PROGS) \
		$(PIPELINE) $(PROSPECTUS) $(DEPS) deps $(FITS_UTILS) $(ALL_OBJ) $(IMAGE2XY) \
		$(NODEP_OBJS) plot-constellations fitsverify plotquad plotxy \
		$(ALL_EXECS) $(GENERATED_FILES) $(ALL_TESTS_CLEAN) \
		$(BACKEND_LIB) $(BACKEND_SO)



