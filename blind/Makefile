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

$(DATA)/%.txt: ;
$(DATA)/%.dat: ;

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

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(QFITS_INC)
CFLAGS += $(CFITS_INC)
CFLAGS += $(LIBKD_INC)
CFLAGS += $(ANUTILS_INC)
CFLAGS += $(ANFILES_INC)
CFLAGS += $(GSL_INC)

# For Keir's fancy Python bindings:
CFLAGS += -fPIC

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

QFITS_UTILS := fitsgetext subtable tabsort column-merge add-healpix-column \
		resort-xylist

CFITS_UTILS := tablist modhead fitscopy tabmerge fitstomatlab liststruc \
		listhead imcopy imarith imstat

FITS_UTILS := $(QFITS_UTILS) $(CFITS_UTILS)

MISC_EXECS := project-usnob diffractionFlag_check \
	hpgrid read_nomad tycho2tostellarium

UTIL_OBJS := fileutil.o boilerplate.o

OTHER_OBJS := usnob.o usnob-fits.o tycho2.o tycho2-fits.o catalog.o \
	codefile.o histogram.o tic.o codekd.o svd.o verify.o \
	solver.o matchfile.o solvedclient.o solvedfile.o intmap.o pnpoly.o \
	blind_wcs.o index.o tweak.o log.o blind-main.o \
	plot-constellations.o quadcenters.o startree2rdls.o \
	brightstars.o ngcic-accurate.o nomad.o nomad-fits.o \
	ngc2000.o noisesim.o noise.o scriptutils.o hd.o

2MASSOBJ := 2masstofits.o 2mass-fits.o 2mass.o

PIPELINE := usnobtofits tycho2tofits build-an-catalog hpowned cut-an \
	startree hpquads codetree unpermute-quads unpermute-stars blind \
	solvedserver printsolved xylist2fits mergesolved rdlsinfo xylsinfo \
	wcsinfo fits-guess-scale agreeable new-wcs get-wcs \
	backend augment-xylist solve-field plotcat wcs-xy2rd wcs-rd2xy \
	certifiable wcs-annotate get-healpix 2masstofits nomadtofits

PROSPECTUS := quadidx codeprojections quadlocations quadscales quadsperstar \
	whynot quadcenters startree2rdls

# old and miscellaneous executables that aren't part of the pipeline.
OLDEXECS := plotquads rawstartree checkquads

CAIROEXECS := plotquad plotxy plot-constellations add-text

OLDEXECS_OBJS := catalog.o histogram.o verify.o intmap.o tic.o matchfile.o \
	codekd.o $(UTIL_OBJS)

PIPELINE_MAIN_OBJ := $(addsuffix .o,$(PIPELINE))
PROSPECTUS_MAIN_OBJ := $(addsuffix .o,$(PROSPECTUS))
FITS_UTILS_MAIN_OBJ := $(addsuffix .o,$(FITS_UTILS))

SIMPLEXY_OBJ := dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o dobjects.o dpeaks.o \
	dselip.o dsigma.o dsmooth.o image2xy.o simplexy.o radix.o \
	ctmf.o

SIMPLEXY2_OBJ := dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o dobjects.o dpeaks.o \
	dselip.o dsigma.o dsmooth.o image2xy.o simplexy2.o radix.o \
	ctmf.o

SIMPLEXY_SHARED_OBJ := simplexy.o dsigma.o dmedsmooth.o dallpeaks.o dpeaks.o \
	dobjects.o dselip.o dsmooth.o dcen3x3.o dfind.o radix.o

FIXAXIS_OBJ := fixaxis.o

TEST_DFIND_OBJ := dfind.o test_dfind.o

IMAGE2XY := image2xy fixaxis test_dfind

INSTALL_EXECS := $(FITS_UTILS) $(PIPELINE) image2xy
INSTALL_CAIRO_EXECS := $(CAIROEXECS)

ALL_OBJ := $(UTIL_OBJS) $(KDTREE_OBJS) $(QFITS_OBJ) \
	$(PIPELINE_MAIN_OBJ) $(PROSPECTUS_MAIN_OBJ) $(FITS_UTILS_MAIN_OBJ) \
	$(OTHER_OBJS) $(2MASSOBJ) $(SIMPLEXY_OBJ) $(SIMPLEXY_SHARED_OBJ) $(FIXAXIS_OBJ) \
	$(TEST_DFIND_OBJ) $(SIMPLEXY2_OBJ)
ALL_EXECS :=

all: $(QFITS_LIB) $(CFITS_LIB) $(LIBKD_LIB) \
	$(REMAKE_LIBKD) $(REMAKE_ANUTILS) $(REMAKE_ANFILES) \
	$(PIPELINE) $(PROSPECTUS) $(IMAGE2XY) $(FITS_UTILS) fitsverify

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

test-solver: test-solver.o solver_test.o blind_wcs.o verify.o log.o \
		matchfile.o tic.o svd.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

NODEP_OBJS += solver_test.o
ALL_OBJ += test-solver.o

solver_test.o: solver.c
	$(CC) $(CFLAGS) -DTESTING=1 -o $@ -c $<

#######################################

# Add the basename of your test sources here...
ALL_TEST_FILES = test_dfind test_ctmf test_simplexy test_dsmooth test_2mass \
	test_tycho2 test_usnob test_nomad test_matchfile test_hd test_scriptutils \
	test_dcen3x3

ALL_TEST_EXTRA_OBJS :=
ALL_TEST_LIBS :=

# Add the dependencies here...
TEST_DFIND_OBJS := dfind.o log.o
ALL_TEST_EXTRA_OBJS += $(TEST_DFIND_OBJS)
test_dfind: $(TEST_DFIND_OBJS) $(ANUTILS_LIB)

TEST_CTMF_OBJS := ctmf.o
ALL_TEST_EXTRA_OBJS += $(TEST_CTMF_OBJS)
test_ctmf: $(TEST_CTMF_OBJS)

TEST_SIMPLEXY_OBJS := simplexy.o ctmf.o dsigma.o dobjects.o dselip.o \
		dallpeaks.o radix.o dsmooth.o dfind.o \
		dmedsmooth.o dpeaks.o dcen3x3.o
ALL_TEST_EXTRA_OBJS += $(TEST_SIMPLEXY_OBJS)
test_simplexy: $(TEST_SIMPLEXY_OBJS) log.o $(ANUTILS_LIB)

TEST_DSMOOTH_OBJS := dsmooth.o
ALL_TEST_EXTRA_OBJS += $(TEST_DSMOOTH_OBJS)
test_dsmooth: $(TEST_DSMOOTH_OBJS)

test_2mass: 2mass.o 2mass-fits.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += 2mass-fits.o 2mass.o

test_tycho2: tycho2-fits.o tycho2.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += tycho2-fits.o tycho2.o

test_usnob: usnob-fits.o usnob.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += usnob-fits.o usnob.o

test_nomad: nomad-fits.o nomad.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += nomad-fits.o nomad.o

test_matchfile: matchfile.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += matchfile.o

test_hd: hd.o $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_TEST_EXTRA_OBJS += hd.o

test_scriptutils: scriptutils.o log.o $(ANUTILS_LIB)
ALL_TEST_EXTRA_OBJS += scriptutils.o log.o

test_dcen3x3: dcen3x3.o
ALL_TEST_EXTRA_OBJS += dcen3x3.o

ALL_TEST_LIBS += $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

include $(COMMON)/makefile.tests

ALL_OBJ += $(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O)
ALL_OBJ += $(ALL_TEST_EXTRA_OBJS) test.o

GENERATED_FILES += $(ALL_TEST_FILES_MAIN_C)

ALL_EXECS += $(ALL_TEST_FILES) test

# END OF KEIR'S SUPER SIMPLE TESTING FRAMEWORK
######################################################

# This is Jon ignoring the awesome testing framework
demo_simplexy: demo_simplexy.o dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o dobjects.o dpeaks.o ctmf.o\
	dselip.o dsigma.o dsmooth.o simplexy.o radix.o \
	$(COMMON)/cairoutils.o boilerplate.o $(ANUTILS_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) -pg

demo_dsmooth: demo_dsmooth.o dsmooth.o \
	$(COMMON)/cairoutils.o boilerplate.o $(ANUTILS_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) -pg

#ctmf.o: ctmf.c

image2xy: image2xy-main.o log.o $(SIMPLEXY2_OBJ) $(CFITS_LIB) $(ANUTILS_LIB)

ALL_OBJ += image2xy-main.o

PNG2XY_OBJ := dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o dobjects.o dpeaks.o \
	dselip.o dsigma.o dsmooth.o png2xy.o simplexy.o radix.o \
	ctmf.o $(COMMON)/cairoutils.o
png2xy: $(PNG2XY_OBJ) $(CFITS_LIB) $(ANUTILS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB) -pg

fixaxis: $(FIXAXIS_OBJ) $(ANUTILS_LIB) $(CFITS_LIB)

simplexy_shared: _simplexy.so
.PHONY: simplexy_shared

_simplexy.so: $(SIMPLEXY_SHARED_OBJ)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

new-wcs: new-wcs-main.o new-wcs.o log.o $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += new-wcs-main.o

rdls2hpls: rdls2hpls.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

rdlstohealpix: rdlstohealpix.o \
		$(ANUTILS_LIB) $(QFITS_LIB)

xylist2fits: xylist2fits.o boilerplate.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

ngc2000.o: ngc2000.c ngc2000names.c ngc2000entries.c

ngc2000names.c: parse-ngc2000names.awk $(DATA)/ngc2000names.dat
	$(AWK) -f parse-ngc2000names.awk < $(DATA)/ngc2000names.dat  > $@

ngc2000entries.c: parse-ngc2000.awk $(DATA)/ngc2000.dat
	$(AWK) -f parse-ngc2000.awk < $(DATA)/ngc2000.dat  > $@

ngcic-accurate-entries.c: parse-ngcic-accurate.awk $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt
	cat $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt | $(AWK) -f parse-ngcic-accurate.awk > $@

certifiable: certifiable.o histogram.o matchfile.o solvedfile.o \
		$(UTILS_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

fits-guess-scale: fits-guess-scale-main.o fits-guess-scale.o log.o \
		$(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += fits-guess-scale-main.o

get-wcs: get-wcs.o \
		$(ANUTILS_LIB) $(QFITS_LIB)

wcs-annotate: wcs-annotate.o ngc2000.o \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

wcs-xy2rd: wcs-xy2rd.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

wcs-rd2xy: wcs-rd2xy-main.o wcs-rd2xy.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += wcs-rd2xy-main.o

noisesim: noisesim.o noise.o $(ANUTILS_LIB)

noisesim2: noisesim2.o noise.o svd.o $(ANUTILS_LIB)

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

tycho2tofits: tycho2tofits.o tycho2-fits.o tycho2.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

hpowned: hpowned.o boilerplate.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

build-an-catalog: build-an-catalog.o usnob-fits.o tycho2-fits.o \
		tycho2.o usnob.o 2mass-fits.o 2mass.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

cut-an: cut-an.o catalog.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

build-hd-tree: build-hd-tree.o tycho2-fits.o tycho2.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

startree: startree.o catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

hpquads: hpquads.o codefile.o pnpoly.o histogram.o tic.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

codetree: codetree.o codekd.o codefile.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

unpermute-stars: unpermute-stars.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

unpermute-quads: unpermute-quads.o codekd.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

backend-old: backend-old.o scriptutils.o fileutil.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

backend: backend.o scriptutils.o blind.o solver.o codekd.o \
		matchfile.o tic.o solvedclient.o solvedfile.o verify.o \
		svd.o intmap.o blind_wcs.o index.o log.o histogram.o tweak.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB) $(GSL_LIB)

solve-field: solve-field.o augment-xylist.o scriptutils.o matchfile.o \
		fileutil.o image2xy.o resort-xylist.o tabsort.o \
		fits-guess-scale.o wcs-rd2xy.o new-wcs.o log.o \
		$(SIMPLEXY2_OBJ) $(CFITS_LIB) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

augment-xylist: augment-xylist-main.o augment-xylist.o tabsort.o fits-guess-scale.o image2xy.o \
		resort-xylist.o \
		scriptutils.o fileutil.o log.o $(SIMPLEXY2_OBJ) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(CFITS_LIB)

ALL_OBJ += augment-xylist-main.o

blind: blind-main.o blind.o solver.o codekd.o matchfile.o tic.o \
		solvedclient.o solvedfile.o verify.o svd.o \
		intmap.o blind_wcs.o index.o log.o histogram.o tweak.o \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@.tmp $^
	mv $@.tmp $@

whynot: whynot.o index.o codekd.o matchfile.o tic.o \
		solvedclient.o solvedfile.o \
		verify.o svd.o intmap.o blind_wcs.o log.o \
		$(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

quadcenters: quadcenters.o fileutil.o boilerplate.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

startree2rdls: startree2rdls.o fileutil.o boilerplate.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB)

agreeable: agreeable.o matchfile.o solvedfile.o solvedclient.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

rdlsinfo: rdlsinfo.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

xylsinfo: xylsinfo.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

wcsinfo: wcsinfo.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) \
		$(QFITS_LIB)

printsolved: printsolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB) 

mergesolved: mergesolved.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

solvedserver: solvedserver.o solvedfile.o $(UTIL_OBJS) $(ANUTILS_LIB) \
		$(QFITS_LIB)
	$(CC) $(CFLAGS) -o $@ $^ -lm

plotcat: plotcat.o catalog.o tycho2-fits.o tycho2.o usnob-fits.o usnob.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

plotquad.o: plotquad.c 
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c $< -o $@

$(COMMON)/cairoutils.o:
	$(MAKE) -C $(COMMON) cairoutils.o

onepixel: onepixel.o $(COMMON)/cairoutils.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

plotquad: plotquad.o matchfile.o $(COMMON)/cairoutils.o boilerplate.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

plot-constellations.o: plot-constellations.c
	$(CC) $(CFLAGS) $(CAIRO_INC) $(NETPBM_INC) -c -o $@ $<

plot-constellations: plot-constellations.o $(COMMON)/cairoutils.o \
		ngc2000.o ngcic-accurate.o brightstars.o boilerplate.o \
		hd.o \
		$(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(CAIRO_LIB) $(NETPBM_LIB) $(PNG_LIB) $(JPEG_LIB)

add-text: add-text.o $(COMMON)/cairoutils.o $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(CAIRO_LIB) $(NETPBM_LIB) $(PNG_LIB) $(JPEG_LIB)

plotxy.o: plotxy.c
	$(CC) $(CFLAGS) $(CAIRO_INC) -c -o $@ $<

plotxy: plotxy.o $(COMMON)/cairoutils.o boilerplate.o $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

quadidx: quadidx.o $(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

codeprojections: codeprojections.o codekd.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

quadlocations: quadlocations.o catalog.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

quadscales: quadscales.o histogram.o \
		$(UTIL_OBJS) $(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) \
		$(QFITS_LIB)

quadsperstar: quadsperstar.o \
		$(UTIL_OBJS) $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) \
		$(QFITS_LIB)

fitsgetext: fitsgetext.o $(ANUTILS_LIB) $(QFITS_LIB)

subtable: subtable.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

tabsort: tabsort-main.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += tabsort-main.o

column-merge: column-merge.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

add-healpix-column: add-healpix-column.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)

resort-xylist: resort-xylist-main.o resort-xylist.o fileutil.o $(UTIL_OBJS) \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += resort-xylist-main.o

bgsubtract: bgsubtract.o ctmf.o $(COMMON)/cairoutils.o $(UTIL_OBJS) $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(CAIRO_LIB) $(PNG_LIB) $(NETPBM_LIB) $(JPEG_LIB)

tycho2tostellarium: tycho2tostellarium.o tycho2-fits.o $(UTIL_OBJS) $(ANUTILS_LIB) \
		$(QFITS_LIB)

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
	rm -f $(EXECS) $(EXTRA_EXECS) $(SOLVER_EXECS) $(MISC_EXECS) \
		$(PIPELINE) $(PROSPECTUS) $(DEPS) deps $(FITS_UTILS) $(ALL_OBJ) $(IMAGE2XY) \
		$(NODEP_OBJS) plot-constellations fitsverify plotquad plotxy \
		$(ALL_EXECS) $(GENERATED_FILES)



