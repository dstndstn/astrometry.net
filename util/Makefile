# This file is part of the Astrometry.net suite.
# Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
# Copyright 2008, 2009, 2010, 2011, 2012 Dustin Lang.
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
COMMON := .

all:
.PHONY: all

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $<

# You can build stripped-down versions of the AN libraries that do not
# depend on libkd or libqfits by defining the following in your Makefile
# before calling "make" in the "util" directory:
#   NO_KDTREE := 1
#   export NO_KDTREE
#   NO_QFITS := 1
#   export NO_QFITS

include $(COMMON)/makefile.common

ANUTILS_LIB := libanutils.a
ANUTILS_OBJ := starutil.o mathutil.o bl.o bt.o healpix-utils.o healpix.o \
	permutedsort.o ioutils.o sip-utils.o fit-wcs.o sip.o md5.o \
	os-features.o an-endian.o errors.o an-opts.o svn.o tic.o log.o datalog.o \
	gslutils.o \
	boilerplate.o tycho2.o intmap.o histogram.o histogram2d.o \
	anwcs.o sparsematrix.o lsqr.o coadd.o convolve-image.o resample.o wcs-resample.o

ANFILES_LIB := libanfiles.a
# Things that it depends on but that aren't linked in
ANFILES_DEPS :=
ANFILES_OBJ := 

ifndef NO_QFITS
ANFILES_OBJ += multiindex.o index.o codekd.o starkd.o rdlist.o xylist.o \
	starxy.o an-catalog.o qidxfile.o quadfile.o scamp-catalog.o \
	hd.o
ANFILES_DEPS += $(QFITS_LIB)
endif

ifndef NO_QFITS
include $(COMMON)/makefile.qfits
endif
ifndef NO_KDTREE
include $(COMMON)/makefile.libkd
endif

include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg
include $(COMMON)/makefile.png
include $(COMMON)/makefile.gsl
include $(COMMON)/makefile.wcslib

include $(COMMON)/makefile.netpbm
# D'oh
ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),reconfig)
    include $(COMMON)/makefile.os-features
  endif
endif

NETPBM_INC_ORIG := $(NETPBM_INC)
NETPBM_LIB_ORIG := $(NETPBM_LIB)

ifeq ($(HAVE_NETPBM),yes)
else
NETPBM_INC :=
NETPBM_LIB :=
endif

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(CAIRO_INC)
CFLAGS += $(JPEG_INC)
CFLAGS += $(PNG_INC)
CFLAGS += $(GSL_INC)
CFLAGS += $(WCSLIB_INC)
CFLAGS += $(NETPBM_INC)

ifdef WCSLIB_EXISTS
  ifdef NOWCSLIB
  else
    CFLAGS += -DWCSLIB_EXISTS
  endif
endif

####
# WCSTOOLS_EXISTS := 1

ifdef WCSTOOLS_EXISTS
  CFLAGS += -DWCSTOOLS_EXISTS
  WCSTOOLS_LIB ?= libwcs.a
endif

WCS_LIB := $(WCSTOOLS_LIB) $(WCSLIB_LIB)
WCS_LIB_DEP :=

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += -lm

ifndef NO_QFITS
CFLAGS += $(QFITS_INC)
endif
ifndef NO_KDTREE
CFLAGS += $(LIBKD_INC)
endif
CFLAGS += -I.

PYTHON_LIBS := _sip.so _healpix.so

PROGS := an-fitstopnm an-pnmtofits downsample-fits pad-file tycho2tofits fits-flip-endian smooth-image search-index hpsplit wcs-to-tan get-healpix add-healpix-column
MAIN_PROGS := wcs-resample fit-wcs

# tycho2tostellarium build-hd-tree

all: $(ANUTILS_LIB) $(ANFILES_LIB) $(PYTHON_LIBS) $(PROGS) $(MAIN_PROGS)

# Things that it depends on but that aren't linked in
ANUTILS_DEPS :=

# ANUTILS_OBJ -- in makefile.utils

# Actually there are ANFILES_H mixed in here too....
ANUTILS_H := an-bool.h an-catalog.h  an-endian.h an-opts.h an-thread-pthreads.h \
	an-thread.h anwcs.h bl.h bl.inc bl.ph bl-nl.h bl-nl.inc bl-nl.ph  bt.h cairoutils.h \
	codekd.h \
	errors.h fitsbin.h fitsfile.h fitsioutils.h \
	fitstable.h os-features-config.h os-features.h gslutils.h hd.h \
	healpix-utils.h healpix.h index.h intmap.h ioutils.h keywords.h log.h \
	mathutil.h permutedsort.h qidxfile.h \
	quadfile.h rdlist.h scamp-catalog.h fit-wcs.h sip-utils.h sip.h sip_qfits.h \
	starkd.h starutil.h starutil.inc starxy.h svn.h tic.h tycho2-fits.h tycho2.h \
	xylist.h coadd.h convolve-image.h resample.h multiindex.h

ifndef NO_QFITS
ANUTILS_OBJ += fitsioutils.o sip_qfits.o fitstable.o fitsbin.o fitsfile.o \
	tycho2-fits.o tic.o
ANUTILS_DEPS += $(QFITS_LIB)
endif

## ANFILES_DEPS, ANFILES_OBJ -- in makefile.anfiles

ALL_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)

# svn.c depends on all the .c and .h files in ANUTILS_OBJ
SVN_C_DEPS := $(subst svn.o,,$(ANUTILS_OBJ))
SVN_C_DEPS := $(subst .o,.c,$(SVN_C_DEPS)) $(subst .o,.h,$(SVN_C_DEPS))

DEP_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)
DEP_PREREQS :=

$(ANUTILS_LIB): $(ANUTILS_OBJ) $(ANUTILS_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(ANUTILS_OBJ)
	$(RANLIB) $@

$(ANFILES_LIB): $(ANFILES_OBJ) $(ANFILES_DEPS)
	-rm -f $@
	$(AR) rc $@ $(ANFILES_OBJ)
	$(RANLIB) $@

svn.c: $(SVN_C_DEPS)
	echo "//" >> svn.c

resample: resample-main.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(WCS_LIB_DEP)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCS_LIB) $(LDLIBS)
ALL_OBJ += resampe.o

tycho2tofits: tycho2tofits.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += tycho2tofits.o

tycho2tostellarium: tycho2tostellarium.o $(ANUTILS_LIB) $(QFITS_LIB)

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

_sip.so: sip-utils.o gslutils.o sip.o starutil.o mathutil.o sip_qfits.o \
		fitsioutils.o errors.o ioutils.o bl.o os-features.o an-endian.o \
		log.o tic.o $(QFITS_LIB) $(GSL_LIB)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

_healpix.so: healpix-utils.o healpix.o permutedsort.o starutil.o mathutil.o errors.o ioutils.o bl.o log.o tic.o os-features.o
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

util.py util_wrap.c: util.i $(ANUTILS_H)
	swig -python -I. $<

_util.so: util_wrap.c setup-util.py \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)
	WCSLIB_LIB="$(WCSLIB_LIB)" GSL_LIB="$(GSL_LIB)" \
	python setup-util.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_util.so $@

PYUTIL := _util.so util.py
pyutil: $(PYUTIL)
.PHONY: pyutil
py: $(PYUTIL)
.PHONY: py

index_wrap.c: index.i index.h
	swig -python -I. $<

_index_util.so _index_c.so: index_wrap.c index_pyutils.c setup-index.py \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
	python setup-index.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_index_c.so _index_c.so
	cp build/lib/_index_util.so _index_util.so

coadd: coadd-main.o $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB) $(WCS_LIB_DEP)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCS_LIB) $(LDLIBS)
ALL_OBJ += coadd.o coadd-main.o

wcs-resample: wcs-resample-main.o $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) $(WCS_LIB_DEP)
	$(CC) -o $@ $^ $(LDFLAGS) $(WCS_LIB) $(LDLIBS)
ALL_OBJ += wcs-resample-main.o

fit-wcs: fit-wcs-main.o $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) $(WCS_LIB_DEP)
	$(CC) -o $@ $^ $(LDFLAGS) $(WCS_LIB) $(LDLIBS)
ALL_OBJ += fit-wcs-main.o

get-healpix: get-healpix.o $(ANUTILS_LIB) $(GSL_LIB)
ALL_OBJ += get-healpix.o

wcs-to-tan: wcs-to-tan.o $(ANUTILS_LIB) $(GSL_LIB) $(QFITS_LIB) $(WCS_LIB_DEP)
	$(CC) -o $@ $^ $(LDFLAGS) $(WCS_LIB) $(LDLIBS)
ALL_OBJ += wcs-to-tan.o

pad-file: pad-file.o $(ANUTILS_LIB)
ALL_OBJ += pad-file.o

an-fitstopnm: an-fitstopnm.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += an-fitstopnm.o

add-healpix-column: add-healpix-column.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += add-healpix-column.o

an-pnmtofits: an-pnmtofits.o $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(NETPBM_LIB) $(LDLIBS)
ALL_OBJ += an-pnmtofits.o

downsample-fits: downsample-fits.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += downsample-fits.o

smooth-image: smooth-image.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += smooth-image.o

fits-flip-endian: fits-flip-endian.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += fits-flip-endian.o

search-index: search-index.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += search-index.o

hpsplit: hpsplit.o \
		$(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += hpsplit.o

PYTHON_INSTALL := fits2fits.py image2pnm.py addpath.py healpix.py sip.py \
	starutil.py starutil_numpy.py \
	shell.py __init__.py file.py run_command.py w3c_validator.py \
	filetype.py pyfits_utils.py fix_sdss_idr.py removelines.py \
	uniformize.py lsstutils.py casjobs.py \
	sdss_das.py sdss_filenames.py sdss_get_catalog.py sdss_noise.py \
	sdss_psf.py sdss_psfield.py sdss_radec_to_rcf.py sqlcl.py \
	usnob_catalog.py usnob_get_image.py usnob_get_region.py \
	find_data_file.py plotshift.py plotutils.py miscutils.py multiproc.py yanny.py \
	$(PYTHON_LIBS)

PYTHON_EXECS := fits2fits.py image2pnm.py removelines.py uniformize.py

PY_INSTALL_DIR := $(INSTALL_DIR)/python/astrometry/util
BIN_INSTALL_DIR := $(INSTALL_DIR)/bin
INCLUDE_INSTALL_DIR := $(INSTALL_DIR)/include
LIB_INSTALL_DIR := $(INSTALL_DIR)/lib
LINK_DIR := ../python/astrometry/util

install: $(PYTHON_LIBS) $(PYTHON_EXECS) $(PROGS) $(MAIN_PROGS) $(ANUTILS_H) $(ANFILES_LIB) $(ANUTILS_LIB)
	@echo Installing in base directory '$(INSTALL_DIR)'
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
	@echo Installing executables into '$(BIN_INSTALL_DIR)'...
	@for x in $(PROGS) $(MAIN_PROGS); do \
		echo cp $$x '$(BIN_INSTALL_DIR)/'$$x; \
		cp $$x '$(BIN_INSTALL_DIR)/'$$x; \
	done
	@for x in $(ANUTILS_H); do \
		echo cp $$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
		cp $$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
	done
	@for x in $(ANFILES_LIB) $(ANUTILS_LIB); do \
		echo cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
		cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
	done
	@echo "The following copy commands may fail; they are optional."
	-cp _util.so '$(PY_INSTALL_DIR)'
	-cp util.py '$(PY_INSTALL_DIR)'
	@echo ok

.PHONY: install

os-features-config.h: os-features-test.c
	@echo
	@echo "---- Error messages in the next few commands are not necessarily bugs ----"
	@echo "     (we're checking how things works on your computer)"
	-rm -f os-features.log
	-rm -f $@ $@.tmp
	@echo
	@echo "Testing canonicalize_file_name..."
	@echo "Testing canonicalize_file_name..." >> os-features.log
	(($(CC) -o os-features-test-canonicalize \
	  $(CFLAGS) -DTEST_CANONICALIZE_FILE_NAME $^ $(LDFLAGS) >> os-features.log && \
	  ./os-features-test-canonicalize >> os-features.log && \
	  echo "#define NEED_CANONICALIZE_FILE_NAME 0") \
	|| echo "#define NEED_CANONICALIZE_FILE_NAME 1") >> $@.tmp
	@echo
	@echo "Testing declare qsort_r..."
	@echo "Testing declare qsort_r..." >> os-features.log
	(($(CC) -o os-features-test-declare-qsort \
	   $(CFLAGS) -DTEST_DECLARE_QSORT_R $^ $(LDFLAGS) >> os-features.log && \
	   ./os-features-test-declare-qsort >> os-features.log && \
	   echo "#define NEED_DECLARE_QSORT_R 1") \
	|| echo "#define NEED_DECLARE_QSORT_R 0") >> $@.tmp
	@echo
	@echo "Testing qsort_r..."
	@echo "Testing qsort_r..." >> os-features.log
	(($(CC) -o os-features-test-qsort \
	   $(CFLAGS) -DTEST_QSORT_R $^ $(LDFLAGS) >> os-features.log && \
	   ./os-features-test-qsort >> os-features.log && \
	   echo "#define NEED_QSORT_R 0") \
	|| echo "#define NEED_QSORT_R 1") >> $@.tmp
	@echo
	@echo "Testing swap qsort_r..."
	@echo "Testing swap qsort_r..." >> os-features.log
	(($(CC) -Werror -o os-features-test-swap-qsort \
	   $(CFLAGS) -DTEST_SWAP_QSORT_R $^ $(LDFLAGS) >> os-features.log && \
	   ./os-features-test-swap-qsort >> os-features.log && \
	   echo "#define NEED_SWAP_QSORT_R 0") \
	|| echo "#define NEED_SWAP_QSORT_R 1") >> $@.tmp
	@echo
	@echo "Testing netpbm..."
	@echo "   NETPBM_INC is $(NETPBM_INC_ORIG)"
	@echo "   NETPBM_LIB is $(NETPBM_LIB_ORIG)"
	@echo "Testing netpbm..." >> os-features.log
	(($(CC) -o os-features-test-netpbm \
	   $(CFLAGS) -DTEST_NETPBM $(NETPBM_INC_ORIG) $^ $(LDFLAGS) $(NETPBM_LIB_ORIG) >> os-features.log && \
	   ./os-features-test-netpbm >> os-features.log && \
	   echo "#define HAVE_NETPBM 1") \
	|| echo "#define HAVE_NETPBM 0") >> $@.tmp
	@echo "--------------- End of expected error messages -----------------"
	@echo
	mv $@.tmp $@
	@echo
	@echo "Config results:"
	@echo "------------------------------"
	@echo
	cat $@
	@echo
	@echo "------------------------------"
	@echo

makefile.os-features: os-features-test.c
	@echo
	@echo "---- Error messages in the next few commands are not necessarily bugs ----"
	@echo "     (we're checking how things works on your computer)"
	-rm -f os-features-makefile.log
	( \
	 echo "# This file is generated by util/Makefile."; \
	 (($(CC) -o os-features-test-netpbm-make \
	   $(CFLAGS) -DTEST_NETPBM_MAKE $(NETPBM_INC_ORIG) $^ $(LDFLAGS) $(NETPBM_LIB_ORIG) >> os-features-makefile.log && \
	   ./os-features-test-netpbm-make >> os-features-makefile.log && \
	   echo "HAVE_NETPBM := yes") \
	|| (echo "# Astrometry.net didn't find netpbm; not setting HAVE_NETPBM."; \
		echo "# See os-features-makefile.log for details."; \
		echo "# To re-run this test, do 'make reconfig; make makefile.os-features' (in the 'util' directory)"; \
		echo "# Or to do it yourself, just uncomment this line:"; \
		echo "# HAVE_NETPBM := yes")) \
	; \
	echo) > $@.tmp
	@echo "--------------- End of expected error messages -----------------"
	@echo
	mv $@.tmp $@
	@echo
	@echo "Config results:"
	@echo "------------------------------"
	@echo
	cat $@
	@echo "------------------------------"
	@echo
	@echo "And, by the way, is WCSlib support being compiled in?"
	@echo
	pkg-config --exists wcslib && echo yes || echo no
	@echo
	@echo "  WCSLIB_INC: $(WCSLIB_INC)"
	@echo "  WCSLIB_LIB: $(WCSLIB_LIB)"
	@echo
	@echo "------------------------------"

reconfig:
	-rm -f os-features-config.h makefile.os-features
.PHONY: reconfig

config: os-features-config.h makefile.os-features
.PHONY: config

ALL_OBJ += os-features.o
ALL_TARGETS := os-features-test.o \
	os-features-test-canonicalize \
	os-features-test-declare-qsort \
	os-features-test-qsort \
	os-features-test-netpbm \
	os-features-test-netpbm-make \
	os-features-config.h



ifndef NO_QFITS
DEP_PREREQS += $(QFITS_LIB)
endif

PROG_MAIN_OBJS := $(addsuffix .o,$(PROGS)) $(addsuffix -main.o,$(MAIN_PROGS))
DEP_OBJ += $(PROG_MAIN_OBJS)

hd1.fits: henry-draper.tsv build-hd-tree
	build-hd-tree -s -R 16 henry-draper.tsv $@

hd.fits: henry-draper.tsv build-hd-tree tycho2.fits tyc2_hd.dat
	build-hd-tree -s -R 16 -T tycho2.fits -X tyc2_hd.dat henry-draper.tsv $@

tyc2_hd.dat:
	wget "ftp://cdsarc.u-strasbg.fr/pub/cats/IV/25/tyc2_hd.dat.gz" -O $@.gz
	gunzip $@.gz

henry-draper.tsv:
	wget "http://trac.astrometry.net/browser/binary/henry-draper/henry-draper.tsv?format=txt" -O $@

tycho2.fits: catalog.dat suppl_1.dat
	tycho2tofits -o $@ $^

catalog.dat:
	wget -c http://www.astro.ku.dk/~cf/CD/data/catalog.dat

suppl_1.dat:
	wget -c http://www.astro.ku.dk/~cf/CD/data/suppl_1.dat

build-hd-tree: build-hd-tree.o tycho2-fits.o tycho2.o \
		$(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += build-hd-tree.o

### TESTS

ALL_TEST_FILES = test_bl test_bt test_endian test_fitstable test_xylist \
	test_rdlist test_healpix test_fitsioutils test_fitsbin test_log test_ioutils \
	test_scamp_catalog test_starutil test_svd test_hd \
	test_tycho2 test_anwcs test_sip-utils test_errors test_multiindex \
	test_convolve_image test_qsort_r test_wcs test_big_tables
# test_hd depends on hd.fits...
ALL_TEST_EXTRA_OBJS = 
ALL_TEST_LIBS = $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)
ALL_TEST_EXTRA_LDFLAGS = $(WCSLIB_LIB)

include $(COMMON)/makefile.tests

DEP_OBJ += $(ALL_TEST_FILES_O)

test_big_tables: $(ANUTILS_LIB) $(QFITS_LIB)

test_qsort_r: $(ANUTILS_LIB)

test_convolve_image: $(ANUTILS_LIB) $(QFITS_LIB)

test_multiindex: $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_errors: $(ANUTILS_LIB)

test_sip-utils: $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)

test_anwcs: $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCSLIB_LIB)

test_wcs: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(WCSLIB_LIB)

test_tycho2: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_hd: $(ANFILES_LIB) $(LIBKD_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_fitstable: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_fitsbin: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_fitsioutils: $(ANUTILS_LIB) $(QFITS_LIB)

test_xylist: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_rdlist: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_bl: $(ANUTILS_LIB)

test_bt: $(ANUTILS_LIB)

test_endian: $(ANUTILS_LIB)

test_healpix: $(ANUTILS_LIB)

test_log: $(ANUTILS_LIB)

test_ioutils: $(ANUTILS_LIB)

test_scamp_catalog: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_starutil: $(ANUTILS_LIB)

test_svd: $(ANUTILS_LIB) $(GSL_LIB)

### /TESTS

# we'll just tack this on here... It depends on cairo, libjpeg, etc, so it's not part
# of the anX.a libraries.
ifeq ($(MAKECMDGOALS),cairoutils.o)

include cairoutils.dep

endif

.PHONY: clean

clean:
	rm -f $(ANUTILS_LIB) $(ANFILES_LIB) $(ALL_OBJ) \
		$(DEPS) deps cairoutils.o $(PYTHON_LIBS) \
		grab-stellarium-constellations \
		$(PROGS) $(MAIN_PROGS) $(ALL_TARGETS) $(ALL_TESTS_CLEAN) \
		cairoutils.dep makefile.os-features *.o *~ *.dep *.so deps
	rm -Rf build


# Try to compensate for clang's lack of -MG to handle generated header
# files by making everything depend explicitly on os-features-config.h.

# the bar "|" indicates an "order-only" dependency: must be built
# before, but doesn't imply a rebuild is necessary.
# Should this perhaps go on the .dep files rather than the .c?
DEP_SRC := $(subst .o,.c,$(DEP_OBJ))
$(DEP_SRC): | os-features-config.h
os-features.h: | os-features-config.h
$(DEP_OBJ): os-features-config.h

ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),reconfig)
    ifneq ($(MAKECMDGOALS),makefile.os-features)
      include $(COMMON)/makefile.deps
    endif
  endif
endif
