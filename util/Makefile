# This file is part of the Astrometry.net suite.
# Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
# Copyright 2008, 2009, 2010 Dustin Lang.
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
DATA := $(BASEDIR)/data

$(DATA)/%.txt: ;
$(DATA)/%.dat: ;

all:
.PHONY: all

# You can build stripped-down versions of the AN libraries that do not
# depend on libkd or libqfits by defining the following in your Makefile
# before calling "make" in the "util" directory:
#   NO_KDTREE := 1
#   export NO_KDTREE
#   NO_QFITS := 1
#   export NO_QFITS

include $(COMMON)/makefile.common
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
include $(COMMON)/makefile.os-features

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
CFLAGS += $(GSL_INC)
CFLAGS += $(WCSLIB_INC)
CFLAGS += $(NETPBM_INC)

ifdef WCSLIB_EXISTS
CFLAGS += -DWCSLIB_EXISTS
endif

LDFLAGS += $(LDFLAGS_DEF)
LDFLAGS += -lm

ifndef NO_QFITS
CFLAGS += $(QFITS_INC)
endif
ifndef NO_KDTREE
CFLAGS += $(LIBKD_INC)
endif
CFLAGS += -I.

ANFILES_LIB := libanfiles.a
ANUTILS_LIB := libanutils.a

PYTHON_LIBS := _sip.so _healpix.so

PROGS := an-fitstopnm an-pnmtofits downsample-fits pad-file tycho2tofits fits-flip-endian smooth-image
# tycho2tostellarium build-hd-tree

all: $(ANUTILS_LIB) $(ANFILES_LIB) $(PYTHON_LIBS) $(PROGS)

# Things that it depends on but which aren't linked in
ANUTILS_DEPS :=

ANUTILS_OBJ := starutil.o mathutil.o bl.o bt.o healpix-utils.o healpix.o \
	permutedsort.o ioutils.o sip.o sip-utils.o md5.o constellations.o \
	os-features.o an-endian.o errors.o an-opts.o svn.o tic.o log.o \
	gslutils.o ngc2000.o ngcic-accurate.o brightstars.o \
	boilerplate.o tycho2.o intmap.o histogram.o histogram2d.o \
	anwcs.o sparsematrix.o lsqr.o

# Actually there are ANFILES_H mixed in here too....
ANUTILS_H := an-bool.h an-catalog.h  an-endian.h an-opts.h an-thread-pthreads.h \
	an-thread.h bl.h bl.inc bl.ph bl-nl.h bl-nl.inc bl-nl.ph brightstars.h bt.h cairoutils.h \
	codekd.h constellations.h dualtree.h dualtree_nearestneighbour.h \
	dualtree_rangesearch.h errors.h fitsbin.h fitsfile.h fitsioutils.h \
	fitstable.h os-features-config.h os-features.h gslutils.h hd.h \
	healpix-utils.h healpix.h index.h intmap.h ioutils.h keywords.h log.h \
	mathutil.h ngc2000.h ngcic-accurate.h permutedsort.h qidxfile.h \
	quadfile.h rdlist.h scamp-catalog.h sip-utils.h sip.h sip_qfits.h \
	starkd.h starutil.h starutil.inc starxy.h svn.h tic.h tycho2-fits.h tycho2.h \
	xylist.h
#cfitsutils.h
#boilerplate.h
#cutest.h
#histogram.h
#histogram2d.h
#md5.h

ifndef NO_QFITS
ANUTILS_OBJ += fitsioutils.o sip_qfits.o fitstable.o fitsbin.o fitsfile.o \
	tycho2-fits.o tic.o
ANUTILS_DEPS += $(QFITS_LIB)
endif
ifndef NO_KDTREE
ANUTILS_OBJ += dualtree.o dualtree_rangesearch.o dualtree_nearestneighbour.o
endif

# Things that it depends on but which aren't linked in
ANFILES_DEPS :=

ANFILES_OBJ := 
ifndef NO_QFITS
ANFILES_OBJ += index.o codekd.o starkd.o rdlist.o xylist.o \
	starxy.o an-catalog.o qidxfile.o quadfile.o scamp-catalog.o \
	hd.o
ANFILES_DEPS += $(QFITS_LIB)
endif

MISC_OBJ := grab-stellarium-constellations.o 

ALL_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)

# svn.c depends on all the .c and .h files in ANUTILS_OBJ
SVN_C_DEPS := $(subst svn.o,,$(ANUTILS_OBJ))
SVN_C_DEPS := $(subst .o,.c,$(SVN_C_DEPS)) $(subst .o,.h,$(SVN_C_DEPS))

svn.c: $(SVN_C_DEPS)
	echo "//" >> svn.c

ngc2000.o: ngc2000.c ngc2000names.c ngc2000entries.c

ngc2000names.c: parse-ngc2000names.awk $(DATA)/ngc2000names.dat
	$(AWK) -f parse-ngc2000names.awk < $(DATA)/ngc2000names.dat  > $@

ngc2000entries.py: parse-ngc2000-py.awk $(DATA)/ngc2000.dat
	$(AWK) -f parse-ngc2000-py.awk < $(DATA)/ngc2000.dat  > $@

ngc2000accurate.py: parse-ngcic-accurate-py.awk $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt
	cat $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt | $(AWK) -f parse-ngcic-accurate-py.awk > $@

ngc2000.py: ngc2000-pre.py ngc2000entries.py ngc2000-mid.py ngc2000accurate.py ngc2000-post.py
	cat $^ > $@

ngc2000entries.c: parse-ngc2000.awk $(DATA)/ngc2000.dat
	$(AWK) -f parse-ngc2000.awk < $(DATA)/ngc2000.dat  > $@

ngcic-accurate-entries.c: parse-ngcic-accurate.awk $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt
	cat $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt | $(AWK) -f parse-ngcic-accurate.awk > $@

tycho2tofits: tycho2tofits.o \
		$(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += tycho2tofits.o

tycho2tostellarium: tycho2tostellarium.o $(ANUTILS_LIB) $(QFITS_LIB)

$(ANUTILS_LIB): $(ANUTILS_OBJ) $(ANUTILS_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(ANUTILS_OBJ)
	$(RANLIB) $@

$(ANFILES_LIB): $(ANFILES_OBJ) $(ANFILES_DEPS)
	-rm -f $@
	$(AR) rc $@ $(ANFILES_OBJ)
	$(RANLIB) $@

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

_sip.so: sip-utils.o sip.o starutil.o mathutil.o sip_qfits.o \
		fitsioutils.o ioutils.o bl.o os-features.o an-endian.o \
		errors.o log.o $(QFITS_LIB)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

_healpix.so: healpix-utils.o healpix.o permutedsort.o starutil.o mathutil.o errors.o bl.o log.o os-features.o
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

pad-file: pad-file.o $(ANUTILS_LIB)
ALL_OBJ += pad-file.o

an-fitstopnm: an-fitstopnm.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += an-fitstopnm.o

an-pnmtofits: an-pnmtofits.o $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(NETPBM_LIB)
ALL_OBJ += an-pnmtofits.o

downsample-fits: downsample-fits.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += downsample-fits.o

smooth-image: smooth-image.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += smooth-image.o

fits-flip-endian: fits-flip-endian.o $(ANUTILS_LIB) $(QFITS_LIB)
ALL_OBJ += fits-flip-endian.o

PYTHON_INSTALL := fits2fits.py image2pnm.py healpix.py sip.py starutil.py \
	shell.py __init__.py file.py run_command.py w3c_validator.py \
	filetype.py pyfits_utils.py fix_sdss_idr.py removelines.py \
	ngc2000.py uniformize.py $(PYTHON_LIBS)

PYTHON_EXECS := fits2fits.py image2pnm.py removelines.py uniformize.py

PY_INSTALL_DIR := $(INSTALL_DIR)/python/astrometry/util
BIN_INSTALL_DIR := $(INSTALL_DIR)/bin
INCLUDE_INSTALL_DIR := $(INSTALL_DIR)/include
LIB_INSTALL_DIR := $(INSTALL_DIR)/lib
LINK_DIR := ../python/astrometry/util

install: $(PYTHON_LIBS) $(PYTHON_EXECS) $(PROGS) $(ANUTILS_H) $(ANFILES_LIB) $(ANUTILS_LIB)
	@echo Installing in base directory $(INSTALL_DIR)
	mkdir -p $(PY_INSTALL_DIR)
	@for x in $(PYTHON_INSTALL); do \
		echo cp $$x $(PY_INSTALL_DIR)/$$x; \
		cp $$x $(PY_INSTALL_DIR)/$$x; \
	done
	@echo Making symlinks in directory $(BIN_INSTALL_DIR)
	mkdir -p $(BIN_INSTALL_DIR)
	@for x in $(PYTHON_EXECS); do \
		echo ln -f -s $(PY_INSTALL_DIR)/$$x $(BIN_INSTALL_DIR)/$$x; \
		ln -f -s $(LINK_DIR)/$$x $(BIN_INSTALL_DIR)/$$x; \
	done
	@echo Installing executables into $(BIN_INSTALL_DIR)...
	@for x in $(PROGS); do \
		echo cp $$x $(BIN_INSTALL_DIR)/$$x; \
		cp $$x $(BIN_INSTALL_DIR)/$$x; \
	done
	@for x in $(ANUTILS_H); do \
		echo cp $$x $(INCLUDE_INSTALL_DIR)/$$x; \
		cp $$x $(INCLUDE_INSTALL_DIR)/$$x; \
	done
	@for x in $(ANFILES_LIB) $(ANUTILS_LIB); do \
		echo cp $$x $(LIB_INSTALL_DIR)/$$x; \
		cp $$x $(LIB_INSTALL_DIR)/$$x; \
	done
.PHONY: install

os-features-config.h: os-features-test.c
	( \
	(($(CC) -o os-features-test-canonicalize \
		$(CFLAGS) -DTEST_CANONICALIZE_FILE_NAME $(LDFLAGS) $^  > /dev/null && \
	  ./os-features-test-canonicalize) \
	|| echo "#define NEED_CANONICALIZE_FILE_NAME 1") \
	; \
	 (($(CC) -o os-features-test-declare-qsort \
		$(CFLAGS) -DTEST_DECLARE_QSORT_R $(LDFLAGS) $^ > /dev/null && \
	  ./os-features-test-declare-qsort) \
	|| echo "#define NEED_DECLARE_QSORT_R 0") \
	; \
	 (($(CC) -o os-features-test-qsort \
		$(CFLAGS) -DTEST_QSORT_R $(LDFLAGS) $^ > /dev/null && \
	  ./os-features-test-qsort) \
	|| echo "#define NEED_QSORT_R 1") \
	; \
	 (($(CC) -o os-features-test-swap-qsort \
		$(CFLAGS) -DTEST_SWAP_QSORT_R $(LDFLAGS) $^ > /dev/null && \
	  ./os-features-test-swap-qsort) \
	|| echo "#define NEED_SWAP_QSORT_R 0") \
	; \
	 (($(CC) -o os-features-test-netpbm \
		$(CFLAGS) -DTEST_NETPBM $(LDFLAGS) $(NETPBM_INC_ORIG) $(NETPBM_LIB_ORIG) $^ > /dev/null && \
	  ./os-features-test-netpbm) \
	|| echo "#define HAVE_NETPBM 0") \
	; \
	echo) | grep -v "Segmentation" > $@

makefile.os-features: os-features-test.c
	( \
	 (($(CC) -o os-features-test-netpbm-make \
		$(CFLAGS) -DTEST_NETPBM_MAKE $(LDFLAGS) $(NETPBM_INC_ORIG) $(NETPBM_LIB_ORIG) $^ > /dev/null && \
	  ./os-features-test-netpbm-make) \
	|| echo "# Don't HAVE_NETPBM.") \
	; \
	echo) | grep -v "Segmentation" > $@

ALL_OBJ += os-features.o
ALL_TARGETS := os-features-test.o \
	os-features-test-canonicalize \
	os-features-test-declare-qsort \
	os-features-test-qsort \
	os-features-test-netpbm \
	os-features-test-netpbm-make \
	os-features-config.h



DEP_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)
ifndef NO_QFITS
DEP_PREREQS := $(QFITS_LIB)
endif

PROG_MAIN_OBJS := $(addsuffix .o,$(PROGS))
DEP_OBJ += $(PROG_MAIN_OBJS)

stellarium-constellations.c:
	-$(MAKE) grab-stellarium-constellations
	-(./grab-stellarium-constellations > $@.tmp  &&  mv $@.tmp $@) || touch $@

grab-stellarium-constellations: grab-stellarium-constellations.o \
		starutil.o mathutil.o bl.o an-endian.o
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ -lm

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
	test_dualtree_nn test_scamp_catalog test_starutil test_svd test_hd \
	test_tycho2 test_anwcs
# test_hd depends on hd.fits...
ALL_TEST_EXTRA_OBJS = 
ALL_TEST_LIBS = $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB) $(GSL_LIB) $(WCSLIB_LIB)

include $(COMMON)/makefile.tests

DEP_OBJ += $(ALL_TEST_FILES_O)

test_anwcs: $(ANUTILS_LIB) $(QFITS_LIB)
	$(CC) -o $@ $^ $(WCSLIB_LIB)

test_tycho2: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_hd: $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

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

test_dualtree_nn: $(LIBKD_LIB) $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_scamp_catalog: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_starutil: $(ANUTILS_LIB)

test_svd: $(ANUTILS_LIB) $(GSL_LIB)

### /TESTS

# we'll just tack this on here... It depends on cairo, libjpeg, etc, so it's not part
# of the anX.a libraries.
ifeq ($(MAKECMDGOALS),cairoutils.o)
  include cairoutils.dep
endif

.PHONY: allclean realclean clean

allclean: clean
	rm -f *.o *~ *.dep *.so

realclean: allclean

clean:
	rm -f $(ANUTILS_LIB) $(ANFILES_LIB) $(ALL_OBJ) \
		$(DEPS) deps cairoutils.o $(PYTHON_LIBS) \
		grab-stellarium-constellations \
		$(PROGS) $(ALL_TARGETS) $(ALL_TESTS_CLEAN) \
		cairoutils.dep makefile.os-features

include $(COMMON)/makefile.deps
