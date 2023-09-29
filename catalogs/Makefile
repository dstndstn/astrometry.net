# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

BASEDIR := ..
COMMON := $(BASEDIR)/util
DATA := .

all:
.PHONY: all

# Detect GSL -- this minimum version was chosen to match the version in gsl-an.
# Earlier versions would probably work fine.
SYSTEM_GSL ?= $(shell (pkg-config --atleast-version=1.14 gsl && echo "yes") || echo "no")
# Make this variable visible to recursive "make" calls
export SYSTEM_GSL

include $(COMMON)/makefile.common

# required for build-hd-tree
include $(COMMON)/makefile.anfiles
# 2masstofits
include $(COMMON)/makefile.zlib
#include $(COMMON)/makefile.libkd
#include $(COMMON)/makefile.anutils

$(DATA)/%.txt: ;
$(DATA)/NGC.csv: ;

openngc-entries.csv: openngc-entries-csv.awk $(DATA)/NGC.csv
	$(AWK) -F\; -f openngc-entries-csv.awk < $(DATA)/NGC.csv > $@

openngc-names.csv: openngc-names-csv.awk $(DATA)/NGC.csv
	$(AWK) -F\; -f openngc-names-csv.awk < $(DATA)/NGC.csv > $@

openngc.o: openngc.c openngc-names.c openngc-entries.c

openngc-names.c: openngc-names-c.awk openngc-names.csv
	$(AWK) -F\; -f openngc-names-c.awk < openngc-names.csv > $@

openngc-entries.c: openngc-entries-c.awk openngc-entries.csv
	$(AWK) -F\; -f openngc-entries-c.awk < openngc-entries.csv > $@

stellarium-constellations.c:
	-$(MAKE) grab-stellarium-constellations
	-(./grab-stellarium-constellations > $@.tmp  &&  mv $@.tmp $@) || touch $@

grab-stellarium-constellations: grab-stellarium-constellations.o \
		starutil.o mathutil.o bl.o an-endian.o errors.o log.o ioutils.o qsort_reentrant.o tic.o
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)

# brightstars-data.c and brightstars.fits:
# I fetched hip.fits by visiting
# http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I/239/hip_main
# and selecting FITS binary output, and columns HIP, Vmag, _RA.icrs, _DE.icrs
# the latter of which are ICRS, epoch 2000 coordinates.
#
# Then ran:
# python brightstars.py /Applications/Stellarium.app/Contents/Resources/skycultures/western/star_names.fab hip.fits brightstars.fits --letternames /Applications/Stellarium.app/Contents/Resources/stars/default/name.fab --brightstars brightstars-data.c --multi


OBJS := openngc.o brightstars.o constellations.o \
	tycho2-fits.o tycho2.o usnob-fits.o usnob.o nomad.o nomad-fits.o \
	ucac3-fits.o ucac3.o ucac4-fits.o ucac4.o ucac5-fits.o ucac5.o \
	2mass-fits.o 2mass.o hd.o constellation-boundaries.o

HEADERS := brightstars.h constellations.h openngc.h \
	tycho2.h tycho2-fits.h usnob-fits.h usnob.h nomad-fits.h nomad.h \
	2mass-fits.h 2mass.h hd.h ucac3.h ucac4.h ucac5.h constellation-boundaries.h

HEADERS_PATH := $(addprefix $(INCLUDE_DIR)/,$(HEADERS))

PYTHON_INSTALL := __init__.py

PY_INSTALL_DIR := $(PY_BASE_INSTALL_DIR)/catalogs
LINK_DIR := $(PY_BASE_LINK_DIR)/catalogs

# MISC_OBJ := grab-stellarium-constellations.o

DEP_OBJ := $(OBJS)
DEP_PREREQS :=
DEP_PREREQS += openngc-names.c openngc-entries.c

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(LIBKD_INC)
CFLAGS += $(ANUTILS_INC)

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(ANFILES_LIB)
LDLIBS += -lm

CFLAGS += -I.
CFLAGS += -I$(COMMON)
CFLAGS += $(ZLIB_INC)

LIBCAT := libcatalogs.a

SLIB := $(LIBCAT)
SLIB += $(ANFILES_SLIB)

$(LIBCAT): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@

PROGS := build-hd-tree tycho2tofits usnobtofits nomadtofits \
	2masstofits ucac5tofits
	#ucac3tofits ucac4tofits

# not built by default
MISC_PROGS := usnob-scamp-catalog read_nomad

ucac3tofits: ucac3tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac3tofits.o

ucac4tofits: ucac4tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac4tofits.o

ucac5tofits: ucac5tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac5tofits.o

2masstofits: 2masstofits.o $(SLIB)
	echo ZLIB_LIB is $(ZLIB_LIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) $(ZLIB_LIB)
ALL_OBJ += 2masstofits.o

$(PROGS): %: %.o $(SLIB)
ALL_OBJ += $(addsuffix .o,$(PROGS))

all: $(LIBCAT) $(PROGS)

hd1.fits: henry-draper.tsv build-hd-tree
	build-hd-tree -s -R 16 henry-draper.tsv $@

# hd.fits: you can grab a pre-built one from:
#    http://data.astrometry.net/hd.fits
# The instructions below are commented-out because some of the URLs have disappeared.
#
# hd.fits: henry-draper.tsv build-hd-tree tycho2.fits tyc2_hd.dat
# 	build-hd-tree -s -R 16 -T tycho2.fits -X tyc2_hd.dat henry-draper.tsv $@
#
# tyc2_hd.dat:
# 	wget "ftp://cdsarc.u-strasbg.fr/pub/cats/IV/25/tyc2_hd.dat.gz" -O $@.gz
# 	gunzip $@.gz
#
# henry-draper.tsv:
# 	wget "http://trac.astrometry.net/browser/binary/henry-draper/henry-draper.tsv?format=txt" -O $@
#
# tycho2.fits: catalog.dat suppl_1.dat
# 	tycho2tofits -o $@ $^
#
# catalog.dat:
# 	wget -c http://www.astro.ku.dk/~cf/CD/data/catalog.dat
#
# suppl_1.dat:
# 	wget -c http://www.astro.ku.dk/~cf/CD/data/suppl_1.dat

install: $(PYTHON_INSTALL) $(LIBCAT) $(HEADERS_PATH)
	@echo Installing in base directory '$(INSTALL_DIR)'
	$(MKDIR) '$(PY_INSTALL_DIR)'
	$(MKDIR) '$(INCLUDE_INSTALL_DIR)'
	$(MKDIR) '$(LIB_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo $(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@for x in $(HEADERS); do \
		echo $(CP) '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
		$(CP) '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
	done
	@for x in $(LIBCAT); do \
		echo $(CP) $$x '$(LIB_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(LIB_INSTALL_DIR)/'$$x; \
	done
	@echo ok

.PHONY: install

pyinstall: $(PYTHON_INSTALL)
	$(MKDIR) '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo $(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
.PHONY: pyinstall

ALL_TEST_FILES = test_tycho2 test_usnob test_nomad test_2mass test_hd \
	test_boundaries
ALL_TEST_EXTRA_OBJS =
ALL_TEST_LIBS = $(SLIB)
ALL_TEST_EXTRA_LDFLAGS =
include $(COMMON)/makefile.tests

$(ALL_TEST_FILES): $(SLIB)

tests: $(ALL_TEST_FILES)
.PHONY: tests

clean:
	rm -f $(LIBCAT) $(OBJS) $(ALL_OBJ) $(DEPS) $(PROGS) *.dep deps \
		grab-stellarium-constellations \
		openngc-entries.csv openngc-names.csv \
		openngc-entries.c openngc-names.c \

ifneq ($(MAKECMDGOALS),clean)
          include $(COMMON)/makefile.deps
endif

