# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

BASEDIR := ..
COMMON := $(BASEDIR)/util
DATA := .

all:
.PHONY: all

include $(COMMON)/makefile.common

# required for build-hd-tree
include $(COMMON)/makefile.anfiles
# 2masstofits
include $(COMMON)/makefile.zlib
#include $(COMMON)/makefile.libkd
#include $(COMMON)/makefile.anutils

$(DATA)/%.txt: ;
$(DATA)/%.dat: ;

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

stellarium-constellations.c:
	-$(MAKE) grab-stellarium-constellations
	-(./grab-stellarium-constellations > $@.tmp  &&  mv $@.tmp $@) || touch $@

grab-stellarium-constellations: grab-stellarium-constellations.o \
		starutil.o mathutil.o bl.o an-endian.o errors.o log.o ioutils.o qsort_reentrant.o tic.o
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)


OBJS := ngc2000.o ngcic-accurate.o brightstars.o constellations.o \
	tycho2-fits.o tycho2.o usnob-fits.o usnob.o nomad.o nomad-fits.o \
	ucac3-fits.o ucac3.o ucac4-fits.o ucac4.o 2mass-fits.o 2mass.o hd.o \
	constellation-boundaries.o

HEADERS := brightstars.h constellations.h ngc2000.h ngcic-accurate.h \
	tycho2.h tycho2-fits.h usnob-fits.h usnob.h nomad-fits.h nomad.h \
	2mass-fits.h 2mass.h hd.h ucac3.h ucac4.h constellation-boundaries.h

HEADERS_PATH := $(addprefix $(INCLUDE_DIR)/,$(HEADERS))

PYTHON_INSTALL := __init__.py ngc2000.py 

PY_INSTALL_DIR := $(PY_BASE_INSTALL_DIR)/catalogs
LINK_DIR := $(PY_BASE_LINK_DIR)/catalogs

# MISC_OBJ := grab-stellarium-constellations.o 

DEP_OBJ := $(OBJS)
DEP_PREREQS :=
DEP_PREREQS += ngc2000names.c ngc2000entries.c ngcic-accurate-entries.c

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
	2masstofits
	#ucac3tofits ucac4tofits

# not built by default
MISC_PROGS := usnob-scamp-catalog read_nomad

ucac3tofits: ucac3tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac3tofits.o

ucac4tofits: ucac4tofits.o $(SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) -lbz2
ALL_OBJ += ucac4tofits.o

2masstofits: 2masstofits.o $(SLIB)
	echo ZLIB_LIB is $(ZLIB_LIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS) $(ZLIB_LIB)
ALL_OBJ += 2masstofits.o

$(PROGS): %: %.o $(SLIB)
ALL_OBJ += $(addsuffix .o,$(PROGS))

all: ngc2000.py $(LIBCAT) $(PROGS)

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
	rm -f $(LIBCAT) $(OBJS) $(ALL_OBJ) $(DEPS) *.dep deps \
		grab-stellarium-constellations \
		ngc2000names.c ngc2000entries.py ngc2000.py ngc2000entries.c \
		ngcic-accurate-entries.c

ifneq ($(MAKECMDGOALS),clean)
          include $(COMMON)/makefile.deps
endif

