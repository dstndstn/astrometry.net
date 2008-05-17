# This file is part of the Astrometry.net suite.
# Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
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

# You can build stripped-down versions of the AN libraries that do not
# depend on libkd or libqfits by defining the following in your Makefile
# before calling make in the an-common directory:
#   NO_KDTREE := 1
#   export NO_KDTREE
#   NO_QFITS := 1
#   export NO_QFITS

%.c: ;
%.h: ;
Makefile: ;

include $(COMMON)/makefile.common
$(COMMON)/makefile.common: ;
ifndef NO_QFITS
include $(COMMON)/makefile.qfits
$(COMMON)/makefile.qfits : ;
endif
ifndef NO_KDTREE
include $(COMMON)/makefile.libkd
$(COMMON)/makefile.libkd : ;
endif

include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg
include $(COMMON)/makefile.netpbm

$(COMMON)/makefile.deps  : ;
$(COMMON)/makefile.netpbm: ;
$(COMMON)/makefile.jpeg  : ;
$(COMMON)/makefile.cairo : ;

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(CAIRO_INC)
CFLAGS += $(NETPBM_INC)
CFLAGS += $(JPEG_INC)

LDFLAGS += $(LDFLAGS_DEF)
LDFLAGS += -lm

ifndef NO_QFITS
CFLAGS += $(QFITS_INC)
endif
ifndef NO_KDTREE
CFLAGS += $(LIBKD_INC)
endif
CFLAGS += -I.

# for _sip.so:
CFLAGS += -fPIC

ANFILES_LIB := libanfiles.a
ANUTILS_LIB := libanutils.a

PYTHON_LIBS := _sip.so _healpix.so

PROGS := an-fitstopnm

all: $(ANUTILS_LIB) $(ANFILES_LIB) $(PYTHON_LIBS) $(PROGS)

# Things that it depends on but which aren't linked in
ANUTILS_DEPS :=

ANUTILS_OBJ := starutil.o mathutil.o bl.o bt.o healpix.o permutedsort.o \
	qsort_reentrant.o ioutils.o sip.o sip-utils.o md5.o constellations.o \
	gnu-specific.o an-endian.o errors.o an-opts.o svn.o
ifndef NO_QFITS
ANUTILS_OBJ += fitsioutils.o sip_qfits.o
ANUTILS_DEPS += $(QFITS_LIB)
endif
ifndef NO_KDTREE
ANUTILS_OBJ += dualtree.o dualtree_rangesearch.o
endif

# Things that it depends on but which aren't linked in
ANFILES_DEPS :=

ANFILES_OBJ := 
ifndef NO_QFITS
ANFILES_OBJ += starkd.o rdlist.o xylist.o an-catalog.o \
	qidxfile.o quadfile.o fitstable.o fitsbin.o fitsfile.o
ANFILES_DEPS += $(QFITS_LIB)
endif

MISC_OBJ := grab-stellarium-constellations.o 

ALL_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)

# svn.c depends on all the .c and .h files in ANUTILS_OBJ
SVN_C_DEPS := $(subst svn.o,,$(ANUTILS_OBJ))
SVN_C_DEPS := $(subst .o,.c,$(SVN_C_DEPS)) $(subst .o,.h,$(SVN_C_DEPS))

svn.c: $(SVN_C_DEPS)
	echo "//" >> svn.c

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
		fitsioutils.o ioutils.o bl.o gnu-specific.o an-endian.o \
		errors.o $(QFITS_LIB)
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

_healpix.so: healpix.o starutil.o mathutil.o
	$(CC) $(SHAREDLIBFLAGS) -o $@ $^

an-fitstopnm: an-fitstopnm.o $(ANUTILS_LIB) $(QFITS_LIB)

ALL_OBJ += an-fitstopnm.o

PYTHON_INSTALL := fits2fits.py image2pnm.py healpix.py sip.py starutil.py \
	shell.py __init__.py file.py run_command.py w3c_validator.py \
	filetype.py \
	$(PYTHON_LIBS)

PYTHON_EXECS := fits2fits.py image2pnm.py

PY_INSTALL_DIR := $(INSTALL_DIR)/python/astrometry/util
BIN_INSTALL_DIR := $(INSTALL_DIR)/bin
LINK_DIR := ../python/astrometry/util

install: $(PYTHON_LIBS)
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

gnu-specific-config.h: gnu-specific-test.c
	(($(CC) -o gnu-specific-test-canonicalize \
		$(CFLAGS) -DTEST_CANONICALIZE_FILE_NAME $(LDFLAGS) $^ && \
	  ./gnu-specific-test-canonicalize) \
	|| echo "#define NEED_CANONICALIZE_FILE_NAME 1") > $@

ALL_OBJ += gnu-specific.o
ALL_TARGETS := gnu-specific-test.o gnu-specific-test-canonicalize \
	gnu-specific-config.h

DEP_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)
ifndef NO_QFITS
DEP_PREREQS := $(QFITS_LIB)
endif
include $(COMMON)/makefile.deps

stellarium-constellations.c:
	-$(MAKE) grab-stellarium-constellations
	-(./grab-stellarium-constellations > $@.tmp  &&  mv $@.tmp $@) || touch $@

grab-stellarium-constellations: grab-stellarium-constellations.o \
		starutil.o mathutil.o bl.o an-endian.o
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ -lm

### TESTS

ALL_TEST_FILES = test_bl test_bt test_endian test_fitstable test_xylist \
	test_rdlist test_healpix test_fitsioutils
ALL_TEST_EXTRA_OBJS = 
ALL_TEST_LIBS = $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

include $(COMMON)/makefile.tests

$(COMMON)/makefile.tests: ;

DEP_OBJ += $(ALL_TEST_FILES_O)

test_fitstable: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_fitsioutils: $(ANUTILS_LIB) $(QFITS_LIB)

test_xylist: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_rdlist: $(ANFILES_LIB) $(ANUTILS_LIB) $(QFITS_LIB)

test_bl: $(ANUTILS_LIB)

test_bt: $(ANUTILS_LIB)

test_endian: $(ANUTILS_LIB)

test_healpix: $(ANUTILS_LIB)

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
		$(PROGS) $(ALL_TARGETS)
