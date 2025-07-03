# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

BASEDIR := ..
COMMON := .

all:
.PHONY: all

# You can build stripped-down versions of the AN libraries that do not
# depend on libkd or libqfits by defining the following in your Makefile
# before calling "make" in the "util" directory:
#   NO_KDTREE := 1
#   export NO_KDTREE
#   NO_QFITS := 1
#   export NO_QFITS

SYSTEM_GSL ?= $(shell (pkg-config --atleast-version=1.14 gsl && echo "yes") || echo "no")
# Make this variable visible to recursive "make" calls
export SYSTEM_GSL

# Redefine these (to be in the current directory) before including
# makefile.anfiles.
ANBASE_LIB_FILE := libanbase.a
ANUTILS_LIB_FILE := libanutils.a
ANFILES_LIB_FILE := libanfiles.a

include $(COMMON)/makefile.common
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.anutils

SO=$(SHAREDLIB_SUFFIX)

ANBASE_OBJ := starutil.o mathutil.o bl-sort.o bl.o bt.o healpix-utils.o \
	healpix.o permutedsort.o ioutils.o fileutils.o md5.o \
	an-endian.o errors.o an-opts.o tic.o log.o datalog.o \
	sparsematrix.o coadd.o convolve-image.o resample.o \
	intmap.o histogram.o histogram2d.o

ANBASE_DEPS :=

ANUTILS_OBJ :=  sip-utils.o fit-wcs.o sip.o \
	anwcs.o wcs-resample.o gslutils.o wcs-pv2sip.o matchobj.o

# Things that it depends on but that aren't linked in
ANFILES_DEPS :=
ANFILES_OBJ := 

# Things that it depends on but that aren't linked in
ANUTILS_DEPS :=

ifndef NO_QFITS
ANFILES_OBJ += multiindex.o index.o indexset.o \
	codekd.o starkd.o rdlist.o xylist.o \
	starxy.o qidxfile.o quadfile.o scamp.o scamp-catalog.o \
	tabsort.o wcs-xy2rd.o wcs-rd2xy.o matchfile.o
ANFILES_DEPS += $(QFITS_LIB)

ANUTILS_OBJ += fitsioutils.o sip_qfits.o fitstable.o fitsbin.o fitsfile.o \
	tic.o
ANUTILS_DEPS += $(QFITS_LIB)
endif


SIMPLEXY_OBJ := dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o dobjects.o \
	dpeaks.o dselip.o dsigma.o dsmooth.o image2xy.o simplexy.o ctmf.o
ANUTILS_OBJ += $(SIMPLEXY_OBJ)

include $(COMMON)/makefile.cairo
include $(COMMON)/makefile.jpeg
include $(COMMON)/makefile.png
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
CFLAGS += -I$(ANBASE_DIR)
CFLAGS += $(CAIRO_INC)
CFLAGS += $(JPEG_INC)
CFLAGS += $(PNG_INC)
CFLAGS += $(NETPBM_INC)
CFLAGS += $(ANFILES_CFLAGS)
CFLAGS += $(ANFILES_INC)
CFLAGS += $(ANUTILS_CFLAGS)
CFLAGS += $(ANUTILS_INC)
CFLAGS += -I.

%.o: %.c
	$(CC) -o $@ $(CPPFLAGS) $(CFLAGS) -c $<

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(ANFILES_LIB)
LDLIBS += $(ANUTILS_LIB)

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

QFITS_UTILS := fits-column-merge subtable fitsgetext wcsinfo

PROGS := an-fitstopnm an-pnmtofits downsample-fits \
	fits-flip-endian hpsplit wcs-to-tan	\
	get-healpix $(QFITS_UTILS) pad-file wcs-match wcs-pv2sip

MAIN_PROGS := wcs-resample tabsort wcs-xy2rd wcs-rd2xy fit-wcs wcs-pv2sip

all: $(ANBASE_LIB_FILE) $(ANUTILS_LIB_FILE) $(ANFILES_LIB_FILE) $(PROGS) $(MAIN_PROGS)

# Actually there are ANFILES_H mixed in here too....
ANUTILS_H := an-bool.h an-endian.h an-opts.h an-thread-pthreads.h \
	an-thread.h anwcs.h bl.h bl.inc bl.ph bl-nl.h bl-nl.inc bl-nl.ph \
	bl-sort.h  bt.h cairoutils.h \
	codekd.h errors.h fitsbin.h fitsfile.h fitsioutils.h \
	fitstable.h os-features-config.h os-features.h gslutils.h \
	healpix-utils.h healpix.h index.h intmap.h ioutils.h fileutils.h \
	keywords.h log.h \
	mathutil.h permutedsort.h qidxfile.h quadfile.h rdlist.h scamp-catalog.h \
	fit-wcs.h sip-utils.h sip.h sip_qfits.h starkd.h starutil.h starutil.inc \
	starxy.h tic.h \
	xylist.h coadd.h convolve-image.h resample.h multiindex.h scamp.h \
	ctmf.h dimage.h image2xy.h simplexy-common.h simplexy.h \
	tabsort.h wcs-rd2xy.h wcs-xy2rd.h wcs-pv2sip.h matchobj.h matchfile.h

ANUTILS_H_PATH := $(addprefix $(INCLUDE_DIR)/,$(ANUTILS_H))

## ANFILES_DEPS, ANFILES_OBJ -- in makefile.anfiles

ALL_OBJ := $(ANBASE_OBJ) $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)

DEP_OBJ := $(ANUTILS_OBJ) $(ANFILES_OBJ) $(MISC_OBJ)
DEP_PREREQS :=

$(ANBASE_LIB_FILE): $(ANBASE_OBJ) $(ANBASE_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(ANBASE_OBJ)
	$(RANLIB) $@

$(ANUTILS_LIB_FILE): $(ANUTILS_OBJ) $(ANUTILS_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(ANUTILS_OBJ)
	$(RANLIB) $@

$(ANFILES_LIB_FILE): $(ANFILES_OBJ) $(ANFILES_DEPS)
	-rm -f $@
	$(AR) rc $@ $(ANFILES_OBJ)
	$(RANLIB) $@

$(MAIN_PROGS): %: %-main.o rdlist.o xylist.o starxy.o $(ANUTILS_SLIB)
$(PROGS): %: %.o $(ANUTILS_SLIB)

DEP_OBJ += $(addsuffix -main.o,$(MAIN_PROGS))
DEP_OBJ += $(addsuffix .o,$(PROGS))

an-pnmtofits: an-pnmtofits.o $(ANUTILS_SLIB)
	$(CC) -o $@ $(LDFLAGS) $^ $(NETPBM_LIB) $(LDLIBS)
ALL_OBJ += an-pnmtofits.o

_util$(PYTHON_SO_EXT): util.i lanczos.i $(ANFILES_SLIB)
	LDFLAGS="$(LDFLAGS)" LDLIBS="$(LDLIBS)" SLIB="$(ANFILES_SLIB)" \
	INC="$(ANFILES_INC)" CFLAGS="$(CFLAGS)" \
	$(PYTHON) setup.py build_ext -v --inplace --build-temp .
util.py: util.i lanczos.i
	LDFLAGS="$(LDFLAGS)" LDLIBS="$(LDLIBS)" SLIB="$(ANFILES_SLIB)" \
	INC="$(ANFILES_INC)" CFLAGS="$(CFLAGS)" \
	$(PYTHON) setup.py build_ext -v --inplace --build-temp .

PYUTIL := _util$(PYTHON_SO_EXT) util.py
pyutil: $(PYUTIL)
.PHONY: pyutil
py: $(PYUTIL)
.PHONY: py

PYTHON_INSTALL := image2pnm.py addpath.py \
	starutil.py starutil_numpy.py \
	shell.py __init__.py file.py run_command.py \
	filetype.py fits.py fix_sdss_idr.py removelines.py \
	uniformize.py \
	usnob_catalog.py usnob_get_image.py usnob_get_region.py \
	find_data_file.py plotshift.py plotutils.py miscutils.py multiproc.py \
	ttime.py resample.py EXIF.py horizons.py \
	imageutils.py siap.py stages.py timingpool.py

# celestial_mechanics.py (needs license)

PYTHON_DRIVERS := image2pnm removelines uniformize degtohms hmstodeg \
	text2fits votabletofits merge-columns

PYTHON_EXECS := $(addprefix ../bin/,$(PYTHON_DRIVERS))

PY_INSTALL_DIR := $(PY_BASE_INSTALL_DIR)/util
LINK_DIR := $(PY_BASE_LINK_DIR)/util

install: install-core
	$(MAKE) install-core
	@echo "The following copy commands may fail; they are optional."
	-$(MAKE) py && $(CP) _util$(PYTHON_SO_EXT) util.py '$(PY_INSTALL_DIR)'
	@echo ok

.PHONY: install

pyinstall:
	$(MKDIR) '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo $(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	$(MAKE) py && $(CP) _util$(PYTHON_SO_EXT) util.py '$(PY_INSTALL_DIR)'

.PHONY: pyinstall

install-core: $(PYTHON_EXECS) $(PROGS) $(MAIN_PROGS) $(ANUTILS_H_PATH) \
		$(ANFILES_LIB_FILE) $(ANUTILS_LIB_FILE) $(ANBASE_LIB_FILE)
	@echo Installing in base directory '$(INSTALL_DIR)'
	@echo Installing python in base directory '$(PY_INSTALL_DIR)'
	$(MKDIR) '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo $(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@echo Installing python scripts in directory '$(BIN_INSTALL_DIR)'
	@echo PYTHON_SCRIPT executable: '$(PYTHON_SCRIPT)'
	$(MKDIR) '$(BIN_INSTALL_DIR)'
	@for x in $(PYTHON_EXECS); do \
		echo $(SED) 's+$(PYTHON_SCRIPT_DEFAULT)+$(PYTHON_SCRIPT)+' $$x > '$(BIN_INSTALL_DIR)/'$$x; \
		$(SED) 's+$(PYTHON_SCRIPT_DEFAULT)+$(PYTHON_SCRIPT)+' $$x > '$(BIN_INSTALL_DIR)/'$$x; \
		echo $(CHMOD_EXECUTABLE) '$(BIN_INSTALL_DIR)/'$$x; \
		$(CHMOD_EXECUTABLE) '$(BIN_INSTALL_DIR)/'$$x; \
	done
	@echo Installing executables into '$(BIN_INSTALL_DIR)'...
	@for x in $(PROGS) $(MAIN_PROGS); do \
		echo $(CP) $$x '$(BIN_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(BIN_INSTALL_DIR)/'$$x; \
	done
	@echo Installing headers into '$(INCLUDE_INSTALL_DIR)'...
	$(MKDIR) '$(INCLUDE_INSTALL_DIR)'
	@for x in $(ANUTILS_H); do \
		echo $(CP) '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
		$(CP) '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
	done
	@echo Installing libraries into '$(LIB_INSTALL_DIR)'...
	$(MKDIR) '$(LIB_INSTALL_DIR)'
	@for x in $(ANFILES_LIB_FILE) $(ANUTILS_LIB_FILE) $(ANBASE_LIB_FILE); do \
		echo $(CP) $$x '$(LIB_INSTALL_DIR)/'$$x; \
		$(CP) $$x '$(LIB_INSTALL_DIR)/'$$x; \
	done

.PHONY: install-core

$(INCLUDE_DIR)/os-features-config.h: os-features-test.c
	@echo
	@echo "---- Error messages in the next few commands are not necessarily bugs ----"
	@echo "     (we're checking how things works on your computer)"
	-rm -f os-features.log
	-rm -f $@ $@.tmp
	@echo
	@echo "Testing netpbm..."
	@echo "   NETPBM_INC is $(NETPBM_INC_ORIG)"
	@echo "   NETPBM_LIB is $(NETPBM_LIB_ORIG)"
	@echo "Testing netpbm..." >> os-features.log
	(($(CC) -o os-features-test-netpbm \
	   $(CPPFLAGS) $(CFLAGS) $(NETPBM_INC_ORIG) $^ $(LDFLAGS) $(NETPBM_LIB_ORIG) >> os-features.log && \
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
	@echo "Testing netpbm..."
	@echo "   NETPBM_INC_ORIG is $(NETPBM_INC_ORIG)"
	@echo "   NETPBM_LIB_ORIG is $(NETPBM_LIB_ORIG)"
	( \
	 echo "# This file is generated by util/Makefile."; \
	 (($(CC) -o os-features-test-netpbm-make \
	   $(CPPFLAGS) $(CFLAGS) $(NETPBM_INC_ORIG) $^ $(LDFLAGS) $(NETPBM_LIB_ORIG) >> os-features-makefile.log && \
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
	-rm -f '$(INCLUDE_DIR)/'os-features-config.h makefile.os-features
.PHONY: reconfig

config: $(INCLUDE_DIR)/os-features-config.h makefile.os-features
.PHONY: config

ALL_TARGETS := os-features-test.o \
	os-features-test-netpbm \
	$(INCLUDE_DIR)/os-features-config.h

ifndef NO_QFITS
DEP_PREREQS += $(QFITS_LIB)
endif

### TESTS

ALL_TEST_FILES = test_bl test_bt test_endian test_fitstable test_xylist \
	test_rdlist test_healpix test_fitsioutils test_fitsbin test_log \
	test_scamp_catalog test_starutil test_svd test_ioutils \
	test_anwcs test_sip-utils test_errors test_multiindex \
	test_convolve_image test_qsort_r test_wcs test_big_tables \
	test_dfind test_ctmf test_dsmooth test_dcen3x3 test_simplexy \
	test_fit_wcs test_matchfile

# test_quadfile -- takes a long time!

ALL_TEST_EXTRA_OBJS = 
ALL_TEST_LIBS = $(ANFILES_SLIB)
ALL_TEST_EXTRA_LDFLAGS = 

include $(COMMON)/makefile.tests

DEP_OBJ += $(ALL_TEST_FILES_O)

tests: $(ALL_TEST_FILES)
.PHONY: tests

TEST_DFIND_OBJS := dfind.o
ALL_TEST_EXTRA_OBJS += $(TEST_DFIND_OBJS)
test_dfind: $(TEST_DFIND_OBJS) $(ANFILES_SLIB)

TEST_CTMF_OBJS := ctmf.o
ALL_TEST_EXTRA_OBJS += $(TEST_CTMF_OBJS)
test_ctmf: $(TEST_CTMF_OBJS)

TEST_DSMOOTH_OBJS := dsmooth.o
ALL_TEST_EXTRA_OBJS += $(TEST_DSMOOTH_OBJS)
test_dsmooth: $(TEST_DSMOOTH_OBJS)

test_dcen3x3: dcen3x3.o
ALL_TEST_EXTRA_OBJS += dcen3x3.o

test_simplexy: $(SIMPLEXY_OBJ) $(ANFILES_SLIB)
ALL_TEST_EXTRA_OBJS += $(SIMPLEXY_OBJ)

NORMAL_TESTS := test_big_tables test_qsort_r \
	test_convolve_image test_multiindex test_errors test_sip-utils \
	test_anwcs test_wcs test_fitstable test_fitsbin \
	test_fitsioutils test_xylist test_rdlist test_bl test_bt test_endian \
	test_healpix test_log test_ioutils test_scamp_catalog test_starutil \
	test_svd test_fit_wcs test_quadfile

$(NORMAL_TESTS): $(ANFILES_SLIB)

### /TESTS

../qfits-an/libqfits.a:
	(cd ../qfits-an && make libqfits.a)

# we'll just tack this on here... It depends on cairo, libjpeg, etc,
# so it's not part of the anX.a libraries.
ifeq ($(MAKECMDGOALS),cairoutils.o)
  include cairoutils.dep
endif

.PHONY: clean

clean:
	rm -f $(ANUTILS_LIB_FILE) $(ANFILES_LIB_FILE) $(ANBASE_LIB_FILE) \
		$(ALL_OBJ) $(DEPS) deps cairoutils.o \
		grab-stellarium-constellations \
		$(PROGS) $(MAIN_PROGS) $(ALL_TARGETS) $(ALL_TESTS_CLEAN) \
		cairoutils.dep makefile.os-features *.o *~ *.dep _util$(PYTHON_SO_EXT) util_wrap.c util.py deps \
		os-features.log os-features-makefile.log \
		os-features-test-netpbm os-features-test-netpbm-make \
		report.txt

check:
	echo PYTHON_SO_EXT is $(PYTHON_SO_EXT)

# Try to compensate for clang's lack of -MG to handle generated header
# files by making everything depend explicitly on os-features-config.h.

# the bar "|" indicates an "order-only" dependency: must be built
# before, but doesn't imply a rebuild is necessary.
# Should this perhaps go on the .dep files rather than the .c?
DEP_SRC := $(subst .o,.c,$(DEP_OBJ))
$(DEP_SRC): | $(INCLUDE_DIR)/os-features-config.h
os-features.h: | $(INCLUDE_DIR)/os-features-config.h
$(DEP_OBJ): $(INCLUDE_DIR)/os-features-config.h

ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),reconfig)
    ifneq ($(MAKECMDGOALS),makefile.os-features)
      include $(COMMON)/makefile.deps
    endif
  endif
endif
