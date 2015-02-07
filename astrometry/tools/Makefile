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

BASEDIR := ../..
MAKEFILES_DIR := $(BASEDIR)/makefiles

all:
.PHONY: all

include $(MAKEFILES_DIR)/makefile.common
include $(MAKEFILES_DIR)/makefile.blind
include $(MAKEFILES_DIR)/makefile.netpbm

#include $(MAKEFILES_DIR)/makefile.cfitsio

ifneq ($(MAKECMDGOALS),clean)
    include $(MAKEFILES_DIR)/makefile.os-features
endif

LDFLAGS := $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(BLIND_LIB)
LDLIBS += $(NETPBM_LIB)

SLIB := $(BLIND_SLIB)
SLIB += $(NETPBM_SLIB)

INC := $(BLIND_INC)
INC += $(NETPBM_INC)

CFLAGS := $(CFLAGS_DEF)
CFLAGS += $(BLIND_CFLAGS)
CFLAGS += $(NETPBM_CFLAGS)
CFLAGS += $(INC)

SHAREDLIBFLAGS := $(SHAREDLIBFLAGS_DEF)

SIMPLE_PROGS := an-fitstopnm an-pnmtofits control-program downsample-fits \
	fits-column-merge fits-flip-endian fitsgetext get-healpix get-wcs \
	hpowned hpsplit	query-starkd subtable subwcs wcs-pv2sip wcsinfo wcs-to-tan

OTHER_PROGS := fit-wcs fits-guess-scale image2xy new-wcs resort-xylist \
	tabsort wcs-rd2xy wcs-xy2rd

# MAIN_PROGS := fit-wcs
# MAINX_PROGS := 

CFITS_UTILS := fitscopy imarith imcopy imstat listhead liststruc modhead \
	tablist tabmerge

PROGS := $(SIMPLE_PROGS) $(OTHER_PROGS) $(CFITS_UTILS) fitsverify

# bgsubtract index-to-table pad-file search-index tweak wcs-match


ALL_OBJ := 

all: $(PROGS)

# $(MAIN_PROGS): %: %-main.o $(SLIB)
# ALL_OBJ += $(addsuffix -main.o,$(MAIN_PROGS))

#$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS)

$(SIMPLE_PROGS): %: %.o $(SLIB)
ALL_OBJ += $(addsuffix .o,$(SIMPLE_PROGS))

fit-wcs: fit-wcs-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

fits-guess-scale: fits-guess-scale-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

image2xy: image2xy-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

new-wcs: new-wcs-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

resort-xylist: resort-xylist-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

tabsort: tabsort-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

wcs-xy2rd: wcs-xy2rd-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

wcs-rd2xy: wcs-rd2xy-main.o
	$(CC) -o $@ $(LDFLAGS) $^ $(SLIB) $(LDLIBS)

# control-program: control-program.o $(SLIB)
# ALL_OBJ += control-program.o

# image2xy: image2xy-main.o image2xy-files.o $(CFITS_SLIB) $(SLIB)
# 	$(CC) -o $@ $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)
# 
# fit-wcs: fit-wcs-main.o $(SLIB)
# 	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS)
# 
# ALL_OBJ += image2xy-files.o

# wcs-pv2sip: wcs-pv2sip.o tweak.o $(SLIB)
# ALL_OBJ += wcs-pv2sip.o

$(CFITS_UTILS) :: %: %.o $(CFITS_SLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(CFITS_LIB) $(LDLIBS)

fitsverify: ftverify.c fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c $(CFITS_SLIB)
	$(CC) -DSTANDALONE -trigraphs $(CFITS_INC) -o $@ $^ $(CFITS_LIB) -lm

# $(OLDEXECS) :: %: %.o $(OLDEXECS_OBJS) $(SLIB)

DEP_OBJ := $(ALL_OBJ)
DEP_PREREQS := #$(QFITS_LIB)

#CFLAGS += $(CAIRO_INC)

include $(MAKEFILES_DIR)/makefile.deps

.PHONY: clean
clean:
	rm -f $(PROGS)

# EXECS) $(EXTRA_EXECS) $(SOLVER_EXECS) $(MISC_EXECS) $(PROGS) \
# 		$(PIPELINE) $(PROSPECTUS) $(DEPS) deps $(FITS_UTILS) $(ALL_OBJ) \
# 		$(NODEP_OBJS) plot-constellations fitsverify plotquad plotxy \
# 		$(ALL_EXECS) $(GENERATED_FILES) $(ALL_TESTS_CLEAN) \
# 		$(ENGINE_LIB) $(ENGINE_SO) _plotstuff_c$(PYTHON_SO_EXT) *.o *~ *.dep deps \
# 		plotstuff



