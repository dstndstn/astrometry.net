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

BASEDIR := ../..
COMMON := $(BASEDIR)/util

.PHONY: all
all:

include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils

LDFLAGS := $(LDFLAGS_DEF)
LDFLAGS += -lm -lpthread

CFLAGS := $(CFLAGS_DEF)

CFLAGS += $(ANUTILS_INC)

EXECS := watcher simple-daemon

all: $(EXECS)

$(ANUTILS_LIB): $(REMAKE_ANUTILS)

watcher: watcher.o \
		$(ANUTILS_LIB)

simple-daemon: simple-daemon.o

.PHONY: tags
tags:
	etags `find . -name "*.h" -o -name "*.c"`

ALL_OBJS := watcher.o

DEP_OBJ := $(ALL_OBJS)

include $(COMMON)/makefile.deps

.PHONY: allclean realclean clean

allclean: clean
	rm -f *.o *~ *.dep 

realclean: allclean

clean:
	rm -f $(EXECS) $(ALL_OBJS) \
		$(DEPS) deps


