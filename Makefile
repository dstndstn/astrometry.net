# This file is part of the Astrometry.net suite.
# Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
# Copyright 2010, 2011, 2012 Dustin Lang.
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

#
# This is where programs and data files will be installed.
#
INSTALL_DIR ?= /usr/local/astrometry
export INSTALL_DIR


# Turn off optimisation?  If the following line is commented-out, the default
# is to turn optimization on.  See util/makefile.common for details.
#export OPTIMIZE = no

all:

BASEDIR := .
COMMON := $(BASEDIR)/util

.PHONY: Makefile $(COMMON)/makefile.qfits
include $(COMMON)/makefile.common
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.cfitsio

.PHONY: all
all: README
	$(MAKE) remake-qfits
	$(MAKE) -C gsl-an
	$(MAKE) -C util
	$(MAKE) -C libkd
	$(MAKE) -C blind

# Targets that require extra libraries
extra:
	$(MAKE) -C util
	$(MAKE) -C blind cairo

# Targets that require SWIG
py:
	$(MAKE) -C util pyutil
	$(MAKE) -C util cairoutils.o
	$(MAKE) -C blind pyplotstuff
	$(MAKE) -C libkd pyspherematch

install: report.txt
	mkdir -p $(INSTALL_DIR)/data
	mkdir -p $(INSTALL_DIR)/bin
	mkdir -p $(INSTALL_DIR)/doc
	mkdir -p $(INSTALL_DIR)/include
	mkdir -p $(INSTALL_DIR)/lib
	mkdir -p $(INSTALL_DIR)/examples
	mkdir -p $(INSTALL_DIR)/python/astrometry
	mkdir -p $(INSTALL_DIR)/ups
	cp ups/astrometry_net.table-dist $(INSTALL_DIR)/ups/astrometry_net.table
	cp __init__.py $(INSTALL_DIR)/python/astrometry
	cp CREDITS GETTING-INDEXES LICENSE README $(INSTALL_DIR)/doc
	cp report.txt $(INSTALL_DIR)/doc
	cp demo/* $(INSTALL_DIR)/examples
	$(MAKE) -C util  install
	$(MAKE) -C libkd install
	$(MAKE) -C qfits-an install
	$(MAKE) -C blind install
	@echo
	@echo The following command may fail if you don\'t have the cairo, netpbm, and
	@echo png libraries and headers installed.  You will lose out on some eye-candy
	@echo but will still be able to solve images.
	@echo
	-$(MAKE) -C blind install-extra

## FIXME -- pyfits is no longer bundled -- but we could still include a makefile target to retrieve, build, and install it... maybe
#cp pyfits/*.py $(INSTALL_DIR)/python/pyfits
# (cd pyfits-2.4.0 && python setup.py install --install-base=stage --install-lib=stage/lib --install-scripts=stage/bin --install-data=stage/data --install-headers=stage/include && cp -r stage/lib/pyfits $(INSTALL_DIR)/python/pyfits)

install-indexes:
	mkdir -p $(INSTALL_DIR)/data
	@for x in `ls index-*.tar.bz2 2>/dev/null`; do \
		echo Installing $$x in $(INSTALL_DIR)/data...; \
		echo tar xvjf $$x -C $(INSTALL_DIR)/data; \
		tar xvjf $$x -C $(INSTALL_DIR)/data; \
	done
	@for x in `ls index-*.tar.gz 2>/dev/null`; do \
		echo Installing $$x in $(INSTALL_DIR)/data...; \
		echo tar xvzf $$x -C $(INSTALL_DIR)/data; \
		tar xvzf $$x -C $(INSTALL_DIR)/data; \
	done

upgrade-indexes:
	@echo
	@echo
	@echo "Warning: this process will modify the index files that you downloaded"
	@echo "and installed (in the directory $(INSTALL_DIR)/data -- if that's not"
	@echo "correct then edit the INSTALL_DIR variable in the Makefile)"
	@echo
	@echo "This process is not reversible, so you will not be able to use the"
	@echo "old version of the code with the new version of the index files."
	@echo
	@echo "If you want to continue experimenting with the old code, please quit"
	@echo "this process (control-C now), copy the index files in the directory"
	@echo "$(INSTALL_DIR)/data to some safe place, then re-run this command."
	@echo
	@echo
	@echo
	@echo "Waiting 10 seconds for you to read the message above..."
	@echo
	sleep 10
	@echo
	@echo "All right, you had your chance.  Proceeding."
	@echo
	$(MAKE) -C libkd
	@echo
	@echo

	@echo "Upgrading:"
	@ls -1 $(INSTALL_DIR)/data/index-*.skdt.fits $(INSTALL_DIR)/data/index-*.ckdt.fits
	@echo
	@echo "Waiting 5 seconds for you to read that..."
	sleep 5
	@echo
	@echo "Hold on to your socks."
	@echo
	@for x in `ls $(INSTALL_DIR)/data/index-*.skdt.fits $(INSTALL_DIR)/data/index-*.ckdt.fits 2>/dev/null`; do \
		echo "Upgrading $$x in $(INSTALL_DIR)/data ..."; \
		echo; \
		echo ./libkd/fix-bb "$$x" "$$x.tmp"; \
		./libkd/fix-bb "$$x" "$$x.tmp"; \
		if [ $$? -eq 0 ]; then \
			echo mv "$$x.tmp" "$$x"; \
			mv "$$x.tmp" "$$x"; \
		elif [ $$? -eq 1 ]; then \
			echo; \
		else \
			echo; \
			echo "Command failed.  Aborting."; \
			echo; \
			break; \
		fi; \
	done

reconfig:
	-rm -f util/os-features-config.h util/makefile.os-features
.PHONY: reconfig

RELEASE_VER := 0.38
SP_RELEASE_VER := 0.3
RELEASE_DIR := astrometry.net-$(RELEASE_VER)
RELEASE_SVN	:= svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)/astrometry
RELEASE_SUBDIRS := cfitsio qfits-an gsl-an util libkd blind demo data etc ups

README: README.in
	$(SED) 's/$$VERSION/$(RELEASE_VER)/g' $< > $@

release:
	-rm -R $(RELEASE_DIR) $(RELEASE_DIR).tar $(RELEASE_DIR).tar.gz $(RELEASE_DIR).tar.bz2
	svn export -N $(RELEASE_SVN) $(RELEASE_DIR)
	for x in $(RELEASE_SUBDIRS); do \
		svn export $(RELEASE_SVN)/$$x $(RELEASE_DIR)/$$x; \
	done
	tar cf $(RELEASE_DIR).tar $(RELEASE_DIR)
	gzip --best -c $(RELEASE_DIR).tar > $(RELEASE_DIR).tar.gz
	bzip2 --best $(RELEASE_DIR).tar

# spherematch-only release
SP_RELEASE_DIR := pyspherematch-$(SP_RELEASE_VER)
SP_RELEASE_SVN	:= svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER)/astrometry
SP_RELEASE_SUBDIRS := gsl-an qfits-an util libkd data
SP_RELEASE_REMOVE := FAQ GETTING-INDEXES
SP_ONLY := pyspherematch-only

release-pyspherematch:
	-rm -R $(SP_RELEASE_DIR) $(SP_RELEASE_DIR).tar $(SP_RELEASE_DIR).tar.gz $(SP_RELEASE_DIR).tar.bz2
	svn export -N $(SP_RELEASE_SVN) $(SP_RELEASE_DIR)
	for x in $(SP_RELEASE_SUBDIRS); do \
		svn export $(SP_RELEASE_SVN)/$$x $(SP_RELEASE_DIR)/$$x; \
	done

	# replace, add and remove files from spherematch-only release
	cp -r $(SP_ONLY)/* $(SP_RELEASE_DIR)
	for x in $(SP_RELEASE_REMOVE); do \
		rm -v $(SP_RELEASE_DIR)/$$x; \
	done

	tar cf $(SP_RELEASE_DIR).tar $(SP_RELEASE_DIR)
	gzip --best -c $(SP_RELEASE_DIR).tar > $(SP_RELEASE_DIR).tar.gz
	bzip2 --best $(SP_RELEASE_DIR).tar


tag-release:
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)

retag-release:
	-svn rm svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER) \
		-m "Remove old release tag in preparation for re-tagging"
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)


tag-release-pyspherematch:
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER) -m "tag pyspherematch verion $(SP_RELEASE_VER) "
	@echo
	@echo version in $(SP_ONLY)/libkd/setup.py :
	@grep version $(SP_ONLY)/libkd/setup.py 
	@echo version in $(SP_ONLY)/README
	@grep wget $(SP_ONLY)/README


retag-release-pyspherematch:
	-svn rm svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER) \
		-m "Remove old release tag in preparation for re-tagging"
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER)  -m "tag pyspherematch version $(SP_RELEASE_VER) "

SNAPSHOT_SVN := svn+ssh://astrometry.net/svn/trunk/src/astrometry
SNAPSHOT_SUBDIRS := $(RELEASE_SUBDIRS)

snapshot:
	$(AN_SHELL) ./make-snapshot.sh $(SNAPSHOT_SVN) $(shell svn info $(SNAPSHOT_SVN) | $(AWK) -F": " /^Revision/'{print $$2}') "$(SNAPSHOT_SUBDIRS)"

test:
	$(MAKE) -C blind test
	$(MAKE) -C util  test
	$(MAKE) -C libkd test

clean:
	$(MAKE) -C util clean
	-$(MAKE) -C cfitsio distclean
	-$(MAKE) -C qfits-an clean
	-rm __init__.pyc
	$(MAKE) -C gsl-an clean
	$(MAKE) -C libkd clean
	$(MAKE) -C blind clean

realclean:
	$(MAKE) -C util realclean
	-$(MAKE) -C cfitsio distclean
	-$(MAKE) -C qfits-an clean
	-rm __init__.pyc
	$(MAKE) -C gsl-an clean
	$(MAKE) -C libkd realclean
	$(MAKE) -C blind realclean

TAGS:
	etags -I `find . -name "*.c" -o -name "*.h"`

tags:
	ctags-exuberant --fields=+aiKS --c++-kinds=+p --extra=+q -I --file-scope=no -R *

report:
	-uname -m
	-uname -a
	-uname -p
	-uname -s
	@echo "CC is $(CC)"
	-which $(CC)
	-$(CC) --version
	-$(MAKE) --version
	-$(CC) -dM -E - < /dev/null
	-cat /proc/cpuinfo
	-sysctl -a kern.ostype kern.osrelease kern.version kern.osversion hw.machine hw.model hw.ncpu hw.byteorder hw.physmem hw.cpufrequency hw.memsize hw.optional.x86_64 hw.cpu64bit_capable machdep.cpu.brand_string
	-free
	@echo "SHAREDLIBFLAGS_DEF: $(SHAREDLIBFLAGS_DEF)"
	@echo "FLAGS_DEF: $(FLAGS_DEF)"
	@echo "CFLAGS_DEF: $(CFLAGS_DEF)"
	@echo "LDFLAGS_DEF: $(LDFLAGS_DEF)"
	-python -V
	@echo "PYTHONPATH: $${PYTHONPATH}"
	@echo "PATH: $${PATH}"
	@echo "pkg-config --cflags cairo:"
	-pkg-config --cflags cairo
	@echo "pkg-config --libs cairo: "
	-pkg-config --libs cairo

report.txt: Makefile
	$(MAKE) report > $@


.SUFFIXES:            # Delete the default suffixes
#.SUFFIXES: .c .o .h   # Define our suffix list
