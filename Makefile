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

#
# This is where programs and data files will be installed.
#
INSTALL_DIR ?= /usr/local/astrometry
export INSTALL_DIR

all:

BASEDIR := .
COMMON := $(BASEDIR)/util
include $(COMMON)/makefile.common
include $(COMMON)/makefile.qfits

.PHONY: Makefile $(COMMON)/makefile.qfits

all: #$(REMAKE_QFITS)
	$(MAKE) -C util
	$(MAKE) -C libkd
	$(MAKE) -C blind

# Targets that require extra libraries
extra:
	$(MAKE) -C util
	$(MAKE) -C blind cairo

# Targets that support the web service
.PHONY: web
web:
	$(MAKE) -C render
	$(MAKE) -C net/execs

install:
	mkdir -p $(INSTALL_DIR)/data
	mkdir -p $(INSTALL_DIR)/bin
	mkdir -p $(INSTALL_DIR)/doc
	mkdir -p $(INSTALL_DIR)/examples
	mkdir -p $(INSTALL_DIR)/python/astrometry
	cp __init__.py $(INSTALL_DIR)/python/astrometry
	cp pyfits/pyfits.py $(INSTALL_DIR)/python/
	cp pyfits/rec.py $(INSTALL_DIR)/python/
	cp CREDITS GETTING-INDICES LICENSE README $(INSTALL_DIR)/doc
	cp demo/* $(INSTALL_DIR)/examples
	$(MAKE) -C util  install
	$(MAKE) -C libkd install
	$(MAKE) -C blind install
	@echo
	@echo The following command may fail if you don\'t have the cairo, netpbm, and
	@echo png libraries and headers installed.  You will lose out on some eye-candy
	@echo but will still be able to solve images.
	@echo
	-$(MAKE) -C blind install-extra

install-indices:
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

upgrade-indices:
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

	@for x in `ls $(INSTALL_DIR)/data/index-*.{skdt,ckdt}.fits 2>/dev/null`; do \
		echo "Upgrading $$x in $(INSTALL_DIR)/data ..."; \
		echo; \
		echo ./libkd/fix-bb "$$x" "$$x.tmp"; \
		echo; \
		./libkd/fix-bb "$$x" "$$x.tmp"; \
		if [ $$? -ne 0 ]; then \
			echo; \
			echo "Command failed.  Aborting."; \
			echo; \
			break; \
		fi; \
		echo "mv $$x.tmp $$x"; \
		mv "$$x.tmp" "$$x"; \
	done


RELEASE_VER := 0.2-pre
RELEASE_DIR := astrometry.net-$(RELEASE_VER)
RELEASE_SVN	:= svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)
RELEASE_SUBDIRS := cfitsio qfits-an gsl-an util libkd blind demo data pyfits etc

release:
	-rm -R $(RELEASE_DIR)
	svn export -N $(RELEASE_SVN) $(RELEASE_DIR)
	for x in $(RELEASE_SUBDIRS); do \
		svn export $(RELEASE_SVN)/astrometry/$$x $(RELEASE_DIR)/$$x; \
	done
	tar cf $(RELEASE_DIR).tar $(RELEASE_DIR)
	gzip --best -c $(RELEASE_DIR).tar > $(RELEASE_DIR).tar.gz
	bzip2 --best $(RELEASE_DIR).tar

tag-release:
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)

SNAPSHOT_SVN := svn+ssh://astrometry.net/svn/trunk/src/astrometry
#SNAPSHOT_VER := $(shell date "+%Y-%m-%d")
#SNAPSHOT_VER := $(shell svn info $(SNAPSHOT_SVN) | $(AWK) -F": " /^Revision/'{print $$2}')
#SNAPSHOT_DIR := astrometry.net-$(SNAPSHOT_VER)
SNAPSHOT_SUBDIRS := $(RELEASE_SUBDIRS)

snapshot:
	$(AN_SHELL) ./make-snapshot.sh $(SNAPSHOT_SVN) $(shell svn info $(SNAPSHOT_SVN) | $(AWK) -F": " /^Revision/'{print $$2}') "$(SNAPSHOT_SUBDIRS)"

test:
	$(MAKE) -C blind test

clean:
	$(MAKE) -C util clean
	-$(MAKE) -C cfitsio distclean
	-$(MAKE) -C qfits-an distclean
	-rm qfits-an/Makefile
	-rm -R qfits-an/stage
	$(MAKE) -C gsl-an clean
	$(MAKE) -C libkd clean
	$(MAKE) -C blind clean
	-$(MAKE) -C render clean
	-$(MAKE) -C net/execs clean

realclean:
	$(MAKE) -C util realclean
	-$(MAKE) -C cfitsio distclean
	-$(MAKE) -C qfits-an distclean
	-rm -R qfits-an/stage
	$(MAKE) -C gsl-an clean
	$(MAKE) -C libkd realclean
	$(MAKE) -C blind realclean
	-$(MAKE) -C render realclean
	-$(MAKE) -C net/execs realclean

TAGS:
	etags -I `find . -name "*.c" -o -name "*.h"`

tags:
	ctags-exuberant --fields=+aiKS --c++-kinds=+p --extra=+q -I --file-scope=no -R *

report:
	-uname -m
	-uname -a
	@echo "CC is $(CC)"
	-which $(CC)
	-$(CC) --version
	-$(MAKE) --version
	-$(CC) -dM -E - < /dev/null
