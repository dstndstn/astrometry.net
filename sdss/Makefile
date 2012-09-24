all: _cutils.so cutils.py

INSTALL_DIR ?= /usr/local/astrometry
MY_DIR := $(INSTALL_DIR)/python/astrometry/sdss

_cutils.so cutils.py: setup.py cutils.i
	python setup.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_cutils.so .

INSTALL_PY := __init__.py common.py dr7.py dr8.py runList-dr8.par \
	dr9.py runList-dr9.par \
	cutils.py _cutils.so

.PHONY: install
install: _cutils.so cutils.py
	@echo Installing in directory $(MY_DIR)
	mkdir -p $(MY_DIR)
	@for x in $(INSTALL_PY); do \
		echo cp $$x $(MY_DIR); \
		cp $$x $(MY_DIR); \
	done


.PHONY: clean
clean:
	rm -f _cutils.so cutils_wrap.c cutils.py

