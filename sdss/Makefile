.PHONY: all
all: _cutils.so cutils.py

BASEDIR := ..
COMMON := $(BASEDIR)/util
include $(COMMON)/makefile.common

MY_DIR := $(PY_BASE_INSTALL_DIR)/sdss

_cutils.so cutils.py: cutils.i
	python setup.py build_ext -v --inplace --build-temp .

INSTALL_PY := __init__.py common.py dr7.py dr8.py runList-dr8.par \
	dr9.py runList-dr9.par \
	cutils.py _cutils.so

.PHONY: install
install: _cutils.so cutils.py
	@echo Installing in directory '$(MY_DIR)'
	mkdir -p '$(MY_DIR)'
	@for x in $(INSTALL_PY); do \
		echo cp $$x '$(MY_DIR)'; \
		cp $$x '$(MY_DIR)'; \
	done


.PHONY: clean
clean:
	rm -f _cutils.so cutils_wrap.c cutils.py

