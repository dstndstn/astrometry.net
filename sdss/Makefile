all: _cutils.so cutils.py

_cutils.so cutils.py: setup.py cutils.i
	python setup.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_cutils.so .

