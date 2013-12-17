
Change Log:
===========

Version 0.46:
-------------

* Makefile revamp.  Now possible to use system GSL rather than our
  shipped subset, by defining SYSTEM_GSL=yes::

    make SYSTEM_GSL=yes

* Move away from *qfits*, toward *an-qfits*; reorganize *qfits-an*
  directory to be more like the other source directories.

* Web api: add *invert* option

* Add more Intermediate World Coords options to WCS classes

Version 0.47:
-------------

* *solve-field*: add "focalmm" as a supported image scale type.

