**************************
Astrometry.net code README
**************************


| Copyright 2006-2010 Michael Blanton, David W. Hogg, Dustin Lang, Keir Mierle and Sam Roweis.
| Copyright 2011-2013 Dustin Lang and David W. Hogg.

This code is accompanied by the paper:

Lang, D., Hogg, D. W.; Mierle, K., Blanton, M., & Roweis, S., 2010,
Astrometry.net: Blind astrometric calibration of arbitrary
astronomical images, Astronomical Journal 137, 1782–1800.
http://arxiv.org/abs/0910.2233

The original purpose of this code release was to back up the claims in
the paper in the interest of scientific repeatability.  Over the
years, it has become more robust and usable for a wider audience, but
it's still neither totally easy nor bug-free.

This release includes a snapshot of all of the components of our
current research code, including routines to:

* Convert raw USNO-B and Tycho2 into FITS format for easier use
* Uniformize, deduplicate, and cut the FITSified catalogs
* Build index files from these cuts
* Solve the astrometry of images using these index files

The code includes:

* A simple but powerful HEALPIX implementation
* The QFITS library with several modifications
* libkd, a compact and high-performance kdtree library

The code requires *index* files, processed from an astrometric
reference catalog such as USNO-B1 or 2MASS.  (Or, more recently,
Gaia.)  We have released several of these; see
:ref:`getting-index-files`.

Installing
==========

See :ref:`build`.

.. _getting-index-files:

Getting Index Files
===================

See <http://data.astrometry.net/>_ for descriptions of the available
index files.

Each index file is designed to solve images within a narrow range of
scales.  The index files for small (angular size) images are rather
large files, so you probably only want to grab the index files
required for the images you wish to solve.  If you grab extra index
files, the solver will run more slowly, but the results should be the
same.

The files are named like *index-41XX.fits* or *index-52XX-YY.fits*.
*XX* is the "scale", *YY* is the "healpix" number.  These are called
the "4100-series" or "5200-series" index files.

Each index file contains a large number of "skymarks" (landmarks for
the sky) that allow our solver to identify your images.  The skymarks
contained in each index file have sizes (diameters) within a narrow
range.  You probably want to download index files whose quads are,
say, 10% to 100% of the sizes of the images you want to solve.

For example, let's say you have some 1-degree square images.  You
should grab index files that contain skymarks of size 0.1 to 1 degree,
or 6 to 60 arcminutes.  Referring to the table below, you should grab
index files 5203 through 5209.  You might find that the same number of
fields solve, and faster, using just one or two of the index files in
the middle of that range - in our example you might try 5205, 5206 and
5207.

For reference, we used index files 202 alone for our SDSS tests (13x9
arcmin fields); these are the same scale is the new 5202 files.

The medium-angle index files are split into 12 "healpix" tiles; each
one covers 1/12th of the sky.  The small-sized ones are split into 48
healpixes.   See the maps here; you might not need all of them.
https://github.com/dstndstn/astrometry.net/blob/master/util/hp.png
https://github.com/dstndstn/astrometry.net/blob/master/util/hp2.png

+-----------------------+-----------------------------------------+
| Index Filename        | Range of skymark diameters (arcminutes) |
+=======================+=========================================+
| ``index-4119.fits``   |      1400--2000                         |
+-----------------------+-----------------------------------------+
| ``index-4118.fits``   |      1000--1400                         |
+-----------------------+-----------------------------------------+
| ``index-4117.fits``   |       680--1000                         |
+-----------------------+-----------------------------------------+
| ``index-4116.fits``   |       480--680                          |
+-----------------------+-----------------------------------------+
| ``index-4115.fits``   |       340--480                          |
+-----------------------+-----------------------------------------+
| ``index-4114.fits``   |       240--340                          |
+-----------------------+-----------------------------------------+
| ``index-4113.fits``   |       170--240                          |
+-----------------------+-----------------------------------------+
| ``index-4112.fits``   |       120--170                          |
+-----------------------+-----------------------------------------+
| ``index-4111.fits``   |        85--120                          |
+-----------------------+-----------------------------------------+
| ``index-4110.fits``   |        60---85                          |
+-----------------------+-----------------------------------------+
| ``index-4109.fits``   |        42--60                           |
+-----------------------+-----------------------------------------+
| ``index-4108.fits``   |        30--42                           |
+-----------------------+-----------------------------------------+
| ``index-4107-*.fits`` |        22--30                           |
+-----------------------+-----------------------------------------+
| ``index-5206-*.fits`` |        16--22                           |
+-----------------------+-----------------------------------------+
| ``index-5205-*.fits`` |        11--16                           |
+-----------------------+-----------------------------------------+
| ``index-5204-*.fits`` |         8--11                           |
+-----------------------+-----------------------------------------+
| ``index-5203-*.fits`` |         5.6--8.0                        |
+-----------------------+-----------------------------------------+
| ``index-5202-*.fits`` |         4.0--5.6                        |
+-----------------------+-----------------------------------------+
| ``index-5201-*.fits`` |         2.8--4.0                        |
+-----------------------+-----------------------------------------+
| ``index-5200-*.fits`` |         2.0--2.8                        |
+-----------------------+-----------------------------------------+

Download the index files you need and then either:

* Copy the files to the ``data`` directory wherever you installed the
  Astrometry.net code (``INSTALL_DIR/data``, perhaps
  ``/usr/local/astrometry/data``); OR

* Copy the files to the top-level (astrometry-$VERSION) source
  directory, and run::

      $ make install-indexes

Next, you can (optionally) configure the solver by editing the file::

   INSTALL_DIR/etc/astrometry.cfg



Big-Endian Machines
-------------------

Most CPUs these days are little-endian.  If you have an Intel or AMD
chip, you can skip this section.  The most common big-endian CPU in
recent times is the PowerPC used in Macs.  (I am leaving that previous
sentence there for the amusement of people old enough to remember
that.)  In more recent years, some ARM architecture chips are also
big-endian (but Macs with the "Apple silicon" M1/2 chips are run in
little-endian mode).  If you have one of these, read on.

The index files we are distributing are for little-endian machines.
For big-endian machines, you must do the following::

    cd /usr/local/astrometry/data
    for f in index-*.fits; do
      fits-flip-endian -i $f -o flip-$f -e 1 -s 4 -e 3 -s 4 -e 4 -s 2 -e 5 -s 8 -e 6 -s 2 -e 8 -s 4 -e 9 -s 4 -e 10 -s 8 -e 11 -s 4
      for e in 0 2 7; do
        modhead flip-$f"[$e]" ENDIAN 01:02:03:04
      done
    done

assuming ``fits-flip-endian`` and ``modhead`` are in your path.  The files
``flip-index-*.fits`` will contain the flipped index files.

If that worked, you can swap the flipped ones into place (while
saving the originals) with::

    cd /usr/local/astrometry/data
    mkdir -p orig
    for f in index-*.fits; do
      echo "backing up $f"
      mv -n $f orig/$f
      echo "moving $f into place"
      mv -n flip-$f $f
    done

Solving
=======

Finally, solve some fields.

(If you didn't build the plotting commands, add "--no-plots" to the
command lines below.)

(These lists of index files have not been updated; usually replacing
"2xx" by "42xx" should work; for some of them the exact set that will
solve has changed.)

If you have any of index files 4112 to 4118 (213 to 218)::

   $ solve-field --scale-low 10 demo/apod4.jpg

If you have any of index files 4115 to 4119 (219)::

   $ solve-field --scale-low 45 demo/apod5.jpg

If you have any of index files 4110 to 4114::

   $ solve-field --scale-low 1 demo/apod3.jpg

If you have any of index files 5206, or 4107 to 4111::

   $ solve-field --scale-low 1 demo/apod2.jpg

If you have any of index files 5203 to 5205::

   $ solve-field apod1.jpg

If you have any of index files 5200 to 5203::

   $ solve-field demo/sdss.jpg


Copyrights and credits for the demo images are listed in the file
``demo/CREDITS`` .

Note that you can also give solve-field a URL rather than a file as input::

   $ solve-field --out apod1b --downsample 2 http://antwrp.gsfc.nasa.gov/apod/image/0302/ngc2264_croman_c3.jpg

(this one will work with index file 4108).

If you don't have the netpbm tools (eg jpegtopnm), do this instead:

If you have any of index files 4113 to 4118::

   $ solve-field --scale-low 10 demo/apod4.xyls

If you have index 4119::

   $ solve-field --scale-low 30 demo/apod5.xyls

If you have any of index files 4110 to 4114::

   $ solve-field --scale-low 1 demo/apod3.xyls

If you have any of index files 4107 to 4111::

   $ solve-field --scale-low 1 demo/apod2.xyls

If you have any of index files 5203 to 5205::

   $ solve-field demo/apod1.xyls

If you have any of index files 5200 to 5203::

   $ solve-field demo/sdss.xyls


Output files
------------

+--------------------+-------------------------------------------------------------+
|   <base>-ngc.png   |  an annotation of the image.                                |
+--------------------+-------------------------------------------------------------+
|   <base>.wcs       |  a FITS WCS header for the solution.                        |
+--------------------+-------------------------------------------------------------+
|   <base>.new       |  a new FITS file containing the WCS header.                 |
+--------------------+-------------------------------------------------------------+
|   <base>-objs.png  |  a plot of the sources (stars) we extracted from            |
|                    |  the image.                                                 |
+--------------------+-------------------------------------------------------------+
|   <base>-indx.png  |  sources (red), plus stars from the index (green),          |
|                    |  plus the skymark ("quad") used to solve the                |
|                    |  image.                                                     |
+--------------------+-------------------------------------------------------------+
|   <base>-indx.xyls |  a FITS BINTABLE with the pixel locations of                |
|                    |  stars from the index.                                      |
+--------------------+-------------------------------------------------------------+
|   <base>.rdls      |  a FITS BINTABLE with the RA,Dec of sources we              |
|                    |  extracted from the image.                                  |
+--------------------+-------------------------------------------------------------+
|   <base>.axy       |  a FITS BINTABLE of the sources we extracted, plus          |
|                    |  headers that describe the job (how the image is            |
|                    |  going to be solved).                                       |
+--------------------+-------------------------------------------------------------+
|   <base>.solved    |  exists and contains (binary) 1 if the field solved.        |
+--------------------+-------------------------------------------------------------+
|   <base>.match     |  a FITS BINTABLE describing the quad match that             |
|                    |  solved the image.                                          |
+--------------------+-------------------------------------------------------------+
|   <base>.corr      |  a FITS BINTABLE describing stars that we think match       |
|                    |  between your image and the reference catalog.              |
+--------------------+-------------------------------------------------------------+
|   <base>.kmz       |  (optional) KMZ file for Google Sky-in-Earth.  You need     |
|                    |  to have "wcs2kml" in your PATH.  See                       |
|                    |  http://code.google.com/p/wcs2kml/downloads/list            |
|                    |  http://code.google.com/p/google-gflags/downloads/list      |
+--------------------+-------------------------------------------------------------+


Tricks and Tips
===============

* To lower the CPU time limit before giving up::

    $  solve-field --cpulimit 30 ...

  will make it give up after 30 seconds.

  (Note, however, that the "backend" configuration file (astrometry.cfg)
  puts a limit on the CPU time that is spent on an image; solve-field
  can reduce this but not increase it.)

* Scale of the image: if you provide bounds (lower and upper limits)
  on the size of the image you are trying to solve, solving can be much
  faster.  In the last examples above, for example, we specified that
  the field is at least 30 degrees wide: this means that we don't need
  to search for matches in the index files that contain only tiny
  skymarks.

  Eg, to specify that the image is between 1 and 2 degrees wide::

    $ solve-field --scale-units degwidth --scale-low 1 --scale-high 2 ...

  If you know the pixel scale instead::

    $ solve-field --scale-units arcsecperpix \
        --scale-low 0.386 --scale-high 0.406 ...

  When you tell solve-field the scale of your image, it uses this to
  decide which index files to try to use to solve your image; each index
  file contains quads whose scale is within a certain range, so if these
  quads are too big or too small to be in your image, there is no need
  to look in that index file.  It is also used while matching quads: a
  small quad in your image is not allowed to match a large quad in the
  index file if such a match would cause the image scale to be outside
  the bounds you specified.  However, all these checks are done before
  computing a best-fit WCS solution and polynomial distortion terms, so
  it is possible (though rare) for the final solution to fall outside
  the limits you specified.  This should only happen when the solution
  is correct, but you gave incorrect inputs, so you shouldn't be
  complaining! :)


* Guess the scale: solve-field can try to guess your image's scale
  from a number of different FITS header values.  When it's right, this
  often speeds up solving a lot, and when it's wrong it doesn't cost
  much.  Enable this with::

    $ solve-field --guess-scale ...

* If you've got big images: you might want to downsample them before
  doing source extraction::

    $ solve-field --downsample 2 ...
    $ solve-field --downsample 4 ...

* Depth.  The solver works by looking at sources in your image,
  starting with the brightest.  It searches for all "skymarks" that can
  be built from the N brightest stars before considering star N+1.  When
  using several index files, it can be much faster to search for many
  skymarks in one index file before switching to the next one.  This
  flag lets you control when the solver switches between index files.
  It also lets you control how much effort the solver puts in before
  giving up - by default it looks at all the sources in your image, and
  usually times out before this finishes.

  Eg, to first look at sources 1-20 in all index files, then sources
  21-30 in all index files, then 31-40::

    $ solve-field --depth 20,30,40 ...

  or::

    $ solve-field --depth 1-20 --depth 21-30 --depth 31-40 ...

  Sources are numbered starting at one, and ranges are inclusive.  If
  you don't give a lower limit, it will take 1 + the previous upper
  limit.  To look at a single source, do::

    $ solve-field --depth 42-42 ...


* Our source extractor sometimes estimates the background badly, so
  by default we sort the stars by brightness using a compromise between
  the raw and background-subtracted flux estimates.  For images without
  much nebulosity, you might find that using the background-subtracted
  fluxes yields faster results.  Enable this by::

    $ solve-field --resort ...


* If you've got big images: you might want to downsample them before
  doing source extraction::

    $ solve-field --downsample 2 ...

  or::

    $ solve-field --downsample 4 ...


* When solve-field processes FITS images, it looks for an existing
  WCS header.  If one is found, it tries to verify that header before
  trying to solve the image all-sky.  You can prevent this with::

    $ solve-field --no-verify ...

  Note that currently solve-field only understands a small subset of
  valid WCS headers: essentially just the TAN projection with a CD
  matrix (not CROT).


* If you don't want the plots to be produced::

    $ solve-field --no-plots ...


* "I know where my image is to within 1 arcminute, how can I tell
  solve-field to only look there?"

  ::

    $ solve-field --ra, --dec, --radius

  Tells it to look within "radius" degrees of the given RA,Dec position.

* To convert a list of pixel coordinates to RA,Dec coordinates::

    $ wcs-xy2rd -w wcs-file -i xy-list -o radec-list

  Where xy-list is a FITS BINTABLE of the pixel locations of sources;
  recall that FITS specifies that the center of the first pixel is pixel
  coordinate (1,1).


* To convert from RA,Dec to pixels::

    $ wcs-rd2xy -w wcs-file -i radec-list -o xy-list


* To make cool overlay plots: see ``plotxy``, ``plot-constellations``.


* To change the output filenames when processing multiple input
  files: each of the output filename options listed below can include
  "%s", which will be replaced by the base output filename.  (Eg, the
  default for --wcs is "%s.wcs").  If you really want a "%" character in
  your output filename, you have to put "%%".

  Outputs include:

  * --new-fits
  * --kmz
  * --solved
  * --cancel
  * --match
  * --rdls
  * --corr
  * --wcs
  * --keep-xylist
  *  --pnm

  also included:

  * --solved-in
  * --verify

* Reusing files between runs:

  The first time you run solve-field, save the source extraction
  results::

    $ solve-field --keep-xylist %s.xy input.fits ...

  On subsequent runs, instead of using the original input file, use the
  saved xylist instead.  Also add ``--continue`` to overwrite any output
  file that already exists.

  ::

    $ solve-field input.xy --continue ...

  To skip previously solved inputs (note that this assumes single-HDU
  inputs)::

    $ solve-field --skip-solved ...


Optimizing the code
-------------------

Here are some things you can do to make the code run faster:

  * we try to guess "-mtune" settings that will work for you; if we're
    wrong, you can set the environment variable ARCH_FLAGS before
    compiling:

      $ ARCH_FLAGS="-mtune=nocona" make

    You can find details in the gcc manual:
      http://gcc.gnu.org/onlinedocs/

    You probably want to look in the section:
      "GCC Command Options"
         -> "Hardware Models and Configurations"
             -> "Intel 386 and AMD x86-64 Options"

    http://gcc.gnu.org/onlinedocs/gcc-4.3.0/gcc/i386-and-x86_002d64-Options.html#i386-and-x86_002d64-Options


What are all these programs?
----------------------------

When you "make install", you'll get a bunch of programs in
/usr/local/astrometry/bin.  Here's a brief synopsis of what each one
does.  For more details, run the program without arguments (most of
them give at least a brief summary of what they do).

Image-solving programs:
^^^^^^^^^^^^^^^^^^^^^^^

  * solve-field: main high-level command-line user interface.
  * astrometry-engine: higher-level solver that reads "augmented xylists";
    called by solve-field.
  * augment-xylist: creates "augmented xylists" from images, which
    include star positions and hints and instructions for solving.
  * image2xy: source extractor.

Plotting programs:
^^^^^^^^^^^^^^^^^^

  * plotxy: plots circles, crosses, etc over images.
  * plotquad: draws polygons over images.
  * plot-constellations: annotates images with constellations, bright
    stars, Messier/NGC objects, Henry Draper catalog stars, etc.
  * plotcat: produces density plots given lists of stars.

WCS utilities:
^^^^^^^^^^^^^^

  * new-wcs: merge a WCS solution with existing FITS header cards; can
    be used to create a new image file containing the WCS headers.
  * fits-guess-scale: try to guess the scale of an image based on FITS
    headers.
  * wcsinfo: print simple properties of WCS headers (scale, rotation, etc)
  * wcs-xy2rd, wcs-rd2xy: convert between lists of pixel (x,y) and
    (RA,Dec) positions.
  * wcs-resample: projects one FITS image onto another image.
  * wcs-grab/get-wcs: try to interpret an existing WCS header.

Miscellany:
^^^^^^^^^^^

  * an-fitstopnm: converts FITS images into ugly PNM images.
  * get-healpix: which healpix covers a given RA,Dec?
  * control-program: sample code for how you might use the
    Astrometry.net code in your own software.
  * textfits: converts a text list (eg, CSV) to a FITS binary table.

FITS utilities
^^^^^^^^^^^^^^

  * tablist: list values in a FITS binary table.
  * modhead: print or modify FITS header cards.
  * fitscopy: general FITS image / table copier.
  * tabmerge: combines rows in two FITS tables.
  * liststruc: shows the structure of a FITS file.
  * listhead: prints FITS header cards.
  * imcopy: copies FITS images.
  * imarith: does (very) simple arithmetic on FITS images.
  * imstat: computes statistics on FITS images.
  * fitsgetext: pull out individual header or data blocks from
    multi-HDU FITS files.
  * subtable: pull out a set of columns from a many-column FITS binary
    table.
  * tabsort: sort a FITS binary table based on values in one column.
  * merge-colums: create a FITS binary table that includes columns
    from two input tables.
  * resort-xylist: used by solve-field to sort a list of stars using a
    compromise between background-subtracted and non-background-subtracted
    flux (because our source extractor sometimes messes up the background
    subtraction).
  * fits-flip-endian: does endian-swapping of FITS binary tables.

Index-building programs
^^^^^^^^^^^^^^^^^^^^^^^

* build-astrometry-index: given a FITS binary table with RA,Dec, build
  an index file.  This is the "easy", recent way.  The old way uses
  the rest of these programs:

  * usnobtofits, tycho2tofits, nomadtofits, 2masstofits: convert
    catalogs into FITS binary tables.
  * startree: build a star kdtree from a catalog.
  * hpquads: find a bright, uniform set of N-star features.
  * codetree: build a kdtree from N-star shape descriptors.
  * unpermute-quads, unpermute-stars: reorder index files for
    efficiency.
  * hpsplit: splits a list of FITS tables into healpix tiles


Source lists ("xylists")
------------------------

The solve-field program accepts either images or "xylists" (xyls),
which are just FITS BINTABLE files which contain two columns (float or
double (E or D) format) which list the pixel coordinates of sources
(stars, etc) in the image.

To specify the column names (eg, "XIMAGE" and "YIMAGE")::

  $ solve-field --x-column XIMAGE --y-column YIMAGE ...

Our solver assumes that the sources are listed in order of brightness,
with the brightest sources first.  If your files aren't sorted, you
can specify a column by which the file should be sorted.

::

  $ solve-field --sort-column FLUX ...

By default it sorts with the largest value first (so it works
correctly if the column contains FLUX values), but you can reverse
that by::

  $ solve-field --sort-ascending --sort-column MAG ...

When using xylists, you should also specify the original width and
height of the image, in pixels::

  $ solve-field --width 2000 --height 1500 ...

Alternatively, if the FITS header contains "IMAGEW" and "IMAGEH" keys,
these will be used.

The solver can deal with multi-extension xylists; indeed, this is a
convenient way to solve a large number of fields at once.  You can
tell it which extensions it should solve by::

  $ solve-field --fields 1-100,120,130-200

(Ranges of fields are inclusive, and the first FITS extension is 1, as
per the FITS standard.)

Unfortunately, the plotting code isn't smart about handling multiple
fields, so if you're using multi-extension xylists you probably want
to turn off plotting::

  $ solve-field --no-plots ...


Backend config
--------------

Because we also operate a web service using most of the same software,
the local version of the solver is a bit more complicated than it
really needs to be.  The "solve-field" program takes your input files,
does source extraction on them to produce an "xylist" -- a FITS
BINTABLE of source positions -- then takes the information you
supplied about your fields on the command-line and adds FITS headers
encoding this information.  We call this file an "augmented xylist";
we use the filename suffix ".axy".  "solve-field" then calls the
"backend" program, passing it your axy file.  "backend" reads a config
file (by default /usr/local/astrometry/etc/astrometry.cfg) that describes
things like where to find index files, whether to load all the index
files at once or run them one at a time, how long to spend on each
field, and so on.  If you want to force only a certain set of index
files to load, you can copy the astrometry.cfg file to a local version
and change the list of index files that are loaded, and then tell
solve-field to use this config file::

   $ solve-field --config myastrometry.cfg ...


Source Extractor
----------------
http://www.astromatic.net/software/sextractor

The "Source Extractor" program by Emmanuel Bertin can
be used to do source extraction if you don't want to use our own
bundled "image2xy" program.

NOTE: users have reported that Source Extractor 2.4.4 (available in some
Ubuntu distributions) DOES NOT WORK -- it prints out correct source
positions as it runs, but the "xyls" output file it produces contains
all (0,0).  We haven't looked into why this is or how to work around
it.  Later versions of Source Extractor such as 2.8.6 work fine.

You can tell solve-field to use Source Extractor like this::

  $ solve-field --use-source-extractor ...

By default we use almost all Source Extractor's default settings.  The
exceptions are:

  1) We write a PARAMETERS_NAME file containing:
         X_IMAGE
         Y_IMAGE
         MAG_AUTO

  2) We write a FILTER_NAME file containing a Gaussian PSF with FWHM
     of 2 pixels.  (See solver/augment-xylist.c "filterstr" for the
     exact string.)

  3) We set CATALOG_TYPE FITS_1.0

  4) We set CATALOG_NAME to a temp filename.


If you want to override any of the settings we use, you can use::

  $ solve-field --use-source-extractor --source-extractor-config <se.conf>

In order to reproduce the default behavior, you must::

  1) Create a parameters file like the one we make, and set
     PARAMETERS_NAME to its filename

  2) Set::

  $ solve-field --x-column X_IMAGE --y-column Y_IMAGE \
       --sort-column MAG_AUTO --sort-ascending

  3) Create a filter file like the one we make, and set FILTER_NAME to
     its filename


Note that you can tell solve-field where to find Source Extractor with::

  $ solve-field --use-source-extractor --source-extractor-path <path-to-se-executable>



Workarounds
-----------
* No python

  There are two places we use python: handling images, and filtering source lists
  before solving.

  You can avoid the image-handling code by doing source extraction
  yourself; see the "No netpbm" section below.

  You can avoid filtering FITS files by using the "--no-remove-lines"
  and "--uniformize 0" option to solve-field.

* No netpbm

  We use the netpbm tools (jpegtopnm, pnmtofits, etc) to convert from
  all sorts of image formats to PNM and FITS.

  If you don't have these programs installed, you must do source
  extraction yourself and use "xylists" rather than images as the input
  to solve-field.  See SOURCE EXTRACTOR and XYLIST sections above.

ERROR MESSAGES during compiling
-------------------------------

1. ``/bin/sh: line 1: /dev/null: No such file or directory``

   We've seen this happen on Macs a couple of times.  Reboot and it goes
   away...

2. ``makefile.deps:40: deps: No such file or directory``

   Not a problem.  We use automatic dependency tracking: "make" keeps
   track of which source files depend on which other source files.  These
   dependencies get stored in a file named "deps"; when it doesn't exist,
   "make" tries to rebuild it, but not before printing this message.

3. ::

     os-features-test.c: In function 'main':
     os-features-test.c:23: warning: implicit declaration of function 'canonicalize_file_name'
     os-features-test.c:23: warning: initialization makes pointer from integer without a cast
     /usr/bin/ld: Undefined symbols:
     _canonicalize_file_name
     collect2: ld returned 1 exit status

   Not a problem.  We provide replacements for a couple of OS-specific
   functions, but we need to decide whether to use them or not.  We do
   that by trying to build a test program and checking whether it works.
   This failure tells us your OS doesn't provide the
   canonicalize_file_name() function, so we plug in a replacement.

4. ::

     configure: WARNING: cfitsio: == No acceptable f77 found in $PATH
     configure: WARNING: cfitsio: == Cfitsio will be built without Fortran wrapper support
     drvrfile.c: In function 'file_truncate':
     drvrfile.c:360: warning: implicit declaration of function 'ftruncate'
     drvrnet.c: In function 'http_open':
     drvrnet.c:300: warning: implicit declaration of function 'alarm'
     drvrnet.c: In function 'http_open_network':
     drvrnet.c:810: warning: implicit declaration of function 'close'
     drvrsmem.c: In function 'shared_cleanup':
     drvrsmem.c:154: warning: implicit declaration of function 'close'
     group.c: In function 'fits_get_cwd':
     group.c:5439: warning: implicit declaration of function 'getcwd'
     ar: creating archive libcfitsio.a

   Not a problem; these errors come from cfitsio and we just haven't fixed them.


License
=======

The Astrometry.net code suite is free software licensed under the GNU
GPL, version 2.  See the file LICENSE for the full terms of the GNU
GPL.

The index files come with their own license conditions.  See the file
GETTING-INDEXES for details.

Contact
=======

You can post questions (or maybe even find the answer to your
questions) at https://groups.google.com/u/1/g/astrometry .  If you
post there, it is often very useful to see an example image that
you're working with, so if you are willing to, you could try
submitting one to the https://nova.astrometry.net web service, and
include the link in your post.

