.. _code:

Astrometry.net code structure
=============================

This is meant to be an introduction to what parts of the codebase run
during a solve.

*blind/solve-field.c*
---------------------

* parses command-line args
* chooses output filenames
* downloads URL inputs
* decides whether inputs are FITS x,y lists or images
* calls *augment_xylist()* to produce *.axy file
* runs *astrometry-engine* to actually do the solve
* produces plots (*-objs.png, *-ngc.png, etc)

*blind/augment-xylist.c*
------------------------

A field to solve is encapsulated in an "axy" file, which is a FITS
binary table containing X,Y star positions, and well as FITS header
cards that describe the command-line arguments and other information
we have about the image.  "axy" is short for "augmented x,y list", and
we often abbreviate "x,y list" to "xylist".  *augment-xylist.c*
creates these "axy" files.

* run *image2pnm.py* to uncompress and convert images to PNM.
* for non-FITS images, run *ppmtopgm* and *an-pnmtofits* to produce FITS
* run *image2xy* (or SourceExtractor) to generate list of (x,y) star coordinates (xylist)
* for FITS files, run *fits2fits.py* to clean file
* run *removelines.py* to remove lines of sources from the xylist
* run *resort_xylist()* to sort by a combination of brightness and background
* run *uniformize.py* to select a spatially uniform subset of stars
* add headers to xylist to create axy file

*blind/engine-main.c*
---------------------

This is the *astrometry-engine* executable.

* reads *astrometry.cfg* file
* finds index files
* reads axy file
* runs *engine_run_job()* to actually do the solve

*blind/engine.c*
----------------

*engine_run_job()*

* parses axy file
* based on range of image scales, selects index files to use
* calls *blind_run()*

*blind/blind.c*
---------------

*blind_run()*

Runs a set of fields with a set of index files.

* reads xylist
* runs any WCS headers to verify (solve-field --verify)
* depending on whether running with *inparallel* or not, loads one or all index files and calls *solve_fields()*
* records good matches that are found (writes WCS, rdls, match, corr files)

*solve_fields()*

* calls *solver_preprocess_field()*
* calls *solver_run()*

*blind/solver.c*
----------------

Runs a single field with a set of index files.

*solver_run()*

* load index files
* compute scale ranges of field and index files
* looks at pairs of stars A,B forming the "backbone" of the quadrangle, precomputing geometry and deciding which stars can be C,D
* adds one star at a time, forming all quadrangles where that star is A,B or C,D, and for each index, calls *add_stars()*

*add_stars()*

* select stars that will form the quadrangle (or triangle or pentagon)
* calls *TRY_ALL_CODES()* = *try_all_codes()*

*try_all_codes()*

* tests permutations of the C,D stars that are valid (satisfy Cx<Dx
  constraints), with different parities
* calls *try_all_codes_2*

*try_all_codes_2()*

* tries different permutations of A,B stars
* calls *try_permutations()*

*try_permutations()*

* recursive
* tries different permutations of C,D stars, checking for cx <= dx constraint
* searches code KD-tree for matches, calls *resolve_matches()* if found

*resolve_matches()*

* given a code match between a field quadrangle and the index,
* looks up the index star numbers forming that quadrangle (in the quadfile)
* retrieves the index star RA,Dec positions for these stars (in the star KD-tree)
* fits a TAN projection to the matched quadrangle
* calls *solver_handle_hit()*

*solver_handle_hit()*

* calls *verify_hit()* to confirm the match
* if matched, calls *solver_tweak2()* to compute SIP coefficients

*blind/verify.c*
----------------

*verify_hit()*

* searches for stars within the field in the star KD-tree
* calls *real_verify_star_lists()* to do the model comparison between true match and false match.





