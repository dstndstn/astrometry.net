
Once-Asked Questions
====================

Q: I don't have *numpy*.  What do I do?
---------------------------------------

"import error: no module named numpy"

A: Disable things that require numpy.
-------------------------------------

Some parts of the code need the "numpy" python package.  To disable things that need numpy::

    solve-field --no-fits2fits --no-remove-lines --uniformize 0  [....usual arguments...]


Q: Is there a way to plot a grid of RA and Dec on the images?
-------------------------------------------------------------

A: Yes
------

You'll have to run the "plot-constellations" program
separately.  For example, if you have an image 1.jpg and WCS 1.wcs:

    jpegtopnm 1.jpg | plot-constellations -w 1.wcs -o grid.png -i - -N -C -G 60

will plot an RA,Dec grid with 60-arcminute spacings.  Unfortunately
they're not labelled...

[Note, see *plotann.py* also for more annotation options.]

Q: Is there a way to get out the center of the image (RA,Dec) and pixel scale of the image?
-------------------------------------------------------------------------------------------

A: Yes, with the *wcsinfo* program
----------------------------------

Yes, run the "wcsinfo" program on a WCS file -- it prints out a bunch
of stats, in a form that's meant to be easy to parse by programs (so
it's not particularly friendly for people).  "ra_center" and
"dec_center" (in degrees) and "pixscale" (in arcsec/pixel) are what
you want.

Q: Is there a way to plot N and E vectors on the image?
-------------------------------------------------------

A: Not yet.
-----------


Q: Is there a way to plot a list of your own objects on the image by inputing RA,Dec?
-------------------------------------------------------------------------------------

A: Check out *plotann.py*, or try these older instructions...
-------------------------------------------------------------

Yes -- but it's roundabout...

First, project your RA,Dec objects into pixel x,y positions:

    wcs-rd2xy -w 1.wcs -i your-objs.rd -o your-objs.xy

Then plot them over the image (or annotated image).  There's not
currently a way to label them.

:

    pngtopnm grid.png | plotxy -i your-objs.xy -I - -x 1 -y 1 -s X -C green -b black > objs.png

The "-x 1 -y 1" compensate for the fact that FITS calls the center of
the first pixel (1,1) rather than (0,0).


Q: Would your code work on all-sky images?
------------------------------------------

A: Not very well
----------------

We assume a TAN projection, so all-sky images typically don't work,
but it should certainly be possible with a bit of tweaking, since
all-sky is really a much easier recognition problem!  One thing you
can try, if your image is big enough, is to cut out a small section
near the middle.

Q: I want to build an index from my own catalog.  How do I proceed?
-------------------------------------------------------------------

A: See :ref:`genindex`
----------------------




