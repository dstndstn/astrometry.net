
.. _nova_api:

Nova.astrometry.net: API
========================

We provide a web-services API that uses JSON to encode the parameters
and results.

We have code-as-documentation in the `client.py
<http://trac.astrometry.net/browser/trunk/src/astrometry/net/client/client.py>`_
file, but will try to keep this documentation up to date as well.

JSON encoding
-------------

The first message you send to the server will be to log in using your
API key.  The formatting steps are:

  * Form the JSON *object*::

       {"apikey": "XXXXXX"}

  * Form the *x-www-form-encoded* string::

       request-json={"apikey": "XXXXXX"}

    which becomes::

	   request-json=%7B%22apikey%22%3A+%22XXXXXX%22%7D

  * Send that to the "login" API URL as a POST request:

       http://nova.astrometry.net/api/login

This form demonstrates how the request must be encoded, and what the result looks like.

.. raw:: html

   <form action="http://staging.astrometry.net/api/login" method="POST">
   <input type="text" name="request-json" size=50 value="{&#34;apikey&#34;: &#34;XXXXXXXX&#34;}" />
   <input type="submit" value="Submit" />
   </form>

Session key
-----------

After logging in with your API key, you will get back a "session key"::

      {"status": "success", "message": "authenticated user: ", "session": "575d80cf44c0aba5491645a6818589c6"}

You have to include that "session" value in all subsequent requests.
The session will remain alive for some period of time, or until you
log out.

Submitting a URL
----------------

API URL::

    http://nova.astrometry.net/api/url_upload

Arguments:

  * ``session``: string, requried.  Your session key, required in all requests
  * ``url``: string, required.  The URL you want to submit to be solved
  * ``allow_commercial_use``: string: "d" (default), "y", "n": licensing terms
  * ``allow_modifications``: string: "d" (default), "y", "n", "sa" (share-alike): licensing terms
  * ``publicly_visible``: string: "y", "n"
  * ``scale_units``: string: "degwidth" (default), "arcminwidth", "arcsecperpix".  The units for the "scale_lower" and "scale_upper" arguments; becomes the "--scale-units" argument to "solve-field" on the server side.
  * ``scale_type``: string, "ul" (default) or "ev".  Set "ul" if you are going to provide "scale_lower" and "scale_upper" arguments, or "ev" if you are going to provide "scale_est" (estimate) and "scale_err" (error percentage) arguments.
  * ``scale_lower``: float.  The lower-bound of the scale of the image.
  * ``scale_upper``: float.  The upper-bound of the scale of the image.
  * ``scale_est``: float.  The estimated scale of the image.
  * ``scale_err``: float, 0 to 100.  The error (percentage) on the estimated scale of the image.
  * ``center_ra``: float, 0 to 360, in degrees.  The position of the center of the image.
  * ``center_dec``: float, -90 to 90, in degrees.  The position of the center of the image.
  * ``radius``: float, in degrees.  Used with ``center_ra``,``center_dec`` to specify that you know roughly where your image is on the sky.
  * ``downsample_factor``: float, >1.  Downsample (bin) your image by this factor before performing source detection.  This often helps with saturated images, noisy images, and large images.  2 and 4 are commonly-useful values.
  * ``tweak_order``: int.  Polynomial degree (order) for distortion correction.  Default is 2.  Higher orders may produce totally bogus results (high-order polynomials are strange beasts).
  * ``use_sextractor``: boolean.  Use the `SourceExtractor <http://www.astromatic.net/software/sextractor>`_ program to detect stars, not our built-in program.
  * ``crpix_center``: boolean.  Set the WCS reference position to be the center pixel in the image?  By default the center is the center of the quadrangle of stars used to identify the image.
  * ``parity``: int, 0, 1 or 2.  Default 2 means "try both".  0 means that the sign of the determinant of the WCS CD matrix is positive, 1 means negative.  The short answer is, "try both and see which one works" if you are interested in using this option.  It results in searching half as many matches so can be helpful speed-wise.
  * ``image_width``: int, only necessary if you are submitting an "x,y list" of source positions.
  * ``image_height``: int, ditto.
  * ``positional_error``: float, expected error on the positions of stars in your image.  Default is 1.

Example:

..   <input type="text" name="request-json1" size=50 value="{&#34;session&#34;: &#34;575d80cf44c0aba5491645a6818589c6&#34;, &#34;url&#34;: &#34;http://apod.nasa.gov/apod/image/1206/ldn673s_block1123.jpg&#34;, &#34;scale_units&#34;: &#34;degwidth&#34;, &#34;scale_lower&#34;: 0.5, &#34;scale_upper: 1.0, &#34;center_ra&#34;: 290, &#34;center_dec&#34;: 11, &#34;radius&#34;: 2.0 }" />

.. raw:: html

   <form action="http://staging.astrometry.net/api/url_upload" method="POST">
   <textarea name="request-json" rows=5 cols=80>
   {"session": "####", "url": "http://apod.nasa.gov/apod/image/1206/ldn673s_block1123.jpg", "scale_units": "degwidth", "scale_lower": 0.5, "scale_upper": 1.0, "center_ra": 290, "center_dec": 11, "radius": 2.0 }
   </textarea>
   <input type="submit" value="Submit" />
   </form>

And you will get back a response such as::

    {"status": "success", "subid": 16714, "hash": "6024b45a16bfb5af7a73735cbabdf2b462c11214"}

The ``subid`` is the Submission number.  The ``hash`` is the ``sha-1`` hash of the contents of the URL you specified.

