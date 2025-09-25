
.. _nova_api:

Nova.astrometry.net: API
========================

We provide a web-services API that uses JSON to encode the parameters
and results.

We have code-as-documentation in the `client.py <https://github.com/dstndstn/astrometry.net/blob/main/net/client/client.py>`_
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

       https://nova.astrometry.net/api/login

Using the `requests` library, this looks like::

    import requests
    import json
    R = requests.post('https://nova.astrometry.net/api/login', data={'request-json': json.dumps({"apikey": "XXXXXXXX"})})
    print(R.text)
    >> u'{"status": "success", "message": "authenticated user: ", "session": "0ps9ztf2kmplhc2gupfne2em5qfn0joy"}'

Note the *request-json=VALUE*: we're not sending raw JSON, we're sending the
JSON-encoded string as though it were a text field called *request-json*.

All the other API calls use this same pattern.  It may be more complicated on
the client side, but it makes testing and the server side easier.

This form demonstrates how the request must be encoded, and what the result looks like.

.. raw:: html

   <form action="https://nova.astrometry.net/api/login" method="POST">
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

API URL:

    https://nova.astrometry.net/api/url_upload

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
  * ``use_sextractor``: boolean.  Use the `SourceExtractor <https://www.astromatic.net/software/sextractor>`_ program to detect stars, not our built-in program.
  * ``crpix_center``: boolean.  Set the WCS reference position to be the center pixel in the image?  By default the center is the center of the quadrangle of stars used to identify the image.
  * ``parity``: int, 0, 1 or 2.  Default 2 means "try both".  0 means that the sign of the determinant of the WCS CD matrix is positive, 1 means negative.  The short answer is, "try both and see which one works" if you are interested in using this option.  It results in searching half as many matches so can be helpful speed-wise.
  * ``image_width``: int, only necessary if you are submitting an "x,y list" of source positions.
  * ``image_height``: int, ditto.
  * ``positional_error``: float, expected error on the positions of stars in your image.  Default is 1.

Example:

..   <input type="text" name="request-json1" size=50 value="{&#34;session&#34;: &#34;575d80cf44c0aba5491645a6818589c6&#34;, &#34;url&#34;: &#34;https://apod.nasa.gov/apod/image/1206/ldn673s_block1123.jpg&#34;, &#34;scale_units&#34;: &#34;degwidth&#34;, &#34;scale_lower&#34;: 0.5, &#34;scale_upper: 1.0, &#34;center_ra&#34;: 290, &#34;center_dec&#34;: 11, &#34;radius&#34;: 2.0 }" />

.. raw:: html

   <form action="https://nova.astrometry.net/api/url_upload" method="POST">
   <textarea name="request-json" rows=5 cols=80>
   {"session": "####", "url": "https://apod.nasa.gov/apod/image/1206/ldn673s_block1123.jpg", "scale_units": "degwidth", "scale_lower": 0.5, "scale_upper": 1.0, "center_ra": 290, "center_dec": 11, "radius": 2.0 }
   </textarea>
   <input type="submit" value="Submit" />
   </form>

And you will get back a response such as::

    {"status": "success", "subid": 16714, "hash": "6024b45a16bfb5af7a73735cbabdf2b462c11214"}

The ``subid`` is the Submission number.  The ``hash`` is the ``sha-1`` hash of the contents of the URL you specified.



Submitting a file
-----------------

Submitting a file is somewhat complicated, because it has to be formatted
as a *multipart/form-data* form.  This is exactly the same way an HTML form
with text fields and a file upload field would do it.  If you are working in
python, it will probably be helpful to look at the *client.py* code.

Specifically, the *multipart/form-data* data you send must have two
parts:
 * The first contains a text field, *request-json*, just like the rest of the API calls.  The value of this field is the JSON-encoded string.  It should have MIME type *text/plain*, and *Content-disposition: form-data; name="request-json"*
 * The second part contains the file data, and should have MIME type *octet-stream*, with *Content-disposition: form-data; name="file"; *filename="XXX"* where XXX* is a filename of your choice.

For example, uploading a file containing the text "Hello World", the data sent in the POST would look like this::

    --===============2521702492343980833==
    Content-Type: text/plain
    MIME-Version: 1.0
    Content-disposition: form-data; name="request-json"
    
    {"publicly_visible": "y", "allow_modifications": "d", "session": "XXXXXX", "allow_commercial_use": "d"}
    --===============2521702492343980833==
    Content-Type: application/octet-stream
    MIME-Version: 1.0
    Content-disposition: form-data; name="file"; filename="myfile.txt"
    
    Hello World
    
    --===============2521702492343980833==--


API URL:

    https://nova.astrometry.net/api/upload

Arguments:

    Same as URL upload, above.


Getting submission status
-------------------------

When you submit a URL or file, you will get back a *subid* submission
identifier.  You can use this to query the status of your submission
as it gets queued and run.  Each submission can have 0 or more "jobs"
associated with it; a job corresponds to a run of the *solve-field*
program on your data.

API URL:

    https://nova.astrometry.net/api/submissions/SUBID

Example:

    https://nova.astrometry.net/api/submissions/1019520

Arguments:

    None required.

Returns (example)::

    {"processing_started": "2016-03-29 11:02:11.967627", "job_calibrations": [[1493115, 785516]],
    "jobs": [1493115], "processing_finished": "2016-03-29 11:02:13.010625",
    "user": 1, "user_images": [1051223]}

If the job has not started yet, the *jobs* array may be empty.  If the
*job_calibrations* array is not empty, then we solved your image.


Getting job status
------------------

API URL:

    https://nova.astrometry.net/api/jobs/JOBID

Example:

    https://nova.astrometry.net/api/jobs/1493115

Arguments:

* None required

Returns (example):

    {"status": "success"}


Getting job results: calibration
--------------------------------

API URL:

    https://nova.astrometry.net/api/jobs/JOBID/calibration/

Example:

    https://nova.astrometry.net/api/jobs/1493115/calibration/

Arguments:

* None required

Returns (example)::

    {"parity": 1.0, "orientation": 105.74942079091929,
    "pixscale": 1.0906710701159739, "radius": 0.8106715896625917,
    "ra": 169.96633791366915, "dec": 13.221011585315143}


Getting job results: tagged objects in your image
-------------------------------------------------

You can get either all tags (including those added by random people), or
just the tags added by the web service automatically (known objects in your
field).

API URL:

* https://nova.astrometry.net/api/jobs/JOBID/tags/
* https://nova.astrometry.net/api/jobs/JOBID/machine_tags/

Example:

* https://nova.astrometry.net/api/jobs/1493115/tags/
* https://nova.astrometry.net/api/jobs/1493115/machine_tags/

Arguments:

* None required

Returns (example)::

    {"tags": ["NGC 3628", "M 66", "NGC 3627", "M 65", "NGC 3623"]}


Getting job results: known objects in your image
------------------------------------------------

API URL:

    https://nova.astrometry.net/api/jobs/JOBID/objects_in_field/

Example:

    https://nova.astrometry.net/api/jobs/1493115/objects_in_field/

Arguments:

* None required

Returns (example)::

    {"objects_in_field": ["NGC 3628", "M 66", "NGC 3627", "M 65", "NGC 3623"]}


Getting job results: known objects in your image, with coordinates
------------------------------------------------------------------

API URL:

    https://nova.astrometry.net/api/jobs/JOBID/annotations/

Example:

    https://nova.astrometry.net/api/jobs/1493115/annotations/

Arguments:

* None required

Returns (example, cut)::

    {"annotations": [
      {"radius": 0.0, "type": "ic", "names": ["IC 2728"],
       "pixelx": 1604.1727638846828, "pixely": 1344.045125738614},
      {"radius": 0.0, "type": "hd", "names": ["HD 98388"],
       "pixelx": 1930.2719762446786, "pixely": 625.1110603737037}
     ]}

Returns a list of objects in your image, including NGC/IC galaxies,
Henry Draper catalog stars, etc.  These should be the same list of
objects annotated in our plots.


Getting job results
-------------------

API URL:

    https://nova.astrometry.net/api/jobs/JOBID/info/

Example:

    https://nova.astrometry.net/api/jobs/1493115/info/

Arguments:

* None required

Returns (example, cut)::

    {"status": "success",
     "machine_tags": ["NGC 3628", "M 66", "NGC 3627", "M 65", "NGC 3623"],
     "calibration": {"parity": 1.0, "orientation": 105.74942079091929, 
        "pixscale": 1.0906710701159739, "radius": 0.8106715896625917, 
        "ra": 169.96633791366915, "dec": 13.221011585315143},
     "tags": ["NGC 3628", "M 66", "NGC 3627", "M 65", "NGC 3623"],
     "original_filename": "Leo Triplet-1.jpg",
     "objects_in_field": ["NGC 3628", "M 66", "NGC 3627", "M 65", "NGC 3623"]}


Getting job results: results files
----------------------------------

Note that when using the API, you can still request regular URLs to,
for example, retrieve the WCS file or overlay plots.  Images submitted
via the API go through exactly the same processing as images submitted
through the browser interface, so you can find the status or results
pages and discover the URLs of various data products that we haven't
documented here.

Note that you _must_ set this HTTP header::

    Referer: https://nova.astrometry.net/api/login


URLs:

* https://nova.astrometry.net/wcs_file/JOBID
* https://nova.astrometry.net/new_fits_file/JOBID
* https://nova.astrometry.net/rdls_file/JOBID
* https://nova.astrometry.net/axy_file/JOBID
* https://nova.astrometry.net/corr_file/JOBID
* https://nova.astrometry.net/annotated_display/JOBID
* https://nova.astrometry.net/red_green_image_display/JOBID
* https://nova.astrometry.net/extraction_image_display/JOBID

Examples:

* https://nova.astrometry.net/wcs_file/1493115
* https://nova.astrometry.net/new_fits_file/1493115
* https://nova.astrometry.net/rdls_file/1493115
* https://nova.astrometry.net/axy_file/1493115
* https://nova.astrometry.net/corr_file/1493115
* https://nova.astrometry.net/annotated_display/1493115
* https://nova.astrometry.net/red_green_image_display/1493115
* https://nova.astrometry.net/extraction_image_display/1493115

Misc Notes
----------

The API and other URLs are defined here:

    httpss://github.com/dstndstn/astrometry.net/blob/main/net/urls.py#L146
