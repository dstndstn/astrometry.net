
Once-Asked Questions
====================

Q: I don't have *numpy*.  What do I do?
---------------------------------------

"import error: no module named numpy"

A: Disable things that require numpy.
-------------------------------------

Some parts of the code need the "numpy" python package.  To disable things that need numpy::

    solve-field --no-fits2fits --no-remove-lines --uniformize 0  [....usual arguments...]

