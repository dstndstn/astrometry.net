.. _svn_checkout:

svn checkout the Astrometry.net code
------------------------------------

This requires that you have an account on ``astrometry.net``.

On your laptop:

Set up ``ssh``: if your username on your laptop is different than your
account at ``astrometry.net``, you're probably fine as-is.  Try::

    ssh astrometry.net

If that doesn't work, edit your ``~/.ssh/config`` file, adding a
stanza like this::

    Host astrometry.net
    User you   # if your account is you@astrometry.net
    IdentityFile ~/.ssh/id_rsa  # or whatever your private key is

Then, check out the code with::

    svn co svn+ssh://astrometry.net/svn/trunk/src/astrometry

You should see a bunch of filenames go by, and when it's done you'll
have an ``astrometry`` directory containing a checked-out copy of the
code.

You can read more about subversion in `The Subversion Book
<http://svnbook.red-bean.com/en/1.7/index.html>`_.  If you read the
tutorial there (chapter 2), you should skip the parts about creating a
repository and importing files -- we've done that already, and the
step above just did the `Creating a working copy
<http://svnbook.red-bean.com/en/1.7/svn.tour.initial.html>`_ step.
Carry on to `Basic work cycle
<http://svnbook.red-bean.com/en/1.7/svn.tour.cycle.html>`_.

