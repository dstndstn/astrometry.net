With this document, we try to collect all the useful code snipplets to help you
using `Astrometry.net <http://astrometry.net/>`_, which will now be referred to
by *AN*. While there is already some limited documentation around for quite
specific aspects of all the components used, an overview helping a beginner to
solve astronomical images was missing---until now.

.. contents::

***********************************************
Building Your Local Astrometry.net Installation
***********************************************

There are several ways to use the *AN* functionality:

#. Build your own local installation;
#. use one of the projects using it, eg.

	- The Flickr Astro group, where all uploaded images are automatically
	  fetched and fed through the solver;
	- `AstroTortilla <http://sourceforge.net/projects/astrotortilla/>`_, which
	  uses *AN* internally;

#. or finally, you can use *AN*'s web service to process your images.

This section will describe how to prepare a local installation by getting and
building the source code. You'll need additional *index* files to actually
solve images, but that will be described later. (They can be downloaded or
prepared locally.)

Getting the Source Code
=======================
The following examples always work on the *complete* SVN repository, though you
only need the ``./src/astrometry/`` subdirectory. Thus, you'd always add that to
save quite some bandwidth.

Downloading Release Tarballs or Daily Snapshots
-----------------------------------------------

For details, please refer to the `AN Usage page
<http://astrometry.net/use.html>`_.  Snapshots as well as release tarballs can
be found at `http://astrometry.net/downloads/
<http://astrometry.net/downloads/>`_.  Please look at the file names carefully:
Some contain version numbers, some SVN revision numbers. As usual, newer is
better. Your choice!

As the project currently (20140623) switches over to GitHub, you'll find
all upcoming releases there, too, listed at
`https://github.com/dstndstn/astrometry.net/releases <https://github.com/dstndstn/astrometry.net/releases>`_.

Current Sources
---------------

The current source tree is managed in GIT on GitHub. Anonymous users will clone
from there::

	git clone https://github.com/dstndstn/astrometry.net.git

while authenticated users (you need your SSH keys in place!) can clone with::

	git clone git@github.com:dstndstn/astrometry.net.git

Historic SVN Tree
-----------------

With SVN
^^^^^^^^
You can anonymously download the *AN* sources like this::

	svn co http://astrometry.net/svn/trunk astrometry

If you have an account (with a few useful patches posted to the mailing
list, you'll usually get one if you ask for it), you can of course fetch
as an authenticated user and will be able to commit your changes::

	svn co svn+ssh://<username>@astrometry.net/svn/trunk astrometry

With GIT-SVN
^^^^^^^^^^^^
Anonymously, this is done by::

	git svn clone http://astrometry.net/svn/trunk astrometry

As an authenticated user, you'd use::

	git svn clone svn+ssh://<username>@astrometry.net/svn/trunk astrometry

Both variants will place a GIT-SVN converted copy of the whole repository into
the local ``./astrometry/`` directory, which you can update with ``git svn
rebase``.

With commit rights (using authenticated access), you can also easily commit
to SVN: Just commit them locally (ie. using ``git update-index``, ``git
add``, ``git rm`` etc., finally followed by ``git commit`` to commit your
changes) and then push them into the upstream SVN repository using ``git svn
dcommit``.


********************
Using Astrometry.net
********************

With or without a local *AN* installation, you can use it to solve your
astronomical images. It will accept `FITS <http://fits.gsfc.nasa.gov/>`_
files as well as other commonly used digital image formats. You can even feed
it with preprocessed (X,\ Y) coordinates you got other astronomical software
stacks, like `SExtractor <http://sextractor.sourceforge.net/>`_.

Astrometry.net Flickr Group
===========================

We've a bot installed watching the `Astrometry Flickr Group
<https://www.flickr.com/groups/astrometry>`_. Every image uploaded to this group
will be solved automatically. Some comments will be added describing the image's location,
rotation, and important objects found.

Astrometry.net Web Service
==========================

You can upload your images to the `Nova Web Service
<http://nova.astrometry.net/upload>`_ found at
``http://nova.astrometry.net/upload``. Keep in mind that this is a public web
service; others may see your captures. We're doing lots of work there, the
service may be unstable from time to time. For every image, you'll find a page
linking to all important files with the WCS solution etc.

Local *AN* Installation
=======================

Other Projects Using Astrometry.net
===================================

AstroTortilla
-------------

AstroTortilla (`SourceForge project page
<http://sourceforge.net/projects/astrotortilla/>`_, `home page
<http://sourceforge.net/p/astrotortilla/home/Home/>`_) is Windows software
shipping a `Cygwin <https://www.cygwin.com/>`_ build of *AN*, which it uses
internally.
