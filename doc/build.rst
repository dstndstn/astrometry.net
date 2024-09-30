.. _build:

Building/installing the Astrometry.net code
===========================================

Grab the code::

   wget http://astrometry.net/downloads/astrometry.net-latest.tar.gz
   tar xvzf astrometry.net-latest.tar.gz
   cd astrometry.net-*

Build it.  The short version::

   make
   make py
   make extra
   make install  # to put it in /usr/local/astrometry
   # or:
   make install INSTALL_DIR=/some/other/place


The long version:

Prerequisites
-------------

For full functionality, you will need:
  * GNU build tools (gcc/clang, make, etc.)
  * cairo
  * netpbm
  * libpng
  * libjpeg
  * libz
  * bzip2
  * python (3.x preferred)
  * numpy
  * swig (>= 2.0)
  * fitsio https://github.com/esheldon/fitsio or astropy http://www.astropy.org/ or pyfits: http://www.stsci.edu/resources/software_hardware/pyfits (version >= 3.1)
  * cfitsio: http://heasarc.gsfc.nasa.gov/fitsio/
 

Ubuntu or Debian-like systems:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::


    $ sudo apt-get install libcairo2-dev libnetpbm10-dev netpbm \
                           libpng2-dev libjpeg-dev python3-numpy \
                           python3-dev python3-pip zlib1g-dev \
                           libbz2-dev swig cfitsio-dev
    $ pip install fitsio # or astropy

For example, in Debian 9 (Stretch):: 

    $ sudo apt-get install libcairo2-dev libnetpbm10-dev netpbm \
                           libpng-dev libjpeg-dev python-numpy \
                           python-pyfits python-dev zlib1g-dev \
                           libbz2-dev swig libcfitsio-dev

In Ubunutu 20.04::

    $ sudo apt install build-essential curl git file pkg-config swig \
           libcairo2-dev libnetpbm10-dev netpbm libpng-dev libjpeg-dev \
           zlib1g-dev libbz2-dev libcfitsio-dev wcslib-dev \
           python3 python3-pip python3-dev \
           python3-numpy python3-scipy python3-pil
    $ pip install fitsio # or astropy

As of April 2019, the script doc/install_astrometry_on_linux.sh will install all dependencies along with astrometry.net on Linux, and download 4200/ index files.


CentOS 6.5 / Fedora / RedHat / RHEL -- Detailed Instructions:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See these `instructions from James Chamberlain <http://plaidhat.com/code/astrometry.php>`_.


Arch Linux
^^^^^^^^^^

A package can be installed from the `Arch Linux (AUR)
<https://aur.archlinux.org/packages/astrometry.net/>`_.


Mac OS X using homebrew:
^^^^^^^^^^^^^^^^^^^^^^^^

First set up homebrew, as described at `Homebrew https://brew.sh/`_.

The "formula" for installing Astrometry.net is called `astrometry-net` and is included in the "core" package
repository, so you just need to

    $ brew install astrometry-net


Mac OS X using homebrew (ancient instructions):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These instructions *Worked For Me* as of September 2012 on OSX 10.8.

First set up homebrew:
  * grab `XCode <https://developer.apple.com/xcode/>`_ (from the Apps Store.  Free, but you still need a credit card.  Argh.)
  * grab `XCode Command-line utilities <https://developer.apple.com/downloads/index.action>`_
  * grab `XQuartz <http://xquartz.macosforge.org/landing/>`_
  * grab `Homebrew <http://mxcl.github.com/homebrew/>`_
  * grab `pip <http://www.pip-installer.org/en/latest/installing.html>`_ if you don't have it already

Optionally, grab some handy homebrew packages::

    $ brew install cfitsio --with-examples
    $ brew install md5sha1sum     # OSX doesn't come with this?!  For shame
 
Install:

    $ brew install astrometry-net

Mac OS X using Fink:
^^^^^^^^^^^^^^^^^^^^

Use apt-get install as per the Debian instructions above (leaving out
``zlib1g-dev`` because it's already included with OSX).  Note that to
use Fink you will need to add something like this in your
``~/.profile`` or ``~/.bashrc`` file::

    . /sw/bin/init.sh
    export CFLAGS="-I/usr/local/include -I/sw/include"
    export LDFLAGS="-L/usr/local/lib -L/sw/lib"

Windows 10/11:
^^^^^^^^^^^^^^^^^^^^

Since there is `Windows Subsystem for Linux (WSL) <https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux>` compatibility layer available on Windows 10/11 OS, you can follow the WSL install `guide <https://docs.microsoft.com/en-us/windows/wsl/install>`.
Once WSL is installed, the build steps are the same as for Debian/Ubuntu-like systems. See above.

Getting/Building
----------------

If you don't have and can't get these libraries, you should still be
able to compile and use the core parts of the solver, but you will
miss out on some eye-candy.

Build the solving system::

  $ make

If you installed the libraries listed above, build the plotting code::

  $ make extra

Install it::

  $ make install

You might see some error message during compilation; see the section
ERROR MESSAGES below for fixes to common problems.

By default it will be installed in  ``/usr/local/astrometry`` .
You can override this by either:

* editing the top-level Makefile (look for INSTALL_DIR); or
* defining INSTALL_DIR on the command-line:

  For bash shell::

    $ export INSTALL_DIR=/path/to/astrometry
    $ make install

  or::

    $ INSTALL_DIR=/path/to/astrometry make install

  For tcsh shell::

    $ setenv INSTALL_DIR /path/to/astrometry
    $ make install

The astrometry solver is composed of several executables.  You may
want to add the INSTALL_DIR/bin directory to your path:

   For bash shell::

     $ export PATH="$PATH:/usr/local/astrometry/bin"

   For tcsh shell::

     $ setenv PATH "$PATH:/usr/local/astrometry/bin"

Some of the scripts are written in Python and are run using the `python` from the user's environment via `env python`.
To override this and use a python executable of your choice, you can use the `PYTHON_SCRIPT` variable, eg,::

     $ make install INSTALL_DIR=/your/install/directory PYTHON_SCRIPT="/usr/bin/env python3.6"'

or::

     $ make install INSTALL_DIR=/your/install/directory PYTHON_SCRIPT="/usr/local/bin/python3.6"'

Astrometry.net code includes some useful utilities copied from Cfitsio for working with fits files
(fitscopy, fitsverify, imcopy). These utilities are not built and installed by default, as in modern linux
distributions are usually provided by appropriate packages (for example `libcfitsio-bin` for Ubuntu/Debian,
`cfitsio-utils` for Fedora, `fitsverify` for both).
However, If you'd like to build and install them from the astrometry copies, run:

    $ make cfitsio-utils

and:

    $ make install-cfitsio-utils


Auto-config
-----------

We use a do-it-yourself auto-config system that tries to detect what
is available on your machine.  It is called ``os-features``, and it
works by trying to compile, link, and run a number of executables to
detect:

 * whether the "netpbm" library is available
 * whether certain GNU-specific function calls exist

You can change the flags used to compile and link "netpbm" by either:

* editing util/makefile.netpbm
* setting NETPBM_INC or NETPBM_LIB, like this::

    $ make NETPBM_INC="-I/tmp" NETPBM_LIB="-L/tmp -lnetpbm"

You can see whether netpbm was successfully detected by::

    $ cat util/makefile.os-features
    # This file is generated by util/Makefile.
    HAVE_NETPBM := yes

You can force a re-detection either by deleting util/makefile.os-features
and util/os-features-config.h, or running::

  $ make reconfig

(which just deletes those files)


Overriding Things
-----------------

For most of the libraries we use, there is a file called
``util/makefile.*`` where we try to auto-configure where the headers
and libraries can be found.  We use ``pkg-config`` when possible, but
you can override things.

``*_INC`` are the compile flags (eg, for the include files).

``*_LIB`` is for libraries.

``*_SLIB``, when used, is for static libraries (.a files).

gsl:
^^^^

You can either use your system's GSL (GNU scientific library)
libraries, or the subset we ship.  (You don't need to do anything
special to use the shipped version.)

System::

    make SYSTEM_GSL=yes

Or specify static lib::

    make SYSTEM_GSL=yes GSL_INC="-I/to/gsl/include" GSL_SLIB="/to/gsl/lib/libgsl.a"

Or specify dynamic lib::

    make SYSTEM_GSL=yes GSL_INC="-I/to/gsl/include" GSL_LIB="-L/to/gsl/lib -lgsl"



cfitsio:
^^^^^^^^

For dynamic libs::

    make CFITS_INC="-I/to/cfitsio/include" CFITS_LIB="-L/to/cfitsio/lib -lcfitsio"

Or for static lib::

    make CFITS_INC="-I/to/cfitsio" CFITS_SLIB="/to/cfitsio/lib/libcfitsio.a"


netpbm:
^^^^^^^

::

    make NETPBM_INC="-I/to/netpbm" NETPBM_LIB="-L/to/netpbm/lib -lnetpbm"

wcslib:
^^^^^^^

Ditto, with ``WCSLIB_INC``, ``WCSLIB_LIB``, ``WCS_SLIB``

cairo:
^^^^^^

``CAIRO_INC``, ``CAIRO_LIB``

jpeg:
^^^^^

``JPEG_INC``, ``JPEG_LIB``

png:
^^^^

``PNG_INC``, ``PNG_LIB``


zlib:
^^^^^

``ZLIB_INC``, ``ZLIB_LIB``

