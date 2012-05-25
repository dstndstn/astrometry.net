.. _nova_orient:

Nova.astrometry.net orientation
===============================

Getting started:

* you'll need an account on ``astrometry.net`` (for svn), and
  to have your ssh public key added to the authorized_keys for
  ``nova@oven.cosmo.fas.nyu.edu``
* can you ssh in as ``nova@oven.cosmo.fas.nyu.edu``?
* can you ``svn checkout`` the Astrometry.net code?


As for the nova website code: if you svn checked out the Astrometry.net 
code, you should have a ``net`` directory -- that's where the server-side 
code that runs the nova.astrometry.net site lives.  It's written in python 
using the Django framework, which provides nice web and database routines.

We actually run three copies of the nova website:

* supernova.astrometry.net -- "trunk", for testing and development
* staging.astrometry.net -- test-flight area when we think we have a releaseable version.
* nova.astrometry.net -- the production site.  This runs off an "svn tagged" version -- not trunk

The web sites are run via the apache web server.  You probably won't have 
to deal with that side at all.  Each one runs out of a directory within 
``nova@oven``'s home directory.  They're called (this may shock you), 
"supernova", "staging", and "nova".  There is a server-side process that 
has to be running in order for the web site to do anything with images 
that people submit.  That's called ``process_submissions.py`` (you can see 
the code in ``net/process_submissions.py``), and for each site we run it in a 
``screen``.  ``screen`` is an old-school utility that lets you create a 
"virtual terminal" that you can "attach" to and "detach" from.  It's a 
handy way of keeping a program running on a server when you're not logged 
in.  Anyway, if you're logged in as ``nova@oven`` you can do::

    screen -ls

to see the currently-running screens;

::

    nova@oven:~$ screen -ls
    There are screens on:
     	20579.supernova	(03/18/2012 08:15:43 PM)	(Detached)
     	27698.nova	(03/14/2012 06:11:25 PM)	(Detached)
    2 Sockets in /var/run/screen/S-nova.

which says there are two screens, ``supernova`` and ``nova``.

You can "attach" or "rejoin" with::

    screen -R supernova

and you should see "process_submissions.py" doing its thing::

    Checking for new Submissions
    Found 0 unstarted submissions
    Checking for UserImages without Jobs
    Found 0 userimages without Jobs
    Submissions running: 0
    Jobs running: 0

anyway, to "detach" from that screen, do::

    ctrl-a d


Ok, so coming back to the code, let's find out where the data live.

Each site has a ``settings`` file.  So, as ``nova@oven``::

    nova@oven:~$ cd supernova/net
    nova@oven:~/supernova/net$ ls -l settings.py
    lrwxrwxrwx 1 nova nova 21 2011-06-09 19:16 settings.py -> settings_supernova.py

So in the ``supernova`` directory, ``settings.py`` is a symlink to
``settings_supernova.py`` .  Let's look in there::

    # settings_supernova.py
    from settings_common import *
    
    DATABASES['default']['NAME'] = 'an-supernova'
    
    LOGGING['loggers']['django.request']['level'] = 'WARN'
    
    SESSION_COOKIE_NAME = 'SupernovaAstrometrySession'
    
    # Used in process_submissions
    ssh_solver_config = 'an-supernova'
    sitename = 'supernova'

oh, it hardly has anything.  But at the top it imports everything from 
settigs_common.py.  Looking there, we see::

    ...
    WEB_DIR = os.path.dirname(astrometry.net.__file__) + '/'
    DATADIR = os.path.join(WEB_DIR, 'data')
    ...

so it figures out the directory it is currently running in based on the 
location of the ``astrometry.net`` python package (``WEB_DIR``), which for 
supernova will be::

    /home/nova/supernova/net

and the ``DATADIR`` is that + ``data``.  If you look in that ``data`` directory, you'll see::

    nova@oven:~/supernova/net$ ls data
    00  08  12  19  25  2c  34  3a  41  49  50  57  60  67  6f  74  7e  8a  91  9a  a3  ad  b4  bf  c6  d4  da  e0  eb  f5
    03  0b  14  1a  27  2e  36  3b  43  4a  52  59  62  6a  70  75  85  8b  92  9d  a6  ae  b5  c0  c7  d5  db  e2  ef  fa
    04  0c  15  20  29  2f  37  3d  44  4b  53  5a  63  6b  71  76  86  8c  94  9e  aa  b0  b7  c3  ca  d6  dc  e4  f0  fb
    06  0d  16  23  2a  32  38  3e  45  4c  55  5b  65  6c  72  79  87  8d  98  9f  ab  b1  b9  c4  cb  d7  dd  e6  f2  fd
    07  11  18  24  2b  33  39  3f  46  4e  56  5e  66  6d  73  7a  89  8f  99  a2  ac  b3  ba  c5  d1  d8  de  e9  f3  ff

ok, a bunch of directories.  What's in them?

::

    nova@oven:~/supernova/net$ find data/00
    data/00
    data/00/32
    data/00/32/84
    data/00/32/84/00328489cbdfce1a99ebbf1078c95669e39fa8a7

::

    nova@oven:~/supernova/net$ file data/00/32/84/00328489cbdfce1a99ebbf1078c95669e39fa8a7
    data/00/32/84/00328489cbdfce1a99ebbf1078c95669e39fa8a7: JPEG image data, JFIF standard 1.01

And check this out::

    nova@oven:~/supernova/net$ sha1sum data/00/32/84/00328489cbdfce1a99ebbf1078c95669e39fa8a7
    00328489cbdfce1a99ebbf1078c95669e39fa8a7  data/00/32/84/00328489cbdfce1a99ebbf1078c95669e39fa8a7
    
So the files are named according to a cryptographic hash of their contents 
(SHA-1), and sorted into subdirectories according to the first three pairs 
of hexadecimal digits::

    AA/BB/CC/AABBCC....

(We sort them into subdirectories like that to avoid having a huge number 
of files in a single directory.)

