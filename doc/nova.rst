Setting up a copy of the web service nova.astrometry.net
========================================================

These are instructions for how to set up a web service like our
https://nova.astrometry.net .  This requires a bit of sysadmin savvy.

Roadmap
-------

The code for the web service lives in the "net" subdirectory of the
git repository;
https://github.com/dstndstn/astrometry.net/tree/main/net .  It is
*not* included in the source code releases, so you'll need to *git
clone* the code.

The web service has several parts:
  * the web front-end.  This is a Django app that we run in Apache via
    WSGI.  Other deployment options are possible but untested.
  * the database.  The Django app uses a database.  We use postgres.
    Other databases (sqlite3, mysql) would work but are, you guessed
    it, untested.
  * front-end processing.  The front-end server has to do some
    asynchronous processing.  That is, the web-facing part of it
    queues submissions that are processed in a separate process,
    called `*process-submissions.py*
    <https://github.com/dstndstn/astrometry.net/blob/main/net/process_submissions.py>`_.
    On nova, we run this inside a *screen* process on the web server.
  * the solve server.  On nova, we have the web front-end running on
    one machine, and the "engine" running on another machine; the web
    server runs *ssh* to connect to the solve server.


Setup -- web front-end
----------------------

I would recommend creating a new user on your system, running the
Apache server as that user, and creating a database account for that
user.  On nova that user is called (you guessed it), "nova".

As that user, you'll want to check out the code, eg::

    cd
    git clone https://github.com/dstndstn/astrometry.net.git nova

See :ref:`web_local` for how to run the web server using sqlite3 and
Django's development web server.

For "real" use, you may want to set up a postgres database and run the
web service via Apache.

Notice that in the *net/* directory we have a bunch of
*settings_XXX.py* files.  Each of these describes the setup of a
deployment of the web site.  We use symlinks to determine which one
runs, eg, on the nova server we have::

    ln -s settings_nova.py settings.py

Note also that we store the database secrets in a separate, secret SVN
repository, which we check out into the directory *net/secrets*; ie,
on the nova server::

    $ ls -1 net/secrets/
    django_db.py
    __init__.py

where *__init__.py* is empty, and *django_db.py* contains::

    DATABASE_USER = 'nova'
    DATABASE_PASSWORD = 'SOSECRET'
    DATABASE_HOST = 'localhost'
    DATABASE_PORT = ''

Setting up your database is sort of beyond the scope of this document.
The django docs should have some material to get you started.  

The following *may* work to set up a postgres database::

    # as root, run the "psql" program, and enter:
    create role nova;
    alter role nova createdb;

    # as user "nova", run "psql" and
    create database "an-nova";

Then, to initialize the database, cd into the *astrometry/net*
directory and run the Django setup scripts::

    python manage.py syncdb
    python manage.py migrate

and test it out::

    python manage.py runserver


You probably want to run the web app under Apache.  The Apache
configuration files for nova are not public, but your *apache2.conf*
file might end up containing entries such as::

    User nova
    Group nova
    Include /etc/apache2/mods-available/wsgi.load
    Include /etc/apache2/mods-available/wsgi.conf
    WSGIScriptAlias / /home/nova/nova/net/nova.wsgi

See the Django deployment docs for much more detailed setup help.


Setup -- front-end processing
-----------------------------

You need to run the *process_submissions.py* script on the web server
to actually process user submissions.  On nova, we run this inside a
*screen* session; unfortunately this means we have to manually start
it whenever the web server is rebooted.  cd into the *astrometry/net*
subdirectory and run, eg::

    python -u process_submissions.py --jobthreads=8 --subthreads=4 < /dev/null >> proc.log 2>&1 &



Setup -- solve-server processing
--------------------------------

For actually running the astrometry engine, the web server uses *ssh*
to connect to a solve server.  One could probably use a local version
(running on the web server) without *too* many changes to the code,
but that has not been implemented yet.

When the web server wants to run the astrometry engine, it executes
the following crazy code
(https://github.com/dstndstn/astrometry.net/blob/main/net/process_submissions.py#L288)::

    cmd = ('(echo %(jobid)s; '
           'tar cf - --ignore-failed-read -C %(jobdir)s %(axyfile)s) | '
           'ssh -x -T %(sshconfig)s 2>>%(logfile)s | '
           'tar xf - --atime-preserve -m --exclude=%(axyfile)s -C %(jobdir)s '
           '>>%(logfile)s 2>&1' %
           dict(jobid='job-%s-%i' % (settings.sitename, job.id),
                axyfile=axyfn, jobdir=jobdir,
                sshconfig=settings.ssh_solver_config,
                logfile=logfn))

So it first sends the job id, then a *tar* stream of the required
input files, and pipes that to *ssh*.  It streams the error output to
the *logfile*, and pipes the standard out to *tar* to receive the
results.  It's sweet.

Notice that *sshconfig* string there, which come from the
*settings.py* file.  For nova, for example, *ssh_solver_config =
'an-nova'*.  We then have an entry in the *nova* user's
*~/.ssh/config* file::

    Host an-nova
    Hostname solveserver.domain.org
    User solve
    IdentityFile ~/.ssh/id_nova_backend

And, naturally, we use SSH keys to automate the login.

On the solve server, the *solve* user has an entry in
*~/.ssh/authorized_keys* for the *id_nova_backend.pub* public key,
that tells the *ssh* server what should be run when that key is used
to log in::

    # id_nova_backend
    #
    command="cd /home/solve/nova/solver; ../net/testscript-astro",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-rsa AAAA[.....] nova@webserver

That script
(https://github.com/dstndstn/astrometry.net/blob/main/net/testscript-astro)
first reads the job id, creates a working directory for the job, uses
*tar* to receive the input files, and then runs the
*astrometry-engine* program to actually run the requested job.
Finally, it uses *tar* to bundle up and send back the results.

(Note that, at present, the *testscript-astro* script still tries to
run the *astrometry-engine* by its old name, *backend* ... we haven't
updated that script in a while.  That script also includes hard-coded
paths, so you will have to edit for your site.)


