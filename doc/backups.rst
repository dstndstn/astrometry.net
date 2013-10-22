.. _backups:

Backups of Astrometry.net stuff
-------------------------------

* **Subversion** (svn) repo.  dstn maintains a mirror 
  (at ``dstn@cs.toronto.edu:svn-backup-astrometry``)
  using the code in ``trunk/scripts/svnsync``
  (`here <http://trac.astrometry.net/browser/trunk/scripts/svnsync>`_).
  It is updated and md5sum-checked hourly, via a cron job on
  ``apps3.cs.toronto.edu``::

    # Sync hourly
    @hourly   /u/dstn/svn-backup-astrometry/sync.sh
    # Check hourly, at 5 minutes past (to let the sync finish)
    5 * * * *   /u/dstn/svn-backup-astrometry/sync.sh && /u/dstn/svn-backup-astrometry/check.sh

  That directory also contains dumps of series of 1000 revs, named
  like ``svn-backup-10k-to-11k.dump.bz2``

* **Trac**.  Backed up in ``svn:secure/trac``.  The database gets
  dumped and committed daily at 4am by a cron job by
  ``astrometry@astrometry.net``.  Attachments do **not** get automatically added.

* **Forum**.  Nothing?

* **Live**.  dstn has backups in ``dstn@broiler:/data1/live``.  These
  are monthly tarballs, created manually as part of the perpetual
  data-juggling on oven.

  Hogg has an independent backup of this.... RIGHT?

* **Nova**.  dstn has a snapshot (2012-06-05) of the nova/net/data
  directory at ``dstn@broiler:/data2/dstn/BACKUP-nova/nova-data``::

    cd /data2/dstn/BACKUP-nova
    rsync --progress -Rarvz nova:nova/net/./data/ nova-data
    rsync --progress -Rarvz nova:nova/net/jobs/./ nova-jobs

  (There is a snapshot dump of the nova database there too.)

  There is a snapshot (snapshot-2013-10-14.tgz) on bbq, as well as daily
  backups, in ``bbq:/export/bbq1/nova/nova-data``::

    # Backup nova@broiler nova.astrometry.net data
    0 4 * * * rsync -ar /data2/nova/nova-data nova-data-rsync:nova-data-backup/
    0 4 * * * pg_dump an-nova | ssh nova-data-rsync 'cat > nova-data-backup/nova-database/an-nova.sql && cd nova-data-backup/nova-database && git commit -a -m "database snapshot: $(date)"'

