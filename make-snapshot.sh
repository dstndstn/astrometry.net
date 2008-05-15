#! /bin/bash

SNAPSHOT_SVN=$1
SNAPSHOT_VER=$2
SNAPSHOT_SUBDIRS=$3

SNAPSHOT_DIR=astrometry.net-${SNAPSHOT_VER}
rm -R ${SNAPSHOT_DIR}
svn export -N ${SNAPSHOT_SVN} ${SNAPSHOT_DIR};
for x in ${SNAPSHOT_SUBDIRS}; do
    svn export ${SNAPSHOT_SVN}/$x ${SNAPSHOT_DIR}/$x;
done
tar cf ${SNAPSHOT_DIR}.tar ${SNAPSHOT_DIR}
gzip --best -c ${SNAPSHOT_DIR}.tar > ${SNAPSHOT_DIR}.tar.gz
echo "Created ${SNAPSHOT_DIR}.tar.gz"
rm ${SNAPSHOT_DIR}.tar.bz2
bzip2 --best ${SNAPSHOT_DIR}.tar
echo "Created ${SNAPSHOT_DIR}.tar.bz2"

