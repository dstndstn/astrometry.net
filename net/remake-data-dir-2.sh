#! /bin/bash
set -e
set -u

NOVADATA=/home/nova/nova/net/data

SUPERSTAGINGDATA=/home/nova/superstaging/net/data
OVERLAYDATA=/data2/nova/superstaging-data-overlay

mkdir -p ${SUPERSTAGINGDATA}

# Unmount, if already mounted
(grep -q "unionfs-fuse ${SUPERSTAGINGDATA}" /etc/mtab &&
 echo Unmounting ${SUPERSTAGINGDATA} &&
 fusermount -u ${SUPERSTAGINGDATA}) || true

# Clear/create the overlay area
echo "Creating overlay directories..."
rm -Rf ${OVERLAYDATA}
mkdir ${OVERLAYDATA}

## create the union filesystem:
echo Mounting ${SUPERSTAGINGDATA}
echo unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYDATA}=RW:${NOVADATA}=RO" "${SUPERSTAGINGDATA}"
unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYDATA}=RW:${NOVADATA}=RO" "${SUPERSTAGINGDATA}"

