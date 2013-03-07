#! /bin/bash
set -e
set -u

NOVADATA=/home/nova/nova/net/data
STAGINGDATA=/home/nova/staging/net/data
OVERLAYDATA=/data2/nova/staging-data-overlay

mkdir -p ${STAGINGDATA}

# Unmount, if already mounted
(grep -q "unionfs-fuse ${STAGINGDATA}" /etc/mtab &&
 echo Unmounting ${STAGINGDATA} &&
 fusermount -u ${STAGINGDATA}) || true

# Clear/create the overlay area
echo "Creating overlay directories..."
rm -Rf ${OVERLAYDATA}
mkdir ${OVERLAYDATA}

## create the union filesystem:
echo Mounting ${STAGINGDATA}
echo unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYDATA}=RW:${NOVADATA}=RO" "${STAGINGDATA}"
unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYDATA}=RW:${NOVADATA}=RO" "${STAGINGDATA}"

