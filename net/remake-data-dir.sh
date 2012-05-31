#! /bin/bash
set -e
set -u

NOVADATA=/home/nova/nova/net/data
STAGINGDATA=/home/nova/staging/net/data
OVERLAY=/home/nova/staging/net/staging-overlay

# Unmount, if already mounted
(grep -q "unionfs-fuse ${STAGINGDATA}" /etc/mtab &&
 echo Unmounting ${STAGINGDATA} &&
 fusermount -u ${STAGINGDATA}) || true

# Clear/create the overlay area
rm -Rf ${OVERLAY}
mkdir ${OVERLAY}

## create the union filesystem:
echo Mounting ${STAGINGDATA}
unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAY}=RW:${NOVADATA}=RO" "${STAGINGDATA}"

#    -o max_files=10000
#    -o allow_root       
#    -o readdir_ino   

