#! /bin/bash
set -e
set -u

NOVADATA=/home/nova/nova/net/data
STAGINGDATA=/home/nova/staging/net/data
OVERLAYDATA=/home/nova/staging/net/staging-data-overlay

NOVAJOBS=/home/nova/nova/net/jobs
STAGINGJOBS=/home/nova/staging/net/jobs
OVERLAYJOBS=/home/nova/staging/net/staging-jobs-overlay

mkdir -p ${STAGINGDATA}
mkdir -p ${STAGINGJOBS}

# Unmount, if already mounted
(grep -q "unionfs-fuse ${STAGINGDATA}" /etc/mtab &&
 echo Unmounting ${STAGINGDATA} &&
 fusermount -u ${STAGINGDATA}) || true

(grep -q "unionfs-fuse ${STAGINGJOBS}" /etc/mtab &&
 echo Unmounting ${STAGINGJOBS} &&
 fusermount -u ${STAGINGJOBS}) || true

# Clear/create the overlay area
echo "Creating overlay directories..."
rm -Rf ${OVERLAYDATA}
mkdir ${OVERLAYDATA}

rm -Rf ${OVERLAYJOBS}
mkdir ${OVERLAYJOBS}

## create the union filesystem:
echo Mounting ${STAGINGDATA}
echo unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYDATA}=RW:${NOVADATA}=RO" "${STAGINGDATA}"
unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYDATA}=RW:${NOVADATA}=RO" "${STAGINGDATA}"

echo Mounting ${STAGINGJOBS}
echo unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYJOBS}=RW:${NOVAJOBS}=RO" "${STAGINGJOBS}"
unionfs-fuse -o cow,use_ino,suid,dev,nonempty,allow_other,default_permissions \
    "${OVERLAYJOBS}=RW:${NOVAJOBS}=RO" "${STAGINGJOBS}"

#    -o max_files=10000
#    -o allow_root       
#    -o readdir_ino   

