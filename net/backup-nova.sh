#! /bin/bash
set -e

#rsync -Wqarz /home/nova/nova/net/data/ broiler:/data2/nova/BACKUP-data/
echo "Data..."
#rsync -arz --progress /home/nova/nova/net/data/ broiler:BACKUP-data/
#rsync -arz --progress /data1/nova/data/ broiler:BACKUP-data/

EX=$(pwd)/exclude.lst
ssh broiler "cd BACKUP-data/; find ." > ${EX}
#echo "Excluding $(wc -l exclude.lst) existing files from backup"
#(cd /home/nova/nova/net/data/; rsync -WOqarz --exclude-from=${EX} . broiler:/data2/nova/BACKUP-data/)
(cd /home/nova/nova/net/data/; rsync -WOvarz --exclude-from=${EX} . broiler:/data2/nova/BACKUP-data/)

#for ((i=0; i<256; i++)); do
#    X=$(printf %02x $i);
#    echo -n .;
#    rsync -WOarz /home/nova/nova/net/data/$X/ broiler:BACKUP-data/$X/;
#done

echo "Jobs..."
rsync -arz --progress /home/nova/nova/net/jobs/ broiler:BACKUP-jobs/

echo "Database..."
pg_dump an-nova | ssh broiler "cat > BACKUP-database/an-nova.sql &&  cd BACKUP-database && git commit -a -m 'database snapshot'"
