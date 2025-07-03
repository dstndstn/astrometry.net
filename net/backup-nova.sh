#! /bin/bash
set -e

echo "Database..."
date=$(date)
cd /data/nova/an-nova-database-backup
ssh db pg_dumpall > an-nova.sql
git commit an-nova.sql -m "database snapshot: $date"
