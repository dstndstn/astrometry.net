#! /bin/bash

base="/data1/dstn/dsolver"
daemon="${base}/astrometry/blind/backend_daemon.py"
config="${base}/backend-config/backend-$(hostname -s).cfg"
log="${base}/logs/daemon-$(hostname -s).log"
echo "running ${daemon}"
echo "using config file ${config}"
echo "logging to ${log}"
touch ${log}
python ${daemon} -c ${config} >> ${log} 2>> ${log}

