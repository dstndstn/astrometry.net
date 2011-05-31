#! /bin/bash

#!@#$%^
source ~/.bashrc

base="/data1/dstn/dsolver"
daemon="${base}/astrometry/blind/backend_daemon.py"
config="${base}/backend-config/backend-$(hostname -s).cfg"
log="${base}/logs/daemon-$(hostname -s).log"

echo "killing old daemon..."
pids=$(pgrep -U dstn -f "python .*backend_daemon.py")
for pid in $pids; do
  parent=$(ps -o "ppid" $pid | tail +2)
  kill $pid $parent
done

#echo "pids are ${pids}"
#if [ "x${pids}" != "x" ]; then
#  echo "killall -gwq ${pids}"
#  killall -gwq ${pids}
#fi
#killall -gwq -r "python .*backend_daemon.py"

echo "running ${daemon}"
echo "using config file ${config}"
echo "logging to ${log}"
echo "pythonpath is ${PYTHONPATH}"
touch ${log}
python ${daemon} -c ${config} >> ${log} 2>&1 #> ${log}
echo "daemon exiting"

