#! /bin/bash

#DIR=$(pwd)
#SOLVER=${DIR}/solver.py
#daemon python $SOLVER -d $DIR /nobackup/dstn/solver-indexes

#CMD=/nobackup30/dstn/an/astrometry/net/server/run-solver.sh
echo "i am $0"
DIR=$(dirname $0)
CMD=${DIR}/run-solver.sh
echo "Launching command $CMD as a daemon."
#echo "daemon is $(which daemon)"
#DERRLOG=${DIR}/daemon-${PPID}-err.log
#DDBGLOG=${DIR}/daemon-${PPID}-dbg.log
#COUT2LOG=${DIR}/client-${PPID}-out.log
#COUTLOG=${DIR}/client-${PPID}-out.log
#CERRLOG=${DIR}/client-${PPID}-err.log
#DAEMON="daemon -N -D ${DIR} -l ${DERRLOG} -b ${DDBGLOG} -o ${COUT2LOG} -O ${COUTLOG} -E ${CERRLOG} -X ${CMD}"
#echo "daemon command: ${DAEMON}"
#${DAEMON}

/nobackup30/dstn/an/astrometry/net/execs/simple-daemon ${CMD}

echo "Daemon launched."
