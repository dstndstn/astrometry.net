#! /bin/bash

BASE=/home/nova/astrometry
cd $BASE/net

export PYTHONPATH=${PYTHONPATH}:$BASE
export PATH=${PATH}:$BASE/util:$BASE/solver:$BASE/plot

#export PATH=${PATH}:~/.local/bin:$BASE/util:$BASE/solver:$BASE/plot
# unset PYTHONPATH
# which uwsgi
# uwsgi --version
# echo $PATH
# export PYTHONPATH=${PYTHONPATH}:${CONDA_PREFIX}

uwsgi -s :3030 --wsgi-file wsgi.py --touch-reload wsgi.py --processes 8 --reload-on-rss 768 -d /data/nova/uwsgi.log --limit-post 500000000


