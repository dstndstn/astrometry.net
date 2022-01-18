#! /bin/bash

BASE=/home/nova/nova
cd $BASE/net

export PYTHONPATH=${PYTHONPATH}:$BASE
export PATH=${PATH}:~/.local/bin:$BASE/util:$BASE/solver:$BASE/plot
# unset PYTHONPATH
# export PATH=~/miniconda3b/bin:${PATH}
# #source activate viewer-conda-2
# source activate viewer-conda-3
# which uwsgi
# uwsgi --version
# echo $PATH
# export PYTHONPATH=${PYTHONPATH}:${CONDA_PREFIX}
# export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

uwsgi -s :3030 --wsgi-file wsgi.py --touch-reload wsgi.py --processes 8 --reload-on-rss 768 -d /data2/nova/uwsgi.log --limit-post 500000000


