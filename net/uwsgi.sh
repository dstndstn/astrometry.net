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

uwsgi -s :3030 --wsgi-file wsgi.py --touch-reload wsgi.py --processes 8 --reload-on-rss 768 -d /data/nova/uwsgi.log --limit-post 500000000 --stats 127.0.0.1:1717 \
      --log-format "[pid: %(pid)|worker: %(wid)|req: -/-] %(addr) [%(ctime)] %(method) %(uri) => generated %(rsize) bytes in %(msecs) msecs (%(proto) %(status)) %(headers) headers in %(hsize) bytes (%(switches) switches on core %(core))" \
      --show-config \
      --harakiri 600 \
      --harakiri-verbose

#      --enable-metrics
#      --log-format = "[pid: %(pid)|app: -|req: -/-] %(addr) (%(user)) {%(vars) vars in %(pktsize) bytes} [%(ctime)] %(method) %(uri) => generated %(rsize) bytes in %(msecs) msecs (%(proto) %(status)) %(headers) headers in %(hsize) bytes (%(switches) switches on core %(core))"

# [pid: 2607665|app: 0|req: 415/1850] 66.249.69.248 () {66 vars in 1417 bytes} [Tue May 31 14:50:46 2022] GET /status/2846101 => generated 12287 bytes in 30 msecs (HTTP/1.1 200) 7 headers in 223 bytes (1 switches on core 0)

#/usr/local/bin/uwsgi -s :3030 --wsgi-file wsgi.py --touch-reload wsgi.py --processes 8 --reload-on-rss 768 -d /data/nova/uwsgi.log --limit-post 500000000
