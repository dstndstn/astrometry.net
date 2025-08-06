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

uwsgi -s :3030 \
      --wsgi-file wsgi.py --touch-reload wsgi.py \
      --processes 100 --reload-on-rss 768 \
      -d /data/nova/uwsgi.log \
      --limit-post 500000000 \
      --stats 127.0.0.1:1717 --stats-http \
      --log-format "[pid: %(pid)|worker: %(wid)|req: -/-] %(addr) [%(ctime)] %(method) %(uri) => generated %(rsize) bytes in %(msecs) msecs (%(proto) %(status)) %(headers) headers in %(hsize) bytes (%(switches) switches on core %(core))" \
      --show-config \
      --harakiri 600 \
      --harakiri-verbose


# --cheaper-busyness-max                  set the cheaper busyness high percent limit, above that value worker is considered loaded (default 50)
# --cheaper-busyness-min                  set the cheaper busyness low percent limit, below that value worker is considered idle (default 25)
# --cheaper-busyness-multiplier           set initial cheaper multiplier, worker needs to be idle for cheaper-overload*multiplier seconds to be cheaped (default 10)
# --cheaper-busyness-penalty              penalty for respawning workers to fast, it will be added to the current multiplier value if worker is cheaped and than respawned back too fast (default 2)
# --cheaper-busyness-verbose              enable verbose log messages from busyness algorithm
# --cheaper-busyness-backlog-alert        spawn emergency worker(s) if any time listen queue is higher than this value (default 33)
# --cheaper-busyness-backlog-multiplier   set cheaper multiplier used for emergency workers (default 3)
# --cheaper-busyness-backlog-step         number of emergency workers to spawn at a time (default 1)
#     --cheaper-busyness-backlog-nonzero      spawn emergency worker(s) if backlog is > 0 for more then N seconds (default 60)

# cheaper-algo = busyness
# processes = 128                      ; Maximum number of workers allowed
# cheaper = 8                          ; Minimum number of workers allowed
# cheaper-initial = 16                 ; Workers created at startup
# cheaper-overload = 1                 ; Length of a cycle in seconds
# cheaper-step = 16                    ; How many workers to spawn at a time
# 
# cheaper-busyness-multiplier = 30     ; How many cycles to wait before killing workers
# cheaper-busyness-min = 20            ; Below this threshold, kill workers (if stable for multiplier cycles)
# cheaper-busyness-max = 70            ; Above this threshold, spawn new workers
# cheaper-busyness-backlog-alert = 16  ; Spawn emergency workers if more than this many requests are waiting in the queue
# cheaper-busyness-backlog-step = 2    ; How many emergency workers to create if there are too many requests in the queue

#      --enable-metrics
#      --log-format = "[pid: %(pid)|app: -|req: -/-] %(addr) (%(user)) {%(vars) vars in %(pktsize) bytes} [%(ctime)] %(method) %(uri) => generated %(rsize) bytes in %(msecs) msecs (%(proto) %(status)) %(headers) headers in %(hsize) bytes (%(switches) switches on core %(core))"

# [pid: 2607665|app: 0|req: 415/1850] 66.249.69.248 () {66 vars in 1417 bytes} [Tue May 31 14:50:46 2022] GET /status/2846101 => generated 12287 bytes in 30 msecs (HTTP/1.1 200) 7 headers in 223 bytes (1 switches on core 0)

#/usr/local/bin/uwsgi -s :3030 --wsgi-file wsgi.py --touch-reload wsgi.py --processes 8 --reload-on-rss 768 -d /data/nova/uwsgi.log --limit-post 500000000
