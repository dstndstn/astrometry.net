#! /bin/bash

cd /src/astrometry/net

# solvescript-docker.sh reads the config file /index/docker.cfg.  If that file does not exist,
# create a default version.
if [ ! -r /index/docker.cfg ]; then
    echo "Creating default /index/docker.cfg file..."
    cat <<EOF > /index/docker.cfg
add_path /index
autoindex
inparallel
EOF
fi

python -u manage.py runserver 0.0.0.0:8000 &
python -u process_submissions.py --solve-locally=/src/astrometry/docker/webservice/solvescript-docker.sh &
wait
