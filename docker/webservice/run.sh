#! /bin/bash

python manage.py runserver 0.0.0.0:8000 &

python -u process_submissions.py --solve-locally=/src/astrometry/net/solvescript-docker.sh &

wait
