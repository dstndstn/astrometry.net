#! /bin/bash

python manage.py runserver 0.0.0.0:8000 &

python process_submissions.py &

wait
