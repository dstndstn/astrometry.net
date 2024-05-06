FROM astrometrynet/solver:latest

#USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt install -y --no-install-recommends \
    apache2 \
    libapache2-mod-wsgi-py3 \
    less \
    emacs-nox \
    tmux \
    systemctl \
    sqlite3

RUN pip3 install --no-cache-dir \
    django \
    social-auth-core django-social-auth3 social-auth-app-django \
    uwsgi

WORKDIR /src/astrometry/net

#USER nova
# For local development, re-copy the current directory into the container.
#ADD --chown=nova:nova . /src/astrometry

ADD . /src/astrometry

RUN ln -sf settings_test.py settings.py

ENV PYTHONPATH /src:${PYTHONPATH}

RUN mkdir appsecrets && \
    touch appsecrets/__init__.py && \
    touch appsecrets/auth.py && \
    cp appsecrets-example/django.py appsecrets

COPY docker/webservice/django_db.py appsecrets/
#RUN (echo -n "DJANGO_SECRET_KEY = '"; tr -dc A-Za-z0-9 </dev/urandom | head -c 52; echo "'") > /src/astrometry/net/appsecrets/django.py

RUN mv migrations/* /tmp && \
    python manage.py makemigrations && \
    python manage.py migrate && \
    python manage.py makemigrations net && \
    python manage.py migrate net && \
    python manage.py loaddata fixtures/initial_data.json && \
    python manage.py loaddata fixtures/flags.json

#USER root

# If you wanted to use the systemctl scripts instead of run.sh, you
# would need to edit these files a bit first (changing /home/nova to
# /src, removing User=nova, or enable the nova user in the Dockerfile), and then:
# RUN cp nova-jobs.service  /etc/systemd/system
# RUN cp nova-uwsgi.service /etc/systemd/system
# RUN systemctl enable nova-jobs
# RUN systemctl enable nova-uwsgi

WORKDIR /src

ENV WSGI_LOG_FILE=

#USER root
CMD ["/bin/bash", "./astrometry/docker/webservice/run.sh"]

EXPOSE 8000
