FROM astrometrynet/solver:latest

RUN pip3 install --no-cache-dir Django

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt install -y --no-install-recommends \
    apache2 \
    libapache2-mod-wsgi-py3 \
    less \
    emacs-nox \
    ssh \
    openssh-server

# Config the ssh server and client that are required for the web service
RUN ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519
RUN ssh-keygen -y -f ~/.ssh/id_ed25519 >> ~/.ssh/authorized_keys 
RUN cp /src/astrometry/docker/webservice/config ~/.ssh/
RUN service ssh start
RUN update-rc.d ssh defaults 

RUN pip3 install --no-cache-dir \
    social-auth-core django-social-auth3 social-auth-app-django

WORKDIR /src/astrometry/net

RUN ln -s settings_test.py settings.py

# Yuck! The installed 'astrometry' package conflicts with '.', so paste it in...
RUN rm -R /usr/local/lib/python/astrometry/net && \
    ln -s /src/astrometry/net /usr/local/lib/python/astrometry/net

RUN mv migrations/* /tmp && \
    python manage.py makemigrations && \
    python manage.py migrate && \
    python manage.py makemigrations net && \
    python manage.py migrate net && \
    python manage.py loaddata fixtures/initial_data.json && \
    python manage.py loaddata fixtures/flags.json

# Create the log dirs to avoid errors of not able to write to log files.
RUN mkdir /data
RUN mkdir /data/nova

# Listen to all the bound addresses
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

EXPOSE 8000
