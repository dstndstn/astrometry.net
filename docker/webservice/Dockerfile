FROM astrometrynet/solver:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt install -y --no-install-recommends \
    apache2 \
    libapache2-mod-wsgi-py3 \
    less \
    emacs-nox \
    tmux \
    systemctl

# Yuck! The installed 'astrometry' package conflicts with '.', so paste it in...
RUN rm -R /usr/local/lib/python/astrometry/net && \
    ln -s /home/nova/astrometry/net /usr/local/lib/python/astrometry/net

# Create the log dirs to avoid errors of not able to write to log files.
RUN mkdir /data && \
    mkdir /data/nova && \
    mkdir /data1 && \
    mkdir /data1/nova && \
    mkdir /data1/nova/tmp && \
    chown -R nova:nova /data && \
    chown -R nova:nova /data1

USER nova
RUN pip3 install --no-cache-dir \
    django \
    social-auth-core django-social-auth3 social-auth-app-django \
    astropy \
    fitsio \
    uwsgi

WORKDIR /home/nova/astrometry/net
# For some reason need to recompile to resolve some weird GLIBC errors...
RUN cd .. && \
    make clean && \
    make -j && \
    make extra -j
RUN ln -s settings_test.py settings.py
RUN mv migrations/* /tmp && \
    python manage.py makemigrations && \
    python manage.py migrate && \
    python manage.py makemigrations net && \
    python manage.py migrate net && \
    python manage.py loaddata fixtures/initial_data.json && \
    python manage.py loaddata fixtures/flags.json

USER root
RUN cd .. && make install INSTALL_DIR=/usr/local
RUN ln -s /home/nova/.local/bin/uwsgi /usr/local/bin/uwsgi  
RUN cp nova-jobs.service /etc/systemd/system
RUN cp nova-uwsgi.service /etc/systemd/system
RUN systemctl enable nova-jobs
RUN systemctl enable nova-uwsgi
RUN systemctl start nova-jobs
RUN systemctl start nova-uwsgi

USER nova
CMD ["/bin/bash", "./launch.sh"]

EXPOSE 8000