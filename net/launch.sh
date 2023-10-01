sudo systemctl start nova-jobs
sudo systemctl start nova-uwsgi

# For the sake of a minimal deployment, we generate a very simple nova.cfg file on the fly.
# For more detailed usage of the cfg file, check /usr/local/astrometry.cfg or nova.cfg in net folder.
# Disable or modify this if more advanced features are demanded.
echo "add_path /data/INDEXES/" > nova.cfg
ls /data/INDEXES | awk '{printf("index %s\n", $1)}' >> nova.cfg

python manage.py runserver 0.0.0.0:8000