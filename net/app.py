from django.apps import AppConfig

class AstrometryNetConfig(AppConfig):
    name = 'astrometry.net'

    def ready(self):
        from astrometry.net.signals import add_user_profile

