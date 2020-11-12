from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from astrometry.net.models import UserProfile

@receiver(post_save, sender=User)
def add_user_profile(sender, instance, created, raw, **kwargs):
    print('add_user_profile() called.  sender', sender)
    print('inst', instance)
    print('created', created)
    print('raw', raw)

    if created and not raw:
        user = instance
        try:
            print('profile exists:', user.profile)
        except:
            print('profile does not exist -- creating one!')
            from astrometry.net.models import create_new_user_profile
            profile = create_new_user_profile(user)
            profile.save()
            print('Created', user.profile)
