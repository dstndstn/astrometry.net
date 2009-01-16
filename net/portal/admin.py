from django.contrib import admin

from astrometry.net.portal.models import PendingAccount

class PendingAccountAdmin(admin.ModelAdmin):
    pass

#admin.site.register(PendingAccount)

admin.site.register(PendingAccount, PendingAccountAdmin)

