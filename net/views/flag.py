from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from astrometry.net.models import *

@login_required
def update_flags(req, category=None, recipient_id=None):
    redirect_url = '/'
    if req.method == 'POST':
        recipient = None
        if category == 'user_image':
            user_image = get_object_or_404(UserImage, pk=recipient_id)
            selected_flags = req.POST.getlist('flags')
            user_image.update_flags(selected_flags, req.user)
            user_image.save()
            redirect_url = req.POST.get('next','/')

    return redirect(redirect_url)
