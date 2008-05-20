def solve(request):
    log('master.solve')

    if not 'axy' in request.POST:
        return HttpResponse('no axy')
    if not 'jobid' in request.POST:
        return HttpResponse('no jobid')

    jobid = request.POST['jobid']
    axy = request.POST['axy']
    # FIXME
    axy = axy.decode('base64_codec')




def cancel(request):
    jobid = request.GET.get('jobid')
    if not jobid:
        return HttpResponseBadRequest('no jobid')
