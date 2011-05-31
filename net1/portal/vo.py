@login_required
@needs_job
def publishtovo(request):
    job = request.job
    if job.submission.user != request.user:
        return HttpResponse('not your job')
    if not job.solved():
        return HttpResponse('job is not solved')
    wcs = job.tanwcs
    if not wcs:
        return HttpResponse('no wcs')


    # BIG HACK! - look through LD_LIBRARY_PATH if this is still needed...
    if not sip.libraryloaded():
        sip.loadlibrary('/home/gmaps/test/an-common/_sip.so')
    

    img = voImage(user = request.user,
                  field = job.field,
                  image_title = 'Field_%i' % (job.field.id),
                  instrument = '',
                  jdate = 0,
                  wcs = wcs,
                  )
    tanwcs = wcs.to_tanwcs()
    (ra, dec) = tanwcs.pixelxy2radec(wcs.imagew/2, wcs.imageh/2)
    (img.ra_center, img.dec_center) = (ra, dec)
    log('tanwcs: ' + str(tanwcs))
    log('(ra, dec) center: (%f, %f)' % (ra, dec))
    (ramin, ramax, decmin, decmax) = tanwcs.radec_bounds(10)
    (img.ra_min, img.ra_max, img.dec_min, img.dec_max) = (ramin, ramax, decmin, decmax)
    log('Saving vo.Image: ', img)
    img.save()

    return HttpResponse('Done.')
