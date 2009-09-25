import os.path



def sdss_filename(filetype, run, camcol, field, band=None, rerun=None):
    x = dict(run=run, band=band, camcol=camcol, field=field, rerun=rerun)
    if filetype == 'fpC':
        return 'fpC-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit' % x
    elif filetype == 'psField':
        return 'psField-%(run)06i-%(camcol)i-%(field)04i.fit' % x
    else:
        return None

def sdss_path(filetype, run, camcol, field, band=None, rerun=None):
    x = dict(run=run, band=band, camcol=camcol, field=field, rerun=rerun)
    y = (run, camcol, field, band, rerun)
    if filetype == 'fpC':
        return '%(run)i/%(rerun)i/corr/%(camcol)i/' % x + sdss_filename(filetype, *y)
    elif filetype == 'psField':
        return '%(run)i/%(rerun)i/objcs/%(camcol)i/' % x + sdss_filename(filetype, *y)

def sdss_find_file(filetype, run, camcol, field, band=None, reruns=None, datadir=None, reduxdir=None):
    if filetype == 'psField':
        basedir = datadir
        for rerun in reruns:
            pth = os.path.join(basedir, sdss_path(filetype, run, camcol, field, rerun=rerun))
            print 'trying path', pth
            if os.path.exists(pth):
                return pth
    return None

