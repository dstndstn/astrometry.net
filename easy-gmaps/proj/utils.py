import math

def asinh(x):
    return math.log(x + math.sqrt(1. + x*x))

def get_bb(request):
    try:
        bb = request.GET['bb']
    except (KeyError):
        raise KeyError('No bb')
    bbvals = bb.split(',')
    if (len(bbvals) != 4):
        raise KeyError('Bad bb')
    longmin  = float(bbvals[0])
    latmin   = float(bbvals[1])
    longmax  = float(bbvals[2])
    latmax   = float(bbvals[3])
    pi = math.pi
    latminrad = latmin * pi / 180.0
    latmaxrad = latmax * pi / 180.0
    xmin = longmin / 360.0
    xmax = longmax / 360.0
    if xmin < 0:
        xmin += 1.0
        xmax += 1.0
    ymin = 0.5 + (asinh(math.tan(latminrad)) / (2.0 * pi))
    ymax = 0.5 + (asinh(math.tan(latmaxrad)) / (2.0 * pi))
    #newymax = 1. - ymin
    #ymin = 1. - ymax
    #ymax = newymax
    return (xmin, xmax, ymin, ymax)


def get_imagesize(request):
    try:
        imw = int(request.GET['w'])
        imh = int(request.GET['h'])
    except (KeyError):
        raise KeyError('No w/h')
    if (imw == 0 or imh == 0):
        raise KeyError('No w/h')
    return (imw, imh)

