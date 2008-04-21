from astrometry.util.starutil import arcmin2rad
from astrometry.net.server.models import Index

def load_indexes():

    hpinds = [
        #(500, 2.0, 2.8),
        #(501, 2.8, 4.0),
        (502, 4. , 5.6),
        (503, 5.6, 8.),
        (504, 8. , 11.),
        ]

    allskyinds = [
        (505, 11., 16.),
        (506, 16., 22.),
        (507, 22., 30.),
        (508, 30., 42.),
        (509, 42., 60.),
        (510, 60., 85.),
        (511, 85., 120.),
        (512, 120., 170.),
        (513, 170., 240.),
        (514, 240., 340.),
        (515, 340., 480.),
        (516, 480., 680.),
        (517, 680., 1000.),
        (518, 1000., 1400.),
        (519, 1400., 2000.),
        ]

    for (indexid, scalelo, scalehi) in hpinds:
        for hp in range(12):
            slo = arcmin2rad(scalelo)
            shi = arcmin2rad(scalehi)
            (i,nil) = Index.objects.get_or_create(indexid=indexid,
                                                  healpix=hp,
                                                  healpix_nside=1,
                                                  defaults={'scalelo': slo,
                                                            'scalehi': shi,})

    for (indexid, scalelo, scalehi) in allskyinds:
        hp = -1
        slo = arcmin2rad(scalelo)
        shi = arcmin2rad(scalehi)
        (i,nil) = Index.objects.get_or_create(indexid=indexid,
                                              healpix=hp,
                                              healpix_nside=1,
                                              defaults={'scalelo': slo,
                                                        'scalehi': shi,})

