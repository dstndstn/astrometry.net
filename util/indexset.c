#include <math.h>
#include "indexset.h"
#include "ioutils.h"
#include "index.h"

void indexset_get(const char* name, bl* indexlist) {
    index_t ind;
    memset(&ind, 0, sizeof(ind));

    if (streq(name, "5000")) {
        int scale, hp, maxhp, nside;
        //[ 2.        ,  2.82842712,  4.        ,  5.65685425,  8.        ,
        //11.3137085 , 16.        , 22.627417  , 32.        ])
        double scales[] = { 2.0, 2.8, 4., 5.6, 8., 11., 16., 22., 30., 42. };
        for (scale=0; scale<8; scale++) {
            if (scale < 5) {
                maxhp = 48;
                nside = 2;
            } else {
                maxhp = 12;
                nside = 1;
            }
            for (hp=0; hp<maxhp; hp++) {
                char* iname;
                asprintf_safe(&iname, "index-%i-%02i.fits", 5000 + scale, hp);
                ind.indexname = iname;
                ind.indexid = 5000 + scale;
                ind.healpix = hp;
                ind.hpnside = nside;
                ind.index_scale_lower = 60. * scales[scale];
                ind.index_scale_upper = 60. * scales[scale+1];
                //ind.index_jitter = ;
                //ind.cutnside = ;

                ind.circle = TRUE;
                ind.cx_less_than_dx = TRUE;
                ind.meanx_less_than_half = TRUE;
                ind.dimquads = 4;

                bl_append(indexlist, &ind);
            }
        }
    }
}

