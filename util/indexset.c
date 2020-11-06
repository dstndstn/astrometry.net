#include <math.h>
#include "indexset.h"
#include "ioutils.h"
#include "index.h"

void indexset_get(const char* name, pl* indexlist) {

    if (streq(name, "5000")) {
        int scale, hp, maxhp, nside;
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
                index_t* ind = calloc(1, sizeof(index_t));
                char* iname;
                asprintf_safe(&iname, "index-%i-%02i.fits", 5000 + scale, hp);
                ind->indexname = iname;
                ind->indexid = 5000 + scale;
                ind->healpix = hp;
                ind->hpnside = nside;
                ind->index_scale_lower = 60. * scales[scale];
                ind->index_scale_upper = 60. * scales[scale+1];
                ind->circle = TRUE;
                ind->cx_less_than_dx = TRUE;
                ind->meanx_less_than_half = TRUE;
                ind->dimquads = 4;
                pl_append(indexlist, ind);
            }
        }
    } else if (streq(name, "5200")) {
        // scales 00 through 06, all hp nside=2
        int scale, hp, maxhp, nside;
        double scales[] = { 2.0, 2.8, 4., 5.6, 8., 11., 16., 22. };
        for (scale=0; scale<=6; scale++) {
  	    maxhp = 48;
	    nside = 2;
            for (hp=0; hp<maxhp; hp++) {
                index_t* ind = calloc(1, sizeof(index_t));
                char* iname;
                asprintf_safe(&iname, "index-%i-%02i.fits", 5200 + scale, hp);
                ind->indexname = iname;
                ind->indexid = 5200 + scale;
                ind->healpix = hp;
                ind->hpnside = nside;
                ind->index_scale_lower = 60. * scales[scale];
                ind->index_scale_upper = 60. * scales[scale+1];
                ind->circle = TRUE;
                ind->cx_less_than_dx = TRUE;
                ind->meanx_less_than_half = TRUE;
                ind->dimquads = 4;
                pl_append(indexlist, ind);
            }
        }
    } else if (streq(name, "5400")) {
        // scales 00 through 06, all hp nside=4
        int scale, hp, maxhp, nside;
        double scales[] = { 2.0, 2.8, 4., 5.6, 8., 11., 16., 22. };
        for (scale=0; scale<=6; scale++) {
  	    maxhp = 192;
	    nside = 4;
            for (hp=0; hp<maxhp; hp++) {
                index_t* ind = calloc(1, sizeof(index_t));
                char* iname;
                asprintf_safe(&iname, "index-%i-%03i.fits", 5400 + scale, hp);
                ind->indexname = iname;
                ind->indexid = 5400 + scale;
                ind->healpix = hp;
                ind->hpnside = nside;
                ind->index_scale_lower = 60. * scales[scale];
                ind->index_scale_upper = 60. * scales[scale+1];
                ind->circle = TRUE;
                ind->cx_less_than_dx = TRUE;
                ind->meanx_less_than_half = TRUE;
                ind->dimquads = 4;
                pl_append(indexlist, ind);
            }
        }
    }
}

