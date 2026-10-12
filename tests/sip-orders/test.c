/* BSD-3-Clause; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "astrometry/fit-wcs.h"
#include "astrometry/sip-utils.h"
#include "astrometry/tweak.h"
#include "astrometry/tweak2.h"
#include "astrometry/wcs-pv2sip.h"
#include "astrometry/fitstable.h"
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
int main(int argc,char** argv) {
 tan_t tan; sip_t out,saved,initial;double xy[242],xyz[363],ra[121],dec[121];memset(&tan,0,sizeof(tan));tan.crval[0]=27;tan.crval[1]=13;tan.cd[0][0]=tan.cd[1][1]=.02;tan.imagew=tan.imageh=2;
 for(int i=0;i<121;i++){xy[2*i]=(i%11)/5.0-1;xy[2*i+1]=(i/11)/5.0-1;tan_pixelxy2xyzarr(&tan,xy[2*i],xy[2*i+1],xyz+3*i);tan_pixelxy2radec(&tan,xy[2*i],xy[2*i+1],ra+i,dec+i);}
 if(argc>1 && !strcmp(argv[1],"reproduce")){int r=fit_sip_wcs_2(xyz,xy,NULL,121,9,10,2,2,0,NULL,0,&out);CHECK(r==-1);puts("PASS reported forward-9 / inverse-10 request rejected");return 0;}
 if(argc>2 && !strcmp(argv[1],"fixture")){fitstable_t* t=fitstable_open_for_writing(argv[2]);CHECK(t);tfits_type d=fitscolumn_double_type();char* names[]={"FIELD_X","FIELD_Y","INDEX_RA","INDEX_DEC"};for(int j=0;j<4;j++)fitstable_add_write_column(t,d,names[j],"");CHECK(!fitstable_write_primary_header(t));CHECK(!fitstable_write_header(t));for(int i=0;i<121;i++)CHECK(!fitstable_write_row(t,xy+2*i,xy+2*i+1,ra+i,dec+i));CHECK(!fitstable_fix_header(t));fitstable_close(t);return 0;}
 int bad[]={-1,SIP_MAXORDER,SIP_MAXORDER+2,INT_MIN,INT_MAX};
 for(int j=0;j<5;j++)for(int side=0;side<2;side++){int f=side?2:bad[j],inv=side?bad[j]:2;memset(&out,0x53,sizeof(out));saved=out;CHECK(fit_sip_wcs(NULL,NULL,NULL,0,NULL,f,inv,0,&out)==-1);CHECK(!memcmp(&out,&saved,sizeof(out)));CHECK(fit_sip_coefficients(NULL,NULL,NULL,0,NULL,f,inv,&out)==-1);CHECK(!memcmp(&out,&saved,sizeof(out)));CHECK(fit_sip_wcs_2(NULL,NULL,NULL,0,f,inv,0,0,0,NULL,0,&out)==-1);CHECK(!memcmp(&out,&saved,sizeof(out)));CHECK(!tweak_just_do_it(NULL,NULL,NULL,NULL,NULL,NULL,0,0,f,inv,0,FALSE,FALSE));CHECK(!tweak2(NULL,0,0,0,0,NULL,0,0,NULL,0,0,0,f,inv,NULL,&out,NULL,NULL,NULL,NULL,NULL,NULL,1));}
 for(int j=0;j<5;j++){CHECK(!wcs_pv2sip_header(NULL,NULL,0,0,0,0,0,0,0,0,bad[j],FALSE,0));CHECK(wcs_pv2sip("nonexistent",0,"unused",FALSE,NULL,0,0,0,0,0,0,0,0,bad[j],FALSE,0)==-1);}
 for(int field=0;field<4;field++){memset(&initial,0,sizeof(initial));int* orders[]={&initial.a_order,&initial.b_order,&initial.ap_order,&initial.bp_order};*orders[field]=10;saved=initial;CHECK(sip_ensure_inverse_polynomials(&initial)==-1);CHECK(sip_compute_inverse_polynomials(&initial,0,0,0,0,0,0)==-1);CHECK(!memcmp(&initial,&saved,sizeof(saved)));tweak_t* t=tweak_new();*t->sip=initial;CHECK(!tweak_advance_to(t,TWEAK_HAS_IMAGE_AD));tweak_go_to(t,TWEAK_HAS_IMAGE_AD);tweak_iterate_to_order(t,2,1);CHECK(!memcmp(t->sip,&initial,sizeof(initial)));tweak_free(t);}
 for(int j=0;j<5;j++){memset(&initial,0,sizeof(initial));CHECK(!tweak2(NULL,0,0,0,0,NULL,0,0,NULL,0,0,0,2,2,&initial,&out,NULL,NULL,NULL,NULL,NULL,NULL,bad[j]));tweak_t* t=tweak_new();tweak_iterate_to_order(t,bad[j],0);tweak_free(t);}
 puts("PASS invalid API orders rejected before input access, allocation or output mutation");
 for(int order=1;order<SIP_MAXORDER;order++){CHECK(!fit_sip_coefficients(xyz,xy,NULL,121,&tan,order,order,&out));CHECK(out.a_order==order&&out.ap_order==order);}
 memset(&initial,0,sizeof(initial));initial.wcstan=tan;initial.a_order=initial.b_order=9;CHECK(!sip_ensure_inverse_polynomials(&initial));CHECK(initial.ap_order==9&&initial.bp_order==9);
 CHECK(!fit_sip_wcs_2(xyz,xy,NULL,121,9,9,2,2,0,NULL,0,&out));CHECK(out.a_order==9&&out.ap_order==9);
 puts("PASS valid fit orders 1..9 and capped automatic inverse order 9");return 0;
}
