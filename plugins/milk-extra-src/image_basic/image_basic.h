#ifndef _BASIC_H
#define _BASIC_H

#include "image_basic/image_add.h"

#include "image_basic/cubecollapse.h"
#include "image_basic/extrapolate_nearestpixel.h"
#include "image_basic/im3Dto2D.h"
#include "image_basic/imcontract.h"
#include "image_basic/imexpand.h"
#include "image_basic/imgetcircasym.h"
#include "image_basic/imgetcircsym.h"
#include "image_basic/imresize.h"
#include "image_basic/imrotate.h"
#include "image_basic/imstretch.h"
#include "image_basic/imswapaxis2D.h"
#include "image_basic/indexmap.h"
#include "image_basic/loadfitsimgcube.h"
#include "image_basic/measure_transl.h"
#include "image_basic/naninf2zero.h"
#include "image_basic/streamfeed.h"
#include "image_basic/streamrecord.h"
#include "image_basic/tableto2Dim.h"

void __attribute__((constructor)) libinit_image_basic();

/*

int basic_lmin_im(const char *ID_name, const char *out_name);

int basic_lmax_im(const char *ID_name, const char *out_name);


long basic_diff(const char *ID1_name, const char *ID2_name,
                const char *ID3_name, long off1, long off2);

imageID basic_extract(const char *ID_in_name, const char *ID_out_name, long n1,
                   long n2, long n3, long n4);

int basic_trunc_circ(const char *ID_name, float f1);



imageID basic_zoom2(const char *ID_name, const char *ID_name_out);



long basic_average_column(const char *ID_name, const char *IDout_name);

long basic_padd(const char *ID_name, const char *ID_name_out, int n1, int n2);

long basic_fliph(const char *ID_name);

long basic_flipv(const char *ID_name);

long basic_fliphv(const char *ID_name);

int basic_median(const char *ID_name, const char *options);

long basic_renorm_max(const char *ID_name);



int basic_translate(const char *ID_name, const char *ID_out, float xtransl,
                    float ytransl);

float basic_correlation(const char *ID_name1, const char *ID_name2);

long IMAGE_BASIC_get_assym_component(const char *ID_name,
                                     const char *ID_out_name, float xcenter, float ycenter, const char *options);






int gauss_histo_image(const char *ID_name, const char *ID_out_name, float sigma,
                      float center);

long load_fitsimages(const char *strfilter);

long basic_cube_center(const char *ID_in_name, const char *ID_out_name);

long cube_average(const char *ID_in_name, const char *ID_out_name, float alpha);



long basic_addimagesfiles(const char *strfilter, const char *outname);

long basic_pasteimages(const char *prefix, long NBcol, const char *IDout_name);

long basic_aveimagesfiles(const char *strfilter, const char *outname);

long basic_addimages(const char *prefix, const char *ID_out);

long basic_averageimages(const char *prefix, const char *ID_out);





imageID image_basic_3Dto2D(
    const char *IDname
);

*/

#endif
