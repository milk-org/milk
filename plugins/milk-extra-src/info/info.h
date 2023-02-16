#if !defined(INFO_H)
#define INFO_H

#include "info/imagemon.h"

void __attribute__((constructor)) libinit_info();

#include "info/cubeMatchMatrix.h"
#include "info/cubestats.h"
#include "info/image_stats.h"
#include "info/imagemon.h"
#include "info/improfile.h"
#include "info/kbdhit.h"
#include "info/percentile.h"
#include "info/print_header.h"

/*
long brighter(
    const char *ID_name,
    double      value
);
*/

/*
errno_t img_nbpix_flux(
    const char *ID_name
);
*/

// int img_histoc(const char *ID_name, const char *fname);

/*
errno_t make_histogram(
    const char *ID_name,
    const char *ID_out_name,
    double      min,
    double      max,
    long        nbsteps
);
*/

// double ssquare(const char *ID_name);

// double rms_dev(const char *ID_name);

/*
double img_min(const char *ID_name);

double img_max(const char *ID_name);
*/

/*
errno_t printpix(
    const char *ID_name,
    const char *filename
);
*/

/*
double background_photon_noise(
    const char *ID_name
);
*/

/*
int test_structure_function(const char *ID_name, long NBpoints,
                            const char *fname);

imageID full_structure_function(
    const char *ID_name,
    long        NBpoints,
    const char *ID_out
);
*/

imageID info_cubeMatchMatrix(const char *IDin_name, const char *IDout_name);

#endif
