#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"

errno_t gaussfilter_addCLIcmd();

imageID gauss_filter(const char *__restrict ID_name,
                     const char *__restrict out_name,
                     float sigma,
                     int   filter_size);

imageID gauss_3Dfilter(const char *__restrict ID_name,
                       const char *__restrict out_name,
                       float sigma,
                       int   filter_size);

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define GAUSSFILT_PARAMS(X) \
    X( \
        ".in_name", \
        &gaussfilt_inimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "input image" \
    ) \
    X( \
        ".out_name", \
        &gaussfilt_outimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "output image" \
    ) \
    X( \
        ".sigma", \
        &gaussfilt_sigma, \
        FPTYPE_FLOAT32, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "gaussian sigma" \
    ) \
    X( \
        ".filter_size", \
        &gaussfilt_filtersize, \
        FPTYPE_INT32, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "filter box size" \
    )

extern char  *gaussfilt_inimname;
extern char  *gaussfilt_outimname;
extern float *gaussfilt_sigma;
extern int   *gaussfilt_filtersize;

#define GAUSSFILT_HELPTEXT \
    "gaussfilt: gaussian 2D filtering\n" \
    "===============================\n" \
    "Applies a Gaussian low-pass filter to a 2D image.\n"

