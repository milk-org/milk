#include "fps.h"
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
    X(    FPTYPE_STREAMNAME, char*,  ".in_name",     "input image",            "im1",  "im1", &gaussfilt_inimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(    FPTYPE_STREAMNAME, char*,  ".out_name",    "output image",           "out1", "out1", &gaussfilt_outimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_FLOAT32,    float,  ".sigma",       "gaussian sigma",         "2.0",  2.0,   &gaussfilt_sigma, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(  FPTYPE_INT32,      int,    ".filter_size", "filter box size",        "5",    5,     &gaussfilt_filtersize, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

extern char  *gaussfilt_inimname;
extern char  *gaussfilt_outimname;
extern float *gaussfilt_sigma;
extern int   *gaussfilt_filtersize;

#define GAUSSFILT_HELPTEXT \
    "gaussfilt: gaussian 2D filtering\n" \
    "===============================\n" \
    "Applies a Gaussian low-pass filter to a 2D image.\n"

