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
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*,  ".in_name",     "input image",            "im1",  "im1", &gaussfilt_inimname,   (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR,     FPTYPE_STREAMNAME, char*,  ".out_name",    "output image",           "out1", "out1", &gaussfilt_outimname,  (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_FLOAT32, FPTYPE_FLOAT32,    float,  ".sigma",       "gaussian sigma",         "2.0",  2.0,   &gaussfilt_sigma,      (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_INT32,   FPTYPE_INT32,      int,    ".filter_size", "filter box size",        "5",    5,     &gaussfilt_filtersize, (void*)&val, CLIARG_VISIBLE_DEFAULT)

extern char  *gaussfilt_inimname;
extern char  *gaussfilt_outimname;
extern float *gaussfilt_sigma;
extern int   *gaussfilt_filtersize;

#define GAUSSFILT_HELPTEXT \
    "gaussfilt: gaussian 2D filtering\n" \
    "===============================\n" \
    "Applies a Gaussian low-pass filter to a 2D image.\n"

