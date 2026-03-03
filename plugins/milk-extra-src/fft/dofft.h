#ifndef FFT_DOFFT_H
#define FFT_DOFFT_H

#include "CLIcore.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"

errno_t dofft_addCLIcmd();

#define DOFFT_PARAMS(X) \
    X( \
        ".in_name", \
        &dofft_inimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "input complex image" \
    ) \
    X( \
        ".out_name", \
        &dofft_outimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "output complex image" \
    ) \
    X( \
        ".dir", \
        &dofft_dir, \
        FPTYPE_INT32, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "FFT direction" \
    )

extern char *dofft_inimname;
extern char *dofft_outimname;
extern int  *dofft_dir;

imageID do1dfft(const char *in_name, const char *out_name);
imageID do1drfft(const char *in_name, const char *out_name);
imageID do1dffti(const char *in_name, const char *out_name);
imageID do2dfft(const char *in_name, const char *out_name);
imageID do2dffti(const char *in_name, const char *out_name);
imageID do2drfft(const char *in_name, const char *out_name);
imageID do2drffti(const char *in_name, const char *out_name);

#define DOFFT_HELPTEXT \
    "dofft: perform 2D FFT\n" \
    "====================\n"

#endif