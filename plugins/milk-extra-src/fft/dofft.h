#ifndef FFT_DOFFT_H
#define FFT_DOFFT_H

#include "CLIcore.h"
#include "fps.h"
#include "processinfo.h"

errno_t dofft_addCLIcmd();

#define DOFFT_PARAMS(X) \
    X(CLIARG_IMG, FPTYPE_STREAMNAME, char*, ".in_name",  "input complex image",  "im1", "im1", &dofft_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR, FPTYPE_STREAMNAME, char*, ".out_name", "output complex image", "out1","out1",&dofft_outimname,(void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_INT32, FPTYPE_INT32,    int,   ".dir",      "FFT direction",        "-1",  -1,    &dofft_dir,       (void*)&val, CLIARG_VISIBLE_DEFAULT)

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