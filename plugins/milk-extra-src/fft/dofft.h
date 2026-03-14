/**
 * @file    dofft.h
 * @brief   FFT operations.
 */

#ifndef FFT_DOFFT_H
#define FFT_DOFFT_H

#include "CLIcore.h"

errno_t CLIADDCMD_milkfft__dofft();

imageID do1dfft(
    const char *in_name,
    const char *out_name
);

imageID do1drfft(
    const char *in_name,
    const char *out_name
);

imageID do1dffti(
    const char *in_name,
    const char *out_name
);

imageID do2dfft(
    const char *in_name,
    const char *out_name
);

imageID do2dffti(
    const char *in_name,
    const char *out_name
);

imageID do2drfft(
    const char *in_name,
    const char *out_name
);

imageID do2drffti(
    const char *in_name,
    const char *out_name
);

#endif