// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    dofft.h
 * @brief   FFT operations.
 */

#ifndef FFT_DOFFT_H
#define FFT_DOFFT_H

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif

imageID do1dfft(const char *in_name, const char *out_name);

imageID do1drfft(const char *in_name, const char *out_name);

imageID do1dffti(const char *in_name, const char *out_name);

imageID do2dfft(const char *in_name, const char *out_name);

imageID do2dffti(const char *in_name, const char *out_name);

imageID do2drfft(const char *in_name, const char *out_name);

imageID do2drffti(const char *in_name, const char *out_name);

#endif
