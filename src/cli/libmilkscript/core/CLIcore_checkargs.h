// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_checkargs.h
 *
 * @brief Check CLI command line arguments
 *
 */

#ifndef CLICORE_CHECKARGS_H
#define CLICORE_CHECKARGS_H


#include "fps_types.h"
#include "milkDebugTools.h"

#include "libmilkdata/milkdata_clicmd.h"

#define CLICMD_SUCCESS 0
#define CLICMD_INVALID_ARG 1
#define CLICMD_ERROR 2

/* Function declarations — only in full CLI mode.
 * In standalone (MILK_NO_CLI), these are provided
 * as static inline stubs in CLIcore_standalone.h.
 */
#ifndef MILK_NO_CLI

int CLI_checkarg(int argnum, uint32_t argtype);

int CLI_checkarg_noerrmsg(int argnum, uint32_t argtype);

errno_t CLI_checkarg_array(CLICMDARGDEF fpscliarg[], int nbarg);

int CLIargs_to_FPSparams_setval(CLICMDARGDEF fpscliarg[], int nbarg, FPS *fps);

int CMDargs_to_FPSparams_create(FPS *fps);

void *get_farg_ptr(char *tag, long *fpsi);

#endif /* !MILK_NO_CLI */

#endif
