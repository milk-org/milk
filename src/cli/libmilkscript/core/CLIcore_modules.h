// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_modules.h
 *
 * @brief Modules functions
 *
 */

#ifndef CLICORE_MODULES_H
#define CLICORE_MODULES_H


typedef const char *__restrict CONST_WORD;

errno_t load_sharedobj(CONST_WORD libname);

errno_t load_module_shared(CONST_WORD modulename);

struct MILK_RUNNABLE_;

errno_t RegisterModule(CONST_WORD FileName,
                       CONST_WORD PackageName,
                       CONST_WORD InfoString,
                       int        versionmajor,
                       int        versionminor,
                       int        versionpatch);

void milk_register_runnable(const struct MILK_RUNNABLE_ *r);

#endif
