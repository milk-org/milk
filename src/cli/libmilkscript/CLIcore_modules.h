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

errno_t load_module_shared_local();

errno_t RegisterModule(CONST_WORD FileName,
                       CONST_WORD PackageName,
                       CONST_WORD InfoString,
                       int        versionmajor,
                       int        versionminor,
                       int        versionpatch);

uint32_t RegisterCLIcommand(CONST_WORD CLIkey,
                            CONST_WORD CLImodulesrc,
                            errno_t (*CLIfptr)(),
                            CONST_WORD CLIinfo,
                            CONST_WORD CLIsyntax,
                            CONST_WORD CLIexample,
                            CONST_WORD CLICcall);

uint32_t RegisterCLIcmd(CLICMDDATA CLIcmddata, errno_t (*CLIfptr)());

#endif
