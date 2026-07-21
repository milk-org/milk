// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_setup.h
 * @brief Processinfo setup module
 */

#ifndef _PROCESSINFO_SETUP_H
#define _PROCESSINFO_SETUP_H

#include "processinfo_shm_create.h"
#include <string.h>

PROCESSINFO *processinfo_setup(char       *pinfoname,
                               const char *descriptionstring,
                               const char *msgstring,
                               const char *functionname,
                               const char *filename,
                               int         linenumber);

errno_t processinfo_error(PROCESSINFO *processinfo, char *errmsgstring);

errno_t processinfo_loopstart(PROCESSINFO *processinfo);

/**
 * @brief Initialize an auxiliary PROCESSINFO struct with source coordinates.
 *
 * Use this for networking or sub-thread loops that need separate
 * SHM tracking without overriding the executable's main static processinfo.
 *
 * @param pinfo_ptr  (PROCESSINFO *) pointer variable to assign.
 * @param pinfoname  (const char *) name of the processinfo instance.
 * @param descr      (const char *) description (can be NULL).
 * @param msg        (const char *) status message (can be NULL).
 *
 * Each argument is evaluated exactly once via local temporaries,
 * making the macro safe when arguments have side effects.
 */
#define PROCESSINFO_AUX_SETUP(pinfo_ptr, pinfoname, descr, msg)                             \
    do                                                                                      \
    {                                                                                       \
        const char *const _paux_name  = (pinfoname);                                        \
        const char *const _paux_descr = (descr);                                            \
        const char *const _paux_msg   = (msg);                                              \
        PROCESSINFO      *_paux_pi    = processinfo_shm_create(_paux_name, 0);              \
        (pinfo_ptr)                   = _paux_pi;                                           \
        if (_paux_pi != NULL)                                                               \
        {                                                                                   \
            _paux_pi->loopstat = 0;                                                         \
            strncpy(_paux_pi->source_FUNCTION, __FUNCTION__,                                \
                    STRINGMAXLEN_PROCESSINFO_SRCFUNC - 1);                                  \
            _paux_pi->source_FUNCTION[STRINGMAXLEN_PROCESSINFO_SRCFUNC - 1] = '\0';         \
            strncpy(_paux_pi->source_FILE, __FILE__, STRINGMAXLEN_PROCESSINFO_SRCFILE - 1); \
            _paux_pi->source_FILE[STRINGMAXLEN_PROCESSINFO_SRCFILE - 1] = '\0';             \
            _paux_pi->source_LINE                                       = __LINE__;         \
            if (_paux_descr)                                                                \
            {                                                                               \
                strncpy(_paux_pi->description, _paux_descr,                                 \
                        STRINGMAXLEN_PROCESSINFO_DESCRIPTION - 1);                          \
                _paux_pi->description[STRINGMAXLEN_PROCESSINFO_DESCRIPTION - 1] = '\0';     \
            }                                                                               \
            if (_paux_msg)                                                                  \
            {                                                                               \
                processinfo_WriteMessage(_paux_pi, _paux_msg);                              \
            }                                                                               \
        }                                                                                   \
    } while (0)

#endif
