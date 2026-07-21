// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file logfunc.h
 * @brief Logfunc module
 */

/**
 * @file logfunc.h
 */

void CORE_logFunctionCall(const int   funclevel,
                          const int   loglevel,
                          const int   logfuncMODE,
                          const char *FileName,
                          const char *FunctionName,
                          const long  line,
                          char       *comments);
