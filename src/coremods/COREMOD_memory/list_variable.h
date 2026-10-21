// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file list_variable.h
 * @brief List variables in memory
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_listvar;
extern const MILK_RUNNABLE runnable_COREMOD_memory_listvarf;

/** @brief List variables matching optional regex */
errno_t list_variable_ID(const char *regexstr);

errno_t list_variable_ID_file(const char *fname);
