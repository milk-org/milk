// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file shmim_purge.h
 * @brief Shmim purge module
 */

#ifndef COREMOD_MEMORY_SHMIM_PURGE_H
#define COREMOD_MEMORY_SHMIM_PURGE_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_shmimpurge;

errno_t shmim_purge(const char *strfilter);

#endif
