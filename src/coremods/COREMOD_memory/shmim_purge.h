// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file shmim_purge.h
 * @brief Shmim purge module
 */

#ifndef COREMOD_MEMORY_SHMIM_PURGE_H
#define COREMOD_MEMORY_SHMIM_PURGE_H

errno_t CLIADDCMD_COREMOD_memory__shmim_purge();

errno_t shmim_purge(const char *strfilter);

#endif
