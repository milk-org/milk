// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_omp_team.h
 * @brief How many threads a parallel loop gets: by work size, capped at the P-cores
 *
 * The count sizes OpenMP parallel regions (num_threads) and the calls to the persistent worker
 * team of COREMOD_arith (arith_team_run(), arith_team.h), which loops that run on every frame use
 * to avoid OpenMP's fork/join and sleep/wake costs.
 */

#ifndef MILK_OMP_TEAM_H
#define MILK_OMP_TEAM_H

#include <stdint.h>

/**
 * @brief Compute a size-aware OpenMP thread count capped at physical P-cores
 */
int milk_omp_team_size(uint64_t nel, uint64_t per_thread);

/** Make milk_omp_team_size() return 1 from now on (forked children: the OpenMP pool is not theirs). */
void milk_omp_force_serial(void);

#endif
