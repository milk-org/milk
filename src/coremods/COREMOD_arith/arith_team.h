// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file arith_team.h
 * @brief Persistent spin-barrier worker team for low-latency parallel block dispatch.
 *
 * Runs a loop on threads that stay alive between calls instead of opening an OpenMP parallel
 * region: no fork/join per call, and the workers spin for MILK_CALC_SPIN_US between jobs before
 * sleeping, so loops that run on every frame (calculator passes, reductions, spatial kernels)
 * avoid OpenMP's sleep/wake cost. The caller picks the thread count, usually with
 * milk_omp_team_size() (libmilkcommon/milk_omp_team.h).
 */

#ifndef COREMOD_ARITH_TEAM_H
#define COREMOD_ARITH_TEAM_H

#include <stdint.h>

/**
 * arith_team_job_fn - Worker callback handling item range `[lo, hi)` on thread `tid` of `nth`
 */
typedef void (*arith_team_job_fn)(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi);

/* Run @fn over items [0, @nitems) in @nth static partitions, with caller as thread 0 */
int arith_team_run(int nth, uint64_t nitems, arith_team_job_fn fn, void *ctx);

/* Run @fn over items [0, @nitems) dynamically handed out in chunks of @chunk */
int arith_team_run_chunks(int               nth,
                          uint64_t          nitems,
                          uint64_t          chunk,
                          arith_team_job_fn fn,
                          void             *ctx);

/* Enter/leave the team's critical section for merging per-thread partial accumulators */
void arith_team_lock(void);
void arith_team_unlock(void);

#endif /* COREMOD_ARITH_TEAM_H */
