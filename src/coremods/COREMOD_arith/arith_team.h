// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file arith_team.h
 * @brief Persistent spin-barrier worker team for low-latency parallel block dispatch.
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
