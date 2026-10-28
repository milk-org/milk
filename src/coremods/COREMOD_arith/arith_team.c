// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file arith_team.c
 * @brief Persistent spin-barrier worker team: one dispatch per block loop, no fork/join.
 *
 * Role & Goal:
 *   Provides a process-wide persistent worker pool for `COREMOD_arith` and `libmilkscript/calc`
 *   block evaluators, reductions, and spatial pre-stages. The caller participates as thread 0,
 *   and workers spin for `MILK_CALC_SPIN_US` (default 2000 us) after a job before sleeping on a
 *   condition variable, avoiding libgomp sleep/wake penalties in real-time loops.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE /* pthread_setname_np */
#endif
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arith_team.h"

#define TEAM_MAX 64

typedef struct
{
    arith_team_job_fn fn;
    void             *ctx;
    uint64_t          nitems;
    uint64_t          chunk; /* 0: static partition; else items handed out in chunks of this size */
    int               nth;
} team_job_t;

static pthread_t       workers[TEAM_MAX];
static int             nworkers; /* worker i runs as thread i + 1 */
static team_job_t      job;
static uint64_t        job_gen;    /* bumped to publish a job */
static uint64_t        next_chunk; /* chunked jobs: the next chunk to hand out */
static int             done;       /* participants that finished the current job */
static int             sleepers;   /* workers waiting on the condition variable */
static int64_t         spin_ns   = -1;
static pthread_mutex_t mtx       = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cv        = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t merge_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  once      = PTHREAD_ONCE_INIT;

static inline int64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000000LL + t.tv_nsec;
}

static inline void cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

/** run_part() - do thread @tid's share of the published job (nothing when it has no share). */
static void run_part(int tid)
{
    const team_job_t j = job;
    if (tid >= j.nth)
    {
        return;
    }
    if (j.chunk == 0)
    {
        uint64_t lo = j.nitems * (uint64_t) tid / (uint64_t) j.nth;
        uint64_t hi = j.nitems * (uint64_t) (tid + 1) / (uint64_t) j.nth;
        if (hi > lo)
        {
            j.fn(j.ctx, tid, j.nth, lo, hi);
        }
        return;
    }
    for (;;)
    {
        uint64_t c  = __atomic_fetch_add(&next_chunk, 1, __ATOMIC_RELAXED);
        uint64_t lo = c * j.chunk;
        if (lo >= j.nitems)
        {
            return;
        }
        uint64_t hi = (lo + j.chunk < j.nitems) ? lo + j.chunk : j.nitems;
        j.fn(j.ctx, tid, j.nth, lo, hi);
    }
}

static void *worker_main(void *arg)
{
    const int tid  = (int) (intptr_t) arg;
    uint64_t  seen = 0;
    for (;;)
    {
        uint64_t g;
        int64_t  t0 = -1;
        while ((g = __atomic_load_n(&job_gen, __ATOMIC_ACQUIRE)) == seen)
        {
            cpu_relax();
            if (spin_ns < 0)
            {
                continue; /* always spin */
            }
            if (t0 < 0)
            {
                t0 = now_ns();
            }
            else if (now_ns() - t0 > spin_ns)
            {
                pthread_mutex_lock(&mtx);
                sleepers++;
                while (__atomic_load_n(&job_gen, __ATOMIC_ACQUIRE) == seen)
                {
                    pthread_cond_wait(&cv, &mtx);
                }
                sleepers--;
                pthread_mutex_unlock(&mtx);
                t0 = -1;
            }
        }
        seen = g;
        if (tid < job.nth)
        {
            run_part(tid);
            __atomic_fetch_add(&done, 1, __ATOMIC_RELEASE);
        }
    }
    return NULL;
}

/** after_fork_child() - a child has no workers: start over on its first job. */
static void after_fork_child(void)
{
    nworkers = 0;
    sleepers = 0;
    done     = 0;
    pthread_mutex_init(&mtx, NULL);
    pthread_cond_init(&cv, NULL);
    pthread_mutex_init(&merge_mtx, NULL);
}

static void team_init(void)
{
    const char *env = getenv("MILK_CALC_SPIN_US");
    spin_ns         = 2000LL * 1000LL;
    if (env != NULL && env[0] != '\0')
    {
        long us = atol(env);
        spin_ns = (us < 0) ? -1 : us * 1000LL;
    }
    pthread_atfork(NULL, NULL, after_fork_child);
}

/** ensure_workers() - have at least @nth - 1 workers; the team size the process can offer. */
static int ensure_workers(int nth)
{
    pthread_once(&once, team_init);
    while (nworkers < nth - 1 && nworkers < TEAM_MAX)
    {
        pthread_t th;
        if (pthread_create(&th, NULL, worker_main, (void *) (intptr_t) (nworkers + 1)) != 0)
        {
            break;
        }
        char name[16];
        snprintf(name, sizeof(name), "milkcalc-%d", nworkers + 1);
        pthread_setname_np(th, name);
        workers[nworkers++] = th;
    }
    return (nth - 1 <= nworkers) ? nth : nworkers + 1;
}

/** dispatch() - publish a job, run thread 0's share here, wait for the others. */
static int dispatch(int nth, uint64_t nitems, uint64_t chunk, arith_team_job_fn fn, void *ctx)
{
    if (nitems == 0)
    {
        return 1;
    }
    if (nth > 1 && chunk == 0 && (uint64_t) nth > nitems)
    {
        nth = (int) nitems;
    }
    if (nth > 1)
    {
        nth = ensure_workers(nth);
    }
    if (nth <= 1)
    {
        if (chunk == 0)
        {
            fn(ctx, 0, 1, 0, nitems);
        }
        else
        {
            for (uint64_t lo = 0; lo < nitems; lo += chunk)
            {
                fn(ctx, 0, 1, lo, (lo + chunk < nitems) ? lo + chunk : nitems);
            }
        }
        return 1;
    }
    job.fn     = fn;
    job.ctx    = ctx;
    job.nitems = nitems;
    job.chunk  = chunk;
    job.nth    = nth;
    __atomic_store_n(&next_chunk, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&done, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&job_gen, 1, __ATOMIC_RELEASE);
    if (__atomic_load_n(&sleepers, __ATOMIC_ACQUIRE) > 0)
    {
        pthread_mutex_lock(&mtx);
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mtx);
    }
    run_part(0);
    while (__atomic_load_n(&done, __ATOMIC_ACQUIRE) < nth - 1)
    {
        cpu_relax();
    }
    return nth;
}

/**
 * arith_team_run - Run @fn over items [0, @nitems) in @nth static ranges, this thread as 0
 * @nth:    Team size wanted (1 runs @fn inline)
 * @nitems: Number of items (blocks, rows, outputs)
 * @fn:     Job: `fn(ctx, tid, nth, lo, hi)` handles items [lo, hi)
 * @ctx:    Passed to @fn
 *
 * Return: The team size used (`<= nth`): every `tid` seen by @fn is below it.
 */
int arith_team_run(int nth, uint64_t nitems, arith_team_job_fn fn, void *ctx)
{
    return dispatch(nth, nitems, 0, fn, ctx);
}

/**
 * arith_team_run_chunks - Run @fn over items [0, @nitems) handed out in chunks of @chunk
 * @nth:    Team size wanted (1 runs @fn inline)
 * @nitems: Number of items
 * @chunk:  Items per call of @fn (uneven work: bins of a segmented reduction)
 * @fn:     Job: `fn(ctx, tid, nth, lo, hi)` handles items [lo, hi)
 * @ctx:    Passed to @fn
 *
 * Return: The team size used.
 */
int arith_team_run_chunks(int nth, uint64_t nitems, uint64_t chunk, arith_team_job_fn fn, void *ctx)
{
    return dispatch(nth, nitems, (chunk > 0) ? chunk : 1, fn, ctx);
}

/**
 * arith_team_lock - Enter the team's merge section (per-thread partials into the result)
 *
 * Return: void
 */
void arith_team_lock(void)
{
    pthread_mutex_lock(&merge_mtx);
}

/**
 * arith_team_unlock - Leave the team's merge section
 *
 * Return: void
 */
void arith_team_unlock(void)
{
    pthread_mutex_unlock(&merge_mtx);
}
