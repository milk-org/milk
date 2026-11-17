// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trigmerge.c
 * @brief   Merge the triggers of several streams into one trigger stream
 *
 * Waits on up to 16 input streams (any-of or all-of) and, each time the
 * condition is met, writes and posts a small int64 output stream. A unit
 * triggers on that stream with its usual single trigger. The output layout
 * is documented in docs/guide/procinfo.md ("Merging triggers").
 *
 * The inputs are waited on in the loop body: the unit's own trigger is
 * IMMEDIATE, or DELAY (-loopd) to cap the merged rate. Clients that fall
 * behind skip frames (no backpressure).
 */

#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "trigmerge.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

#define TRIGMERGE_NIN 16
#define TRIGMERGE_OUTSIZE (4 + 3 * TRIGMERGE_NIN)
#define TRIGMERGE_SLICE_NS 100000000L /* longest single wait, so stop requests are seen */

static char   trigmerge_in[TRIGMERGE_NIN][FUNCTION_PARAMETER_STRMAXLEN];
static char   trigmerge_out[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char   trigmerge_logic[FUNCTION_PARAMETER_STRMAXLEN] = "any";
static double trigmerge_timeout                             = 0.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define TRIGMERGE_IN(X, n) \
    X(".in" #n, trigmerge_in[n], FPTYPE_STREAMNAME, 0, FPFLAG_DEFAULT_INPUT, "input stream " #n)

#define FPS_PARAMS(X)                                                                             \
    X(".in0", trigmerge_in[0], FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream 0")      \
    X(".in1", trigmerge_in[1], FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream 1")      \
    X(".out", trigmerge_out, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "merged trigger stream") \
    TRIGMERGE_IN(X, 2)                                                                            \
    TRIGMERGE_IN(X, 3)                                                                            \
    TRIGMERGE_IN(X, 4)                                                                            \
    TRIGMERGE_IN(X, 5)                                                                            \
    TRIGMERGE_IN(X, 6)                                                                            \
    TRIGMERGE_IN(X, 7)                                                                            \
    TRIGMERGE_IN(X, 8)                                                                            \
    TRIGMERGE_IN(X, 9)                                                                            \
    TRIGMERGE_IN(X, 10)                                                                           \
    TRIGMERGE_IN(X, 11)                                                                           \
    TRIGMERGE_IN(X, 12)                                                                           \
    TRIGMERGE_IN(X, 13)                                                                           \
    TRIGMERGE_IN(X, 14)                                                                           \
    TRIGMERGE_IN(X, 15)                                                                           \
    X(".logic", trigmerge_logic, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "any or all")            \
    X(".timeout", &trigmerge_timeout, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                    \
      "fire anyway after this many seconds without a merge (0: never)")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

typedef struct
{
    char      name[FUNCTION_PARAMETER_STRMAXLEN];
    IMAGE    *im;   /* NULL: not configured or not found */
    int       sem;  /* this reader's semaphore index */
    uint64_t  last; /* cnt0 at the last fire */
    pthread_t thr;  /* any-of waiter */
} trigmerge_input;

static trigmerge_input tm[TRIGMERGE_NIN];
static char            tm_logic[FUNCTION_PARAMETER_STRMAXLEN];
static int             tm_configured; /* 0: connect on the next pass */
static struct timespec tm_retry;      /* next reconnect attempt while an input is missing */
static int             tm_all;        /* 1: all-of, 0: any-of */
static uint64_t        tm_cfg;        /* configured inputs (non-empty names) */
static uint64_t        tm_missing;    /* configured but not found */
static uint64_t        tm_got;        /* all-of: inputs that moved since the last fire */
static uint64_t        tm_posted;     /* any-of: inputs whose waiter woke, set by the threads */
static sem_t           tm_wake;       /* any-of: posted by the waiters */
static int             tm_threads;    /* 1 while the waiters run */
static int             tm_run;        /* waiters exit when 0 */
static struct timespec tm_lastfire;

static void tm_after(struct timespec *ts, long ns)
{
    clock_gettime(CLOCK_REALTIME, ts);
    ts->tv_nsec += ns;
    while (ts->tv_nsec >= 1000000000L)
    {
        ts->tv_nsec -= 1000000000L;
        ts->tv_sec++;
    }
}

static int tm_before(const struct timespec *a, const struct timespec *b)
{
    return (a->tv_sec < b->tv_sec) || (a->tv_sec == b->tv_sec && a->tv_nsec < b->tv_nsec);
}

static void *tm_waiter(void *arg)
{
    int              k  = (int) (intptr_t) arg;
    trigmerge_input *in = &tm[k];
    while (__atomic_load_n(&tm_run, __ATOMIC_ACQUIRE))
    {
        struct timespec ts;
        tm_after(&ts, TRIGMERGE_SLICE_NS);
        if (ImageStreamIO_semtimedwait(in->im, in->sem, &ts) == 0)
        {
            __atomic_fetch_or(&tm_posted, (uint64_t) 1 << k, __ATOMIC_RELEASE);
            sem_post(&tm_wake);
        }
    }
    return NULL;
}

static void tm_stop_threads(void)
{
    if (!tm_threads)
    {
        return;
    }
    __atomic_store_n(&tm_run, 0, __ATOMIC_RELEASE);
    for (int k = 0; k < TRIGMERGE_NIN; k++)
    {
        if (tm[k].im != NULL)
        {
            pthread_join(tm[k].thr, NULL);
        }
    }
    sem_destroy(&tm_wake);
    tm_threads = 0;
}

static void tm_drain(int k)
{
    while (ImageStreamIO_semtrywait(tm[k].im, tm[k].sem) == 0)
    {
    }
}

/* (Re)connect the inputs when a parameter changed (@set), and every second while one is missing. */
static void tm_configure(void)
{
    struct timespec now;
    tm_after(&now, 0);
    int changed = !tm_configured || (strcmp(tm_logic, trigmerge_logic) != 0) ||
                  (tm_missing != 0 && !tm_before(&now, &tm_retry));
    for (int k = 0; k < TRIGMERGE_NIN && !changed; k++)
    {
        changed = (strcmp(tm[k].name, trigmerge_in[k]) != 0);
    }
    if (!changed)
    {
        return;
    }

    tm_stop_threads();
    tm_configured = 1;
    tm_after(&tm_retry, 1000000000L);
    strcpy(tm_logic, trigmerge_logic);
    tm_all     = (strcmp(trigmerge_logic, "all") == 0);
    tm_cfg     = 0;
    tm_missing = 0;
    tm_got     = 0;
    if (!tm_all && strcmp(trigmerge_logic, "any") != 0)
    {
        fprintf(stderr, "trigmerge: .logic '%s' is neither any nor all: using any\n",
                trigmerge_logic);
    }
    for (int k = 0; k < TRIGMERGE_NIN; k++)
    {
        trigmerge_input *in = &tm[k];
        strcpy(in->name, trigmerge_in[k]);
        in->im = NULL;
        if (in->name[0] == '\0')
        {
            continue;
        }
        tm_cfg |= (uint64_t) 1 << k;
        IMGID id = imgid_make_from_name(in->name);
        resolveIMGID(&id, ERRMODE_NULL, dcimg, dcnimg);
        in->sem = (id.im != NULL) ? ImageStreamIO_getsemwaitindex(id.im, -1) : -1;
        if (in->sem < 0)
        {
            fprintf(stderr, "trigmerge: input %d '%s' %s\n", k, in->name,
                    (id.im == NULL) ? "not found" : "has no free semaphore");
            tm_missing |= (uint64_t) 1 << k;
            continue;
        }
        in->im                      = id.im;
        in->im->semReadPID[in->sem] = getpid();
        tm_drain(k);
        in->last = in->im->md->cnt0;
    }

    if (!tm_all)
    {
        sem_init(&tm_wake, 0, 0);
        __atomic_store_n(&tm_posted, 0, __ATOMIC_RELEASE);
        __atomic_store_n(&tm_run, 1, __ATOMIC_RELEASE);
        for (int k = 0; k < TRIGMERGE_NIN; k++)
        {
            if (tm[k].im != NULL)
            {
                pthread_create(&tm[k].thr, NULL, tm_waiter, (void *) (intptr_t) k);
            }
        }
        tm_threads = 1;
    }
}

/* Write and post the merged trigger: layout in docs/guide/procinfo.md. */
static void tm_fire(PROCESSINFO *processinfo,
                    IMAGE       *out,
                    uint64_t     fired,
                    uint64_t     timedout,
                    int64_t      reason)
{
    int64_t *v     = out->array.SI64;
    out->md->write = 1;
    v[0]           = (int64_t) fired;
    v[1]           = (int64_t) tm_cfg;
    v[2]           = (int64_t) (timedout | tm_missing);
    v[3]           = reason;
    for (int k = 0; k < TRIGMERGE_NIN; k++)
    {
        IMAGE   *im              = tm[k].im;
        uint64_t cnt             = (im != NULL) ? im->md->cnt0 : 0;
        v[4 + k]                 = (int64_t) cnt;
        v[4 + TRIGMERGE_NIN + k] = (im != NULL) ? (int64_t) (cnt - tm[k].last) : 0;
        v[4 + 2 * TRIGMERGE_NIN + k] =
            (im != NULL)
                ? (int64_t) im->md->writetime.tv_sec * 1000000000LL + im->md->writetime.tv_nsec
                : 0;
        tm[k].last = cnt;
    }
    processinfo_update_output_stream(processinfo, out, NULL);
    tm_after(&tm_lastfire, 0);
}

/**
 * @brief Wait up to one slice for the merge condition, and fire if it is met
 *
 * @return 1 if it fired, 0 if the slice ended first.
 */
static MILK_HOT int fpsexec(PROCESSINFO *processinfo, IMAGE *out)
{
    tm_configure();

    struct timespec until, firedeadline = { 0, 0 };
    tm_after(&until, TRIGMERGE_SLICE_NS);
    int has_deadline = (trigmerge_timeout > 0.0);
    if (has_deadline)
    {
        long ns      = (long) (trigmerge_timeout * 1e9);
        firedeadline = tm_lastfire;
        firedeadline.tv_sec += ns / 1000000000L;
        firedeadline.tv_nsec += ns % 1000000000L;
        if (firedeadline.tv_nsec >= 1000000000L)
        {
            firedeadline.tv_nsec -= 1000000000L;
            firedeadline.tv_sec++;
        }
        if (tm_before(&firedeadline, &until))
        {
            until = firedeadline;
        }
    }
    uint64_t connected = tm_cfg & ~tm_missing;

    if (!tm_all)
    {
        /* no input connected: the wait just times out (nothing posts tm_wake) */
        if (tm_threads && sem_timedwait(&tm_wake, &until) == 0)
        {
            while (sem_trywait(&tm_wake) == 0)
            {
            }
            uint64_t fired = __atomic_exchange_n(&tm_posted, 0, __ATOMIC_ACQ_REL);
            for (int k = 0; k < TRIGMERGE_NIN; k++)
            {
                if (tm[k].im != NULL && tm[k].im->md->cnt0 != tm[k].last)
                {
                    fired |= (uint64_t) 1 << k;
                }
            }
            if (fired != 0)
            {
                tm_fire(processinfo, out, fired, 0, 0);
                return 1;
            }
        }
        else if (!tm_threads)
        {
            clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &until, NULL);
        }
    }
    else
    {
        for (int k = 0; k < TRIGMERGE_NIN; k++)
        {
            uint64_t bit = (uint64_t) 1 << k;
            if (!(connected & bit) || (tm_got & bit))
            {
                continue;
            }
            if (tm[k].im->md->cnt0 != tm[k].last ||
                ImageStreamIO_semtrywait(tm[k].im, tm[k].sem) == 0 ||
                ImageStreamIO_semtimedwait(tm[k].im, tm[k].sem, &until) == 0)
            {
                tm_got |= bit;
            }
            else
            {
                break; /* slice or deadline over: resume with this input on the next pass */
            }
        }
        if (connected != 0 && tm_got == connected)
        {
            for (int k = 0; k < TRIGMERGE_NIN; k++)
            {
                if (tm[k].im != NULL)
                {
                    tm_drain(k);
                }
            }
            uint64_t got = tm_got;
            tm_got       = 0;
            tm_fire(processinfo, out, got, 0, 0);
            return 1;
        }
    }

    struct timespec now;
    tm_after(&now, 0);
    if (has_deadline && !tm_before(&now, &firedeadline))
    {
        uint64_t got = tm_all ? tm_got : 0;
        tm_got       = 0;
        tm_fire(processinfo, out, got, connected & ~got, 1);
        return 1;
    }
    return 0;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(trigmerge, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t trigmerge_compute(const MILK_RUNNABLE *self)
{
    if (dcfpsptr != NULL && dcfpsptr->cmdset.triggermode != PROCESSINFO_TRIGGERMODE_DELAY)
    {
        /* the inputs are waited on below: only the delay mode paces the loop */
        functionparameter_SetParamValue_INT64(dcfpsptr, ".procinfo.triggermode",
                                              PROCESSINFO_TRIGGERMODE_IMMEDIATE);
        dcfpsptr->cmdset.triggermode = PROCESSINFO_TRIGGERMODE_IMMEDIATE;
    }

    memset(tm, 0, sizeof(tm));
    tm_configured = 0;
    tm_threads    = 0;
    tm_after(&tm_lastfire, 0);

    IMGID out = stream_connect_create_2D(trigmerge_out, TRIGMERGE_OUTSIZE, 1, _DATATYPE_INT64);
    if (out.im == NULL)
    {
        return RETURN_FAILURE;
    }

    int fired = 0;
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fired = fpsexec(processinfo, out.im);
    while (processinfo == NULL && !fired)
    {
        /* one-shot (no processinfo): the body runs once, so wait here for the first merge */
        fired = fpsexec(processinfo, out.im);
    }
    MILK_PROCINFO_END(self)

    tm_stop_threads();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_trigmerge = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "trigmerge",
                 .cmdkey      = "trigmerge",
                 .description = "merge stream triggers",
                 .description_long =
                     "Wait on up to 16 input streams (.logic any or all) and write a small int64 "
                     "stream each time the condition is met, so a unit can trigger on several "
                     "streams. The output carries which inputs fired, their counters, how many "
                     "frames each moved and their write times. Use -loopd to cap the merged "
                     "rate." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(trigmerge),
    .compute = trigmerge_compute,
};
EXPORT_FPSEXEC(milk, "mem-trigmerge", runnable_COREMOD_memory_trigmerge);
