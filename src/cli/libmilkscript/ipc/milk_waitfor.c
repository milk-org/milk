// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_waitfor.c
 * @brief `@waitfor` target parsing, state aliasing and wait loops shared by the client and the server.
 *
 * Frame waits block on a stream semaphore; a stream that does not exist yet is waited for with
 * inotify on the shm directory. Entity conditions are polled with a short back-off.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <math.h>
#include <poll.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/inotify.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"
#include "milk_waitfor.h"

#define WAIT_SEM_STEP 0.05 /* longest single semaphore wait: bounds cancel/timeout latency */

/** proc_state_name() - canonical processinfo state for a user-supplied state word or number. */
static const char *proc_state_name(const char *st)
{
    static const struct
    {
        const char *alias;
        const char *state;
    } tab[] = { { "ACTIVE", "ACTIVE" }, { "RUN", "ACTIVE" },   { "RUNNING", "ACTIVE" },
                { "1", "ACTIVE" },      { "PAUSE", "PAUSED" }, { "PAUSED", "PAUSED" },
                { "2", "PAUSED" },      { "STOP", "STOPPED" }, { "STOPPED", "STOPPED" },
                { "EXIT", "STOPPED" },  { "3", "STOPPED" },    { "INIT", "INIT" },
                { "0", "INIT" },        { "ERROR", "ERROR" },  { "4", "ERROR" },
                { NULL, NULL } };

    for (int i = 0; tab[i].alias != NULL; i++)
    {
        if (strcasecmp(st, tab[i].alias) == 0)
        {
            return tab[i].state;
        }
    }
    return st;
}

/**
 * milk_waitfor_parse() - classify @target into @t; @state is the optional `p.<name>` state (default STOPPED).
 */
milk_wait_kind_t milk_waitfor_parse(const char *target, const char *state, milk_wait_target_t *t)
{
    memset(t, 0, sizeof(*t));
    while (target != NULL && *target == '@')
    {
        target++;
    }
    if (target == NULL || target[0] == '\0')
    {
        return t->kind = MILK_WAIT_BAD;
    }
    if (strpbrk(target, "<>") != NULL || strstr(target, "==") != NULL ||
        strstr(target, "!=") != NULL || strstr(target, "&&") != NULL ||
        strstr(target, "||") != NULL)
    {
        return t->kind = MILK_WAIT_EXPR;
    }

    const char    *rest = target;
    milk_ns_kind_t kind = milk_entity_ns_kind(target, &rest);
    const char    *eq   = strchr(target, '=');
    if (eq != NULL)
    {
        size_t lhslen = (size_t) (eq - target);
        if (lhslen == 0 || lhslen >= sizeof(t->lhs))
        {
            return t->kind = MILK_WAIT_BAD;
        }
        memcpy(t->lhs, target, lhslen);
        t->lhs[lhslen]       = '\0';
        const char *prop     = strrchr(t->lhs, '.');
        int         is_pstat = (kind == MILK_NS_PROC && prop != NULL &&
                                (!strcmp(prop, ".status") || !strcmp(prop, ".state")));
        snprintf(t->expect, sizeof(t->expect), "%s", is_pstat ? proc_state_name(eq + 1) : eq + 1);
        return t->kind = MILK_WAIT_EQ;
    }
    if (kind == MILK_NS_PROC && rest[0] != '\0' && strchr(rest, '.') == NULL)
    {
        snprintf(t->lhs, sizeof(t->lhs), "p.%s.status", rest);
        snprintf(t->expect, sizeof(t->expect), "%s",
                 proc_state_name(state != NULL ? state : "STOPPED"));
        return t->kind = MILK_WAIT_EQ;
    }

    t->local = (strncmp(target, "L:", 2) == 0 || kind == MILK_NS_LOCAL);
    snprintf(t->stream, sizeof(t->stream), "%s",
             (strncmp(target, "S:", 2) == 0) ? target + 2
             : (kind == MILK_NS_STREAM)      ? rest
                                             : target);
    return t->kind = MILK_WAIT_STREAM;
}

/** milk_waitfor_value_matches() - 1 if @val satisfies @expect (exact, RUN/RUNNING alias, or numerically equal). */
int milk_waitfor_value_matches(const char *val, const char *expect)
{
    if (strcmp(val, expect) == 0)
    {
        return 1;
    }
    if ((strcmp(val, "RUN") == 0 && strcmp(expect, "RUNNING") == 0) ||
        (strcmp(val, "RUNNING") == 0 && strcmp(expect, "RUN") == 0))
    {
        return 1;
    }
    char  *e1 = NULL;
    char  *e2 = NULL;
    double d1 = strtod(val, &e1);
    double d2 = strtod(expect, &e2);
    return val[0] != '\0' && expect[0] != '\0' && *e1 == '\0' && *e2 == '\0' &&
           fabs(d1 - d2) <= 1.0e-9 * (1.0 + fabs(d2));
}

/** milk_wait_now() - CLOCK_MONOTONIC time in seconds. */
double milk_wait_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

/** milk_wait_backoff() - sleep one poll step of a condition wait; *@iter counts the steps. */
void milk_wait_backoff(int *iter)
{
    struct timespec slp = { 0, ((*iter)++ < 20) ? 1000000L : 5000000L }; /* 1 ms x20, then 5 ms */
    nanosleep(&slp, NULL);
}

/** wait_stop() - MILK_WAIT_CANCEL / MILK_WAIT_TIMEOUT if the wait must end now, else 0. */
static int wait_stop(const milk_wait_ops_t *ops, double t_start, double timeout_sec)
{
    if (ops != NULL && ops->cancelled != NULL && ops->cancelled(ops->ctx))
    {
        return MILK_WAIT_CANCEL;
    }
    if (timeout_sec > 0.0 && milk_wait_now() - t_start >= timeout_sec)
    {
        return MILK_WAIT_TIMEOUT;
    }
    return 0;
}

/** milk_waitfor_eq() - poll entity @lhs until it matches @expect; 0, MILK_WAIT_TIMEOUT or MILK_WAIT_CANCEL. */
int milk_waitfor_eq(const char            *lhs,
                    const char            *expect,
                    double                 t_start,
                    double                 timeout_sec,
                    const milk_wait_ops_t *ops)
{
    for (int iter = 0;;)
    {
        int rc = wait_stop(ops, t_start, timeout_sec);
        if (rc != 0)
        {
            return rc;
        }
        char val[256];
        if (ops != NULL && ops->lock != NULL)
        {
            ops->lock(ops->ctx);
        }
        int got = milk_shm_prop_get(lhs, val, sizeof(val));
        if (ops != NULL && ops->unlock != NULL)
        {
            ops->unlock(ops->ctx);
        }
        if (got == 0 && milk_waitfor_value_matches(val, expect))
        {
            return 0;
        }
        milk_wait_backoff(&iter);
    }
}

/** wait_created() - open @sname into @img, waiting (inotify) until it exists; 0 when open. */
static int wait_created(IMAGE                 *img,
                        const char            *sname,
                        double                 t_start,
                        double                 timeout_sec,
                        const milk_wait_ops_t *ops,
                        int                   *created)
{
    int  ifd = -1; /* inotify on the shm directory; -2: unavailable, poll every 10 ms */
    int  rc  = 0;
    char shm[1024];
    int  named = (ImageStreamIO_filename(shm, sizeof(shm), sname) == IMAGESTREAMIO_SUCCESS);

    *created = 0;
    /* no open attempt while there is no file (ImageStreamIO would warn) */
    while ((named && access(shm, F_OK) != 0) ||
           ImageStreamIO_openIm(img, sname) != IMAGESTREAMIO_SUCCESS || img->md == NULL)
    {
        *created = 1;
        if ((rc = wait_stop(ops, t_start, timeout_sec)) != 0)
        {
            break;
        }
        if (ifd == -1 && (ifd = inotify_init1(IN_CLOEXEC | IN_NONBLOCK)) >= 0 &&
            inotify_add_watch(ifd, milk_shm_prop_shmdir(), IN_CREATE | IN_MOVED_TO) < 0)
        {
            close(ifd);
            ifd = -2;
        }
        if (ifd >= 0)
        {
            /* a file that does not open yet is still being written: try again soon. 200 ms bounds the
             * cancel/timeout checks */
            struct pollfd pfd = { ifd, POLLIN, 0 };
            if (poll(&pfd, 1, (named && access(shm, F_OK) == 0) ? 2 : 200) > 0)
            {
                char    evbuf[4096];
                ssize_t rr = read(ifd, evbuf, sizeof(evbuf));
                (void) rr;
            }
        }
        else
        {
            struct timespec slp = { 0, 10000000L };
            nanosleep(&slp, NULL);
        }
    }
    if (ifd >= 0)
    {
        close(ifd);
    }
    return rc;
}

/**
 * milk_waitfor_stream() - wait until stream @sname has advanced @nframes (or reached *@abs_cnt) on semaphore
 * @req_sem (-1: any free one). A missing stream is waited for when @wait_create, else MILK_WAIT_ABSENT.
 */
int milk_waitfor_stream(const char            *sname,
                        long                   nframes,
                        int                    req_sem,
                        const uint64_t        *abs_cnt,
                        int                    wait_create,
                        double                 t_start,
                        double                 timeout_sec,
                        const milk_wait_ops_t *ops)
{
    IMAGE img;
    memset(&img, 0, sizeof(img));
    if (!wait_create)
    {
        if (sname[0] == '\0' || ImageStreamIO_openIm(&img, sname) != IMAGESTREAMIO_SUCCESS ||
            img.md == NULL)
        {
            return MILK_WAIT_ABSENT;
        }
    }
    int created = 0;
    if (wait_create)
    {
        int rc = wait_created(&img, sname, t_start, timeout_sec, ops, &created);
        if (rc != 0)
        {
            return rc;
        }
    }

    /* a stream created during the wait: its frames count from its creation, not from when it opened */
    uint64_t target_cnt =
        (abs_cnt != NULL) ? *abs_cnt
                          : (created ? 0 : img.md->cnt0) + (uint64_t) (nframes < 1 ? 1 : nframes);
    int semindex = (req_sem >= 0) ? req_sem : ImageStreamIO_getsemwaitindex(&img, -1);
    int use_sem  = (semindex >= 0 && semindex < img.md->sem);
    if (use_sem)
    {
        while (sem_trywait(img.semptr[semindex]) == 0)
        {
        }
    }

    int rc = 0;
    while (img.md->cnt0 < target_cnt && (rc = wait_stop(ops, t_start, timeout_sec)) == 0)
    {
        if (!use_sem)
        {
            struct timespec slp = { 0, 1000000L };
            nanosleep(&slp, NULL);
            continue;
        }
        double step = WAIT_SEM_STEP;
        if (timeout_sec > 0.0 && timeout_sec - (milk_wait_now() - t_start) < step)
        {
            step = timeout_sec - (milk_wait_now() - t_start);
        }
        if (step < 1.0e-4)
        {
            step = 1.0e-4;
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        long add_ns = (long) (step * 1.0e9);
        ts.tv_nsec += add_ns;
        if (ts.tv_nsec >= 1000000000L)
        {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000L;
        }
        sem_timedwait(img.semptr[semindex], &ts);
    }

    if (req_sem < 0 && use_sem)
    {
        img.semReadPID[semindex] = 0;
    }
    ImageStreamIO_closeIm(&img);
    return rc;
}
