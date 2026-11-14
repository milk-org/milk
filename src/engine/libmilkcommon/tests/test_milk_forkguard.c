// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest: a guarded mutex that another thread holds makes fork() wait, and is free in the child. */

#include "milk_forkguard.h"

#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            return 1;                                                    \
        }                                                                \
    } while (0)

static pthread_mutex_t g_lock  = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_plain = PTHREAD_MUTEX_INITIALIZER; /* not guarded */
static volatile int    g_held  = 0;
static volatile int    g_whole = 1; /* 0 while the holder is in the middle of its update */

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

/* Holds both locks for 0.3 s, as a command in the middle of an update does, and the plain one
 * for 0.3 s more */
static void *holder(void *arg)
{
    struct timespec d = { 0, 300000000 };
    (void) arg;
    pthread_mutex_lock(&g_plain);
    pthread_mutex_lock(&g_lock);
    g_whole = 0;
    g_held  = 1;
    nanosleep(&d, NULL);
    g_whole = 1;
    pthread_mutex_unlock(&g_lock);
    nanosleep(&d, NULL);
    pthread_mutex_unlock(&g_plain);
    return NULL;
}

int main(void)
{
    pthread_t       th;
    struct timespec tick = { 0, 1000000 };
    int             st   = 0;

    CHECK(milk_fork_guard(&g_lock) == 0);
    CHECK(milk_fork_guard(&g_lock) == 0); /* again: nothing */
    CHECK(pthread_create(&th, NULL, holder, NULL) == 0);
    while (!g_held)
    {
        nanosleep(&tick, NULL);
    }

    double t0  = now();
    pid_t  pid = fork(); /* waits for the holder */
    if (pid == 0)
    {
        /* the guarded mutex is free and its data whole; the plain one would block for ever */
        int ok = (g_whole == 1 && pthread_mutex_trylock(&g_lock) == 0 &&
                  pthread_mutex_unlock(&g_lock) == 0 && pthread_mutex_trylock(&g_plain) != 0);
        _exit(ok ? 0 : 1);
    }
    CHECK(pid > 0);
    CHECK(now() - t0 > 0.2);
    CHECK(waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0);
    CHECK(pthread_join(th, NULL) == 0);
    CHECK(pthread_mutex_trylock(&g_lock) == 0 && pthread_mutex_unlock(&g_lock) == 0); /* parent */
    return 0;
}
