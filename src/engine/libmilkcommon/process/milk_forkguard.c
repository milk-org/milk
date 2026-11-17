// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_forkguard.c
 * @brief Mutexes that stay usable in the child of a fork() (see milk_forkguard.h), and the hook
 *        of milk_safepoint.h: what a process with several users of its data needs.
 */

#include "milk_forkguard.h"
#include "milk_safepoint.h"

void (*milk_safe_point_fn)(void) = NULL;

#define FORK_GUARD_MAX 32

static pthread_mutex_t *g_guard[FORK_GUARD_MAX];
static int              g_nguard     = 0;
static pthread_mutex_t  g_guard_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t   g_guard_once = PTHREAD_ONCE_INIT;

/* Before fork(), in the forking thread: wait for every guarded mutex */
static void guard_prepare(void)
{
    pthread_mutex_lock(&g_guard_lock);
    for (int i = 0; i < g_nguard; i++)
    {
        pthread_mutex_lock(g_guard[i]);
    }
}

/* After fork(), in the parent: release them */
static void guard_parent(void)
{
    for (int i = g_nguard - 1; i >= 0; i--)
    {
        pthread_mutex_unlock(g_guard[i]);
    }
    pthread_mutex_unlock(&g_guard_lock);
}

/* After fork(), in the child: its only thread holds them all; start afresh */
static void guard_child(void)
{
    for (int i = 0; i < g_nguard; i++)
    {
        pthread_mutex_init(g_guard[i], NULL);
    }
    pthread_mutex_init(&g_guard_lock, NULL);
}

static void guard_register(void)
{
    pthread_atfork(guard_prepare, guard_parent, guard_child);
}

int milk_fork_guard(pthread_mutex_t *m)
{
    int rc = 0;
    int i  = 0;

    pthread_once(&g_guard_once, guard_register);
    pthread_mutex_lock(&g_guard_lock);
    while (i < g_nguard && g_guard[i] != m)
    {
        i++;
    }
    if (i == g_nguard)
    {
        if (g_nguard < FORK_GUARD_MAX)
        {
            g_guard[g_nguard++] = m;
        }
        else
        {
            rc = -1;
        }
    }
    pthread_mutex_unlock(&g_guard_lock);
    return rc;
}
