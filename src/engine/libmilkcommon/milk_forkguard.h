// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_forkguard.h
 * @brief Mutexes that stay usable in the child of a fork()
 *
 * A process with several threads that calls fork() gives its child a copy of every mutex as it
 * is at that instant: one that another thread holds stays locked for ever in the child, and what
 * it protects may be half-updated. A session's server forks while a command runs (a read, a
 * pipeline stage, an `--async` command), so the locks that the running command takes are
 * registered here.
 */

#ifndef MILK_FORKGUARD_H
#define MILK_FORKGUARD_H

#include <pthread.h>

/**
 * milk_fork_guard - Make mutex @m safe across fork()
 * @m: A default mutex that lives as long as the process (a static one)
 *
 * From then on fork() waits until no thread holds @m, so that the data it protects is whole in
 * the child, and the child gets @m unlocked. A thread must not hold two guarded mutexes at once,
 * nor call this while it holds one: fork() takes them all, in the order they were registered.
 * Registering a mutex again does nothing.
 *
 * Return: 0, or -1 when the table is full (32 mutexes)
 */
int milk_fork_guard(pthread_mutex_t *m);

#endif
