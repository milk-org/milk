// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_cache.h
 * @brief Process-wide cache of mapped shared-memory streams and FPS, checked against the file on use.
 *
 * Entries stay mapped between calls; each lookup stat()s the backing file and remaps when the
 * stream or FPS was re-created (different inode) or removed. Thread-safe via an internal mutex;
 * long-lived callers (such as `@waitfor`) pin stream entries across blocking waits with
 * milk_shm_cache_pin_stream() / milk_shm_cache_unpin_stream().
 */

#ifndef MILK_SHM_CACHE_H
#define MILK_SHM_CACHE_H

#include "CLIcore.h"
#include "fps_types.h"

IMAGE *milk_shm_cache_stream(const char *name);

IMAGE *milk_shm_cache_pin_stream(const char *name, int req_sem, int *out_semindex);

void milk_shm_cache_unpin_stream(IMAGE *im, int req_sem, int semindex);

/** milk_shm_cache_full() - 1 if this thread's last stream lookup failed because all 64 slots are pinned. */
int milk_shm_cache_full(void);

/** milk_shm_sem_claim() - take a free semaphore slot of @im for this process (never one it already holds,
 * so concurrent waiters never share one); -1 if none. milk_shm_sem_release() gives it back. */
int  milk_shm_sem_claim(IMAGE *im);
void milk_shm_sem_release(IMAGE *im, int slot);

FPS *milk_shm_cache_fps(const char *name);

#endif /* MILK_SHM_CACHE_H */
