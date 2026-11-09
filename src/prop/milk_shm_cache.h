// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_cache.h
 * @brief Process-wide cache of mapped shared-memory streams and FPSs, checked against the file
 *        on each use, and the reader semaphore slots of a stream.
 *
 * Purpose:     reading properties or waiting on a stream should not map it again on every call,
 *              yet must notice a stream or FPS that was re-created or removed.
 * Does:        keeps up to 64 streams and 16 FPSs mapped between calls; each lookup stat()s the
 *              file and maps it again when it changed (another inode) or is gone. A waiter pins
 *              its stream across a blocking wait and gets a semaphore slot of its own;
 *              milk_shm_sem_claim() and milk_shm_sem_release() take and give back a slot.
 * Used by:     milkprop (milk_shm_prop.c, milk_shm_prop_stream.c, milk_shm_prop_fps.c);
 *              milk_waitfor.c (`@waitfor` pins its stream); for semaphore slots,
 *              milk_client_stream.c (`@waitany`, `@waitfor "<expr>"`) and the server's
 *              milk_server_stream.c (`@grab`).
 * Uses:        ImageStreamIO, libfps (fps_connect()), milk_shm_prop_shmdir().
 * See also:    milk_shm_cache.c.
 * Constraints: thread-safe (one mutex). A stream that is not pinned can be unmapped by a later
 *              lookup (after 64 other streams, or when its file was re-created): a long-lived
 *              user pins it.
 */

#ifndef MILK_SHM_CACHE_H
#define MILK_SHM_CACHE_H

#include "ImageStreamIO/ImageStreamIO.h"
#include "libfps/fps.h"
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
