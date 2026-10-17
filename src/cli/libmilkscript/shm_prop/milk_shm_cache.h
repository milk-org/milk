// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_cache.h
 * @brief Process-wide cache of mapped shared-memory streams and FPS, checked against the file on use.
 *
 * Entries stay mapped between calls; each lookup stat()s the backing file and remaps when the
 * stream or FPS was re-created (different inode) or removed. Not thread-safe: in the server, call it
 * with the command lock held. Returned pointers are valid until the next lookup of another name.
 */

#ifndef MILK_SHM_CACHE_H
#define MILK_SHM_CACHE_H

#include "CLIcore.h"
#include "fps_types.h"

IMAGE *milk_shm_cache_stream(const char *name);

FPS *milk_shm_cache_fps(const char *name);

void milk_shm_cache_drop_all(void);

#endif /* MILK_SHM_CACHE_H */
