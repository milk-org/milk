// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_frame.h
 * @brief One torn-frame rule for readers of shared streams (calculator passes, stream statistics,
 *        `@grab`)
 *
 * A read of a shared stream is torn when a write was under way as it began or ended, or when the
 * stream's frame counter moved during it. A reader re-reads a torn frame up to MILK_FRAME_TRIES
 * times in all, MILK_FRAME_WAIT_NS apart, and then uses the last read.
 */

#ifndef MILK_FRAME_H
#define MILK_FRAME_H

#include <stdint.h>
#include <time.h>

#include "ImageStreamIO/ImageStruct.h"

#define MILK_FRAME_TRIES 4         /* reads of a frame before a torn one is used */
#define MILK_FRAME_WAIT_NS 100000L /* pause before a re-read: the writer finishes its frame */

/**
 * struct milk_frame_mark_t - A stream's frame counter and write flag as a read begins
 * @cnt0:  Frame counter
 * @write: Write flag
 */
typedef struct
{
    uint64_t cnt0;
    uint8_t  write;
} milk_frame_mark_t;

/** milk_frame_mark - Note @im's frame counter and write flag before reading its pixels. */
static inline milk_frame_mark_t milk_frame_mark(const IMAGE *im)
{
    milk_frame_mark_t m = { __atomic_load_n(&im->md->cnt0, __ATOMIC_ACQUIRE),
                            __atomic_load_n(&im->md->write, __ATOMIC_ACQUIRE) };
    return m;
}

/** milk_frame_clean - Return 1 when no write overlapped the read of @im that began at @m. */
static inline int milk_frame_clean(const IMAGE *im, milk_frame_mark_t m)
{
    return m.write == 0 && __atomic_load_n(&im->md->write, __ATOMIC_ACQUIRE) == 0 &&
           __atomic_load_n(&im->md->cnt0, __ATOMIC_ACQUIRE) == m.cnt0;
}

/** milk_frame_wait - Pause before re-reading a torn frame. */
static inline void milk_frame_wait(void)
{
    struct timespec ts = { 0, MILK_FRAME_WAIT_NS };
    nanosleep(&ts, NULL);
}

#endif /* MILK_FRAME_H */
