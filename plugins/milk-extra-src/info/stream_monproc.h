// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_monproc.h
 * @brief   Header for stream monitor process and shared memory structure
 */

#ifndef _INFO_STREAMMONPROC_H
#define _INFO_STREAMMONPROC_H

#include <stdint.h>
#include <time.h>
#ifndef __STDC_WANT_LIB_EXT1__
typedef int errno_t;
#endif

// Maximum number of frames in the history circular buffer
#define STREAM_MON_MAX_SAMPLES 1024

// Maximum number of bins for the histogram
#define STREAM_MON_MAX_HIST_BINS 128

/**
 * @brief Custom shared memory structure for stream monitoring
 *
 * Holds basic statistics for the last N frames.
 */
typedef struct
{
    uint64_t cnt;    // Total counter of frames processed
    uint32_t cindex; // Current circular buffer index (0 to size-1)
    uint32_t size;   // Active size of buffer (<= STREAM_MON_MAX_SAMPLES)

    // Circular buffers
    double          flux[STREAM_MON_MAX_SAMPLES]; // Total flux (sum of pixels)
    struct timespec time[STREAM_MON_MAX_SAMPLES]; // Receive time

    // Histogram Data
    uint32_t hist_nbins; // Number of active bins
    // Per-frame histogram parameters (to ensure consistency)
    float hist_min_buf[STREAM_MON_MAX_SAMPLES];
    float hist_max_buf[STREAM_MON_MAX_SAMPLES];

    uint32_t hist_target_dist
        [STREAM_MON_MAX_HIST_BINS]; // Target pixel count per bin (unused in linear mode?)
    // Note: hist_thresholds removed/deprecated in favor of per-frame min/max reconstruction
    // float    hist_thresholds[STREAM_MON_MAX_HIST_BINS + 1];
    uint32_t hist_counts[STREAM_MON_MAX_SAMPLES][STREAM_MON_MAX_HIST_BINS]; // Histogram history

} STREAM_MON_STRUCT;


// Shared Memory Access Functions

/**
 * @brief Connect to (or create if in read/write mode) the monitor shared memory
 *
 * @param streamname Name of the stream being monitored
 * @param create     1 to create/reset, 0 to read-only connect
 * @return STREAM_MON_STRUCT* Pointer to mapped memory, or NULL on failure
 */
STREAM_MON_STRUCT *stream_monitor_connect(const char *streamname, int create);

/**
 * @brief Detach from monitor shared memory
 *
 * @param smon Pointer to the struct
 */
void stream_monitor_detach(STREAM_MON_STRUCT *smon);

/**
 * @brief Print help text for the stream monitor
 *
 * @return errno_t RETURN_SUCCESS
 */
errno_t stream_monitor_help();

#endif
