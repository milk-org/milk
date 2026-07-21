// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_scan.c
 * @brief Background scan thread for milk-CTRL
 *
 * Runs ov_model_full_scan() in a loop on a background
 * thread with a configurable sleep interval. Uses
 * triple-buffered models so the display thread always
 * reads a consistent, complete snapshot while the scan
 * thread writes to a private buffer.
 *
 * Triple-buffer protocol:
 *   - scan thread OWNS the "write" slot (writes freely)
 *   - scan thread publishes by swapping write → ready
 *   - display thread picks up ready → display
 *   - old display slot becomes available for next write
 *   - display thread reads "display" slot safely
 *     (scan thread never touches it)
 */

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#include "overview_defs.h"
#include "overview_data.h"

/* =========================================================
 * Triple-buffered model
 * ========================================================= */

static OV_MODEL ov_model_slots[3];

/**
 * Slot indices (protected by mutex):
 *   write_idx   — owned by scan thread (writes here)
 *   ready_idx   — latest complete scan (swap target)
 *   display_idx — owned by display thread (reads here)
 */
static int        ov_write_idx   = 0;
static int        ov_ready_idx   = 1;
static int        ov_display_idx = 2;
static atomic_int ov_new_data    = 0;

static pthread_mutex_t ov_model_mutex = PTHREAD_MUTEX_INITIALIZER;


/* =========================================================
 * Scan thread state
 * ========================================================= */

static pthread_t      ov_scan_thread;
static volatile int   ov_scan_running      = 0;
static volatile float ov_scan_interval_s   = 1.0f;
static atomic_int     ov_force_update_flag = 0;

/**
 * ov_scan_get_interval - get current scan interval.
 *
 * Return: interval in seconds.
 */
float ov_scan_get_interval(void)
{
    return ov_scan_interval_s;
}

/**
 * ov_scan_set_interval - set scan interval.
 * @interval_s: new interval in seconds (min 0.1, max 10)
 */
void ov_scan_set_interval(float interval_s)
{
    if (interval_s < 0.1f)
    {
        interval_s = 0.1f;
    }
    if (interval_s > 10.0f)
    {
        interval_s = 10.0f;
    }
    ov_scan_interval_s = interval_s;
}

/**
 * ov_scan_has_new_data - check if the first scan has completed.
 */
int ov_scan_has_new_data(void)
{
    return atomic_load_explicit(&ov_new_data, memory_order_acquire);
}


/* =========================================================
 * Scan thread main loop
 * ========================================================= */

static void *ov_scan_thread_func(void *arg __attribute__((unused)))
{
    memset(ov_model_slots, 0, sizeof(ov_model_slots));

    while (ov_scan_running && !OV_SIG_ANY_SET())
    {
        /* Scan into our private write slot
         * (no lock needed — display never touches it) */
        ov_model_full_scan(&ov_model_slots[ov_write_idx]);

        /* Publish: swap write → ready.
         * The old ready slot becomes our new write slot. */
        pthread_mutex_lock(&ov_model_mutex);
        {
            int tmp      = ov_ready_idx;
            ov_ready_idx = ov_write_idx;
            ov_write_idx = tmp;
            atomic_store_explicit(&ov_new_data, 1, memory_order_release);
        }
        pthread_mutex_unlock(&ov_model_mutex);

        /* Sleep for the configured interval in small increments
         * so we can exit immediately if requested. */
        {
            float           interval = ov_scan_interval_s;
            struct timespec ts;
            ts.tv_sec  = 0;
            ts.tv_nsec = 10000000L; /* 10 ms */

            int num_sleeps = (int) (interval / 0.01f);
            for (int i = 0; i < num_sleeps; i++)
            {
                if (!ov_scan_running || OV_SIG_ANY_SET() ||
                    atomic_load_explicit(&ov_force_update_flag, memory_order_acquire))
                {
                    atomic_store_explicit(&ov_force_update_flag, 0, memory_order_release);
                    break;
                }
                nanosleep(&ts, NULL);
            }
        }
    }

    return NULL;
}


/* =========================================================
 * Public API
 * ========================================================= */

/**
 * ov_scan_start - launch the background scan thread.
 *
 * Return: 0 on success, -1 on failure.
 */
int ov_scan_start(void)
{
    if (ov_scan_running)
    {
        return 0;
    }
    ov_scan_running = 1;

    if (pthread_create(&ov_scan_thread, NULL, ov_scan_thread_func, NULL) != 0)
    {
        ov_scan_running = 0;
        return -1;
    }
    return 0;
}

/**
 * ov_scan_stop - signal the scan thread to stop and join.
 */
void ov_scan_stop(void)
{
    ov_scan_running = 0;
    pthread_join(ov_scan_thread, NULL);
    ov_scan_cache_cleanup();
}

/**
 * ov_scan_get_model - pick up the latest complete model.
 *
 * Swaps ready → display under lock. The returned pointer
 * remains valid and untouched by the scan thread until
 * the next call to ov_scan_get_model().
 *
 * Return: pointer to the current display model.
 */
const OV_MODEL *ov_scan_get_model(void)
{
    /* Pick up the latest ready buffer ONLY if new data
     * has been published by the scan thread. */
    pthread_mutex_lock(&ov_model_mutex);
    if (atomic_load(&ov_new_data))
    {
        int tmp        = ov_display_idx;
        ov_display_idx = ov_ready_idx;
        ov_ready_idx   = tmp;
        atomic_store(&ov_new_data, 0);
    }
    pthread_mutex_unlock(&ov_model_mutex);

    return &ov_model_slots[ov_display_idx];
}

/**
 * ov_scan_force_update - interrupt sleep to force an immediate scan.
 */
void ov_scan_force_update(void)
{
    atomic_store_explicit(&ov_force_update_flag, 1, memory_order_release);
}
