// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gst_sync.h
 * @brief   Per-frame synchronization of GPU-resident streams, CPU or GPU ingress trigger
 *
 * One GST_SYNC per compute loop. Inside the processinfo loop, for each frame:
 *
 *     gst_input_prep(&sync);                      top    arm the trigger
 *     gst_output_prep(&sync, out);                top    one call per output, main output last
 *     ... enqueue the compute kernels on sync.stream, passing sync.d_trig ...
 *     gst_input_closeout(&sync, processinfo);     bottom blocks until the frame is done
 *     gst_output_closeout(&sync, out, ...);       bottom one call per output, main output last
 *
 * Nothing blocks before gst_input_closeout(): the frame is entirely enqueued, then waited for.
 *
 * Main and auxiliary outputs go through the same two functions. The only difference is the
 * order: the main output is declared and closed last, so that its counter and semaphores are
 * the last ones to move and a consumer triggering on it finds the auxiliary outputs updated.
 */

#ifndef GST_SYNC_H
#define GST_SYNC_H

#include <stdint.h>

#include <cuda_runtime_api.h>

#include "ImageStreamIO/ImageStruct.h"
#include "processinfo.h"

#include "gst_kernels.h"

#define GST_NBOUT_MAX 4

/** @brief Outcome of a frame, returned by gst_input_closeout(). */
#define GST_FRAME_DONE 0    /**< computed and published on the device */
#define GST_FRAME_TIMEOUT 1 /**< GPU trigger timed out: nothing computed, nothing to post */
#define GST_FRAME_ERROR 2   /**< CUDA error */

typedef struct
{
    int          gpusync; /**< 1: trigger is the wait kernel; 0: trigger was done on the host */
    cudaStream_t stream;  /**< the compute kernels of the frame are enqueued here */

    const IMAGE_GPU_SYNC *d_insync; /**< device, sync block of the triggering input */
    GST_TRIGGER          *d_trig;   /**< device, trigger report; pass it to the compute kernels */
    GST_TRIGGER           h_trig;   /**< host copy, valid after gst_input_closeout() */
    uint64_t              last_in_cnt;

    IMAGE *out[GST_NBOUT_MAX]; /**< outputs declared for the current frame */
    int    nbout;
    int    status; /**< GST_FRAME_*, set by gst_input_closeout() */
} GST_SYNC;

/**
 * @brief Select the device of the triggering input, create the stream and the trigger report.
 *
 * The trigger starts from the current device counter of the input: only frames published
 * from now on trigger.
 */
errno_t gst_sync_init(GST_SYNC *sync, IMAGE *in, int gpusync);

/** @brief Release the stream and the trigger report. */
void gst_sync_free(GST_SYNC *sync);

/**
 * @brief Top of loop: arm the trigger of the frame.
 *
 * GPU sync: enqueues the wait kernel. CPU sync: the host trigger already happened, no-op.
 */
void gst_input_prep(GST_SYNC *sync);

/**
 * @brief Top of loop: declare an output written by the frame.
 *
 * CPU sync: raises md->write. Not raised with GPU sync: the frame is enqueued before its
 * trigger, the flag would stay up while waiting.
 */
void gst_output_prep(GST_SYNC *sync, IMAGE *out);

/**
 * @brief Bottom of loop: commit the frame on the device and wait for it.
 *
 * Enqueues the device-side publish of every declared output (in declaration order) and the
 * copy of the trigger report, then synchronizes the stream. With GPU sync, this is where the
 * loop blocks waiting for the input. Trigger information from the device (status, missed
 * frames, input counter) is written to @p processinfo (may be NULL).
 *
 * @return GST_FRAME_DONE, GST_FRAME_TIMEOUT or GST_FRAME_ERROR
 */
int gst_input_closeout(GST_SYNC *sync, PROCESSINFO *processinfo);

/**
 * @brief Bottom of loop: update the CPU metadata of an output and post its semaphores.
 *
 * Does nothing unless the frame is GST_FRAME_DONE.
 *
 * @param in  triggering input, source of the stream processing trace
 */
void gst_output_closeout(GST_SYNC *sync, IMAGE *out, PROCESSINFO *processinfo, IMAGE *in);

#endif // GST_SYNC_H
