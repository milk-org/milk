// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gst_kernels.h
 * @brief   C interface to the CUDA kernels of gpusynctest (gst_kernels.cu)
 *
 * Every function enqueues on a CUDA stream and returns without waiting. Launch errors are
 * collected with cudaGetLastError() by the caller (see gst_input_closeout).
 */

#ifndef GST_KERNELS_H
#define GST_KERNELS_H

#include <stdint.h>

#include <cuda_runtime_api.h>

#include "ImageStreamIO/ImageStruct.h"

/**
 * @brief Upper bound of the wait kernel spin, in polls of the ingress counter.
 *
 * Keeps cudaStreamSynchronize() from blocking forever, so that the host loop sees processinfo
 * stop requests. NOT CALIBRATED: the wall time of one poll depends on the device.
 */
#define GST_WAIT_MAXSPIN (1ULL << 24)

/** @brief Trigger report of one frame: written on the device, copied back to the host. */
typedef struct
{
    uint64_t in_cnt;  /**< ingress IMAGE_GPU_SYNC.cnt0 seen by the wait kernel */
    uint64_t spins;   /**< polls done by the wait kernel */
    uint32_t timeout; /**< 1: the wait kernel gave up, the frame is skipped on the device */
    uint32_t pad;
} GST_TRIGGER;

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Wait kernel: spins until d_insync->cnt0 differs from d_trig->in_cnt (the value seen at
 *        the previous frame) or GST_WAIT_MAXSPIN is reached; fills d_trig.
 */
void gst_enqueue_wait(cudaStream_t stream, const IMAGE_GPU_SYNC *d_insync, GST_TRIGGER *d_trig);

/** @brief Publish kernel: d_outsync->cnt0++, unless d_trig reports a timeout. */
void gst_enqueue_publish(cudaStream_t stream, IMAGE_GPU_SYNC *d_outsync, const GST_TRIGGER *d_trig);

/** @brief Compute kernel of the addone unit: out = in + 1, unless d_trig reports a timeout. */
void gst_enqueue_addone(cudaStream_t       stream,
                        const uint16_t    *d_in,
                        uint16_t          *d_out,
                        uint64_t           nelement,
                        const GST_TRIGGER *d_trig);

#ifdef __cplusplus
}
#endif

#endif // GST_KERNELS_H
