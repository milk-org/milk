// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gst_kernels.cu
 * @brief   CUDA kernels of gpusynctest: device-side wait, device-side publish, out = in + 1
 */

// MILK_CMAKE_MANDATE_CUDA

#include "gst_kernels.h"

#define GST_BLOCKSIZE 256
#define GST_WAIT_NANOSLEEP 50

/* Single thread. The counter is written by another process (another CUDA context): read it as
 * volatile so that every poll is a load. */
__global__ void gst_wait_kernel(const volatile IMAGE_GPU_SYNC *insync,
                                GST_TRIGGER                   *trig,
                                uint64_t                       maxspin)
{
    const uint64_t last  = trig->in_cnt;
    uint64_t       spins = 0;
    uint64_t       cnt   = insync->cnt0;

    while (cnt == last && spins < maxspin)
    {
        spins++;
#if __CUDA_ARCH__ >= 700
        __nanosleep(GST_WAIT_NANOSLEEP);
#endif
        cnt = insync->cnt0;
    }

    trig->spins   = spins;
    trig->timeout = (cnt == last) ? 1 : 0;
    trig->in_cnt  = cnt;
}

/* Single thread, enqueued after the kernels writing the pixels. */
__global__ void gst_publish_kernel(volatile IMAGE_GPU_SYNC *outsync, const GST_TRIGGER *trig)
{
    if (trig->timeout)
    {
        return;
    }

    outsync->cnt0 = outsync->cnt0 + 1;
}

__global__ void gst_addone_kernel(const uint16_t    *in,
                                  uint16_t          *out,
                                  uint64_t           nelement,
                                  const GST_TRIGGER *trig)
{
    if (trig->timeout)
    {
        return;
    }

    const uint64_t ii = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ii < nelement)
    {
        out[ii] = in[ii] + 1;
    }
}

void gst_enqueue_wait(cudaStream_t stream, const IMAGE_GPU_SYNC *d_insync, GST_TRIGGER *d_trig)
{
    gst_wait_kernel<<<1, 1, 0, stream>>>(d_insync, d_trig, GST_WAIT_MAXSPIN);
}

void gst_enqueue_publish(cudaStream_t stream, IMAGE_GPU_SYNC *d_outsync, const GST_TRIGGER *d_trig)
{
    gst_publish_kernel<<<1, 1, 0, stream>>>(d_outsync, d_trig);
}

void gst_enqueue_addone(cudaStream_t       stream,
                        const uint16_t    *d_in,
                        uint16_t          *d_out,
                        uint64_t           nelement,
                        const GST_TRIGGER *d_trig)
{
    const unsigned int nblocks = (unsigned int) ((nelement + GST_BLOCKSIZE - 1) / GST_BLOCKSIZE);
    gst_addone_kernel<<<nblocks, GST_BLOCKSIZE, 0, stream>>>(d_in, d_out, nelement, d_trig);
}
