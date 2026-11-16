// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gst_sync.c
 * @brief   DRAFT, never compiled. Per-frame synchronization of GPU-resident streams
 */

#include "milk_runnable.h"

// MILK_CMAKE_MANDATE_CUDA
#include <cuda_runtime_api.h>

#include "gst_sync.h"

errno_t gst_sync_init(GST_SYNC *sync, IMAGE *in, int gpusync)
{
    memset(sync, 0, sizeof(GST_SYNC));
    sync->gpusync  = gpusync;
    sync->d_insync = ImageStreamIO_get_image_d_sync_ptr(in);

    cudaError_t cerr = cudaSetDevice(in->md->location);
    if (cerr == cudaSuccess)
    {
        cerr = cudaStreamCreate(&sync->stream);
    }
    if (cerr == cudaSuccess)
    {
        cerr = cudaMalloc((void **) &sync->d_trig, sizeof(GST_TRIGGER));
    }
    if (cerr == cudaSuccess)
    {
        IMAGE_GPU_SYNC h_insync;
        cerr =
            cudaMemcpy(&h_insync, sync->d_insync, sizeof(IMAGE_GPU_SYNC), cudaMemcpyDeviceToHost);
        sync->h_trig.in_cnt = h_insync.cnt0;
    }
    if (cerr == cudaSuccess)
    {
        cerr = cudaMemcpy(sync->d_trig, &sync->h_trig, sizeof(GST_TRIGGER), cudaMemcpyHostToDevice);
    }
    if (cerr != cudaSuccess)
    {
        PRINT_ERROR("CUDA setup: %s", cudaGetErrorString(cerr));
        return RETURN_FAILURE;
    }

    sync->last_in_cnt = sync->h_trig.in_cnt;
    return RETURN_SUCCESS;
}


void gst_sync_free(GST_SYNC *sync)
{
    cudaStreamDestroy(sync->stream);
    cudaFree(sync->d_trig);
}


void gst_input_prep(GST_SYNC *sync)
{
    sync->nbout = 0;

    if (sync->gpusync)
    {
        gst_enqueue_wait(sync->stream, sync->d_insync, sync->d_trig);
    }
}


void gst_output_prep(GST_SYNC *sync, IMAGE *out)
{
    if (sync->nbout < GST_NBOUT_MAX)
    {
        sync->out[sync->nbout++] = out;
    }

    if (!sync->gpusync)
    {
        out->md->write = 1;
    }
}


int gst_input_closeout(GST_SYNC *sync, PROCESSINFO *processinfo)
{
    // Device-side publish, after the compute kernels enqueued by the caller
    for (int oi = 0; oi < sync->nbout; oi++)
    {
        gst_enqueue_publish(sync->stream, ImageStreamIO_get_image_d_sync_ptr(sync->out[oi]),
                            sync->d_trig);
    }

    cudaError_t cerr = cudaGetLastError(); // launches of this frame, compute kernels included
    if (cerr == cudaSuccess)
    {
        cerr = cudaMemcpyAsync(&sync->h_trig, sync->d_trig, sizeof(GST_TRIGGER),
                               cudaMemcpyDeviceToHost, sync->stream);
    }
    if (cerr == cudaSuccess)
    {
        cerr = cudaStreamSynchronize(sync->stream);
    }
    if (cerr != cudaSuccess)
    {
        PRINT_ERROR("CUDA: %s", cudaGetErrorString(cerr));
        sync->status = GST_FRAME_ERROR;
        return sync->status;
    }

    if (!sync->gpusync)
    {
        // Trigger information was set on the host by processinfo_waitoninputstream()
        sync->status = GST_FRAME_DONE;
        return sync->status;
    }

    if (sync->h_trig.timeout)
    {
        sync->status = GST_FRAME_TIMEOUT;
        if (processinfo != NULL)
        {
            processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_TIMEDOUT;
            processinfo->trigggertimeoutcnt++;
        }
        return sync->status;
    }

    sync->status = GST_FRAME_DONE;
    if (processinfo != NULL)
    {
        processinfo->triggerstatus      = PROCESSINFO_TRIGGERSTATUS_RECEIVED;
        processinfo->triggermissedframe = (int) (sync->h_trig.in_cnt - sync->last_in_cnt - 1);
        processinfo->triggermissedframe_cumul += processinfo->triggermissedframe;
        processinfo->triggerstreamcnt = sync->h_trig.in_cnt;
    }
    sync->last_in_cnt = sync->h_trig.in_cnt;

    return sync->status;
}


void gst_output_closeout(GST_SYNC *sync, IMAGE *out, PROCESSINFO *processinfo, IMAGE *in)
{
    if (sync->status != GST_FRAME_DONE)
    {
        return;
    }

    // cnt0, times, write = 0, stream processing trace, semaphores
    processinfo_update_output_stream(processinfo, out, in);
}
