// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gst_addone.c
 * @brief   DRAFT, never compiled. GPU-resident stream processing with CPU or GPU ingress sync
 *
 * out = in + 1 on a CUDA stream, between two GPU-resident uint16 streams of the same device.
 *
 *  - .gpusync OFF: host trigger (semaphore of the ingress).
 *  - .gpusync ON : trigger mode is forced to immediate; a wait kernel polls the IMAGE_GPU_SYNC
 *                  block of the ingress.
 *
 * The per-frame synchronization (trigger, device-side publish, CPU metadata and post) is in
 * gst_sync.h; the loop below only enqueues its compute kernel between the prep and closeout
 * calls.
 *
 * Producer and consumer in separate processes need CUDA MPS for the wait kernel to see the
 * counter move (contexts are time-sliced otherwise).
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

// MILK_CMAKE_MANDATE_CUDA
#include <cuda_runtime_api.h>

#include "gst_kernels.h"
#include "gst_sync.h"

static char    addone_in[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char    addone_out[FUNCTION_PARAMETER_STRMAXLEN] = "";
static int32_t addone_gpusync                           = 0;

#define GST_ADDONE_PARAMS(X)                                                            \
    X(".in_name", addone_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM,       \
      "input stream (GPU, uint16)")                                                     \
    X(".out_name", addone_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output stream") \
    X(".gpusync", &addone_gpusync, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,               \
      "ingress sync on GPU (OFF: CPU semaphore)")

MILK_PARAMS_DEFINE(gst_addone, GST_ADDONE_PARAMS);


/**
 * @brief Link the egress stream, or create it with the shape of the ingress, on the same device.
 *
 * Not registered in the image array of the process (IMGID has no GPU creation path).
 */
static errno_t gst_link_or_create_egress(IMAGE *outimage, const char *name, IMAGE_METADATA *inmd)
{
    memset(outimage, 0, sizeof(IMAGE));

    if (ImageStreamIO_openIm(outimage, name) == IMAGESTREAMIO_SUCCESS)
    {
        if (outimage->md->location != inmd->location || outimage->md->datatype != inmd->datatype ||
            outimage->md->naxis != inmd->naxis || outimage->md->nelement != inmd->nelement)
        {
            PRINT_ERROR("stream %s exists and does not match the input", name);
            return RETURN_FAILURE;
        }
        return RETURN_SUCCESS;
    }

    if (ImageStreamIO_createIm_gpu(outimage, name, inmd->naxis, inmd->size, inmd->datatype,
                                   inmd->location, 1, IMAGE_NB_SEMAPHORE, inmd->NBkw,
                                   inmd->imagetype, 0) != IMAGESTREAMIO_SUCCESS)
    {
        PRINT_ERROR("cannot create stream %s", name);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}


static errno_t gst_addone_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    // Host waits in cudaStreamSynchronize() by spinning: lowest latency, one busy CPU core.
    // Only effective if no CUDA context exists yet in this process; error ignored otherwise.
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    IMGID inimg = imgid_make_from_name(addone_in);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (inimg.md->location < 0 || inimg.md->datatype != _DATATYPE_UINT16)
    {
        PRINT_ERROR("stream %s must be GPU-resident and uint16", addone_in);
        return RETURN_FAILURE;
    }

    IMAGE outimage;
    if (gst_link_or_create_egress(&outimage, addone_out, inimg.md) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }

    GST_SYNC sync;
    if (gst_sync_init(&sync, inimg.im, addone_gpusync) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }

    const uint64_t  nelement = inimg.md->nelement;
    const uint16_t *d_in     = (const uint16_t *) ImageStreamIO_get_image_d_ptr(inimg.im);
    uint16_t       *d_out    = (uint16_t *) ImageStreamIO_get_image_d_ptr(&outimage);

    // The flag owns the trigger mode: the wait kernel is the trigger when gpusync is ON
    {
        const int triggermode =
            sync.gpusync ? PROCESSINFO_TRIGGERMODE_IMMEDIATE : PROCESSINFO_TRIGGERMODE_SEMAPHORE;
        self->rt->cmddata.cmdsettings->triggermode = triggermode;
        if (dcfpsptr != NULL)
        {
            dcfpsptr->cmdset.triggermode = triggermode;
            functionparameter_SetParamValue_INT64(dcfpsptr, ".procinfo.triggermode", triggermode);
        }
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        gst_input_prep(&sync);
        gst_output_prep(&sync, &outimage); // auxiliary outputs first, main output last

        gst_enqueue_addone(sync.stream, d_in, d_out, nelement, sync.d_trig);

        if (gst_input_closeout(&sync, processinfo) == GST_FRAME_ERROR && processinfo != NULL)
        {
            processinfo->CTRLval = 3; // exit loop
        }
        gst_output_closeout(&sync, &outimage, processinfo, inimg.im);
    }
    MILK_PROCINFO_END(self)

    gst_sync_free(&sync);
    imgid_free(&inimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// Not declared in a header: the module registry declares it weak (file may be dropped).
const MILK_RUNNABLE runnable_gst_addone = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "gstaddone",
                 .cmdkey           = "addone",
                 .description      = "gpusynctest: out = in + 1 on GPU, CPU or GPU ingress sync",
                 .description_long = "GPU-resident uint16 streams; .gpusync selects where the "
                                     "ingress is synchronized" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(gst_addone),
    .compute = gst_addone_compute,
};
EXPORT_FPSEXEC(milk, "gpusynctest-addone", runnable_gst_addone);
