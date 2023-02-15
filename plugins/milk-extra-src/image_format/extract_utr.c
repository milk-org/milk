/**
 * @file    extract_utr.c
 * @brief   CDS (correlated double sampling) + UTR (sample up-the-ramp) image processing loop for CRED streams
 *
 * Designed for CRED cameras:
 *      Input support int16 / uin16
 *      Relies on counters in the first pixels either in CRED2 or CRED1 formats
 *      Determines from counter behavior if rawimages is on/off and falls back to passthrough mode
 *      Relies on stream keyword DET-NSMP to determine current NDR value
 *
 * Input: raw camera stream name
 * Input: output UTR stream name
 * Input: Saturation threshold for UTR/CDS discard
 *
 * Output: Post UTR reduced stream (float 32)
 */

#include <pthread.h>

#include "CommandLineInterface/CLIcore.h"
#include "extract_utr.h"

// Local variables pointers
static char  *in_imname;
static char  *out_imname;
static float *ptr_sat_value;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &in_imname,
        NULL
    },
    {
        CLIARG_STR_NOT_IMG,
        ".out_name",
        "up-the-ramp image",
        "out2",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &out_imname,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".sat_value",
        "Saturation threshold",
        "satval",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ptr_sat_value,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"cred_cds_utr",
                                "RT compute of CDS/UTR for camera streams",
                                CLICMD_FIELDS_DEFAULTS
                               };

static errno_t help_function()
{
    printf(
        "Perform real-time up-the-ramp data reduction on CRED1/2 streams.\n");
    return RETURN_SUCCESS;
}

/*
THE IMPORTANT, CUSTOM PART
*/

static errno_t copy_cast_SI16TOF(float *out, int16_t *in, int n_val)
{
    for(long ii = 0; ii < n_val; ++ii)
    {
        out[ii] = (float) in[ii];
    }

    return RETURN_SUCCESS;
}

static errno_t copy_cast_UI16TOF(float *out, uint16_t *in, int n_val)
{
    for(long ii = 0; ii < n_val; ++ii)
    {
        out[ii] = (float) in[ii];
    }

    return RETURN_SUCCESS;
}

static errno_t simple_desat_iterate(float  *last_valid,
                                    int    *frame_count,
                                    u_char *frame_valid,
                                    float   sat_val,
                                    IMGID   in_img,
                                    int     reset)
{

    int n_pixels = in_img.md->size[0] * in_img.md->size[1];

    float in_val_px;
    int   k;

    if(reset)
    {
        for(
            int ii = 8; ii < n_pixels;
            ++ii) // For all pixels, including the tags [we could skip the 1st row on the CREDs]
        {
            in_val_px       = (float) in_img.im->array.UI16[ii];
            k               = (in_val_px <= sat_val);
            frame_valid[ii] = k;
            frame_count[ii] = 1;

            last_valid[ii] = k ? in_val_px : 0.0f;
        }
    }
    else
    {
        for(
            int ii = 8; ii < n_pixels;
            ++ii) // For all pixels, including the tags [we could skip the 1st row on the CREDs]
        {
            in_val_px       = (float) in_img.im->array.UI16[ii];
            k               = (in_val_px <= sat_val);
            frame_valid[ii] = k;
            frame_count[ii] += k;

            last_valid[ii] = k ? in_val_px : last_valid[ii];
        }
    }

    return RETURN_SUCCESS;
}

static errno_t utr_iterate(float  *sum_x,
                           float  *sum_y,
                           float  *sum_xy,
                           float  *sum_xx,
                           float  *sum_yy,
                           int    *frame_count,
                           u_char *frame_valid,
                           float   sat_val,
                           IMGID   in_img,
                           int     reset)
{

    int subframe_count = in_img.im->array.UI16[2]; // NDR raw counter
    int n_pixels       = in_img.md->size[0] * in_img.md->size[1];

    float in_val_px;
    int   k;

    if(reset)
    {
        for(
            int ii = 8; ii < n_pixels;
            ++ii) // For all pixels, including the tags [we could skip the 1st row on the CREDs]
        {
            in_val_px = (float) in_img.im->array.UI16[ii];

            // Detect saturation - which can have several forms for CRED1 / CRED2 / clipping to some max

            k               = (in_val_px <= sat_val);
            frame_valid[ii] = k;

            frame_count[ii] = k; // At reset: 0 or 1

            sum_x[ii]  = k * subframe_count;
            sum_y[ii]  = k * in_val_px;
            sum_xy[ii] = (k * subframe_count) * in_val_px;
            sum_xx[ii] = (k * subframe_count) * subframe_count;
            sum_yy[ii] = (k * in_val_px) * in_val_px;
        }
    }
    else
    {
        // not reset
        for(
            int ii = 8; ii < n_pixels;
            ++ii) // For all pixels, including the tags [we could skip the 1st row on the CREDs]
        {
            in_val_px = (float) in_img.im->array.UI16[ii];

            // Detect saturation - which can have several forms for CRED1 / CRED2 / clipping to some max
            k = (in_val_px <= sat_val);

            frame_valid[ii] = k;
            frame_count[ii] += k; // At reset: 0 or 1

            // Only perform those accumulations for unsat pixels, but this avoids if statements.
            sum_x[ii] += k * subframe_count;
            sum_y[ii] += k * in_val_px;
            sum_xy[ii] += (k * subframe_count) * in_val_px;
            sum_xx[ii] += (k * subframe_count) * subframe_count;
            sum_yy[ii] += (k * in_val_px) * in_val_px;
        }
    }

    return RETURN_SUCCESS;
}

static errno_t utr_reset_buffers(float  *sum_x,
                                 float  *sum_y,
                                 float  *sum_xy,
                                 float  *sum_xx,
                                 float  *sum_yy,
                                 int    *frame_count,
                                 u_char *frame_valid,
                                 int     n_pixels)
{
    memset(sum_x, 0, n_pixels * SIZEOF_DATATYPE_FLOAT);
    memset(sum_y, 0, n_pixels * SIZEOF_DATATYPE_FLOAT);
    memset(sum_xy, 0, n_pixels * SIZEOF_DATATYPE_FLOAT);
    memset(sum_xx, 0, n_pixels * SIZEOF_DATATYPE_FLOAT);
    memset(sum_yy, 0, n_pixels * SIZEOF_DATATYPE_FLOAT);

    memset(frame_count, 0, n_pixels * SIZEOF_DATATYPE_INT32);
    memset(frame_valid, 1, n_pixels * SIZEOF_DATATYPE_UINT8);

    return RETURN_SUCCESS;
}

static errno_t utr_finalize(float *sum_x,
                            float *sum_y,
                            float *sum_xy,
                            float *sum_xx,
                            int   *frame_count,
                            int    tot_num_frames,
                            int    n_pixels,
                            float *out_buf)
{

    int   fcii;
    float sxii;

    for(int ii = 0; ii < n_pixels; ++ii)
    {
        fcii = frame_count[ii];
        sxii = sum_x[ii];

        if(fcii > 1)  // Multiple valid readouts
        {
            // There's a minus because x is the decreasing raw number, thus decreases w/ time.
            out_buf[ii] = -tot_num_frames *
                          (fcii * sum_xy[ii] - sxii * sum_y[ii]) /
                          (fcii * sum_xx[ii] - sxii * sxii);
            /*if((frame_count[ii] * sum_xx[ii] - sum_x[ii] * sum_x[ii]) == 0)
            {
                utr_img.im->array.F[ii] = -1;
                // PRINT_WARNING("MADE NANs -- %d, %d, %f, %f", ii, frame_count[ii], sum_xx[ii], sum_x[ii]*sum_x[ii]);
            }*/
        }
        else if(fcii == 1)  // One single valid readout
        {
            out_buf[ii] = tot_num_frames * sum_x[ii];
        }
        else
        {
            out_buf[ii] = 0.0f;
        }
    }

    return RETURN_SUCCESS;
}

static errno_t simple_desat_finalize(float *last_valid,
                                     float *first_read,
                                     int   *frame_count,
                                     int    tot_num_frames,
                                     int    n_pixels,
                                     int    invert,
                                     float *out_buf)
{
    if(!invert)
    {
        for(int ii = 0; ii < n_pixels; ++ii)
        {
            // Avoid no valid frames // We need at least two reads to CDS them.
            out_buf[ii] = frame_count[ii] >= 2
                          ? ((tot_num_frames - 1) *
                             (last_valid[ii] - first_read[ii]) /
                             (frame_count[ii] - 1))
                          : 0.0f;
        }
    }
    else
    {
        // invert
        for(int ii = 0; ii < n_pixels; ++ii)
        {
            // Avoid no valid frames // We need at least two reads to CDS them.
            out_buf[ii] = frame_count[ii] >= 2
                          ? ((tot_num_frames - 1) *
                             (first_read[ii] - last_valid[ii]) /
                             (frame_count[ii] - 1))
                          : 0.0f;
        }
    }

    return RETURN_SUCCESS;
}

/*
BOILERPLATE
*/

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID in_img = mkIMGID_from_name(in_imname);
    resolveIMGID(&in_img, ERRMODE_ABORT);

    // Set in_img to be the trigger
    strcpy(CLIcmddata.cmdsettings->triggerstreamname, in_imname);
    // for FPS mode:
    if(data.fpsptr != NULL)
    {
        strcpy(data.fpsptr->cmdset.triggerstreamname, in_imname);
    }

    // Resolve or create outputs, per need
    IMGID out_img = mkIMGID_from_name(out_imname);
    if(resolveIMGID(&out_img, ERRMODE_WARN))
    {
        PRINT_WARNING("WARNING - output image not found and being created");
        in_img.datatype = _DATATYPE_FLOAT; // To be passed to out_img
        imcreatelikewiseIMGID(&out_img, &in_img);
        resolveIMGID(&out_img, ERRMODE_ABORT);
    }

    /*
     Keyword setup - initialization
    */
    int ndr_kw_loc = -1;

    for(int kw = 0; kw < in_img.md->NBkw; ++kw)
    {
        strcpy(out_img.im->kw[kw].name, in_img.im->kw[kw].name);
        out_img.im->kw[kw].type  = in_img.im->kw[kw].type;
        out_img.im->kw[kw].value = in_img.im->kw[kw].value;
        strcpy(out_img.im->kw[kw].comment, in_img.im->kw[kw].comment);

        if(strcmp(in_img.im->kw[kw].name, "DET-NSMP") == 0)
        {
            // DET-NSMP official fits keyword name for NDR.
            ndr_kw_loc = kw;
        }
    }

    /*
    SETUP
    */
    // For counting NRD reads
    int  cred_counter           = 0;
    int  prev_cred_counter      = 0;
    int  cred_counter_last_init = 0;
    int  cred_counter_repeat    = 0;
    long time_acq_us            = 0; // Time acq embedded at pixel 8

    // For the imagetags
    int px_check = 0;

    // For counting frames and avoiding double processing when catching up with the semaphore
    long frame_counter           = 0;
    long prev_frame_counter      = 0;
    long frame_counter_last_init = 0;

    int ndr_value     = 0;
    int old_ndr_value = 0;

    int  n_pixels = in_img.md->size[0] * in_img.md->size[1];
    long buf_pp   = 0;

    float *sum_x[2];
    float *sum_xx[2];
    float *sum_y[2];
    float *sum_xy[2];
    float *sum_yy[2];

    int    *frame_count[2];
    u_char *frame_valid[2];
    float  *last_valid[2];
    float  *save_first_read[2];

    for(long pp = 0; pp < 2; ++pp)
    {
        sum_x[pp]  = (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);
        sum_xx[pp] = (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);
        sum_y[pp]  = (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);
        sum_xy[pp] = (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);
        sum_yy[pp] = (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);

        frame_count[pp] = (int *) malloc(n_pixels * SIZEOF_DATATYPE_INT32);
        frame_valid[pp] = (u_char *) malloc(n_pixels * SIZEOF_DATATYPE_INT8);
        last_valid[pp]  = (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);
        save_first_read[pp] =
            (float *) malloc(n_pixels * SIZEOF_DATATYPE_FLOAT);

        // Reset the buffers for utr
        utr_reset_buffers(sum_x[pp],
                          sum_y[pp],
                          sum_xy[pp],
                          sum_xx[pp],
                          sum_yy[pp],
                          frame_count[pp],
                          frame_valid[pp],
                          n_pixels);
        // Reset the buffer for simple_desat
        memset(last_valid[pp], 0, n_pixels * SIZEOF_DATATYPE_FLOAT);
    }

    // TELEMETRY
    int just_init  = FALSE;
    int miss_count = 0;

    // Multi-warp finalization
    int pending_fin_warps = FALSE;
    int next_fin_warp;
    int publishable_output;
    int tot_fin_warps = 2;
    int n_pixels_in_warp;
    int warp_offset;

    // FIXME FIXME FIXME FIXME
    PRINT_WARNING("Saturation value: %f", *ptr_sat_value);

    /*
    PROCESSINFO INIT
    */
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    // PROCESSINFO* processinfo now available

    /*
    LOOP
    */

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    {

        old_ndr_value = ndr_value;

        prev_frame_counter = frame_counter;
        frame_counter      = in_img.im->array.UI16[0] |
        (in_img.im->array.UI16[1] << 16); // 32 bit counter

        if(frame_counter <= prev_frame_counter)
        {
            // Do not process the same frame twice if late on the semaphores.
            // This will trigger when the framegrabber garbages out
            // This will trigger when we wraparound after 2**32 frames
            PRINT_WARNING("Continue issued at %ld, %ld",
                          prev_frame_counter,
                          frame_counter);
            continue; // This applies to the loop started and closed in PROCINFO macros
        }

        // if we hit 0 just before, this is the first image, save it for the CDS
        prev_cred_counter = cred_counter;
        cred_counter      = in_img.im->array.UI16[2]; // Counter in px 3

        px_check = in_img.im->array.UI16[3];

        /*
        INITIALIZE NDR FROM KW
        */
        ndr_value =
        (int) in_img.im->kw[ndr_kw_loc]
        .value
        .numl; // This is the TRUE NDR value, per the camera control server.

        /*
        HOUSEKEEPING + HIJACK COUNTER FOR CRED1 NDR2
        Because CRED1 NDR2 counts 0 then 1, rather than the opposite in all other modes.
        TODO actually decide before entering the loop if this is CRED1 or 2 once and for all.
        */
        if(in_img.md->datatype == _DATATYPE_UINT16 && ndr_value == 2)
        {
            cred_counter = 1 - cred_counter;
        }
        if(prev_cred_counter > 0 && cred_counter > prev_cred_counter)
        {
            // PRINT_WARNING("Raw frame 0 missed - a UTR/SDS frame was lost");
        }

        /*
        Complicated branching:
        A / Find if we're in NDR1
        B / Is this the CRED 1 and the CRED 2
        C / Find if we're in rawimages off -> override to ndr_value = 1; for CRED2 this is px_check == ndr_val
        for CRED1 this is
            C.1 = px[2] always = 1 in CDS
            C.2 = px[2] always = 0 in NDR
        D / Find if we've lost sync: CRED2 4th px should match 0x3ff0, CRED1 4th pix should match 0x0000
        */
        // First: CRED1 ndr change accumulator:
        if(cred_counter == prev_cred_counter)
        {
            if(cred_counter_repeat < 10)
            {
                ++cred_counter_repeat;
            }
        }
        else
        {
            cred_counter_repeat = 0;
        }

        just_init = FALSE;
        if(ndr_value == 1 ||
                (in_img.md->datatype == _DATATYPE_UINT16 &&
                 (cred_counter_repeat == 10 || !(px_check == 0))) ||
                (in_img.md->datatype == _DATATYPE_INT16 &&
                 (cred_counter == ndr_value || !((px_check & 0x3ff0) == 0x3ff0))))
        {
            ndr_value               = 1; // Override
            frame_counter_last_init = frame_counter;
            cred_counter_last_init  = cred_counter;
            just_init               = TRUE;
        }
        else if(prev_cred_counter == 0 || cred_counter > prev_cred_counter)
        {
            // Test: we are at the first frame of a burst OR we just missed the last frame of the previous burst
            // Note: ndr_value > 1 here.
            // Backup the first frame for CDS output
            if(in_img.md->datatype == _DATATYPE_UINT16)
            {
                copy_cast_UI16TOF(save_first_read[buf_pp],
                                  in_img.im->array.UI16,
                                  n_pixels);
            }
            else
            {
                copy_cast_SI16TOF(save_first_read[buf_pp],
                                  in_img.im->array.SI16,
                                  n_pixels);
            }
            frame_counter_last_init = frame_counter;
            cred_counter_last_init  = cred_counter;
            just_init               = TRUE;
        }

        // Did we skip a frame ?
        if(ndr_value > 1 && frame_counter != prev_frame_counter + 1)
        {
            // TELEMETRY
            ++miss_count;
        }

        if(old_ndr_value != ndr_value)
        {
            PRINT_WARNING("NDR meas changed from %d to %d",
                          old_ndr_value,
                          ndr_value);
        }

        tot_fin_warps = ndr_value == 1 ? 1 : 2;

        // PRINT_WARNING("%d, %d, %d", in_img.im->array.UI16[0], in_img.im->array.UI16[2], in_img.im->array.UI16[39185]);

        /*
        ACCUMULATE
        */
        if(ndr_value > 1 && ndr_value <= 6)
        {
            simple_desat_iterate(last_valid[buf_pp],
                                 frame_count[buf_pp],
                                 frame_valid[buf_pp],
                                 *ptr_sat_value,
                                 in_img,
                                 just_init);
        }
        else if(ndr_value > 6)
        {
            utr_iterate(sum_x[buf_pp],
                        sum_y[buf_pp],
                        sum_xy[buf_pp],
                        sum_xx[buf_pp],
                        sum_yy[buf_pp],
                        frame_count[buf_pp],
                        frame_valid[buf_pp],
                        *ptr_sat_value,
                        in_img,
                        just_init);
        }

        /*
        PRE - FINALIZE
        */
        if(cred_counter == 0 ||
                ndr_value ==
                1) // If we are hitting 0, compute the UTR, the QL, and post the outputs
        {
            if(pending_fin_warps)
            {
                PRINT_ERROR(
                    "Entering finalize with pending fin_warps from previous "
                    "finalize");
            }
            // Copy the first 4 pixels from the current image
            copy_cast_UI16TOF(out_img.im->array.F, in_img.im->array.UI16, 4);
            // Add some more telemetry
            out_img.im->array.F[4] = (float)
                                     ndr_value; // Value by which stuff is normalized, and type of processing done.
            out_img.im->array.F[5] = (float) cred_counter_last_init;
            out_img.im->array.F[6] =
                ((float) frame_counter_last_init) /
                1e6; // Divide by 1e6 to avoid messing up scaling
            out_img.im->array.F[7] = (float) miss_count;

            // Fetch the time of acquisition that's been embedded by edttake at pixel 8 as a raw long.
            time_acq_us = *((long *) &in_img.im->array.UI16[8]);
            // Store 6 digits per pixel
            out_img.im->array.F[8] = (float)(time_acq_us / 1000000000000L);
            out_img.im->array.F[9] =
                (float)((time_acq_us / 1000000L) % 1000000L);
            out_img.im->array.F[10] = (float)(time_acq_us % 1000000L);

            /*
            Keyword value carry-over
            */
            for(int kw = 0; kw < in_img.md->NBkw; ++kw)
            {
                out_img.im->kw[kw].value = in_img.im->kw[kw].value;
            }

            next_fin_warp      = 0;
            pending_fin_warps  = TRUE;
            publishable_output = TRUE;

            // Ping-pong toggle
            buf_pp = 1 - buf_pp;

            // HOUSEKEEPING
            if(miss_count > 0)
            {
                PRINT_WARNING("UTR/SDS ramp - missing %d/%d frames (cnt0 %ld)",
                              miss_count,
                              ndr_value,
                              in_img.md->cnt0);
                miss_count = 0;
            }

            // TODO ??? Weighted stuff.
        }

        /*
        FINALIZATION WARPS
        */
        if(pending_fin_warps)
        {
            // PREPARE WARP INDICES
            if(next_fin_warp == 0)  // First warp
            {
                warp_offset      = 12; // Skip the telemetry counters
                n_pixels_in_warp = n_pixels / tot_fin_warps - 12;
            }
            else
            {
                warp_offset = next_fin_warp * (n_pixels / tot_fin_warps);
                if(next_fin_warp == tot_fin_warps - 1)  // Final warp
                {
                    n_pixels_in_warp = n_pixels - warp_offset;
                }
                else
                {
                    n_pixels_in_warp = n_pixels / tot_fin_warps;
                }
            }

            // WARP!
            if(ndr_value == 1)  // PASSTHROUGH
            {
                // ndr_value == 1: single reads OR rawimages off passthrough mode
                // Skip 8 meta info pixels
                if(in_img.md->datatype == _DATATYPE_UINT16)
                {
                    copy_cast_UI16TOF(out_img.im->array.F + warp_offset,
                                      in_img.im->array.UI16 + warp_offset,
                                      n_pixels_in_warp);
                }
                else
                {
                    copy_cast_SI16TOF(out_img.im->array.F + warp_offset,
                                      in_img.im->array.SI16 + warp_offset,
                                      n_pixels_in_warp);
                }
            }
            else if(ndr_value <= 6)  // CDS
            {
                if(next_fin_warp == 0 &&
                        frame_counter != frame_counter_last_init + ndr_value - 1)
                {
                    // Did we get two reads to do a proper CDS ?
                    // Compute the exposure scaling in case we missed the first read !
                    // This will be very important in CDS at high speed
                    PRINT_WARNING("CDS / DESAT finalize: not enough reads.");
                    publishable_output = FALSE; // Abort finalization
                    next_fin_warp      = tot_fin_warps - 1;
                }
                else
                {
                    out_img.im->md->write = TRUE;
                    simple_desat_finalize(
                        &last_valid[1 - buf_pp][warp_offset],
                        &save_first_read[1 - buf_pp][warp_offset],
                        &frame_count[1 - buf_pp][warp_offset],
                        ndr_value,
                        n_pixels_in_warp,
                        FALSE, // No inversion even CRED1 CDS
                        & (out_img.im->array.F[warp_offset]));
                }
            }
            else // UTR
            {
                out_img.im->md->write = TRUE;
                utr_finalize(&sum_x[1 - buf_pp][warp_offset],
                             &sum_y[1 - buf_pp][warp_offset],
                             &sum_xy[1 - buf_pp][warp_offset],
                             &sum_xx[1 - buf_pp][warp_offset],
                             &frame_count[1 - buf_pp][warp_offset],
                             ndr_value,
                             n_pixels_in_warp,
                             &(out_img.im->array.F[warp_offset]));
            }

            if(next_fin_warp == tot_fin_warps - 1)
            {
                pending_fin_warps = FALSE;
                if(publishable_output)
                {
                    processinfo_update_output_stream(processinfo, out_img.ID);
                }
            }
            ++next_fin_warp;
        }
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    /*
    TEARDOWN
    */

    for(int pp = 0; pp < 2; ++pp)
    {
        free(sum_x[pp]);
        free(sum_y[pp]);
        free(sum_xy[pp]);
        free(sum_xx[pp]);
        free(sum_yy[pp]);

        free(frame_count[pp]);
        free(frame_valid[pp]);
        free(last_valid[pp]);
        free(save_first_read[pp]);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/*
CLI boilerplate
*/
INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_image_format__cred_cds_utr()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
