// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "processinfo_internal.h"
#include "ImageStreamIO/ImageStreamIO.h"

#ifndef CLOCK_MILK
#    define CLOCK_MILK CLOCK_REALTIME
#endif

static inline void propagate_long_kw(const IMAGE *in_im,
                                     IMAGE       *out_im,
                                     const char  *kw_name,
                                     const char  *kw_comment)
{
    if (in_im == NULL || out_im == NULL || in_im == out_im || in_im->md == NULL ||
        out_im->md == NULL || in_im->kw == NULL || out_im->kw == NULL)
    {
        return;
    }

    int     nb_in  = in_im->md[0].NBkw;
    int     in_idx = -1;
    int64_t in_val = 0;
    for (int k = 0; k < nb_in; k++)
    {
        if (in_im->kw[k].type == 'L' && strcmp(in_im->kw[k].name, kw_name) == 0)
        {
            in_idx = k;
            in_val = in_im->kw[k].value.numl;
            break;
        }
    }
    if (in_idx < 0 || in_val <= 0)
    {
        return;
    }

    int nb_out    = out_im->md[0].NBkw;
    int empty_idx = -1;
    for (int k = 0; k < nb_out; k++)
    {
        if (strcmp(out_im->kw[k].name, kw_name) == 0)
        {
            if (out_im->kw[k].type == 'L' && out_im->kw[k].value.numl >= in_val)
            {
                return;
            }
            out_im->kw[k].type       = 'L';
            out_im->kw[k].value.numl = in_val;
            return;
        }
        if (empty_idx < 0 && out_im->kw[k].name[0] == '\0')
        {
            empty_idx = k;
        }
    }

    if (empty_idx >= 0)
    {
        snprintf(out_im->kw[empty_idx].name, KEYWORD_MAX_STRING, "%s", kw_name);
        out_im->kw[empty_idx].type       = 'L';
        out_im->kw[empty_idx].value.numl = in_val;
        snprintf(out_im->kw[empty_idx].comment, KEYWORD_MAX_COMMENT, "%s", kw_comment);
    }
}

/* getpid() is a real syscall on glibc >= 2.25: cache it, reset in a forked child */
static pid_t g_pid = 0;

static void pid_fork_child(void)
{
    g_pid = 0;
}

__attribute__((constructor)) static void pid_register_atfork(void)
{
    pthread_atfork(NULL, NULL, pid_fork_child);
}

static inline pid_t cached_pid(void)
{
    if (g_pid == 0)
    {
        g_pid = getpid();
    }
    return g_pid;
}

/**
 * @brief Update output stream metadata and telemetry at the end of a loop iteration.
 */
errno_t processinfo_update_output_stream(PROCESSINFO *processinfo,
                                         IMAGE       *output_image,
                                         IMAGE       *input_image)
{
    if (output_image == NULL)
    {
        return RETURN_FAILURE;
    }

    if (output_image->md->shared == 1)
    {
        // Always update PID and timestamp, regardless of processinfo status
        struct timespec ts;
        if (clock_gettime(CLOCK_MILK, &ts) == -1)
        {
            PRINT_ERROR("clock_gettime: %s", strerror(errno));
            exit(EXIT_FAILURE);
        }

        output_image->streamproctrace[0].procwrite_PID   = cached_pid();
        output_image->streamproctrace[0].ts_streamupdate = ts;

        DEBUG_TRACEPOINT(" ");

        if (processinfo != NULL)
        {
            // If input_image is NULL, try to use processinfo->trigger_image
            if (input_image == NULL)
            {
                if (processinfo->trigger_image != NULL)
                {
                    input_image = processinfo->trigger_image;
                }
            }

            if (input_image != NULL)
            {
                int sptisize = input_image->md[0].NBproctrace - 1;
                // Ensure we don't overflow output trace
                int sptosize = output_image->md[0].NBproctrace - 1;
                if (sptisize > sptosize)
                {
                    sptisize = sptosize;
                }

                if (sptisize > 0)
                {
                    // copy streamproctrace from input to output (same image when in place)
                    memmove(&output_image->streamproctrace[1], &input_image->streamproctrace[0],
                            sizeof(STREAM_PROC_TRACE) * sptisize);
                }
            }

            // write first streamproctrace entry
            DEBUG_TRACEPOINT("trigger info");
            output_image->streamproctrace[0].triggermode = processinfo->triggermode;

            output_image->streamproctrace[0].trigger_inode = processinfo->triggerstreaminode;

            output_image->streamproctrace[0].ts_procstart =
                processinfo->texecstart[processinfo->timerindex];

            output_image->streamproctrace[0].trigsemindex = processinfo->triggersem;

            output_image->streamproctrace[0].triggerstatus = processinfo->triggerstatus;

            if (input_image != NULL)
            {
                output_image->streamproctrace[0].cnt0 = input_image->md[0].cnt0;
            }
        }

        if (input_image != NULL)
        {
            propagate_long_kw(input_image, output_image, "_MAQTIME",
                              "Acquisition time [us since Unix epoch]");
            propagate_long_kw(input_image, output_image, "_MTICK",
                              "Master simulation tick index");
        }

        DEBUG_TRACEPOINT(" ");
    }

    ImageStreamIO_UpdateIm(output_image);

    return RETURN_SUCCESS;
}
