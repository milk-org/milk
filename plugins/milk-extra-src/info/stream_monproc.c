// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_monproc.c
 * @brief   monitor stream with multi-level time binning, circular buffer, and
 * dynamic histogram
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <glob.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/processtools_trigger.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "stream_monproc.h"

static char *inimname;

/**
 * @brief Time binning flag
 *
 * Each bit 'i' set in tbinflag enables a time bin of 2^i frames.
 * Example: tbinflag = 48 (binary 110000) enables bins of 16 (2^4) and 32 (2^5)
 * frames.
 */
static uint64_t *tbinflag;
static long      fpi_tbinflag = -1;

static uint32_t *cbbuffersize;
static long      fpi_cbbuffersize = -1;

static CLICMDARGDEF farg[] = {
    { CLIARG_IMG, ".in_name", "input image", "im1", CLIARG_VISIBLE_DEFAULT, (void **) &inimname,
      NULL },
    { CLIARG_UINT64, ".tbinflag", "time binning flag (bit i sets bin 2^i)", "48",
      CLIARG_HIDDEN_DEFAULT, (void **) &tbinflag, &fpi_tbinflag },
    { CLIARG_UINT32, ".cbsize", "circular buffer size", "64", CLIARG_HIDDEN_DEFAULT,
      (void **) &cbbuffersize, &fpi_cbbuffersize }
};

static CLICMDDATA CLIcmddata = { "streammon",
                                 "stream monitor with multi-level time binning and circular buffer",
                                 CLICMD_FIELDS_DEFAULTS };

// Exposed help function
errno_t stream_monitor_help()
{
    printf("Monitor a stream and compute averages and RMS at different time "
           "scales.\n");
    printf("The scales are defined by the tbinflag bitmask.\n");
    printf("Also maintains a circular buffer of the last .cbsize frames.\n");
    printf("Supports input datatypes: uint8, int8, uint16, int16, uint32, int32, "
           "uint64, int64, float, double.\n");
    printf("Creates a custom shared memory file <stream>.mon.shm for basic stats "
           "(Flux, Time) and Histogram.\n");
    return RETURN_SUCCESS;
}

static errno_t help_function()
{
    return stream_monitor_help();
}

// ----------------------------------------------------------------------------
// Shared Memory Helper Functions
// ----------------------------------------------------------------------------

STREAM_MON_STRUCT *stream_monitor_connect(const char *streamname, int create)
{
    char               shmname[STRINGMAXLEN_FULLFILENAME];
    int                fd;
    STREAM_MON_STRUCT *smon = NULL;

    snprintf(shmname, sizeof(shmname), "%s/%s.mon.shm", data.shmdir, streamname);

    int flags = O_RDWR;
    if (create)
    {
        flags |= O_CREAT;
    }

    fd = open(shmname, flags, 0666);
    if (fd == -1)
    {
        if (create)
        {
            perror("Error opening/creating monitor SHM file");
        }
        return NULL;
    }

    if (create)
    {
        if (ftruncate(fd, sizeof(STREAM_MON_STRUCT)) == -1)
        {
            perror("Error truncating monitor SHM file");
            close(fd);
            return NULL;
        }
    }

    smon = (STREAM_MON_STRUCT *) mmap(NULL, sizeof(STREAM_MON_STRUCT), PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0);
    close(fd);

    if (smon == MAP_FAILED)
    {
        perror("Error mapping monitor SHM file");
        return NULL;
    }

    if (create)
    {
        smon->size       = STREAM_MON_MAX_SAMPLES;
        smon->cnt        = 0;
        smon->cindex     = 0;
        smon->hist_nbins = STREAM_MON_MAX_HIST_BINS;
        memset(smon->flux, 0, sizeof(smon->flux));
        memset(smon->time, 0, sizeof(smon->time));
        memset(smon->hist_min_buf, 0, sizeof(smon->hist_min_buf));
        memset(smon->hist_max_buf, 0, sizeof(smon->hist_max_buf));
        memset(smon->hist_counts, 0, sizeof(smon->hist_counts));
    }

    return smon;
}

void stream_monitor_detach(STREAM_MON_STRUCT *smon)
{
    if (smon)
    {
        munmap(smon, sizeof(STREAM_MON_STRUCT));
    }
}

// ----------------------------------------------------------------------------
// Histogram Logic
// ----------------------------------------------------------------------------

// Binary search for bin index
static inline int get_bin_index(float val, float min_val, float max_val, int nbins)
{
    if (val < min_val)
    {
        return 0; // Underflow
    }
    if (val >= max_val)
    {
        return nbins - 1; // Overflow
    }

    float step = (max_val - min_val) / nbins;
    int   idx  = (int) ((val - min_val) / step);
    if (idx < 0)
    {
        idx = 0;
    }
    if (idx >= nbins)
    {
        idx = nbins - 1;
    }
    return idx;
}

// ----------------------------------------------------------------------------
// Compute Function Macros
// ----------------------------------------------------------------------------

#define ACCUMULATE_AND_HIST(CTYPE)                                                     \
    {                                                                                  \
        CTYPE    *ptr      = (CTYPE *) inimg.im->array.raw;                            \
        uint32_t *hist_ptr = smon->hist_counts[mon_idx];                               \
        int       nbins    = smon->hist_nbins;                                         \
                                                                                       \
        for (uint64_t i = 0; i < xysize; i++)                                          \
        {                                                                              \
            double val_d = (double) ptr[i];                                            \
            float  val_f = (float) val_d;                                              \
                                                                                       \
            /* Statistics */                                                           \
            frame_flux += val_d;                                                       \
            arraysum[0][i] += val_d;                                                   \
            arraysumsq[0][i] += val_d * val_d;                                         \
                                                                                       \
            /* Histogram */                                                            \
            int bin = get_bin_index(val_f, current_hist_min, current_hist_max, nbins); \
            hist_ptr[bin]++;                                                           \
                                                                                       \
            /* Track min/max for init */                                               \
            if (val_f < init_min)                                                      \
                init_min = val_f;                                                      \
            if (val_f > init_max)                                                      \
                init_max = val_f;                                                      \
        }                                                                              \
    }

// ----------------------------------------------------------------------------
// Stream Monitor Run Function
// ----------------------------------------------------------------------------

errno_t stream_monitor_run(const char *inimname_arg,
                           uint64_t    tbinflag_arg,
                           uint32_t    cbbuffersize_arg,
                           int         procinfo_flag,
                           int         fps_flag)
{
    DEBUG_TRACE_FSTART();

    // FORCE NULL first to ignore any previous/garbage state
    CLIcmddata.cmdsettings = NULL;

    // Setup CLIcmddata if running standalone
    CMDSETTINGS standalone_settings;
    long        local_loop_limit = -1; // Enforce infinite

    if (CLIcmddata.cmdsettings == NULL)
    {
        memset(&standalone_settings, 0, sizeof(CMDSETTINGS));
        standalone_settings.procinfo_loopcntMax = -1; // Default to infinite loop

        // Semaphore Triggering
        standalone_settings.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
        strncpy(standalone_settings.triggerstreamname, inimname_arg, STRINGMAXLEN_IMAGE_NAME - 1);
        standalone_settings.semindexrequested = -1; // Auto

        if (procinfo_flag)
        {
            standalone_settings.flags |= CLICMDFLAG_PROCINFO;
        }

        CLIcmddata.cmdsettings = &standalone_settings;
    }
    else
    {
        // If reusing settings, read loop limit
        local_loop_limit = CLIcmddata.cmdsettings->procinfo_loopcntMax;
    }

    // Connect to input image
    IMGID inimg = mkIMGID_from_name(inimname_arg);
    resolveIMGID(&inimg, ERRMODE_WARN);

    if (inimg.ID == -1)
    {
        // Not found, try to load from SHM
        read_sharedmem_image(inimname_arg);
        resolveIMGID(&inimg, ERRMODE_ABORT);
    }

    uint32_t xsize    = inimg.size[0];
    uint32_t ysize    = inimg.size[1];
    uint64_t xysize   = (uint64_t) xsize * ysize;
    uint8_t  datatype = inimg.md->datatype;
    int      typesize = ImageStreamIO_typesize(datatype);

    if (typesize <= 0)
    {
        PRINT_ERROR("Unknown or unsupported datatype size");
        return RETURN_FAILURE;
    }

    printf("Starting monitor loop for stream '%s'\n", inimg.name);
    printf("  Trigger mode: SEMAPHORE\n");
    printf("  PID: %d\n", getpid());
    fflush(stdout);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // FORCE infinite loop settings on processinfo if active
    if (processinfo)
    {
        processinfo->loopcntMax = -1;
    }

    // ------------------------------------------------------------------------
    // Circular Buffer Startup Cleanup
    // ------------------------------------------------------------------------
    {
        glob_t glob_result;
        char   pattern[1024];
        snprintf(pattern, sizeof(pattern), "%s/%s.cb*.im.shm", data.shmdir, inimg.name);
        if (glob(pattern, 0, NULL, &glob_result) == 0)
        {
            for (size_t i = 0; i < glob_result.gl_pathc; ++i)
            {
                char *filename = glob_result.gl_pathv[i];
                char *base     = strrchr(filename, '/');
                if (base)
                {
                    base++;
                }
                else
                {
                    base = filename;
                }
                char streamname[STRINGMAXLEN_IMGNAME];
                strncpy(streamname, base, STRINGMAXLEN_IMGNAME - 1);
                streamname[STRINGMAXLEN_IMGNAME - 1] = '\0';
                char *dot                            = strstr(streamname, ".im.shm");
                if (dot)
                {
                    *dot = '\0';
                }
                int   n_val  = -1;
                char *cb_ptr = strstr(streamname, ".cb");
                if (cb_ptr && sscanf(cb_ptr, ".cb%d", &n_val) == 1)
                {
                    if (n_val != cbbuffersize_arg)
                    {
                        delete_image_ID(streamname, DELETE_IMAGE_ERRMODE_WARNING);
                    }
                }
            }
            globfree(&glob_result);
        }
    }

    // ------------------------------------------------------------------------
    // Create Streams
    // ------------------------------------------------------------------------
    char cbname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(cbname, "%s.cb%d", inimg.name, cbbuffersize_arg);
    IMGID cbimg        = stream_connect_create_3D(cbname, xsize, ysize, cbbuffersize_arg, datatype);
    cbimg.md->ownerPID = getpid();
    ImageStreamIO_semflush(cbimg.im, -1);

    char cbtimename[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(cbtimename, "%s.cb%dtime", inimg.name, cbbuffersize_arg);
    IMGID cbtimg = stream_connect_create_2D(cbtimename, 2, cbbuffersize_arg, _DATATYPE_UINT64);
    cbtimg.md->ownerPID = getpid();
    ImageStreamIO_semflush(cbtimg.im, -1);

    // ------------------------------------------------------------------------
    // Monitor Shared Memory
    // ------------------------------------------------------------------------
    STREAM_MON_STRUCT *smon = stream_monitor_connect(inimg.name, 1);
    if (!smon)
    {
        return RETURN_FAILURE;
    }

    // ------------------------------------------------------------------------
    // Binning Setup
    // ------------------------------------------------------------------------
    int numbin = 0;
    int tbinarray[64];
    for (int i = 0; i < 64; i++)
    {
        if ((tbinflag_arg >> i) & 1)
        {
            tbinarray[numbin++] = (int) (1ULL << i);
        }
    }

    if (numbin == 0)
    {
        PRINT_ERROR("No bins selected in tbinflag");
        stream_monitor_detach(smon);
        return RETURN_FAILURE;
    }

    // Allocate accumulation arrays
    double **arraysum     = (double **) malloc(numbin * sizeof(double *));
    double **arraysumsq   = (double **) malloc(numbin * sizeof(double *));
    int     *bincounter   = (int *) calloc(numbin, sizeof(int));
    IMGID   *imgoutbin    = (IMGID *) malloc(numbin * sizeof(IMGID));
    IMGID   *imgoutbinrms = (IMGID *) malloc(numbin * sizeof(IMGID));

    for (int b = 0; b < numbin; b++)
    {
        arraysum[b]   = (double *) calloc(xysize, sizeof(double));
        arraysumsq[b] = (double *) calloc(xysize, sizeof(double));
        char imname[STRINGMAXLEN_IMGNAME];
        WRITE_IMAGENAME(imname, "%s.tbin%d", inimg.name, tbinarray[b]);
        imgoutbin[b]              = stream_connect_create_2D(imname, xsize, ysize, _DATATYPE_FLOAT);
        imgoutbin[b].md->ownerPID = getpid();
        ImageStreamIO_semflush(imgoutbin[b].im, -1);

        WRITE_IMAGENAME(imname, "%s.tbin%d.rms", inimg.name, tbinarray[b]);
        imgoutbinrms[b] = stream_connect_create_2D(imname, xsize, ysize, _DATATYPE_FLOAT);
        imgoutbinrms[b].md->ownerPID = getpid();
        ImageStreamIO_semflush(imgoutbinrms[b].im, -1);
    }

    // ------------------------------------------------------------------------
    // Main Loop
    // ------------------------------------------------------------------------
    uint64_t loopcnt        = 0;
    int      hist_init_done = 0;

    // Histogram State
    float current_hist_min = 0.0;
    float current_hist_max = 1.0;

    while (processloopOK)
    {
        if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
        {
            processloopOK = processinfo_loopstep(processinfo);
            if (processloopOK)
            {
                processinfo_waitoninputstream(processinfo);
                if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT &&
                    processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE)
                {
                    continue;
                }
                processinfo_exec_start(processinfo);
            }
        }
        else
        {
            if (local_loop_limit != -1 && local_loop_limit != 0 && loopcnt >= local_loop_limit)
            {
                processloopOK = 0;
            }
            if (processloopOK)
            {
                if (inimg.im->semptr)
                {
                    ImageStreamIO_semwait(inimg.im, 0);
                }
                else
                {
                    usleep(1000);
                }
            }
        }

        if (!processloopOK)
        {
            break;
        }

        // --------------------------------------------------------------------
        // Update Circular Buffer
        // --------------------------------------------------------------------
        uint64_t cb_idx    = loopcnt % cbbuffersize_arg;
        char    *cbptr_raw = (char *) cbimg.im->array.raw + (cb_idx * xysize * typesize);
        memcpy(cbptr_raw, inimg.im->array.raw, xysize * typesize);
        cbimg.md->cnt1  = cb_idx;
        cbimg.md->write = 1;
        processinfo_update_output_stream(processinfo, cbimg.ID);

        // --------------------------------------------------------------------
        // Update Timing
        // --------------------------------------------------------------------
        struct timespec tnow   = inimg.md->atime;
        uint64_t       *cbtptr = cbtimg.im->array.UI64;
        cbtptr[cb_idx * 2 + 0] = (uint64_t) tnow.tv_sec;
        cbtptr[cb_idx * 2 + 1] = (uint64_t) tnow.tv_nsec;
        cbtimg.md->cnt1        = cb_idx;
        cbtimg.md->write       = 1;
        processinfo_update_output_stream(processinfo, cbtimg.ID);

        // --------------------------------------------------------------------
        // Histogram / Flux / Binning
        // --------------------------------------------------------------------
        double   frame_flux = 0.0;
        uint32_t mon_idx    = smon->cnt % STREAM_MON_MAX_SAMPLES;

        memset(smon->hist_counts[mon_idx], 0, sizeof(uint32_t) * STREAM_MON_MAX_HIST_BINS);

        float init_min = FLT_MAX;
        float init_max = -FLT_MAX;

        // Record the thresholds used for this frame!
        smon->hist_min_buf[mon_idx] = current_hist_min;
        smon->hist_max_buf[mon_idx] = current_hist_max;

        switch (datatype)
        {
        case _DATATYPE_UINT8:
            ACCUMULATE_AND_HIST(uint8_t);
            break;
        case _DATATYPE_INT8:
            ACCUMULATE_AND_HIST(int8_t);
            break;
        case _DATATYPE_UINT16:
            ACCUMULATE_AND_HIST(uint16_t);
            break;
        case _DATATYPE_INT16:
            ACCUMULATE_AND_HIST(int16_t);
            break;
        case _DATATYPE_UINT32:
            ACCUMULATE_AND_HIST(uint32_t);
            break;
        case _DATATYPE_INT32:
            ACCUMULATE_AND_HIST(int32_t);
            break;
        case _DATATYPE_UINT64:
            ACCUMULATE_AND_HIST(uint64_t);
            break;
        case _DATATYPE_INT64:
            ACCUMULATE_AND_HIST(int64_t);
            break;
        case _DATATYPE_FLOAT:
            ACCUMULATE_AND_HIST(float);
            break;
        case _DATATYPE_DOUBLE:
            ACCUMULATE_AND_HIST(double);
            break;
        }

        if (!hist_init_done)
        {
            current_hist_min = init_min;
            current_hist_max = init_max;
            if (current_hist_max <= current_hist_min)
            {
                current_hist_max = current_hist_min + 1.0;
            }
            hist_init_done = 1;
        }
        else
        {
            // ----------------------------------------------------------------
            // Adjust Histogram Thresholds for NEXT frame
            // ----------------------------------------------------------------
            uint32_t c0    = smon->hist_counts[mon_idx][0];
            uint32_t cN    = smon->hist_counts[mon_idx][smon->hist_nbins - 1];
            uint32_t limit = xysize / 100; // 1%

            float range = current_hist_max - current_hist_min;
            if (range <= 0)
            {
                range = 1.0;
            }

            if (c0 > limit)
            {
                current_hist_min -= range * 0.1;
            }
            else if (c0 == 0)
            {
                current_hist_min += range * 0.01;
            }

            if (cN > limit)
            {
                current_hist_max += range * 0.1;
            }
            else if (cN == 0)
            {
                current_hist_max -= range * 0.01;
            }

            if (current_hist_max <= current_hist_min)
            {
                current_hist_max = current_hist_min + 1e-6;
            }
        }

        bincounter[0]++;

        // Cascade bins
        for (int b = 0; b < numbin; b++)
        {
            if (bincounter[b] == tbinarray[b])
            {
                if (b + 1 < numbin)
                {
                    for (uint64_t pixi = 0; pixi < xysize; pixi++)
                    {
                        arraysum[b + 1][pixi] += arraysum[b][pixi];
                        arraysumsq[b + 1][pixi] += arraysumsq[b][pixi];
                    }
                    bincounter[b + 1] += bincounter[b];
                }

                float *outptr    = imgoutbin[b].im->array.F;
                float *outrmsptr = imgoutbinrms[b].im->array.F;
                double invcount  = 1.0 / bincounter[b];

                imgoutbin[b].md->write    = 1;
                imgoutbinrms[b].md->write = 1;

                for (uint64_t pixi = 0; pixi < xysize; pixi++)
                {
                    double avg      = arraysum[b][pixi] * invcount;
                    double sqavg    = arraysumsq[b][pixi] * invcount;
                    double var      = sqavg - avg * avg;
                    outptr[pixi]    = (float) avg;
                    outrmsptr[pixi] = (float) (var > 0 ? sqrt(var) : 0);
                }
                processinfo_update_output_stream(processinfo, imgoutbin[b].ID);
                processinfo_update_output_stream(processinfo, imgoutbinrms[b].ID);

                for (uint64_t pixi = 0; pixi < xysize; pixi++)
                {
                    arraysum[b][pixi]   = 0.0;
                    arraysumsq[b][pixi] = 0.0;
                }
                bincounter[b] = 0;
            }
        }

        // Update Monitor SHM
        smon->flux[mon_idx] = frame_flux;
        smon->time[mon_idx] = tnow;
        smon->cindex        = mon_idx;
        smon->cnt++;

        // End of loop updates
        if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
        {
            if (processinfo != NULL)
            {
                if (processinfo_compute_status(processinfo) == 0)
                {
                    processloopOK = 0;
                }
                processinfo_exec_end(processinfo);
            }
        }
        else
        {
            if (data.signal_INT)
            {
                processloopOK = 0;
            }
        }

        loopcnt++;
    }

    // Clean up processinfo if it was created
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        if (processinfo)
        {
            processinfo_cleanExit(processinfo);
        }
    }

    if (CLIcmddata.cmdsettings == &standalone_settings)
    {
        CLIcmddata.cmdsettings = NULL;
    }

    // Cleanup
    stream_monitor_detach(smon);
    for (int b = 0; b < numbin; b++)
    {
        free(arraysum[b]);
        free(arraysumsq[b]);
    }
    free(arraysum);
    free(arraysumsq);
    free(bincounter);
    free(imgoutbin);
    free(imgoutbinrms);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    // Wrapper for CLI mode
    return stream_monitor_run(inimname, *tbinflag, *cbbuffersize, 0, 0);
}

INSERT_STD_FPSCLIfunctions_DynamicSize

    // Register function in CLI
    errno_t CLIADDCMD_info__stream_monproc()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
