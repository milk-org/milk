// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_procinfo_macros.h
 * @brief   Processinfo loop bodies and image-argument helper macros
 *
 * The _MILK_PROCINFO_*_IMPL bodies are instantiated by MILK_PROCINFO_*(self) in milk_runnable.h.
 */

#ifndef FPS_PROCINFO_MACROS_H
#define FPS_PROCINFO_MACROS_H

#include "milk_types.h"
#include "fps.h"
#include "fps_unit.h"

imageID image_ID(const char *apzname, IMAGE *dcimage_array, long dcnbimg);

typedef const char *__restrict CONST_WORD;

#define RETURN_CLICHECKARGARRAY_SUCCESS 0
#define RETURN_CLICHECKARGARRAY_FAILURE 1
#define RETURN_CLICHECKARGARRAY_FUNCPARAMSET 2
#define RETURN_CLICHECKARGARRAY_HELP 3

#define HELPDETAILSSTRINGSTART "------- DETAILS ------"
#define HELPDETAILSSTRINGEND "-------- END ---------"


typedef struct
{
    char *name;
} LOCVAR_INIMG;

#define FARG_INPUTIM(imkey)                      \
    { CLIARG_STR,                                \
      "." #imkey ".name",                        \
      "input image",                             \
      #imkey,                                    \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
      (void **) &imkey.name }


typedef struct
{
    char     *name;
    uint32_t *xsize;
    uint32_t *ysize;
    uint32_t *datatype;
    uint32_t *shared;
    uint32_t *NBkw;
    uint32_t *CBsize;
} LOCVAR_OUTIMG2D;

#define FARG_OUTIM_NAME(imkey)                   \
    { CLIARG_STR,                                \
      "." #imkey ".name",                        \
      "output image",                            \
      #imkey,                                    \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
      (void **) &imkey.name,                     \
      NULL }

#define FARG_OUTIM_XSIZE(imkey)                  \
    { CLIARG_UINT32,                             \
      "." #imkey ".xsize",                       \
      "x size",                                  \
      "256",                                     \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
      (void **) &imkey.xsize,                    \
      NULL }

#define FARG_OUTIM_YSIZE(imkey)                  \
    { CLIARG_UINT32,                             \
      "." #imkey ".ysize",                       \
      "y size",                                  \
      "256",                                     \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
      (void **) &imkey.ysize,                    \
      NULL }

#define FARG_OUTIM_SHARED(imkey)                                         \
    { CLIARG_UINT32,        "." #imkey ".shared",    "shared flag", "0", \
      FPFLAG_DEFAULT_INPUT, (void **) &imkey.shared, NULL }

#define FARG_OUTIM_NBKW(imkey)                                    \
    { CLIARG_UINT32, "." #imkey ".NBkw",   "number keywords",     \
      "10",          FPFLAG_DEFAULT_INPUT, (void **) &imkey.NBkw, \
      NULL }

#define FARG_OUTIM_CBSIZE(imkey)                                    \
    { CLIARG_UINT32, "." #imkey ".CBsize", "circ buffer size",      \
      "0",           FPFLAG_DEFAULT_INPUT, (void **) &imkey.CBsize, \
      NULL }

#define FARG_OUTIM2D(imkey)                                                   \
    FARG_OUTIM_NAME(imkey), FARG_OUTIM_XSIZE(imkey), FARG_OUTIM_YSIZE(imkey), \
        FARG_OUTIM_SHARED(imkey), FARG_OUTIM_NBKW(imkey), FARG_OUTIM_CBSIZE(imkey)

#define FARG_OUTIM2DCREATE(imkey, img, dt)                                          \
    IMGID img       = imgid_make_from_name(imkey.name);                             \
    img.mdt->shared = *imkey.shared;                                                \
    img.mdt->NBkw   = *imkey.NBkw;                                                  \
    img.mdt->CBsize = *imkey.CBsize;                                                \
    if (*imkey.shared == 1)                                                         \
    {                                                                               \
        img = stream_connect_create_2D(imkey.name, *imkey.xsize, *imkey.ysize, dt); \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        img.mdt->naxis    = 2;                                                      \
        img.mdt->size[0]  = *imkey.xsize;                                           \
        img.mdt->size[1]  = *imkey.ysize;                                           \
        img.mdt->datatype = dt;                                                     \
        createimagefromIMGID(&img);                                                 \
    }                                                                               \
    imcreateIMGID(&img);


/* The body macros take the CLICMDDATA lvalue as parameter (named CLIcmddata so that the bodies stay
 * unchanged); MILK_PROCINFO_*(self) in milk_runnable.h pass the runnable's own. */
#define _MILK_PROCINFO_INIT_IMPL(CLIcmddata)                                                       \
    int          processloopOK = 1;                                                                \
    PROCESSINFO *processinfo   = NULL;                                                             \
    /* set default timeout to 2 sec */                                                             \
    CLIcmddata.cmdsettings->triggertimeout.tv_sec  = 2;                                            \
    CLIcmddata.cmdsettings->triggertimeout.tv_nsec = 0;                                            \
    if (dcfpsptr != NULL)                                                                          \
    {                                                                                              \
        fpsresync_fps_to_modvar(dcfpsptr, dcfpsbindings, dcfpsnbindings, 1);                       \
        CLIcmddata.cmdsettings->flags |= (dcfpsptr->cmdset.flags & CLICMDFLAG_PROCINFO);           \
        CLIcmddata.cmdsettings->RT_priority         = dcfpsptr->cmdset.RT_priority;                \
        CLIcmddata.cmdsettings->procinfo_loopcntMax = dcfpsptr->cmdset.procinfo_loopcntMax;        \
        CLIcmddata.cmdsettings->triggermode         = dcfpsptr->cmdset.triggermode;                \
        strncpy(CLIcmddata.cmdsettings->triggerstreamname, dcfpsptr->cmdset.triggerstreamname,     \
                STRINGMAXLEN_IMAGE_NAME - 1);                                                      \
        CLIcmddata.cmdsettings->semindexrequested    = dcfpsptr->cmdset.semindexrequested;         \
        CLIcmddata.cmdsettings->triggerdelay.tv_sec  = dcfpsptr->cmdset.triggerdelay.tv_sec;       \
        CLIcmddata.cmdsettings->triggerdelay.tv_nsec = dcfpsptr->cmdset.triggerdelay.tv_nsec;      \
        if (dcfpsptr->cmdset.triggertimeout.tv_sec == 0 &&                                         \
            dcfpsptr->cmdset.triggertimeout.tv_nsec == 0)                                          \
        {                                                                                          \
            CLIcmddata.cmdsettings->triggertimeout.tv_sec  = 2;                                    \
            CLIcmddata.cmdsettings->triggertimeout.tv_nsec = 0;                                    \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            CLIcmddata.cmdsettings->triggertimeout.tv_sec =                                        \
                dcfpsptr->cmdset.triggertimeout.tv_sec;                                            \
            CLIcmddata.cmdsettings->triggertimeout.tv_nsec =                                       \
                dcfpsptr->cmdset.triggertimeout.tv_nsec;                                           \
        }                                                                                          \
    }                                                                                              \
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                                       \
    {                                                                                              \
        char pinfodescr[200];                                                                      \
        int  slen = snprintf(pinfodescr, 200, "function %.10s", CLIcmddata.key);                   \
        if (slen < 1 || slen >= 200)                                                               \
        {                                                                                          \
            PRINT_ERROR("snprintf write error (<1 char or string truncation, slen=%d)", slen);     \
            abort();                                                                               \
        }                                                                                          \
        if (dcfpsptr != NULL)                                                                      \
        {                                                                                          \
            const char *_piname = (dcfpsname[0] != '\0') ? dcfpsname : dcfpsptr->md->name;         \
            processinfo = processinfo_setup((char *) _piname, pinfodescr, "startup", __FUNCTION__, \
                                            __FILE__, __LINE__);                                   \
            fps_to_processinfo(dcfpsptr, processinfo);                                             \
            /* Sync CLIcmddata.cmdsettings from processinfo to keep them in sync */                \
            CLIcmddata.cmdsettings->procinfo_loopcntMax    = processinfo->loopcntMax;              \
            CLIcmddata.cmdsettings->triggermode            = processinfo->triggermode;             \
            CLIcmddata.cmdsettings->triggerdelay           = processinfo->triggerdelay;            \
            CLIcmddata.cmdsettings->triggertimeout         = processinfo->triggertimeout;          \
            CLIcmddata.cmdsettings->RT_priority            = processinfo->RT_priority;             \
            CLIcmddata.cmdsettings->procinfo_MeasureTiming = processinfo->MeasureTiming;           \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            processinfo = processinfo_setup(CLIcmddata.key, pinfodescr, "startup", __FUNCTION__,   \
                                            __FILE__, __LINE__);                                   \
        }                                                                                          \
        processinfo->loopcntMax      = CLIcmddata.cmdsettings->procinfo_loopcntMax;                \
        processinfo->triggerstreamID = -2;                                                         \
        processinfo->triggermode     = CLIcmddata.cmdsettings->triggermode;                        \
        strncpy(processinfo->triggerstreamname, CLIcmddata.cmdsettings->triggerstreamname,         \
                STRINGMAXLEN_IMAGE_NAME - 1);                                                      \
        processinfo->triggerdelay    = CLIcmddata.cmdsettings->triggerdelay;                       \
        processinfo->triggertimeout  = CLIcmddata.cmdsettings->triggertimeout;                     \
        processinfo->triggerstreamID = image_ID(processinfo->triggerstreamname, dcimg, dcnimg);    \
        FUNC_CHECK_RETURN(processinfo_waitoninputstream_init(                                      \
            processinfo,                                                                           \
            (processinfo->triggerstreamID > -1 ? &dcimg[processinfo->triggerstreamID] : NULL),     \
            CLIcmddata.cmdsettings->triggermode, CLIcmddata.cmdsettings->semindexrequested));      \
        processinfo->RT_priority   = CLIcmddata.cmdsettings->RT_priority;                          \
        processinfo->CPUmask       = CLIcmddata.cmdsettings->CPUmask;                              \
        processinfo->MeasureTiming = CLIcmddata.cmdsettings->procinfo_MeasureTiming;               \
        processinfo_loopstart(processinfo);                                                        \
        processinfo_CatchSignals(); /* restored by _MILK_PROCINFO_END_IMPL */                      \
    }                                                                                              \
    if (dcfpsptr != NULL)                                                                          \
    {                                                                                              \
        /* proc.<name>.<pid>.shm exists past this point */                                         \
        dcfpsptr->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP;                          \
        fps_unit_notify_ready(); /* a unit's starter returns now (U4) */                           \
    }

#define _MILK_PROCINFO_LOOPSTART_IMPL(CLIcmddata)                                    \
    while (processloopOK == 1)                                                       \
    {                                                                                \
        if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                     \
        {                                                                            \
            DEBUG_TRACEPOINT("loopstep");                                            \
            processloopOK = processinfo_loopstep(processinfo);                       \
            fps_unit_tick(processinfo->loopcnt);                                     \
            /* a stop request ends the loop before the wait and without computing */ \
            if (UNLIKELY(processinfo_exit_requested(processinfo)))                   \
            {                                                                        \
                processloopOK = 0;                                                   \
                continue;                                                            \
            }                                                                        \
            DEBUG_TRACEPOINT("waitoninputstream");                                   \
            processinfo_waitoninputstream(processinfo);                              \
            if (UNLIKELY(processinfo_exit_requested(processinfo)))                   \
            {                                                                        \
                processloopOK = 0;                                                   \
                continue;                                                            \
            }                                                                        \
            if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT &&  \
                processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE)       \
            {                                                                        \
                continue;                                                            \
            }                                                                        \
            DEBUG_TRACEPOINT("exec_start");                                          \
            processinfo_exec_start(processinfo);                                     \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            processloopOK = 0;                                                       \
        }                                                                            \
        int processcompstatus = 1;                                                   \
        if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                     \
        {                                                                            \
            processcompstatus = processinfo_compute_status(processinfo);             \
        }                                                                            \
        if (processcompstatus == 1)                                                  \
        {                                                                            \
            if (dcfpsptr != NULL)                                                    \
            {                                                                        \
                fpsresync_fps_to_modvar(dcfpsptr, dcfpsbindings, dcfpsnbindings, 0); \
            }


#define _MILK_PROCINFO_END_IMPL(CLIcmddata)                                                  \
    }                                                                                        \
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                                 \
    {                                                                                        \
        if (processinfo != NULL)                                                             \
        {                                                                                    \
            if (dcfpsptr != NULL)                                                            \
            {                                                                                \
                fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);            \
                if (dcfpsptr->cmdset.triggermodeptr != NULL)                                 \
                {                                                                            \
                    processinfo->triggermode = *dcfpsptr->cmdset.triggermodeptr;             \
                }                                                                            \
                if (dcfpsptr->cmdset.procinfo_loopcntMax_ptr != NULL)                        \
                {                                                                            \
                    processinfo->loopcntMax = *dcfpsptr->cmdset.procinfo_loopcntMax_ptr;     \
                }                                                                            \
                if (dcfpsptr->cmdset.triggerdelayptr != NULL)                                \
                {                                                                            \
                    processinfo->triggerdelay = dcfpsptr->cmdset.triggerdelayptr[0];         \
                }                                                                            \
                if (dcfpsptr->cmdset.triggertimeoutptr != NULL)                              \
                {                                                                            \
                    if (dcfpsptr->cmdset.triggertimeoutptr[0].tv_sec == 0 &&                 \
                        dcfpsptr->cmdset.triggertimeoutptr[0].tv_nsec == 0)                  \
                    {                                                                        \
                        processinfo->triggertimeout.tv_sec  = 2;                             \
                        processinfo->triggertimeout.tv_nsec = 0;                             \
                    }                                                                        \
                    else                                                                     \
                    {                                                                        \
                        processinfo->triggertimeout = dcfpsptr->cmdset.triggertimeoutptr[0]; \
                    }                                                                        \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        if (processinfo != NULL)                                                             \
        {                                                                                    \
            processinfo_exec_end(processinfo);                                               \
        }                                                                                    \
    }                                                                                        \
    }                                                                                        \
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                                 \
    {                                                                                        \
        processinfo_cleanExit(processinfo);                                                  \
        processinfo_RestoreSignals();                                                        \
    }                                                                                        \
    if (dcfpsptr != NULL)                                                                    \
    {                                                                                        \
        dcfpsptr->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP;                   \
    }

#endif /* FPS_PROCINFO_MACROS_H */
