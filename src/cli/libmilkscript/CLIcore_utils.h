/**
 * @file    CLIcore_utils.h
 * @brief   Util functions and macros for coding convenience
 *
 */

#ifndef CLICORE_UTILS_H
#define CLICORE_UTILS_H

#ifdef __cplusplus
typedef const char *CONST_WORD;
#else
typedef const char *__restrict CONST_WORD;
#endif

#include <string.h>

#include "CLIcore.h"

#include "libfps/IMGID.h"
#include "COREMOD_memory/COREMOD_memory.h"

#define CLICMD_FIELDS_FPSPROC                                                                  \
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg, CLICMDFLAG_FPS | CLICMDFLAG_PROCINFO, \
        NULL, NULL, NULL
#define CLICMD_FIELDS_DEFAULTS \
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg, CLICMDFLAG_FPS, NULL, NULL, NULL
#define CLICMD_FIELDS_NOFPS __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg, 0, NULL, NULL, NULL

#define CLICMD_FIELDS_FPSPROC_W_ARG(farg)                                                      \
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg, CLICMDFLAG_FPS | CLICMDFLAG_PROCINFO, \
        NULL, NULL, NULL
#define CLICMD_FIELDS_DEFAULTS_W_ARG(farg) \
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg, CLICMDFLAG_FPS, NULL, NULL, NULL
#define CLICMD_FIELDS_NOFPS_W_ARG(farg) \
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg, 0, NULL, NULL, NULL


#define CLICMD_FIELDS_NOPARAM __FILE__, 0, NULL, CLICMDFLAG_FPS, NULL, NULL, NULL

// return codes for function CLI_checkarg_array
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


/** @brief Template for ouput image argument to CLI function
 *
 */
#define FARG_OUTIM2D(imkey)                                                   \
    FARG_OUTIM_NAME(imkey), FARG_OUTIM_XSIZE(imkey), FARG_OUTIM_YSIZE(imkey), \
        FARG_OUTIM_SHARED(imkey), FARG_OUTIM_NBKW(imkey), FARG_OUTIM_CBSIZE(imkey)


// connect to and/or create output 2D image/stream
//
#define FARG_OUTIM2DCREATE(imkey, img, data_type)                                          \
    IMGID img       = imgid_make_from_name(imkey.name);                                    \
    img.mdt->shared = *imkey.shared;                                                       \
    img.mdt->NBkw   = *imkey.NBkw;                                                         \
    img.mdt->CBsize = *imkey.CBsize;                                                       \
    if (*imkey.shared == 1)                                                                \
    {                                                                                      \
        img = stream_connect_create_2D(imkey.name, *imkey.xsize, *imkey.ysize, data_type); \
    }                                                                                      \
    else                                                                                   \
    {                                                                                      \
        img.mdt->naxis    = 2;                                                             \
        img.mdt->size[0]  = *imkey.xsize;                                                  \
        img.mdt->size[1]  = *imkey.ysize;                                                  \
        img.mdt->datatype = data_type;                                                     \
        createimagefromIMGID(&img);                                                        \
    }                                                                                      \
    imcreateIMGID(&img);


// binding between variables and function args/params
#define STD_FARG_LINKfunction                                                      \
    for (int argi = 0; argi < (int) (sizeof(farg) / sizeof(CLICMDARGDEF)); argi++) \
    {                                                                              \
        long  fpsi           = -1;                                                 \
        void *ptr            = get_farg_ptr(farg[argi].fpstag, &fpsi);             \
        *(farg[argi].valptr) = ptr;                                                \
        if (farg[argi].indexptr != NULL)                                           \
        {                                                                          \
            *(farg[argi].indexptr) = fpsi;                                         \
        }                                                                          \
    }


/** @brief Standard Function call wrapper
 *
 * CLI argument(s) is(are) parsed and checked with CLI_checkarray(), then
 * passed to the compute function call.
 *
 * Custom code may be added for more complex processing of function arguments.
 *
 * If CLI call arguments check out, go ahead with computation.
 * Arguments not contained in CLI call line are extracted from the
 * command argument list
 */
#define INSERT_STD_CLIfunction                                       \
    static errno_t CLIfunction(void)                                 \
    {                                                                \
        errno_t retval = CLI_checkarg_array(farg, CLIcmddata.nbarg); \
        if (retval == RETURN_SUCCESS)                                \
        {                                                            \
            STD_FARG_LINKfunction return compute_function();         \
        }                                                            \
        if (retval == RETURN_CLICHECKARGARRAY_HELP)                  \
        {                                                            \
            return RETURN_SUCCESS;                                   \
        }                                                            \
        if (retval == RETURN_CLICHECKARGARRAY_FUNCPARAMSET)          \
        {                                                            \
            return RETURN_SUCCESS;                                   \
        }                                                            \
        return retval;                                               \
    }


#define INSERT_STD_PROCINFO_COMPUTEFUNC_INIT                                                      \
    int          processloopOK = 1;                                                               \
    PROCESSINFO *processinfo   = NULL;                                                            \
    /* set default timeout to 2 sec */                                                            \
    CLIcmddata.cmdsettings->triggertimeout.tv_sec  = 2;                                           \
    CLIcmddata.cmdsettings->triggertimeout.tv_nsec = 0;                                           \
    if (dcfpsptr != NULL)                                                                         \
    { /* If FPS mode, then FPS settings override defaults*/                                       \
        /* dcfpsptr->cmset entries are read by fps_connect */                                     \
        /*CLIcmddata.cmdsettings->flags = dcfpsptr->cmdset.flags;*/                               \
        CLIcmddata.cmdsettings->flags |= (dcfpsptr->cmdset.flags & CLICMDFLAG_PROCINFO);          \
        CLIcmddata.cmdsettings->RT_priority         = dcfpsptr->cmdset.RT_priority;               \
        CLIcmddata.cmdsettings->procinfo_loopcntMax = dcfpsptr->cmdset.procinfo_loopcntMax;       \
        CLIcmddata.cmdsettings->triggermode         = dcfpsptr->cmdset.triggermode;               \
        strncpy(CLIcmddata.cmdsettings->triggerstreamname, dcfpsptr->cmdset.triggerstreamname,    \
                STRINGMAXLEN_IMAGE_NAME - 1);                                                     \
        CLIcmddata.cmdsettings->semindexrequested      = dcfpsptr->cmdset.semindexrequested;      \
        CLIcmddata.cmdsettings->triggerdelay.tv_sec    = dcfpsptr->cmdset.triggerdelay.tv_sec;    \
        CLIcmddata.cmdsettings->triggerdelay.tv_nsec   = dcfpsptr->cmdset.triggerdelay.tv_nsec;   \
        CLIcmddata.cmdsettings->triggertimeout.tv_sec  = dcfpsptr->cmdset.triggertimeout.tv_sec;  \
        CLIcmddata.cmdsettings->triggertimeout.tv_nsec = dcfpsptr->cmdset.triggertimeout.tv_nsec; \
    }                                                                                             \
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                                      \
    {                                                                                             \
        char pinfodescr[200];                                                                     \
        int  slen = snprintf(pinfodescr, 200, "function %.10s", CLIcmddata.key);                  \
        if (slen < 1)                                                                             \
        {                                                                                         \
            PRINT_ERROR("snprintf wrote <1 char");                                                \
            abort();                                                                              \
        }                                                                                         \
        if (slen >= 200)                                                                          \
        {                                                                                         \
            PRINT_ERROR("snprintf string truncation");                                            \
            abort();                                                                              \
        }                                                                                         \
        if (dcfpsptr != NULL)                                                                     \
        {                                                                                         \
            /* dcfpsname may be empty when called via fps_generic_run     */                      \
            /* which bypasses CLI arg parsing. Fall back to FPS SHM name. */                      \
            const char *_piname_ = (dcfpsname[0] != '\0') ? dcfpsname : dcfpsptr->md->name;       \
            processinfo          = processinfo_setup((char *) _piname_, pinfodescr, "startup",    \
                                                     __FUNCTION__, __FILE__, __LINE__);           \
            fps_to_processinfo(dcfpsptr, processinfo);                                            \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            processinfo = processinfo_setup(CLIcmddata.key, pinfodescr, "startup", __FUNCTION__,  \
                                            __FILE__, __LINE__);                                  \
        }                                                                                         \
        DEBUG_TRACEPOINT("setting processinfo parameters");                                       \
        processinfo->loopcntMax      = CLIcmddata.cmdsettings->procinfo_loopcntMax;               \
        processinfo->triggerstreamID = -2;                                                        \
        processinfo->triggermode     = CLIcmddata.cmdsettings->triggermode;                       \
        strncpy(processinfo->triggerstreamname, CLIcmddata.cmdsettings->triggerstreamname,        \
                STRINGMAXLEN_IMAGE_NAME - 1);                                                     \
        processinfo->triggerdelay    = CLIcmddata.cmdsettings->triggerdelay;                      \
        processinfo->triggertimeout  = CLIcmddata.cmdsettings->triggertimeout;                    \
        processinfo->triggerstreamID = image_ID(processinfo->triggerstreamname, dcimg, dcnimg);   \
        DEBUG_TRACEPOINT("triggerstreamID = %ld", processinfo->triggerstreamID);                  \
        FUNC_CHECK_RETURN(processinfo_waitoninputstream_init(                                     \
            processinfo,                                                                          \
            (processinfo->triggerstreamID > -1 ? &dcimg[processinfo->triggerstreamID] : NULL),    \
            CLIcmddata.cmdsettings->triggermode, CLIcmddata.cmdsettings->semindexrequested));     \
        DEBUG_TRACEPOINT("setting RT priority to %d", CLIcmddata.cmdsettings->RT_priority);       \
        processinfo->RT_priority   = CLIcmddata.cmdsettings->RT_priority;                         \
        processinfo->CPUmask       = CLIcmddata.cmdsettings->CPUmask;                             \
        processinfo->MeasureTiming = CLIcmddata.cmdsettings->procinfo_MeasureTiming;              \
        DEBUG_TRACEPOINT("loopstart");                                                            \
        processinfo_loopstart(processinfo);                                                       \
    }


#define INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART                                   \
    while (processloopOK == 1)                                                      \
    {                                                                               \
        if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                    \
        {                                                                           \
            DEBUG_TRACEPOINT("loopstep");                                           \
            processloopOK = processinfo_loopstep(processinfo);                      \
            DEBUG_TRACEPOINT("waitoninputstream");                                  \
            processinfo_waitoninputstream(processinfo);                             \
            if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT && \
                processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE)      \
            {                                                                       \
                /* Don't execute loop at all upon semaphore timeout */              \
                /* Except if the trigger is SEMAPHORE_PROP_TIMEOUTS */              \
                /* in which case we avoid this block and keep going */              \
                continue;                                                           \
            }                                                                       \
            DEBUG_TRACEPOINT("exec_start");                                         \
            processinfo_exec_start(processinfo);                                    \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            processloopOK = 0;                                                      \
        }                                                                           \
        int processcompstatus = 1;                                                  \
        if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                    \
        {                                                                           \
            processcompstatus = processinfo_compute_status(processinfo);            \
        }                                                                           \
        if (processcompstatus == 1)                                                 \
        {
#define INSERT_STD_PROCINFO_COMPUTEFUNC_START \
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT      \
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART


#define INSERT_STD_PROCINFO_COMPUTEFUNC_END                                              \
    }                                                                                    \
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                             \
    {                                                                                    \
        if (processinfo != NULL)                                                         \
        {                                                                                \
            if (dcfpsptr != NULL)                                                        \
            {                                                                            \
                if (dcfpsptr->cmdset.triggermodeptr != NULL)                             \
                {                                                                        \
                    processinfo->triggermode = *dcfpsptr->cmdset.triggermodeptr;         \
                }                                                                        \
                if (dcfpsptr->cmdset.procinfo_loopcntMax_ptr != NULL)                    \
                {                                                                        \
                    processinfo->loopcntMax = *dcfpsptr->cmdset.procinfo_loopcntMax_ptr; \
                }                                                                        \
                if (dcfpsptr->cmdset.triggerdelayptr != NULL)                            \
                {                                                                        \
                    processinfo->triggerdelay = dcfpsptr->cmdset.triggerdelayptr[0];     \
                }                                                                        \
                if (dcfpsptr->cmdset.triggertimeoutptr != NULL)                          \
                {                                                                        \
                    processinfo->triggertimeout = dcfpsptr->cmdset.triggertimeoutptr[0]; \
                }                                                                        \
            }                                                                            \
        }                                                                                \
        if (processinfo != NULL)                                                         \
        {                                                                                \
            processinfo_exec_end(processinfo);                                           \
        }                                                                                \
    }                                                                                    \
    }                                                                                    \
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)                             \
    {                                                                                    \
        processinfo_cleanExit(processinfo);                                              \
    }


#define INSERT_STD_CLIREGISTERFUNC                                        \
    {                                                                     \
        if (getenv("MILK_FPSPROCINFO"))                                   \
        {                                                                 \
            CLIcmddata.flags |= CLICMDFLAG_PROCINFO;                      \
        }                                                                 \
        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction); \
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;             \
    }


#endif
