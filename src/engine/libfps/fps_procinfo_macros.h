/**
 * @file    fps_procinfo_macros.h
 * @brief   Standard compute loop and CLI-like argument macros for standalone
 *
 * Provides INSERT_STD_PROCINFO_COMPUTEFUNC_* macros natively to the engine.
 */

#ifndef FPS_PROCINFO_MACROS_H
#define FPS_PROCINFO_MACROS_H

#include "milk_types.h"
#include "fps.h"

imageID image_ID(const char *apzname, IMAGE *dcimage_array, long dcnbimg);

#ifdef __cplusplus

typedef const char *CONST_WORD;
#else
typedef const char *__restrict CONST_WORD;
#endif

#define CLICMD_FIELDS_FPSPROC          \
    __FILE__,                          \
    sizeof(farg) / sizeof(CLICMDARGDEF), \
    farg,                              \
    CLICMDFLAG_FPS | CLICMDFLAG_PROCINFO, \
    NULL, NULL, NULL

#define CLICMD_FIELDS_DEFAULTS         \
    __FILE__,                          \
    sizeof(farg) / sizeof(CLICMDARGDEF), \
    farg, CLICMDFLAG_FPS,              \
    NULL, NULL, NULL

#define CLICMD_FIELDS_NOFPS            \
    __FILE__,                          \
    sizeof(farg) / sizeof(CLICMDARGDEF), \
    farg, 0, NULL, NULL, NULL

#define CLICMD_FIELDS_NOPARAM          \
    __FILE__, 0, NULL,                 \
    CLICMDFLAG_FPS, NULL, NULL, NULL

#define RETURN_CLICHECKARGARRAY_SUCCESS      0
#define RETURN_CLICHECKARGARRAY_FAILURE      1
#define RETURN_CLICHECKARGARRAY_FUNCPARAMSET 2
#define RETURN_CLICHECKARGARRAY_HELP         3

#define HELPDETAILSSTRINGSTART \
    "------- DETAILS ------"
#define HELPDETAILSSTRINGEND \
    "-------- END ---------"


typedef struct
{
    char *name;
} LOCVAR_INIMG;

#define FARG_INPUTIM(imkey) \
    {                       \
        CLIARG_STR,         \
        "." #imkey ".name", \
        "input image",      \
        #imkey,             \
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
        (void **) &imkey.name  \
    }


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

#define FARG_OUTIM_NAME(imkey)     \
    {CLIARG_STR,                   \
     "." #imkey ".name",           \
     "output image",               \
     #imkey,                       \
     (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
     (void **) &imkey.name,        \
     NULL}

#define FARG_OUTIM_XSIZE(imkey)    \
    {CLIARG_UINT32,                \
     "." #imkey ".xsize",          \
     "x size", "256",              \
     (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
     (void **) &imkey.xsize,       \
     NULL}

#define FARG_OUTIM_YSIZE(imkey)    \
    {CLIARG_UINT32,                \
     "." #imkey ".ysize",          \
     "y size", "256",              \
     (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
     (void **) &imkey.ysize,       \
     NULL}

#define FARG_OUTIM_SHARED(imkey)   \
    {CLIARG_UINT32,                \
     "." #imkey ".shared",         \
     "shared flag", "0",           \
     FPFLAG_DEFAULT_INPUT,         \
     (void **) &imkey.shared,      \
     NULL}

#define FARG_OUTIM_NBKW(imkey)     \
    {CLIARG_UINT32,                \
     "." #imkey ".NBkw",           \
     "number keywords", "10",      \
     FPFLAG_DEFAULT_INPUT,         \
     (void **) &imkey.NBkw,        \
     NULL}

#define FARG_OUTIM_CBSIZE(imkey)   \
    {CLIARG_UINT32,                \
     "." #imkey ".CBsize",         \
     "circ buffer size", "0",      \
     FPFLAG_DEFAULT_INPUT,         \
     (void **) &imkey.CBsize,      \
     NULL}

#define FARG_OUTIM2D(imkey)        \
    FARG_OUTIM_NAME(imkey),        \
    FARG_OUTIM_XSIZE(imkey),       \
    FARG_OUTIM_YSIZE(imkey),       \
    FARG_OUTIM_SHARED(imkey),      \
    FARG_OUTIM_NBKW(imkey),        \
    FARG_OUTIM_CBSIZE(imkey)

#define FARG_OUTIM2DCREATE(imkey, img, dt) \
    IMGID img = imgid_make_from_name(    \
        imkey.name);                     \
    img.mdt->shared = *imkey.shared;     \
    img.mdt->NBkw   = *imkey.NBkw;      \
    img.mdt->CBsize = *imkey.CBsize;     \
    if (*imkey.shared == 1) {            \
        img = stream_connect_create_2D(  \
            imkey.name,                  \
            *imkey.xsize,                \
            *imkey.ysize, dt);           \
    } else {                             \
        img.mdt->naxis   = 2;            \
        img.mdt->size[0] = *imkey.xsize; \
        img.mdt->size[1] = *imkey.ysize; \
        img.mdt->datatype = dt;          \
        createimagefromIMGID(&img);       \
    }                                    \
    imcreateIMGID(&img);


#define STD_FARG_LINKfunction            \
    for (int argi = 0;                   \
         argi < (int)(sizeof(farg) /     \
             sizeof(CLICMDARGDEF));      \
         argi++)                         \
    {                                    \
        long  fpsi = -1;                 \
        void *ptr  = get_farg_ptr(       \
            farg[argi].fpstag, &fpsi);   \
        *(farg[argi].valptr) = ptr;      \
        if (farg[argi].indexptr != NULL)  \
        {                                \
            *(farg[argi].indexptr) = fpsi;\
        }                                \
    }


/* INSERT_STD_* macros -- stub versions
 * for standalone compilation. The real
 * versions are in CLIcore_utils.h and
 * reference CLI functions. */

#define INSERT_STD_CLIfunction            \
    static errno_t __attribute__((unused)) CLIfunction(void)     \
    {                                    \
        (void) farg;                     \
        (void) CLIcmddata;               \
        return RETURN_SUCCESS;           \
    }




#define INSERT_STD_CLIREGISTERFUNC {}

/* Process info macros -- these are used by
 * standalone code so we provide the real
 * versions via CLIcore_utils.h-compatible
 * macros that reference CLIcmddata. */

#define INSERT_STD_PROCINFO_COMPUTEFUNC_INIT \
    int          processloopOK = 1;      \
    PROCESSINFO *processinfo   = NULL;   \
    CLIcmddata.cmdsettings->            \
        triggertimeout.tv_sec  = 2;      \
    CLIcmddata.cmdsettings->            \
        triggertimeout.tv_nsec = 0;      \
    if (dcfpsptr != NULL)               \
    {                                    \
        CLIcmddata.cmdsettings->flags   \
            |= (dcfpsptr->cmdset.flags  \
                & CLICMDFLAG_PROCINFO); \
        CLIcmddata.cmdsettings->        \
            RT_priority =                \
            dcfpsptr->cmdset.RT_priority;\
        CLIcmddata.cmdsettings->        \
            procinfo_loopcntMax =        \
            dcfpsptr->cmdset             \
                .procinfo_loopcntMax;    \
        CLIcmddata.cmdsettings->        \
            triggermode =                \
            dcfpsptr->cmdset             \
                .triggermode;            \
        strncpy(CLIcmddata.cmdsettings  \
                    ->triggerstreamname, \
                dcfpsptr->cmdset         \
                    .triggerstreamname,  \
                STRINGMAXLEN_IMAGE_NAME  \
                    - 1);                \
        CLIcmddata.cmdsettings->        \
            semindexrequested =          \
            dcfpsptr->cmdset             \
                .semindexrequested;      \
        CLIcmddata.cmdsettings->        \
            triggerdelay.tv_sec =        \
            dcfpsptr->cmdset             \
                .triggerdelay.tv_sec;    \
        CLIcmddata.cmdsettings->        \
            triggerdelay.tv_nsec =       \
            dcfpsptr->cmdset             \
                .triggerdelay.tv_nsec;   \
        CLIcmddata.cmdsettings->        \
            triggertimeout.tv_sec =      \
            dcfpsptr->cmdset             \
                .triggertimeout.tv_sec;  \
        CLIcmddata.cmdsettings->        \
            triggertimeout.tv_nsec =     \
            dcfpsptr->cmdset             \
                .triggertimeout.tv_nsec; \
    }                                    \
    if (CLIcmddata.cmdsettings->flags   \
        & CLICMDFLAG_PROCINFO)           \
    {                                    \
        char pinfodescr[200];            \
        int slen = snprintf(pinfodescr,  \
            200, "function %.10s",       \
            CLIcmddata.key);             \
        if (slen < 1 || slen >= 200) {   \
            abort();                     \
        }                                \
        if (dcfpsptr != NULL) {         \
            const char *_piname =        \
                (dcfpsname[0] != '\0')   \
                ? dcfpsname              \
                : dcfpsptr->md->name;    \
            processinfo =                \
                processinfo_setup(       \
                    (char*)_piname,      \
                    pinfodescr,          \
                    "startup",           \
                    __FUNCTION__,        \
                    __FILE__, __LINE__); \
            fps_to_processinfo(          \
                dcfpsptr,               \
                processinfo);            \
        } else {                         \
            processinfo =                \
                processinfo_setup(       \
                    CLIcmddata.key,      \
                    pinfodescr,          \
                    "startup",           \
                    __FUNCTION__,        \
                    __FILE__, __LINE__); \
        }                                \
        processinfo->loopcntMax =        \
            CLIcmddata.cmdsettings      \
                ->procinfo_loopcntMax;   \
        processinfo->triggerstreamID     \
            = -2;                        \
        processinfo->triggermode =       \
            CLIcmddata.cmdsettings      \
                ->triggermode;           \
        strncpy(processinfo              \
                    ->triggerstreamname, \
                CLIcmddata.cmdsettings  \
                    ->triggerstreamname, \
                STRINGMAXLEN_IMAGE_NAME  \
                    - 1);                \
        processinfo->triggerdelay =      \
            CLIcmddata.cmdsettings      \
                ->triggerdelay;          \
        processinfo->triggertimeout =    \
            CLIcmddata.cmdsettings      \
                ->triggertimeout;        \
        processinfo->triggerstreamID =   \
            image_ID(processinfo         \
                         ->triggerstreamname, \
                     dcimg, dcnimg);     \
        FUNC_CHECK_RETURN(               \
            processinfo_waitoninputstream_init( \
                processinfo,             \
                (processinfo             \
                         ->triggerstreamID \
                     > -1                \
                     ? &dcimg[processinfo \
                                  ->triggerstreamID] \
                     : NULL),            \
                CLIcmddata.cmdsettings  \
                    ->triggermode,        \
                CLIcmddata.cmdsettings  \
                    ->semindexrequested));\
        processinfo->RT_priority =       \
            CLIcmddata.cmdsettings      \
                ->RT_priority;           \
        processinfo->CPUmask =           \
            CLIcmddata.cmdsettings      \
                ->CPUmask;               \
        processinfo->MeasureTiming =     \
            CLIcmddata.cmdsettings      \
                ->procinfo_MeasureTiming;\
        processinfo_loopstart(           \
            processinfo);                \
    }


#define INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART \
    while (processloopOK == 1)           \
    {                                    \
        if (CLIcmddata.cmdsettings->flags \
            & CLICMDFLAG_PROCINFO) {     \
            processloopOK =              \
                processinfo_loopstep(    \
                    processinfo);        \
            processinfo_waitoninputstream(\
                processinfo);            \
            if (processinfo              \
                    ->triggerstatus ==   \
                PROCESSINFO_TRIGGERSTATUS_TIMEDOUT \
                && processinfo           \
                       ->triggermode ==   \
                   PROCESSINFO_TRIGGERMODE_SEMAPHORE) \
            {                            \
                continue;                \
            }                            \
            processinfo_exec_start(      \
                processinfo);            \
        } else {                         \
            processloopOK = 0;           \
        }                                \
        int processcompstatus = 1;       \
        if (CLIcmddata.cmdsettings->flags \
            & CLICMDFLAG_PROCINFO) {     \
            processcompstatus =          \
                processinfo_compute_status( \
                    processinfo);        \
        }                                \
        if (processcompstatus == 1) {


#define INSERT_STD_PROCINFO_COMPUTEFUNC_START \
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT \
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART


#define INSERT_STD_PROCINFO_COMPUTEFUNC_END \
    }                                    \
    if (CLIcmddata.cmdsettings->flags   \
        & CLICMDFLAG_PROCINFO) {         \
        if (processinfo != NULL) {       \
            if (dcfpsptr != NULL) {     \
                if (dcfpsptr->cmdset    \
                        .triggermodeptr  \
                    != NULL) {           \
                    processinfo          \
                        ->triggermode =  \
                        *dcfpsptr       \
                             ->cmdset    \
                             .triggermodeptr; \
                }                        \
                if (dcfpsptr->cmdset    \
                        .procinfo_loopcntMax_ptr \
                    != NULL) {           \
                    processinfo          \
                        ->loopcntMax =   \
                        *dcfpsptr       \
                             ->cmdset    \
                             .procinfo_loopcntMax_ptr; \
                }                        \
                if (dcfpsptr->cmdset    \
                        .triggerdelayptr \
                    != NULL) {           \
                    processinfo          \
                        ->triggerdelay = \
                        dcfpsptr        \
                            ->cmdset     \
                            .triggerdelayptr[0]; \
                }                        \
                if (dcfpsptr->cmdset    \
                        .triggertimeoutptr \
                    != NULL) {           \
                    processinfo          \
                        ->triggertimeout = \
                        dcfpsptr        \
                            ->cmdset     \
                            .triggertimeoutptr[0]; \
                }                        \
            }                            \
        }                                \
        if (processinfo != NULL) {       \
            processinfo_exec_end(        \
                processinfo);            \
        }                                \
    }                                    \
    }                                    \
    if (CLIcmddata.cmdsettings->flags   \
        & CLICMDFLAG_PROCINFO) {         \
        processinfo_cleanExit(           \
            processinfo);                \
    }

#endif /* FPS_PROCINFO_MACROS_H */
