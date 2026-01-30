#ifndef FPS_H
#define FPS_H

#include <time.h>
#include <sys/types.h>
#include <stdint.h>
#include "cmdsettings.h"
#include "timeutils.h"
#include "processinfo_signals.h"

typedef long imageID;
typedef long variableID;

#include "IMGID.h"

#define STRINGMAXLEN_FPS_LOGMSG  1000
#define STRINGMAXLEN_FPS_CMDLINE 1000

#define NB_FPS_MAX 100

#define MAXNBLEVELS 20

#define FPSCONNECT_SIMPLE 0
#define FPSCONNECT_CONF   1
#define FPSCONNECT_RUN    2

// CMCODE type is uint32_t
#define FPSCMDCODE_CONFSTART     0x00000001 // start configuration process
#define FPSCMDCODE_CONFSTOP      0x00000002 // stop configuration process
#define FPSCMDCODE_FPSINIT       0x00000004 // initialize FPS if does not exist
#define FPSCMDCODE_FPSINITCREATE 0x00000008 // (re-)create FPS even if it exists
#define FPSCMDCODE_RUNSTART      0x00000010 // start run process
#define FPSCMDCODE_RUNSTOP       0x00000020 // stop run process
#define FPSCMDCODE_TMUXSTART     0x00000100 // start tmux sessions
#define FPSCMDCODE_TMUXSTOP      0x00000200 // stop tmux sessions

#define FPSCMDCODE_IGNORE 0x00001000 // do not run anything

#define FUNCTION_PARAMETER_KEYWORD_STRMAXLEN 64
#define FUNCTION_PARAMETER_KEYWORD_MAXLEVEL  20

#define FPTYPE_AUTO    0x00000000 // automatic typing
#define FPTYPE_UNDEF   0x00000001
#define FPTYPE_INT32   0x00000002
#define FPTYPE_UINT32  0x00000004
#define FPTYPE_INT64   0x00000008
#define FPTYPE_UINT64  0x00000010
#define FPTYPE_FLOAT32 0x00000020
#define FPTYPE_FLOAT64 0x00000040

#define FPTYPE_PID      0x00000080
#define FPTYPE_TIMESPEC 0x00000100

#define FPTYPE_FILENAME     0x00000200 // generic filename
#define FPTYPE_FITSFILENAME 0x00000400 // FITS file
#define FPTYPE_EXECFILENAME 0x00000800 // executable file

#define FPTYPE_DIRNAME 0x00001000 // directory name

#define FPTYPE_STREAMNAME 0x00002000

#define FPTYPE_STRING 0x00004000 // generic string

#define FPTYPE_ONOFF  0x00008000

#define FPTYPE_PROCESS 0x00010000

#define FPTYPE_FPSNAME 0x00020000 // connection to another FPS

#define STRINGMAXLEN_FPSTYPE  20

#define FUNCTION_PARAMETER_DESCR_STRMAXLEN 64
#define FUNCTION_PARAMETER_STRMAXLEN       64

#define FPFLAG_ACTIVE       0x0000000000000001
#define FPFLAG_USED         0x0000000000000002
#define FPFLAG_VISIBLE      0x0000000000000004

#define FPFLAG_WRITE        0x0000000000000010
#define FPFLAG_WRITECONF    0x0000000000000020
#define FPFLAG_WRITERUN     0x0000000000000040
#define FPFLAG_WRITESTATUS  0x0000000000000080

#define FPFLAG_LOG          0x0000000000000100 // log on change
#define FPFLAG_SAVEONCHANGE 0x0000000000000200 // save to disk on change
#define FPFLAG_SAVEONCLOSE  0x0000000000000400 // save to disk on close

#define FPFLAG_IMPORTED     0x0000000000001000
#define FPFLAG_FEEDBACK     0x0000000000002000
#define FPFLAG_ONOFF        0x0000000000004000

#define FPFLAG_CHECKINIT    0x0000000000010000
#define FPFLAG_MINLIMIT     0x0000000000020000 // enforce min limit
#define FPFLAG_MAXLIMIT     0x0000000000040000 // enforce max limit
#define FPFLAG_ERROR        0x0000000000080000 // is current parameter value OK ?

#define STREAM_LOAD_SOURCE_NOTFOUND        0
#define STREAM_LOAD_SOURCE_NOTFOUND_STRING "STREAM_LOAD_SOURCE_NOTFOUND"

#define STREAM_LOAD_SOURCE_LOCALMEM        1
#define STREAM_LOAD_SOURCE_LOCALMEM_STRING "STREAM_LOAD_SOURCE_LOCALMEM"

#define STREAM_LOAD_SOURCE_SHAREMEM        2
#define STREAM_LOAD_SOURCE_SHAREMEM_STRING "STREAM_LOAD_SOURCE_SHAREMEM"

#define STREAM_LOAD_SOURCE_CONFFITS        3
#define STREAM_LOAD_SOURCE_CONFFITS_STRING "STREAM_LOAD_SOURCE_CONFFITS"

#define STREAM_LOAD_SOURCE_CONFNAME        4
#define STREAM_LOAD_SOURCE_CONFNAME_STRING "STREAM_LOAD_SOURCE_CONFNAME"

#define STREAM_LOAD_SOURCE_NULL        5
#define STREAM_LOAD_SOURCE_NULL_STRING "STREAM_LOAD_SOURCE_NULL"

#define STREAM_LOAD_SOURCE_EXITFAILURE        -1
#define STREAM_LOAD_SOURCE_EXITFAILURE_STRING "STREAM_LOAD_SOURCE_EXITFAILURE"

#define FPFLAG_STREAM_LOAD_FORCE_LOCALMEM 0x0000000000100000
#define FPFLAG_STREAM_LOAD_FORCE_SHAREMEM 0x0000000000200000
#define FPFLAG_STREAM_LOAD_FORCE_CONFFITS 0x0000000000400000
#define FPFLAG_STREAM_LOAD_FORCE_CONFNAME 0x0000000000800000

#define FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM 0x0000000001000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM 0x0000000002000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS 0x0000000004000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME 0x0000000008000000

#define FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM 0x0000000010000000
#define FPFLAG_STREAM_LOAD_UPDATE_CONFFITS 0x0000000020000000

#define FPFLAG_FILE_CONF_REQUIRED 0x0000000040000000 // file must exist for CONF process to proceed
#define FPFLAG_FILE_RUN_REQUIRED 0x0000000080000000 // file must exist for RUN process to proceed

#define FPFLAG_FPS_CONF_REQUIRED 0x0000000040000000 // file must exist for CONF process to proceed
#define FPFLAG_FPS_RUN_REQUIRED 0x0000000080000000 // file must exist for RUN process to proceed

#define FPFLAG_STREAM_CONF_REQUIRED 0x0000000040000000 // stream has to be in MEM for CONF process to proceed
#define FPFLAG_STREAM_RUN_REQUIRED 0x0000000080000000 // stream has to be in MEM for RUN process to proceed

// Additional notes on load functions in AOloopControl_IOtools
//
/* AOloopControl_IOtools_2Dloadcreate_shmim( const char *name,
    const char *fname,
    long xsize,
    long ysize,
    float DefaultValue)
*/
//

#define FPFLAG_STREAM_ENFORCE_DATATYPE 0x0000000100000000 // enforce stream datatype

#define FPFLAG_STREAM_TEST_DATATYPE_UINT8 0x0000000200000000 // test if stream of type UINT8   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT8 0x0000000400000000 // test if stream of type INT8    (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_UINT16 0x0000000800000000 // test if stream of type UINT16  (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT16 0x0000001000000000 // test if stream of type INT16   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_UINT32 0x0000002000000000 // test if stream of type UINT32  (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT32 0x0000004000000000 // test if stream of type INT32   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_UINT64 0x0000008000000000 // test if stream of type UINT64  (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT64 0x0000010000000000 // test if stream of type INT64   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_HALF 0x0000020000000000 // test if stream of type HALF    (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_FLOAT 0x0000040000000000 // test if stream of type FLOAT   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_DOUBLE 0x0000080000000000 // test if stream of type DOUBLE  (OR test)

#define FPFLAG_STREAM_ENFORCE_1D    0x0000100000000000 // enforce 1D image
#define FPFLAG_STREAM_ENFORCE_2D    0x0000200000000000 // enforce 2D image
#define FPFLAG_STREAM_ENFORCE_3D    0x0000400000000000 // enforce 3D image
#define FPFLAG_STREAM_ENFORCE_XSIZE 0x0008000000000000 // enforce X size
#define FPFLAG_STREAM_ENFORCE_YSIZE 0x0010000000000000 // enforce Y size
#define FPFLAG_STREAM_ENFORCE_ZSIZE 0x0020000000000000 // enforce Z size

#define FPFLAG_CHECKSTREAM 0x0040000000000000 // check and display stream status in GUI
#define FPFLAG_STREAM_MEMLOADREPORT 0x0080000000000000 // Write stream load report (for debugging)

#define FPFLAG_DEFAULT_INPUT (FPFLAG_ACTIVE | FPFLAG_USED | FPFLAG_VISIBLE | FPFLAG_WRITE | FPFLAG_WRITECONF | FPFLAG_SAVEONCHANGE | FPFLAG_FEEDBACK | FPFLAG_WRITESTATUS)
#define FPFLAG_DEFAULT_OUTPUT (FPFLAG_ACTIVE | FPFLAG_USED | FPFLAG_VISIBLE)
#define FPFLAG_DEFAULT_INPUT_STREAM (FPFLAG_DEFAULT_INPUT | FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM)
#define FPFLAG_DEFAULT_OUTPUT_STREAM (FPFLAG_DEFAULT_INPUT | FPFLAG_CHECKSTREAM)

#define FPFLAG_DEFAULT_STATUS (FPFLAG_ACTIVE | FPFLAG_USED | FPFLAG_VISIBLE)

#define FUNCTION_PARAMETER_NBPARAM_DEFAULT 10 // size of dynamically allocated array of parameters

typedef struct
{
    long    streamID;
    uint8_t stream_atype;
    uint32_t stream_naxis[2];
    uint32_t stream_xsize[2];
    uint32_t stream_ysize[2];
    uint32_t stream_zsize[2];
    uint8_t  stream_sourceLocation;
} FUNCTION_PARAMETER_SUBINFO_STREAM;

typedef struct
{
    long FPSNBparamMAX;
    long FPSNBparamActive;
    long FPSNBparamUsed;
} FUNCTION_PARAMETER_SUBINFO_FPS;

typedef struct
{
    uint64_t fpflag;
    uint64_t userflag;

    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel;

    char keywordfrom[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];

    char description[FUNCTION_PARAMETER_DESCR_STRMAXLEN];

    uint32_t type;

    union
    {
        int32_t  i32[4];
        uint32_t ui32[4];
        int64_t  i64[4];
        uint64_t ui64[4];
        double f64[4];
        float  f32[4];
        pid_t pid[2];
        struct timespec ts[2];
        char string[2][FUNCTION_PARAMETER_STRMAXLEN];
    } val;

    union
    {
        FUNCTION_PARAMETER_SUBINFO_STREAM stream;
        FUNCTION_PARAMETER_SUBINFO_FPS    fps;
    } info;

    long cnt0;

} FUNCTION_PARAMETER;

#define STRINGMAXLEN_FPS_NAME 100

#define STRINGMAXLEN_PROCESSINFO_TMUXNAME    100



#define FUNCTION_PARAMETER_STRUCT_MSG_SIZE 500

#define FUNCTION_PARAMETER_STRUCT_STATUS_CONF 0x0001
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUN 0x0002
#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF 0x0010
#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN 0x0020
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP 0x0100
#define FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK 0x0200
#define FUNCTION_PARAMETER_STRUCT_STATUS_CHECKERR 0x0400
#define FUNCTION_PARAMETER_STRUCT_STATUS_SAVE     0x0800

#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF 0x1000
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN  0x2000
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL 0x4000

#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN 0x0001
#define FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE 0x0004
#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED 0x0008

#define FPS_NB_MSG                        100
#define FUNCTION_PARAMETER_STRUCT_MSG_LEN 500

#define FPS_MSG_FLAG_NOTINITIALIZED 0x0001
#define FPS_MSG_FLAG_BELOWMIN       0x0002
#define FPS_MSG_FLAG_ABOVEMAX       0x0004
#define FPS_MSG_FLAG_ERROR 0x0008
#define FPS_MSG_FLAG_INFO  0x0010

#define FPS_CWD_STRLENMAX            200
#define FPS_DIR_STRLENMAX            200
#define FPS_SRCDIR_STRLENMAX         200
#define FPS_PNAME_STRMAXLEN          100
#define FPS_CALLPROGNAME_STRMAXLEN    80
#define FPS_CALLFUNCNAME_STRMAXLEN   100
#define FPS_DESCR_STRMAXLEN          200
#define FPS_KEYWORDARRAY_STRMAXLEN   200
#define STRINGMAXLEN_FPS_DIRNAME     200
#define FPS_HELPTEXT_STRMAXLEN      8192

#define FPS_MAXNB_MODULE     50
#define FPS_MODULE_STRMAXLEN 200

typedef struct
{
    char name[STRINGMAXLEN_FPS_NAME];
    char description[FPS_DESCR_STRMAXLEN];
    char helptext[FPS_HELPTEXT_STRMAXLEN];
    char execfullpath[512];
    char keywordarray[FPS_KEYWORDARRAY_STRMAXLEN];
    char workdir[FPS_CWD_STRLENMAX];
    char datadir[FPS_DIR_STRLENMAX];
    char confdir[FPS_DIR_STRLENMAX];
    char sourcefname[FPS_SRCDIR_STRLENMAX];
    int sourceline;
    // the name and indices are automatically parsed in the following format
    char pname[FPS_PNAME_STRMAXLEN]; // example: pname
    char callprogname[FPS_CALLPROGNAME_STRMAXLEN];
    char callfuncname[FPS_CALLFUNCNAME_STRMAXLEN];
    char tmuxname[STRINGMAXLEN_PROCESSINFO_TMUXNAME];
    char nameindexW[16][10]; // subnames
    int  NBnameindex;        // example: 2
    pid_t           confpid;
    struct timespec confpidstarttime;
    pid_t           runpid;
    struct timespec runpidstarttime;
    int  NBmodule;
    char modulename[FPS_MAXNB_MODULE][FPS_MODULE_STRMAXLEN];
    uint64_t signal;
    uint64_t confwaitus;
    uint32_t status;
    uint64_t processinfo_change_cnt;
    long NBparamMAX;
    char message[FPS_NB_MSG][FUNCTION_PARAMETER_STRUCT_MSG_LEN];
    int msgpindex[FPS_NB_MSG];
    uint32_t msgcode[FPS_NB_MSG];
    long msgcnt;
    uint32_t conferrcnt;
} FUNCTION_PARAMETER_STRUCT_MD;

#define FPS_LOCALSTATUS_CONFLOOP 0x0001

typedef struct
{
    FUNCTION_PARAMETER_STRUCT_MD *md;
    FUNCTION_PARAMETER           *parray;
    uint16_t localstatus;
    int      SMfd;
    uint32_t CMDmode;
    long NBparam;
    long NBparamActive;
    CMDSETTINGS cmdset;
} FUNCTION_PARAMETER_STRUCT;

typedef struct
{
    struct timespec triggerdelay[2];
} FPS2PROCINFOMAP;

#define NB_FPSCTRL_TASK_MAX       5000
#define NB_FPSCTRL_TASK_PURGESIZE 50

#define FPSTASK_STATUS_ACTIVE 0x0000000000000001
#define FPSTASK_STATUS_SHOW      0x0000000000000002
#define FPSTASK_STATUS_RUNNING   0x0000000000000004
#define FPSTASK_STATUS_COMPLETED 0x0000000000000008

#define FPSTASK_STATUS_WAITING      0x0000000000000010
#define FPSTASK_STATUS_RECEIVED     0x0000000000000020
#define FPSTASK_STATUS_CMDNOTFOUND  0x0000000000000040
#define FPSTASK_STATUS_CMDFAIL      0x0000000000000080
#define FPSTASK_STATUS_ERR_ARGTYPE  0x0000000000000100
#define FPSTASK_STATUS_ERR_TYPECONV 0x0000000000000200
#define FPSTASK_STATUS_ERR_NBARG    0x0000000000000400
#define FPSTASK_STATUS_ERR_NOFPS    0x0000000000000800
#define FPSTASK_STATUS_CMDOK        0x0000000000001000

#define FPSTASK_FLAG_WAITONRUN  0x0000000000000001
#define FPSTASK_FLAG_WAITONCONF 0x0000000000000002
#define FPSTASK_FLAG_WAIT_FOR_FPS_NORUN 0x0000000000000004

#define NB_FPSCTRL_TASKQUEUE_MAX 100

typedef struct
{
    int priority;
} FPSCTRL_TASK_QUEUE;

typedef struct
{
    char cmdstring[STRINGMAXLEN_FPS_CMDLINE];
    uint64_t inputindex;
    uint32_t queue;
    uint64_t status;
    uint64_t flag;
    int fpsindex;
    struct timespec creationtime;
    struct timespec activationtime;
    struct timespec completiontime;
} FPSCTRL_TASK_ENTRY;

typedef struct
{
    int      exitloop;
    int      fpsCTRL_DisplayMode;
    int      fpsCTRL_DisplayVerbose;
    uint32_t mode;
    int      NBfps;
    int      NBkwn;
    long     NBindex;
    char     fpsnamemask[100];
    int      nodeSelected;
    int      run_display;
    int      fpsindexSelected;
    int      GUIlineSelected[100];
    int      currentlevel;
    int      directorynodeSelected;
    int      pindexSelected;
    char     fpsCTRLfifoname[200];
    int      fpsCTRLfifofd;
    int      direction;
    int      scheduler_wrowstart;
} FPSCTRL_PROCESS_VARS;

#define NB_KEYWNODE_MAX 6000
#define MAX_NB_CHILD    3000

typedef struct
{
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel;
    int parent_index;
    int NBchild;
    int child[MAX_NB_CHILD];
    int leaf;
    int fpsindex;
    int pindex;
} KEYWORD_TREE_NODE;

int function_parameter_printlist(FUNCTION_PARAMETER *funcparamarray, long NBparamMAX);

#ifdef USE_NCURSES
errno_t functionparameter_CTRLscreen(uint32_t mode,
                                     char    *fpsnamemask,
                                     char    *fpsCTRLfifoname,
                                     double  timeout_sec);
#endif

FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(const char *fpsname, uint32_t mode);
FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup_sized(const char *fpsname, uint32_t mode, long NBparamMAX);
uint16_t function_parameter_FPCONFloopstep(FUNCTION_PARAMETER_STRUCT *fps);
uint16_t function_parameter_FPCONFexit(FUNCTION_PARAMETER_STRUCT *fps);
uint16_t function_parameter_RUNexit(FUNCTION_PARAMETER_STRUCT *fps);

#include "fps_add_entry.h"
#include "fps_checkparameter.h"
#include "fps_connect.h"
#include "fps_connectExternalFPS.h"
#include "fps_disconnect.h"
#include "fps_execFPScmd.h"
#include "fps_GetFileName.h"
#include "fps_getFPSargs.h"
#include "fps_GetParamIndex.h"
#include "fps_GetTypeString.h"
#include "fps_load.h"
#include "fps_loadstream.h"
#include "fps_outlog.h"
#include "fps_paramvalue.h"
// #include "fps_printlist.h" // Removed
#include "fps_PrintParameterInfo.h"
#include "fps_printparameter_valuestring.h"
#include "fps_save2disk.h"
#include "fps_scan.h"
#include "fps_shmdirname.h"
#include "fps_WriteParameterToDisk.h"

#include "fps_CONFstop.h"
#include "fps_RUNstop.h"
#include "fps_processinfo.h"
#include "fps_tmux.h"

#include "fps_processinfo_entries.h"

// ===========================
// CONVENIENT MACROS FOR FPS
// ===========================

/** @defgroup fpsmacro          MACROS: Function parameter structure
 *
 * Frequently used function parameter structure (FPS) operations :
 * - Create / initialize FPS
 * - Add parameters to existing FPS
 *
 * @{
 */

/**
 * @brief Initialize function parameter structure (FPS)
 *
 * @param[in] VARfpsname FPS name
 * @param[in] VARCMDmode command code
 * @param[in] VARNBparamMAX max number of parameters
 */
#define FPS_SETUP_INIT_SIZED(VARfpsname, VARCMDmode, VARNBparamMAX)            \
    FUNCTION_PARAMETER_STRUCT fps;                                             \
    do                                                                         \
    {                                                                          \
        fps.SMfd = -1;                                                         \
        fps      = function_parameter_FPCONFsetup_sized((VARfpsname), (VARCMDmode), (VARNBparamMAX)); \
        strncpy(fps.md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX);          \
        fps.md->sourceline = __LINE__;                                         \
        {                                                                      \
            char msgstring[STRINGMAXLEN_FPS_LOGMSG];                           \
            SNPRINTF_CHECK(msgstring,                                          \
                           STRINGMAXLEN_FPS_LOGMSG,                            \
                           "LOGSTART %s %d %s %d",                             \
                           (VARfpsname),                                       \
                           (VARCMDmode),                                       \
                           fps.md->sourcefname,                                \
                           fps.md->sourceline);                                \
            functionparameter_outlog("FPSINIT", msgstring);                    \
        }                                                                      \
    } while (0)


/**
 * @brief Initialize function parameter structure (FPS)
 *
 * @param[in] VARfpsname FPS name
 * @param[in] VARCMDmode command code
 */
#define FPS_SETUP_INIT(VARfpsname, VARCMDmode)                                 \
    FUNCTION_PARAMETER_STRUCT fps;                                             \
    do                                                                         \
    {                                                                          \
        fps.SMfd = -1;                                                         \
        fps      = function_parameter_FPCONFsetup((VARfpsname), (VARCMDmode)); \
        strncpy(fps.md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX);          \
        fps.md->sourceline = __LINE__;                                         \
        {                                                                      \
            char msgstring[STRINGMAXLEN_FPS_LOGMSG];                           \
            SNPRINTF_CHECK(msgstring,                                          \
                           STRINGMAXLEN_FPS_LOGMSG,                            \
                           "LOGSTART %s %d %s %d",                             \
                           (VARfpsname),                                       \
                           (VARCMDmode),                                       \
                           fps.md->sourcefname,                                \
                           fps.md->sourceline);                                \
            functionparameter_outlog("FPSINIT", msgstring);                    \
        }                                                                      \
    } while (0)



/** @brief Connect to FPS
 *
 *
 */
#define FPS_CONNECT(VARfpsname, VARCMDmode)                                    \
    FUNCTION_PARAMETER_STRUCT fps;                                             \
    do                                                                         \
    {                                                                          \
        fps.SMfd = -1;                                                         \
        if (function_parameter_struct_connect((VARfpsname),                    \
                                              &fps,                            \
                                              (VARCMDmode)) == -1)             \
        {                                                                      \
            printf(                                                            \
                "ERROR: fps \"%s\" does not exist -> running without "         \
                "FPS interface\n",                                             \
                VARfpsname);                                                   \
            return RETURN_FAILURE;                                             \
        }                                                                      \
    } while (0)




/** @brief Start FPS configuration loop
 */
#define FPS_CONFLOOP_START                                                     \
    if (!(fps.localstatus & FPS_LOCALSTATUS_CONFLOOP))                         \
    {                                                                          \
        return RETURN_SUCCESS;                                                 \
    }                                                                          \
    while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP)                         \
    {                                                                          \
        {                                                                      \
            struct timespec treq, trem;                                        \
            treq.tv_sec  = 0;                                                  \
            treq.tv_nsec = 50000;                                              \
            nanosleep(&treq, &trem);                                           \
            if (processinfo_signal_INT == 1)                                   \
            {                                                                  \
                fps.localstatus &= ~FPS_LOCALSTATUS_CONFLOOP;                  \
            }                                                                  \
        }                                                                      \
        if (function_parameter_FPCONFloopstep(&fps) == 1)                      \
        {



/** @brief End FPS configuration loop
 */
#define FPS_CONFLOOP_END                                                       \
    functionparameter_CheckParametersAll(&fps);                                \
    }                                                                          \
    }                                                                          \
    function_parameter_FPCONFexit(&fps);

/** @brief Combine initialization of FPS and procinfo for RUN process
 */

#define FPSPROCINFOLOOP_RUNINIT(...)                                           \
    PROCESSINFO *processinfo = NULL;                                                  \
    int          processloopOK = 1;                                            \
    do                                                                         \
    {                                                                          \
        char pinfodescr[200];                                                  \
        int  slen = snprintf(pinfodescr, 200, __VA_ARGS__);                    \
        if (slen < 1)                                                          \
        {                                                                      \
            PRINT_ERROR("snprintf wrote <1 char");                             \
            abort();                                                           \
        }                                                                      \
        if (slen >= 200)                                                       \
        {                                                                      \
            PRINT_ERROR("snprintf string truncation");                         \
            abort();                                                           \
        }                                                                      \
        processinfo = processinfo_setup(FPS_name,                              \
                                        pinfodescr,                            \
                                        "startup",                             \
                                        __FUNCTION__,                          \
                                        __FILE__,                              \
                                        __LINE__);                             \
        fps_to_processinfo(&fps, processinfo);                                 \
    } while (0)




#define FPS_AUTORUN_SETUP(funcstring, shortname)                               \
    FUNCTION_PARAMETER_STRUCT fps;                                             \
    do                                                                         \
    {                                                                          \
        extern uint32_t FPS_CMDCODE;                                           \
        extern char     FPS_name[STRINGMAXLEN_FPS_NAME];                       \
        extern char     FPS_callprogname[FPS_CALLPROGNAME_STRMAXLEN];          \
        extern char     FPS_callfuncname[FPS_CALLFUNCNAME_STRMAXLEN];          \
        extern FUNCTION_PARAMETER_STRUCT *fpsarray;                            \
        snprintf(FPS_name,STRINGMAXLEN_FPS_NAME,  "%s-%06ld", (shortname), (long) getpid());      \
        FPS_CMDCODE = FPSCMDCODE_FPSINIT;                                 \
        strncpy(FPS_callprogname, "milk", FPS_CALLPROGNAME_STRMAXLEN - 1); \
        strncpy(FPS_callfuncname, "autorun", FPS_CALLFUNCNAME_STRMAXLEN - 1);  \
        FPSCONF_##funcstring();                                                \
        function_parameter_struct_connect(FPS_name,                            \
                                          &fps,                                \
                                          FPSCONNECT_SIMPLE);                  \
    } while (0)




#define FPS_EXECFUNCTION_STD                                                   \
    static errno_t FPSEXECfunction()                                           \
    {                                                                          \
        extern uint32_t FPS_CMDCODE;                                           \
        extern char     FPS_name[STRINGMAXLEN_FPS_NAME];                       \
        extern char     FPS_callprogname[FPS_CALLPROGNAME_STRMAXLEN];          \
        extern char     FPS_callfuncname[FPS_CALLFUNCNAME_STRMAXLEN];          \
        extern FUNCTION_PARAMETER_STRUCT *fpsarray;                            \
        FUNCTION_PARAMETER_STRUCT fps;                                         \
        snprintf(FPS_name, STRINGMAXLEN_FPS_NAME, "%s-%06ld", CLIcmddata.key, (long) getpid());   \
        FPS_CMDCODE = FPSCMDCODE_FPSINIT;                                 \
        strncpy(FPS_callprogname, "milk", FPS_CALLPROGNAME_STRMAXLEN - 1); \
        strncpy(FPS_callfuncname, CLIcmddata.key, FPS_CALLFUNCNAME_STRMAXLEN - 1); \
        FPSCONFfunction();                                                     \
        function_parameter_struct_connect(FPS_name,                            \
                                          &fps,                                \
                                          FPSCONNECT_SIMPLE);                  \
        CLIargs_to_FPSparams_setval(farg, CLIcmddata.nbarg, &fps);             \
        function_parameter_struct_disconnect(&fps);                            \
        FPSRUNfunction();                                                      \
        return RETURN_SUCCESS;                                                 \
    }



#define FPS_CLIFUNCTION_STD                                                    \
    static errno_t FPSCLIfunction(void)                                        \
    {                                                                          \
        extern errno_t (*FPS_CONFfunc)();                                      \
        extern errno_t (*FPS_RUNfunc)();                                       \
        extern uint32_t FPS_CMDCODE;                                           \
        extern char     FPS_name[STRINGMAXLEN_FPS_NAME];                       \
        extern char     FPS_callprogname[FPS_CALLPROGNAME_STRMAXLEN];          \
        extern char     FPS_callfuncname[FPS_CALLFUNCNAME_STRMAXLEN];          \
        extern FUNCTION_PARAMETER_STRUCT *fpsarray;                            \
        function_parameter_getFPSargs_from_CLIfunc(CLIcmddata.key);            \
        if (FPS_CMDCODE != 0)                                                  \
        {                                                                      \
            printf("DEBUG: FPS command detected (code %u)\n",                  \
                   FPS_CMDCODE);                                               \
            FPS_CONFfunc = FPSCONFfunction;                                    \
            FPS_RUNfunc  = FPSRUNfunction;                                     \
            strncpy(FPS_name, FPS_name, STRINGMAXLEN_FPS_NAME - 1);             \
            strncpy(FPS_callprogname, "milk", FPS_CALLPROGNAME_STRMAXLEN - 1); \
            strncpy(FPS_callfuncname, CLIcmddata.key, FPS_CALLFUNCNAME_STRMAXLEN - 1); \
            function_parameter_execFPScmd();                                   \
            return RETURN_SUCCESS;                                             \
        }                                                                      \
        if (CLI_checkarg_array(farg, CLIcmddata.nbarg) == RETURN_SUCCESS)      \
        {                                                                      \
            FPSEXECfunction();                                                 \
            return RETURN_SUCCESS;                                             \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            return CLICMD_INVALID_ARG;                                         \
        }                                                                      \
    }

#define FPS_MAKE_CONF_FUNCNAME(x)      FPSCONF_##x
#define FPSCONF_FUNCTION_NAME(fncname) FPS_MAKE_CONF_FUNCNAME(fncname)

#define FPS_MAKE_RUN_FUNCNAME(x)      FPSRUN_##x
#define FPSRUN_FUNCTION_NAME(fncname) FPS_MAKE_RUN_FUNCNAME(fncname)

#define FPS_MAKE_CLI_FUNCNAME(x)      FPSCLI_##x
#define FPSCLI_FUNCTION_NAME(fncname) FPS_MAKE_CLI_FUNCNAME(fncname)

#define FPS_MAKE_CLIADDCMD_FUNCNAME(x)      FPSCLIADDCMD_##x
#define FPSCLIADDCMD_FUNCTION_NAME(fncname) FPS_MAKE_CLIADDCMD_FUNCNAME(fncname)

/** @brief Macro to generate standalone CONFSTOP function
 */
#define FPS_MAKE_STANDALONE_CONFSTOP(FUNC_SUFFIX) \
int FPSCONFSTOP_##FUNC_SUFFIX(const char *fps_name) { \
    FUNCTION_PARAMETER_STRUCT fps; \
    printf("Stopping configuration process for '%s'\n", fps_name); \
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) { \
        fprintf(stderr, "Error: FPS '%s' not found.\n", fps_name); \
        return 1; \
    } \
    functionparameter_CONFstop(&fps); \
    function_parameter_struct_disconnect(&fps); \
    return 0; \
}

/** @brief Macro to generate standalone RUNSTOP function
 */
#define FPS_MAKE_STANDALONE_RUNSTOP(FUNC_SUFFIX) \
int FPSRUNSTOP_##FUNC_SUFFIX(const char *fps_name) { \
    FUNCTION_PARAMETER_STRUCT fps; \
    printf("Stopping run process for '%s'\n", fps_name); \
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) { \
        fprintf(stderr, "Error: FPS '%s' not found.\n", fps_name); \
        return 1; \
    } \
    functionparameter_RUNstop(&fps); \
    function_parameter_struct_disconnect(&fps); \
    functionparameter_FPS_processinfo_signal(fps_name, 3); \
    return 0; \
}

// Helper for X_HELP_PRINT to extract the first argument from variadic args
#define GET_1ST_ARG_HELPER(arg1, ...) arg1
#define GET_1ST_ARG(args) GET_1ST_ARG_HELPER args

#ifndef CLICMDARG_FLAG_DEFAULT
#define CLICMDARG_FLAG_DEFAULT 0x00000000
#endif
#ifndef CLICMDARG_FLAG_NOCLI
#define CLICMDARG_FLAG_NOCLI 0x00000001
#endif

// Define CLIARG macros if not already defined (to handle standalone compilation without CLIcore headers)
#ifndef CLIARG_VISIBLE_DEFAULT
#define CLIARG_VISIBLE_DEFAULT CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT
#endif
#ifndef CLIARG_HIDDEN_DEFAULT
#define CLIARG_HIDDEN_DEFAULT CLICMDARG_FLAG_NOCLI, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT
#endif
#ifndef CLIARG_OUTPUT_DEFAULT
#define CLIARG_OUTPUT_DEFAULT CLICMDARG_FLAG_NOCLI, FPTYPE_AUTO, FPFLAG_DEFAULT_OUTPUT
#endif

#ifndef COLORRESET
#define COLORRESET     "\033[0m"
#endif
#ifndef COLORARGCLI
#define COLORARGCLI    "\033[36m" // argument part of CLI call: cyan
#endif
#ifndef COLORARGnotCLI
#define COLORARGnotCLI "\033[35m" // argument not part of CLI call: yellow
#endif

#ifndef COLORCOMMAND
#define COLORCOMMAND   "\033[32m" // command: green
#endif
#ifndef COLORHEADER
#define COLORHEADER    "\033[1m"  // header: bold
#endif
#ifndef COLOROPTION
#define COLOROPTION    "\033[33m" // option: yellow
#endif

#define X_HELP_PRINT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, ...) \
    { \
        char valuestring[256]; \
        uint64_t flags = GET_1ST_ARG((__VA_ARGS__)); \
        int is_hidden = (flags & CLICMDARG_FLAG_NOCLI); \
        char type_str[20] = "???"; \
        if (fps_type == FPTYPE_FLOAT32) strcpy(type_str, "float32"); \
        else if (fps_type == FPTYPE_UINT32) strcpy(type_str, "uint32"); \
        else if (fps_type == FPTYPE_INT32) strcpy(type_str, "int32"); \
        else if (fps_type == FPTYPE_INT64) strcpy(type_str, "int64"); \
        else if (fps_type == FPTYPE_UINT64) strcpy(type_str, "uint64"); \
        else if (fps_type == FPTYPE_STRING) strcpy(type_str, "string"); \
        else if (fps_type == FPTYPE_FILENAME) strcpy(type_str, "filename"); \
        else if (fps_type == FPTYPE_STREAMNAME) strcpy(type_str, "stream"); \
        else if (fps_type == FPTYPE_FLOAT64) strcpy(type_str, "float64"); \
        else if (fps_type == FPTYPE_ONOFF) strcpy(type_str, "onoff"); \
        \
        snprintf(valuestring, 256, "[%7s]  %s", type_str, def_str); \
        \
        if (!is_hidden) { \
            if (show_help_color) \
                printf("%6d   " COLORARGCLI "%-16s" COLORRESET " %-24s %s\n", CLIargcnt, key, valuestring, descr); \
            else \
                printf("%6d   %-16s %-24s %s\n", CLIargcnt, key, valuestring, descr); \
        } else { \
            if (show_help_color) \
                printf("[hidden] " COLORARGnotCLI "%-16s" COLORRESET " %-24s %s\n", key, valuestring, descr); \
            else \
                printf("[hidden] %-16s %-24s %s\n", key, valuestring, descr); \
        } \
        CLIargcnt++; \
    }

/** @brief Macro to generate standalone main function
 */
#define FPS_MAIN_STANDALONE(DEFAULT_FPS_NAME, FUNC_PREFIX, HELPTEXT, PARAMS_MACRO) \
int main(int argc, char *argv[]) { \
    char fps_name[STRINGMAXLEN_FPS_NAME] = DEFAULT_FPS_NAME; \
    char arg_fps_name[STRINGMAXLEN_FPS_NAME] = ""; \
    int use_tmux = 0; \
    int show_help = 0; \
    int show_help_color = 0; \
    char *command = NULL; \
    char *keywords = NULL; \
    char *description = NULL; \
    char *colon_pos = NULL; \
    for (int i = 1; i < argc; i++) { \
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { \
            show_help = 1; \
        } else if (strcmp(argv[i], "-hc") == 0 || strcmp(argv[i], "--help-color") == 0) { \
            show_help = 1; \
            show_help_color = 1; \
        } else if (strcmp(argv[i], "-tmux") == 0) { \
            use_tmux = 1; \
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--keywords") == 0) && i + 1 < argc) { \
            keywords = argv[++i]; \
        } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--description") == 0) && i + 1 < argc) { \
            description = argv[++i]; \
        } else if (command == NULL) { \
            command = argv[i]; \
            /* Check for fpsname:command format */ \
            if ((colon_pos = strchr(command, ':')) != NULL) { \
                *colon_pos = '\0'; \
                strncpy(arg_fps_name, command, STRINGMAXLEN_FPS_NAME - 1); \
                command = colon_pos + 1; \
            } \
        } \
    } \
    if (strlen(arg_fps_name) > 0) { \
        strncpy(fps_name, arg_fps_name, STRINGMAXLEN_FPS_NAME - 1); \
    } \
    if (show_help || (argc < 2)) { \
        if (show_help_color) { \
            printf("\n" COLORHEADER "Usage:" COLORRESET " %s " COLOROPTION "[fpsname:]" COLORRESET COLORCOMMAND "<Command>" COLORRESET " " COLOROPTION "[Options]" COLORRESET "\n\n", argv[0]); \
            printf(COLORHEADER "Description:" COLORRESET "\n  Standalone FPS application.\n\n"); \
            printf(COLORHEADER "Commands:" COLORRESET "\n"); \
            printf("  " COLORCOMMAND "fpsinit" COLORRESET "    One-time setup: creates the FPS shared memory segment.\n"); \
            printf("  " COLORCOMMAND "fps" COLORRESET "        Print content of the FPS.\n"); \
            printf("  " COLORCOMMAND "fpslist" COLORRESET "    List all FPS instances matching this executable.\n"); \
            printf("  " COLORCOMMAND "confstart" COLORRESET "  Run the configuration monitoring loop.\n"); \
            printf("  " COLORCOMMAND "confstep" COLORRESET "   Run a single configuration monitoring step.\n"); \
            printf("  " COLORCOMMAND "confstop" COLORRESET "   Stop the configuration monitoring loop.\n"); \
            printf("  " COLORCOMMAND "runstart" COLORRESET "   Run the main processing loop.\n"); \
            printf("  " COLORCOMMAND "runstop" COLORRESET "    Stop the main processing loop.\n\n"); \
            printf(COLORHEADER "Options:" COLORRESET "\n"); \
            printf("  " COLOROPTION "fpsname:" COLORRESET "           Optional prefix to specify FPS name (default: %s).\n", DEFAULT_FPS_NAME); \
            printf("  " COLOROPTION "-k, --keywords KEYWORDS" COLORRESET "  Specify FPS keywords (default: NULL).\n"); \
            printf("  " COLOROPTION "-d, --description DESC" COLORRESET "   Specify FPS description (default: NULL).\n"); \
            printf("  " COLOROPTION "-tmux" COLORRESET "                    Auto-create a tmux session and dispatch commands.\n"); \
            printf("  " COLOROPTION "-h, --help" COLORRESET "               Show this help message.\n"); \
            printf("  " COLOROPTION "-hc, --help-color" COLORRESET "        Show this help message with color.\n\n"); \
        } else { \
            printf("\nUsage: %s [fpsname:]<Command> [Options]\n\n", argv[0]); \
            printf("Description:\n  Standalone FPS application.\n\n"); \
            printf("Commands:\n"); \
            printf("  fpsinit    One-time setup: creates the FPS shared memory segment.\n"); \
            printf("  fps        Print content of the FPS.\n"); \
            printf("  fpslist    List all FPS instances matching this executable.\n"); \
            printf("  confstart  Run the configuration monitoring loop.\n"); \
            printf("  confstep   Run a single configuration monitoring step.\n"); \
            printf("  confstop   Stop the configuration monitoring loop.\n"); \
            printf("  runstart   Run the main processing loop.\n"); \
            printf("  runstop    Stop the main processing loop.\n\n"); \
            printf("Options:\n"); \
            printf("  fpsname:                 Optional prefix to specify FPS name (default: %s).\n", DEFAULT_FPS_NAME); \
            printf("  -k, --keywords KEYWORDS  Specify FPS keywords (default: NULL).\n"); \
            printf("  -d, --description DESC   Specify FPS description (default: NULL).\n"); \
            printf("  -tmux                    Auto-create a tmux session and dispatch commands.\n"); \
            printf("  -h, --help               Show this help message.\n"); \
            printf("  -hc, --help-color        Show this help message with color.\n\n"); \
        } \
        if (HELPTEXT[0] != '\0') { \
            if (show_help_color) printf(COLORHEADER "Detailed Help:" COLORRESET "\n"); \
            else printf("Detailed Help:\n"); \
            printf("--------------\n"); \
            printf("%s\n\n", HELPTEXT); \
        } \
        if (show_help_color) printf(COLORHEADER "CLI call arguments:" COLORRESET "\n"); \
        else printf("CLI call arguments:\n"); \
        int CLIargcnt = 0; \
        PARAMS_MACRO(X_HELP_PRINT) \
        printf("\n"); \
        return 0; \
    } \
    if (command == NULL) { \
        fprintf(stderr, "Error: Missing command argument.\n"); \
        return 1; \
    } \
    if (strcmp(command, "fps") == 0) { \
        char cmd[STRINGMAXLEN_FPS_NAME + 20]; \
        snprintf(cmd, sizeof(cmd), "milk-fps-info %s", fps_name); \
        return system(cmd); \
    } else if (strcmp(command, "fpslist") == 0) { \
        FUNCTION_PARAMETER_STRUCT *fpsarray = (FUNCTION_PARAMETER_STRUCT *) calloc(NB_FPS_MAX, sizeof(FUNCTION_PARAMETER_STRUCT)); \
        if (fpsarray == NULL) return 1; \
        for(int i = 0; i < NB_FPS_MAX; i++) fpsarray[i].SMfd = -1; \
        KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE)); \
        if (keywnode == NULL) { free(fpsarray); return 1; } \
        int NBkwn = 0, NBfps = 0; \
        long NBpindex = 0; \
        functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0); \
        if (NBfps > 0) { \
            char *exec_basename = strrchr(argv[0], '/'); \
            if (exec_basename) exec_basename++; else exec_basename = argv[0]; \
            int found = 0; \
            for(int i = 0; i < NBfps; i++) { \
                char *fps_exec_basename = strrchr(fpsarray[i].md->execfullpath, '/'); \
                if (fps_exec_basename) fps_exec_basename++; else fps_exec_basename = fpsarray[i].md->execfullpath; \
                if (strcmp(exec_basename, fps_exec_basename) == 0) { \
                    if (!found) { \
                        printf("%-30s %-10s %s\n", "FPS Name", "Status", "Description"); \
                        printf("------------------------------------------------------------\n"); \
                        found = 1; \
                    } \
                    char status_str[32] = "UNKNOWN"; \
                    if(fpsarray[i].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF) strcpy(status_str, "CONF"); \
                    else if(fpsarray[i].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUN) strcpy(status_str, "RUN"); \
                    printf("%-30s %-10s %s\n", fpsarray[i].md->name, status_str, fpsarray[i].md->description); \
                } \
                function_parameter_struct_disconnect(&fpsarray[i]); \
            } \
            if (!found) printf("No matching FPS instances found for executable '%s'.\n", exec_basename); \
        } else { \
            printf("No FPS instances found.\n"); \
        } \
        free(keywnode); \
        free(fpsarray); \
        return 0; \
    } \
    if (use_tmux) { \
        char path[1024]; \
        if (functionparameter_FPS_get_executable_path(path, sizeof(path)) == NULL) { \
            if (realpath(argv[0], path) == NULL) strncpy(path, argv[0], 1023); \
        } \
        char name_arg[256] = ""; \
        /* fps name is now part of command, but tmux dispatch expects args? No, tmux dispatch sends command line. */ \
        /* We need to reconstruct command line for tmux */ \
        /* The previous implementation used -n name. Now we prepend name: */ \
        if (strcmp(fps_name, DEFAULT_FPS_NAME) != 0) { \
             /* Should we pass fps_name separately? functionparameter_FPS_tmux_send_dispatch logic check needed */ \
             /* But simpler: pass reconstructed command "fpsname:command" */ \
             /* functionparameter_FPS_tmux_send_dispatch takes (char *fpsname, char *cmd, char *execpath, char *arg) */ \
             /* It constructs "tmux send-keys ... execpath cmd arg" */ \
             /* We want "tmux send-keys ... execpath fpsname:cmd" */ \
             snprintf(name_arg, sizeof(name_arg), " %s:%s", fps_name, command); \
             /* Wait, command is already parsed. */ \
        } else { \
             snprintf(name_arg, sizeof(name_arg), " %s", command); \
        } \
        functionparameter_FPS_tmux_standalone_setup(fps_name); \
        /* Hack: pass empty command to tmux dispatch, and full command in arg */ \
        if (functionparameter_FPS_tmux_send_dispatch(fps_name, "", path, name_arg) == 0) { \
            return 0; \
        } \
        if (strcmp(command, "fpsinit") == 0) { \
            FPSINIT_##FUNC_PREFIX(fps_name, keywords, description); \
        } \
        return 0; \
    } \
    if (strcmp(command, "fpsinit") == 0) { \
        return FPSINIT_##FUNC_PREFIX(fps_name, keywords, description); \
    } else if (strcmp(command, "confstart") == 0) { \
        return FPSCONF_##FUNC_PREFIX(fps_name, 1); \
    } else if (strcmp(command, "confstep") == 0) { \
        return FPSCONF_##FUNC_PREFIX(fps_name, 0); \
    } else if (strcmp(command, "confstop") == 0) { \
        return FPSCONFSTOP_##FUNC_PREFIX(fps_name); \
    } else if (strcmp(command, "runstart") == 0) { \
        return FPSRUN_##FUNC_PREFIX(fps_name); \
    } else if (strcmp(command, "runstop") == 0) { \
        return FPSRUNSTOP_##FUNC_PREFIX(fps_name); \
    } \
    fprintf(stderr, "Invalid command: %s\n", command); \
    return 1; \
}

/**
 * @brief Standard initialization preamble for FPSINIT function
 */
#define FPS_INIT_STD_PREAMBLE(VARfps, VARfps_name, VARkeywords, VARdescription, VARhelptext) \
    printf("Initializing FPS '%s'...\n", VARfps_name); \
    (VARfps) = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_FPSINIT); \
    strncpy((VARfps).md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX - 1); \
    (VARfps).md->sourceline = __LINE__; \
    if ((VARkeywords) != NULL) { \
        strncpy((VARfps).md->keywordarray, (VARkeywords), FPS_KEYWORDARRAY_STRMAXLEN - 1); \
    } \
    if ((VARdescription) != NULL) { \
        strncpy((VARfps).md->description, (VARdescription), FPS_DESCR_STRMAXLEN - 1); \
    } \
    strncpy((VARfps).md->helptext, (VARhelptext), FPS_HELPTEXT_STRMAXLEN - 1);

/**
 * @brief Standard ProcessInfo default settings for FPSINIT
 */
#define FPS_INIT_PROCINFO_DEFAULTS(VARfps, VARtriggerstream, VARtimeout_sec) \
    strncpy((VARfps).cmdset.triggerstreamname, (VARtriggerstream), STRINGMAXLEN_IMAGE_NAME - 1); \
    (VARfps).cmdset.procinfo_loopcntMax = -1; \
    (VARfps).cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE; \
    (VARfps).cmdset.triggertimeout.tv_sec = (VARtimeout_sec); \
    (VARfps).cmdset.triggertimeout.tv_nsec = 0;

/**
 * @brief Standard body for FPSCONF function
 * 
 * @param VARfps_name Name of the FPS
 * @param VARloop Loop flag (1 for loop, 0 for single step)
 * @param BLOCK_VAR_MAP Code block to map parameters (e.g. { ptr = ...; })
 * @param BLOCK_VALIDATE Code block to validate parameters (e.g. { validate(); })
 */
#define FPS_CONF_STD_BODY(VARfps_name, VARloop, BLOCK_VAR_MAP, BLOCK_VALIDATE) \
    FUNCTION_PARAMETER_STRUCT fps; \
    if (VARloop) { \
        printf("Starting configuration process loop for '%s'\n", VARfps_name); \
        fps = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_CONFSTART); \
        BLOCK_VAR_MAP \
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { \
            if (function_parameter_FPCONFloopstep(&fps)) { \
                BLOCK_VALIDATE \
            } \
            usleep(10000); \
        } \
    } else { \
        printf("Running single configuration step for '%s'\n", VARfps_name); \
        fps = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_FPSINIT); \
        BLOCK_VAR_MAP \
        function_parameter_FPCONFloopstep(&fps); \
    } \
    function_parameter_FPCONFexit(&fps);

/**
 * @brief Standard connection and parameter mapping for FPSRUN
 */
#define FPS_RUN_STD_PREAMBLE(VARfps_name, VARfps, BLOCK_VAR_MAP) \
    if (function_parameter_struct_connect(VARfps_name, &(VARfps), FPSCONNECT_RUN) == -1) { \
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", VARfps_name); \
        return 1; \
    } \
    BLOCK_VAR_MAP

/**
 * @brief Standard setup for ProcessInfo in FPSRUN
 */
#define FPS_RUN_PROCESSINFO_SETUP(VARprocessinfo, VARfps_name, VARdesc_short, VARdesc_detail, VARinput_image, VARfps) \
    VARprocessinfo = processinfo_setup((char*)VARfps_name, VARdesc_short, VARdesc_detail, __FUNCTION__, __FILE__, __LINE__); \
    if (!VARprocessinfo) return 1; \
    processinfo_CatchSignals(); \
    processinfo_waitoninputstream_init(VARprocessinfo, VARinput_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1); \
    fps_to_processinfo(&(VARfps), VARprocessinfo); \
    processinfo_loopstart(VARprocessinfo);

/**
 * @brief Standard loop for FPSRUN
 */
#define FPS_RUN_PROCESSINFO_LOOP(VARprocessinfo, VARfps, VARinput_image, VARoutput_image, BLOCK_COMPUTE) \
    int loopOK = 1; \
    while(loopOK) { \
        loopOK = processinfo_loopstep(VARprocessinfo); \
        if(!loopOK) break; \
        processinfo_waitoninputstream(VARprocessinfo); \
        if (VARprocessinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue; \
        processinfo_exec_start(VARprocessinfo); \
        BLOCK_COMPUTE \
        processinfo_exec_end(VARprocessinfo); \
        processinfo_update_output_stream(VARprocessinfo, VARoutput_image, VARinput_image); \
    } \
    processinfo_cleanExit(VARprocessinfo); \
    function_parameter_struct_disconnect(&(VARfps));

/** @} */ // end group fpsmacro

#endif // FPS_H