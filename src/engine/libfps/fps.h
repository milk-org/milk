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

#define FPTYPE_STRING_NOT_STREAM 0x00040000 // string that is NOT a stream/image

#define FPTYPE_IS_STRING(type) \
    ((type == FPTYPE_STRING) || (type == FPTYPE_FILENAME) || (type == FPTYPE_FITSFILENAME) || \
     (type == FPTYPE_EXECFILENAME) || (type == FPTYPE_DIRNAME) || (type == FPTYPE_STREAMNAME) || \
     (type == FPTYPE_PROCESS) || (type == FPTYPE_FPSNAME) || (type == FPTYPE_STRING_NOT_STREAM))

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

#define FPFLAG_PRIMARY_CLI_INPUT 0x0000000000000008 // primary parameter: required CLI arg + TUI highlight
#define FPFLAG_CLI_INPUT 0x0000020000000000 // parameter is expected to be specified by user on CLI
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

#define FUNCTION_PARAMETER_NBPARAM_DEFAULT 100 // size of dynamically allocated array of parameters

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
    uint64_t value_cnt;
    int cli_index; // CLI argument index (-1 if not CLI arg)

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
    int      display_offset[100];
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
#include "fps_SetParamCLIindex.h"
#include "fps_print_info.h"
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

/* FPS-CLI unified framework (V2) */
#include "fps_cli_binding.h"
#include "fps_local_store.h"
#include "fps_cli_init.h"
#include "fps_cli_sync.h"
#include "fps_lifecycle.h"
#include "fps_cli_query.h"
#include "fps_cli_function.h"

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
#ifndef COLORPRIMARY
#define COLORPRIMARY   "\033[1;36m" // primary CLI argument: bold cyan
#endif
#ifndef COLORNORMAL
#define COLORNORMAL    "\033[0;37m" // normal text

#define COLORERROR     "\033[31m"   // error: red
#endif

// Helper for X_HELP_PRINT to extract the first argument from variadic args
#define GET_1ST_ARG_HELPER(arg1, ...) arg1
#define GET_1ST_ARG(args) GET_1ST_ARG_HELPER args

#define X_HELP_MEASURE(fps_type, c_type, key, \
    descr, def_str, ptr_name, get_func, \
    flags, ...) \
    { \
        int _kl = (int) strlen(key); \
        if (_kl > col_kw_w) col_kw_w = _kl; \
        char _ts[20] = "???"; \
        if (fps_type == FPTYPE_FLOAT32) \
            strcpy(_ts, "FLOAT32"); \
        else if (fps_type == FPTYPE_UINT32) \
            strcpy(_ts, "UINT32"); \
        else if (fps_type == FPTYPE_INT32) \
            strcpy(_ts, "INT32"); \
        else if (fps_type == FPTYPE_INT64) \
            strcpy(_ts, "INT64"); \
        else if (fps_type == FPTYPE_UINT64) \
            strcpy(_ts, "UINT64"); \
        else if (fps_type == FPTYPE_FLOAT64) \
            strcpy(_ts, "FLOAT64"); \
        else if (fps_type == FPTYPE_ONOFF) \
            strcpy(_ts, "ONOFF"); \
        else if (fps_type == FPTYPE_STREAMNAME)\
            strcpy(_ts, "STREAMNAME"); \
        else if (fps_type == FPTYPE_FILENAME) \
            strcpy(_ts, "FILENAME"); \
        else if (fps_type == FPTYPE_FITSFILENAME)\
            strcpy(_ts, "FITSFILE"); \
        else if (fps_type == FPTYPE_EXECFILENAME)\
            strcpy(_ts, "EXECFILE"); \
        else if (fps_type == FPTYPE_DIRNAME) \
            strcpy(_ts, "DIRNAME"); \
        else if (fps_type == FPTYPE_FPSNAME) \
            strcpy(_ts, "FPSNAME"); \
        else if (fps_type == FPTYPE_PROCESS) \
            strcpy(_ts, "PROCESS"); \
        else if (fps_type == FPTYPE_STRING_NOT_STREAM)\
            strcpy(_ts, "STRING"); \
        else if (fps_type == FPTYPE_STRING) \
            strcpy(_ts, "STRING"); \
        else if (fps_type == FPTYPE_PID) \
            strcpy(_ts, "PID"); \
        else if (fps_type == FPTYPE_TIMESPEC) \
            strcpy(_ts, "TIMESPEC"); \
        int _tl = (int) strlen(_ts); \
        if (_tl > col_tp_w) col_tp_w = _tl; \
        int _dl = (int) strlen(def_str); \
        if (_dl > col_df_w) col_df_w = _dl; \
    }

#define X_HELP_PRINT(fps_type, c_type, key, \
    descr, def_str, ptr_name, get_func, \
    flags, ...) \
    { \
        char valuestring[256]; \
        int is_hidden = \
            !(flags & FPFLAG_PRIMARY_CLI_INPUT); \
        char type_str[20] = "???"; \
        if (fps_type == FPTYPE_FLOAT32) \
            strcpy(type_str, "FLOAT32"); \
        else if (fps_type == FPTYPE_UINT32) \
            strcpy(type_str, "UINT32"); \
        else if (fps_type == FPTYPE_INT32) \
            strcpy(type_str, "INT32"); \
        else if (fps_type == FPTYPE_INT64) \
            strcpy(type_str, "INT64"); \
        else if (fps_type == FPTYPE_UINT64) \
            strcpy(type_str, "UINT64"); \
        else if (fps_type == FPTYPE_FLOAT64) \
            strcpy(type_str, "FLOAT64"); \
        else if (fps_type == FPTYPE_ONOFF) \
            strcpy(type_str, "ONOFF"); \
        else if (fps_type == FPTYPE_STREAMNAME)\
            strcpy(type_str, "STREAMNAME"); \
        else if (fps_type == FPTYPE_FILENAME) \
            strcpy(type_str, "FILENAME"); \
        else if (fps_type == FPTYPE_FITSFILENAME)\
            strcpy(type_str, "FITSFILE"); \
        else if (fps_type == FPTYPE_EXECFILENAME)\
            strcpy(type_str, "EXECFILE"); \
        else if (fps_type == FPTYPE_DIRNAME) \
            strcpy(type_str, "DIRNAME"); \
        else if (fps_type == FPTYPE_FPSNAME) \
            strcpy(type_str, "FPSNAME"); \
        else if (fps_type == FPTYPE_PROCESS) \
            strcpy(type_str, "PROCESS"); \
        else if (fps_type == FPTYPE_STRING_NOT_STREAM)\
            strcpy(type_str, "STRING"); \
        else if (fps_type == FPTYPE_STRING) \
            strcpy(type_str, "STRING"); \
        else if (fps_type == FPTYPE_PID) \
            strcpy(type_str, "PID"); \
        else if (fps_type == FPTYPE_TIMESPEC) \
            strcpy(type_str, "TIMESPEC"); \
        \
        if (!is_hidden) { \
            if (show_help_color) \
                printf("%8d " COLORARGCLI \
                       "%-*s" COLORRESET \
                       " %-*s %-*s %s\n", \
                       CLIargcnt, \
                       col_kw_w, key, \
                       col_tp_w, type_str, \
                       col_df_w, def_str, \
                       descr); \
            else \
                printf("%8d %-*s %-*s" \
                       " %-*s %s\n", \
                       CLIargcnt, \
                       col_kw_w, key, \
                       col_tp_w, type_str, \
                       col_df_w, def_str, \
                       descr); \
        } else { \
            if (show_help_color) \
                printf("[hidden] " \
                       COLORARGnotCLI \
                       "%-*s" COLORRESET \
                       " %-*s %-*s %s\n", \
                       col_kw_w, key, \
                       col_tp_w, type_str, \
                       col_df_w, def_str, \
                       descr); \
            else \
                printf("[hidden] %-*s %-*s" \
                       " %-*s %s\n", \
                       col_kw_w, key, \
                       col_tp_w, type_str, \
                       col_df_w, def_str, \
                       descr); \
        } \
        CLIargcnt++; \
    }

/** @brief Macro to generate standalone main function
 */
#define FPS_MAIN_STANDALONE(DEFAULT_FPS_NAME, FUNC_PREFIX, HELPTEXT, PARAMS_MACRO) \
int main(int argc, char *argv[]) { \
    char fps_name[STRINGMAXLEN_FPS_NAME] = ""; \
    strncpy(fps_name, DEFAULT_FPS_NAME, STRINGMAXLEN_FPS_NAME - 1); \
    char arg_fps_name[STRINGMAXLEN_FPS_NAME] = ""; \
    int use_tmux = 0; \
    int use_procinfo = 0; \
    int show_help = 0; \
    int show_h1 = 0; \
    int show_help_color = 1; \
    char *command = NULL; \
    char *keywords = NULL; \
    char *description = NULL; \
    char *colon_pos = NULL; \
    for (int i = 1; i < argc; i++) { \
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { \
            show_help = 1; \
        } else if (strcmp(argv[i], "-h1") == 0 || strcmp(argv[i], "--help-oneline") == 0) { \
            show_h1 = 1; \
        } else if (strcmp(argv[i], "-hc") == 0 || strcmp(argv[i], "--help-color") == 0) { \
            show_help = 1; \
            show_help_color = 1; \
        } else if (strcmp(argv[i], "-hnc") == 0 || strcmp(argv[i], "--help-no-color") == 0) { \
            show_help = 1; \
            show_help_color = 0; \
        } else if (strcmp(argv[i], "-tmux") == 0) { \
            use_tmux = 1; \
        } else if (strcmp(argv[i], "-procinfo") == 0 || strcmp(argv[i], "--procinfo") == 0) { \
            use_procinfo = 1; \
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--keywords") == 0) && i + 1 < argc) { \
            keywords = argv[++i]; \
        } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--description") == 0) && i + 1 < argc) { \
            description = argv[++i]; \
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) && i + 1 < argc) { \
            strncpy(arg_fps_name, argv[++i], STRINGMAXLEN_FPS_NAME - 1); \
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
    if (command == NULL) { \
        command = "run"; \
    } else { \
        /* Check if the command is a recognized one. If not, it's likely the first parameter. */ \
        if (strcmp(command, "fpsinit") != 0 && \
            strcmp(command, "fps") != 0 && \
            strcmp(command, "fpslist") != 0 && \
            strcmp(command, "confstart") != 0 && \
            strcmp(command, "confstep") != 0 && \
            strcmp(command, "confstop") != 0 && \
            strcmp(command, "runstart") != 0 && \
            strcmp(command, "runstop") != 0 && \
            strcmp(command, "exec") != 0 && \
            strcmp(command, "set") != 0 && \
            strcmp(command, "run") != 0) { \
            /* It's not a recognized command, so we treat it as a parameter for 'run' */ \
            /* However, if it contains a colon, we already split it into arg_fps_name and command. */ \
            /* If it was 'fpsname:paramvalue', then command is now 'paramvalue'. */ \
            /* We should push it back into the argument list for 'run' to pick up. */ \
            command = "run"; \
        } \
    } \
    (void) use_procinfo; \
    if (strlen(arg_fps_name) > 0) { \
        strncpy(fps_name, arg_fps_name, STRINGMAXLEN_FPS_NAME - 1); \
    } \
    if (show_h1) { \
        printf("%s\n", \
               "Standalone FPS application."); \
        return 0; \
    } \
    if (show_help || (argc < 2)) { \
        if (show_help_color) { \
            printf("\n" COLORHEADER "Usage:" COLORRESET " %s " COLOROPTION "[fpsname:]" COLORRESET COLORCOMMAND "<Command>" COLORRESET " " COLOROPTION "[Options]" COLORRESET "\n", argv[0]); \
            printf("  Compiled: %s %s\n\n", __DATE__, __TIME__); \
            printf(COLORHEADER "Description:" COLORRESET "\n  Standalone FPS application.\n\n"); \
            printf(COLORHEADER "Commands:" COLORRESET "\n"); \
            printf("  " COLORCOMMAND "fpsinit" COLORRESET "    One-time setup: creates the FPS shared memory segment.\n"); \
            printf("  " COLORCOMMAND "fps" COLORRESET "        Print content of the FPS.\n"); \
            printf("  " COLORCOMMAND "fpslist" COLORRESET "    List all FPS instances matching this executable.\n"); \
            printf("  " COLORCOMMAND "confstart" COLORRESET "  Run the configuration monitoring loop.\n"); \
            printf("  " COLORCOMMAND "confstep" COLORRESET "   Run a single configuration monitoring step.\n"); \
            printf("  " COLORCOMMAND "confstop" COLORRESET "   Stop the configuration monitoring loop.\n"); \
            printf("  " COLORCOMMAND "runstart" COLORRESET "   Run the main processing loop.\n"); \
            printf("  " COLORCOMMAND "runstop" COLORRESET "    Stop the main processing loop.\n"); \
            printf("  " COLORCOMMAND "set" COLORRESET " " COLOROPTION "[args]" COLORRESET "  Set positional arguments in the FPS (use . to skip).\n"); \
            printf("  " COLORCOMMAND "exec" COLORRESET " " COLOROPTION "[args]" COLORRESET " Auto-init + set args + run.\n\n"); \
            printf(COLORHEADER "Options:" COLORRESET "\n"); \
            printf("  " COLOROPTION "fpsname:" COLORRESET "           Optional prefix (or use -n) to specify FPS name (default: %s).\n", DEFAULT_FPS_NAME); \
            printf("  " COLOROPTION "-n, --name FPSNAME" COLORRESET "       Specify FPS name.\n"); \
            printf("  " COLOROPTION "-k, --keywords KEYWORDS" COLORRESET "  Specify FPS keywords (default: NULL).\n"); \
            printf("  " COLOROPTION "-d, --description DESC" COLORRESET "   Specify FPS description (default: NULL).\n"); \
            printf("  " COLOROPTION "-tmux" COLORRESET "                    Auto-create a tmux session and dispatch commands.\n"); \
            printf("  " COLOROPTION "-procinfo" COLORRESET "                Enable processinfo support (for fpsinit).\n"); \
            printf("  " COLOROPTION "-h, --help" COLORRESET "               Show this help message (color default).\n"); \
            printf("  " COLOROPTION "-hnc, --help-no-color" COLORRESET "   Show this help message without color.\n\n"); \
            printf(COLORHEADER "Notes:" COLORRESET "\n"); \
            printf("  Alternate ways to perform these operations once the FPS has been created:\n"); \
            printf("    " COLORCOMMAND "milk-fps-confstart" COLORRESET " <fpsname>\n"); \
            printf("    " COLORCOMMAND "milk-fps-confstop" COLORRESET "  <fpsname>\n"); \
            printf("    " COLORCOMMAND "milk-fps-runstart" COLORRESET "  <fpsname>\n"); \
            printf("    " COLORCOMMAND "milk-fps-runstop" COLORRESET "   <fpsname>\n"); \
            printf("    " COLORCOMMAND "milk-fps-confstep" COLORRESET "  <fpsname>\n\n"); \
        } else { \
            printf("\nUsage: %s [fpsname:]<Command> [Options]\n", argv[0]); \
            printf("  Compiled: %s %s\n\n", __DATE__, __TIME__); \
            printf("Description:\n  Standalone FPS application.\n\n"); \
            printf("Commands:\n"); \
            printf("  fpsinit    One-time setup: creates the FPS shared memory segment.\n"); \
            printf("  fps        Print content of the FPS.\n"); \
            printf("  fpslist    List all FPS instances matching this executable.\n"); \
            printf("  confstart  Run the configuration monitoring loop.\n"); \
            printf("  confstep   Run a single configuration monitoring step.\n"); \
            printf("  confstop   Stop the configuration monitoring loop.\n"); \
            printf("  runstart   Run the main processing loop.\n"); \
            printf("  runstop    Stop the main processing loop.\n"); \
            printf("  set [args] Set positional arguments in the FPS (use . to skip).\n"); \
            printf("  exec [args] Auto-init + set args + run.\n\n"); \
            printf("Options:\n"); \
            printf("  fpsname:                 Optional prefix (or use -n) to specify FPS name (default: %s).\n", DEFAULT_FPS_NAME); \
            printf("  -n, --name FPSNAME       Specify FPS name.\n"); \
            printf("  -k, --keywords KEYWORDS  Specify FPS keywords (default: NULL).\n"); \
            printf("  -d, --description DESC   Specify FPS description (default: NULL).\n"); \
            printf("  -tmux                    Auto-create a tmux session and dispatch commands.\n"); \
            printf("  -procinfo                Enable processinfo support (for fpsinit).\n"); \
            printf("  -h, --help               Show this help message (color default).\n"); \
            printf("  -hnc, --help-no-color    Show this help message without color.\n\n"); \
            printf("Notes:\n"); \
            printf("  Alternate ways to perform these operations once the FPS has been created:\n"); \
            printf("    milk-fps-confstart <fpsname>\n"); \
            printf("    milk-fps-confstop  <fpsname>\n"); \
            printf("    milk-fps-runstart  <fpsname>\n"); \
            printf("    milk-fps-runstop   <fpsname>\n"); \
            printf("    milk-fps-confstep  <fpsname>\n\n"); \
        } \
        if (HELPTEXT[0] != '\0') { \
            if (show_help_color) printf(COLORHEADER "Detailed Help:" COLORRESET "\n"); \
            else printf("Detailed Help:\n"); \
            printf("--------------\n"); \
            printf("%s\n\n", HELPTEXT); \
        } \
        int col_kw_w = 7; /* strlen("Keyword") */ \
        int col_tp_w = 4; /* strlen("Type")    */ \
        int col_df_w = 7; /* strlen("Default") */ \
        PARAMS_MACRO(X_HELP_MEASURE) \
        if (show_help_color) { \
            printf(COLORHEADER \
                   "CLI call arguments:" \
                   COLORRESET "\n"); \
        } else { \
            printf("CLI call arguments:\n"); \
        } \
        printf("%8s %-*s %-*s %-*s %s\n", \
               "Idx", col_kw_w, "Keyword", \
               col_tp_w, "Type", \
               col_df_w, "Default", \
               "Description"); \
        printf("%8s %-*s %-*s %-*s %s\n", \
               "---", col_kw_w, "-------", \
               col_tp_w, "----", \
               col_df_w, "-------", \
               "-----------"); \
        int CLIargcnt = 0; \
        (void) CLIargcnt; \
        PARAMS_MACRO(X_HELP_PRINT) \
        printf("\n"); \
        if (show_help_color) \
            printf(COLOROPTION "Run " COLORCOMMAND "milk-fpsexec-help" COLOROPTION " for detailed FPS framework info." COLORRESET "\n\n"); \
        else \
            printf("Run milk-fpsexec-help for detailed FPS framework info.\n\n"); \
        return 0; \
    } \
    if (command == NULL) { \
        fprintf(stderr, "Error: Missing command argument.\n"); \
        return 1; \
    } \
    if (strcmp(command, "exec") != 0) \
        printf("FPS " COLORCOMMAND "%s" COLORRESET " %s\n", fps_name, command); \
    if (strcmp(command, "fps") == 0) { \
        FUNCTION_PARAMETER_STRUCT fps; \
        if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) { \
            fprintf(stderr, "Error: cannot connect to FPS '%s'.\n", fps_name); \
            return 1; \
        } \
        function_parameter_print_info(&fps, 0, 0); \
        function_parameter_struct_disconnect(&fps); \
        return 0; \
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
        if (strcmp(command, "exec") == 0) { \
            FUNCTION_PARAMETER_STRUCT fps; \
            if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) { \
                printf("FPS " COLORCOMMAND "%s" COLORRESET " exec -> " "\033[33m" "NEW" COLORRESET "\n", fps_name); \
                FPSINIT_##FUNC_PREFIX(fps_name, keywords, description); \
            } else { \
                printf("FPS " COLORCOMMAND "%s" COLORRESET " exec -> " COLORCOMMAND "REUSE" COLORRESET "\n", fps_name); \
                function_parameter_struct_disconnect(&fps); \
            } \
            char run_arg[512] = ""; \
            if (strcmp(fps_name, DEFAULT_FPS_NAME) != 0) { \
                snprintf(run_arg, sizeof(run_arg), " %s:runstart", fps_name); \
            } else { \
                snprintf(run_arg, sizeof(run_arg), " runstart"); \
            } \
            if (use_procinfo) { \
                strncat(run_arg, " -procinfo", sizeof(run_arg) - strlen(run_arg) - 1); \
            } \
            functionparameter_FPS_tmux_send_dispatch(fps_name, "runstart", path, run_arg); \
            return 0; \
        } \
        if (functionparameter_FPS_tmux_send_dispatch(fps_name, command, path, name_arg) == 0) { \
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
    } else if (strcmp(command, "set") == 0) { \
        FUNCTION_PARAMETER_STRUCT fps; \
        if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) { \
            fprintf(stderr, "Error: FPS '%s' not found. Run fpsinit first.\n", fps_name); \
            return 1; \
        } \
        { \
            int cmd_pos = -1; \
            for (int j = 1; j < argc; j++) { \
                if (strcmp(argv[j], "set") == 0) { cmd_pos = j; break; } \
            } \
            if (cmd_pos != -1) { \
                int cli_idx = 0; \
                for (long p = 0; p < fps.md->NBparamMAX; p++) { \
                    if (!(fps.parray[p].fpflag & FPFLAG_ACTIVE)) continue; \
                    if (!(fps.parray[p].fpflag & FPFLAG_PRIMARY_CLI_INPUT)) continue; \
                    int arg_idx = cmd_pos + 1 + cli_idx; \
                    cli_idx++; \
                    if (arg_idx >= argc) break; \
                    if (strcmp(argv[arg_idx], ".") != 0) { \
                        functionparameter_SetParamValue_STRING(&fps, fps.parray[p].keyword[0], argv[arg_idx]); \
                    } \
                } \
            } \
        } \
        function_parameter_struct_disconnect(&fps); \
        printf("FPS " COLORCOMMAND "%s" COLORRESET " set done\n", fps_name); \
        return 0; \
    } else if (strcmp(command, "exec") == 0) { \
        FUNCTION_PARAMETER_STRUCT fps; \
        if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) { \
            printf("FPS " COLORCOMMAND "%s" COLORRESET " exec -> " "\033[33m" "NEW" COLORRESET "\n", fps_name); \
            FPSINIT_##FUNC_PREFIX(fps_name, keywords, description); \
        } else { \
            printf("FPS " COLORCOMMAND "%s" COLORRESET " exec -> " COLORCOMMAND "REUSE" COLORRESET "\n", fps_name); \
            function_parameter_struct_disconnect(&fps); \
        } \
        return FPSRUN_##FUNC_PREFIX(fps_name); \
    } else if (strcmp(command, "runstart") == 0 || strcmp(command, "run") == 0) { \
        return FPSRUN_##FUNC_PREFIX(fps_name); \
    } else if (strcmp(command, "runstop") == 0) { \
        return FPSRUNSTOP_##FUNC_PREFIX(fps_name); \
    } \
    fprintf(stderr, "Invalid command: %s\n", command); \
    return 1; \
}

/*
 * ================================================================
 * FPS_MAIN_STANDALONE_V2 — uses generic library functions
 *
 * Usage (new binding format):
 *     FPS_MAIN_STANDALONE_V2(
 *         FPS_app_info,
 *         MY_PARAMS,
 *         my_compute_function)
 *
 * The PARAMS_MACRO uses the 6-arg binding format:
 *     X(keyword, ptr, type, is_primary, fpflag, descr)
 * ================================================================
 */

/**
 * @brief Help-print expansion for the 6-arg binding format.
 */
/* ---- helper: fill type_str for V2 params ---- */
#define X_HELP_V2_FILL_TYPESTR_(type, ts) \
    do { \
        if (type == FPTYPE_INT32) \
            strcpy(ts, "INT32"); \
        else if (type == FPTYPE_UINT32) \
            strcpy(ts, "UINT32"); \
        else if (type == FPTYPE_INT64) \
            strcpy(ts, "INT64"); \
        else if (type == FPTYPE_UINT64) \
            strcpy(ts, "UINT64"); \
        else if (type == FPTYPE_FLOAT32) \
            strcpy(ts, "FLOAT32"); \
        else if (type == FPTYPE_FLOAT64) \
            strcpy(ts, "FLOAT64"); \
        else if (type == FPTYPE_ONOFF) \
            strcpy(ts, "ONOFF"); \
        else if (type == FPTYPE_STREAMNAME) \
            strcpy(ts, "STREAMNAME"); \
        else if (type == FPTYPE_FILENAME) \
            strcpy(ts, "FILENAME"); \
        else if (type == FPTYPE_FITSFILENAME) \
            strcpy(ts, "FITSFILE"); \
        else if (type == FPTYPE_EXECFILENAME) \
            strcpy(ts, "EXECFILE"); \
        else if (type == FPTYPE_DIRNAME) \
            strcpy(ts, "DIRNAME"); \
        else if (type == FPTYPE_FPSNAME) \
            strcpy(ts, "FPSNAME"); \
        else if (type == FPTYPE_PROCESS) \
            strcpy(ts, "PROCESS"); \
        else if (FPTYPE_IS_STRING(type)) \
            strcpy(ts, "STRING"); \
        else if (type == FPTYPE_PID) \
            strcpy(ts, "PID"); \
        else if (type == FPTYPE_TIMESPEC) \
            strcpy(ts, "TIMESPEC"); \
    } while (0)

/* ---- helper: fill val_str for V2 params ---- */
#define X_HELP_V2_FILL_VALSTR_(type, ptr, vs) \
    do { \
        if (type == FPTYPE_INT32) \
            sprintf(vs, "%d", *(int32_t*)ptr); \
        else if (type == FPTYPE_UINT32) \
            sprintf(vs, "%u", *(uint32_t*)ptr); \
        else if (type == FPTYPE_INT64) \
            sprintf(vs, "%ld", *(int64_t*)ptr); \
        else if (type == FPTYPE_UINT64) \
            sprintf(vs, "%lu", *(uint64_t*)ptr); \
        else if (type == FPTYPE_FLOAT32) \
            sprintf(vs, "%f", *(float*)ptr); \
        else if (type == FPTYPE_FLOAT64) \
            sprintf(vs, "%f", *(double*)ptr); \
        else if (type == FPTYPE_ONOFF) \
            sprintf(vs, "%s", \
                (*(int32_t*)ptr) \
                    ? "ON" : "OFF"); \
        else if (type == FPTYPE_PID) \
            sprintf(vs, "%d", \
                (int)*(pid_t*)ptr); \
        else if (type == FPTYPE_TIMESPEC) \
            sprintf(vs, "%ld.%09ld", \
                ((struct timespec*)ptr)->tv_sec,\
                ((struct timespec*)ptr) \
                    ->tv_nsec); \
        else if (FPTYPE_IS_STRING(type) || \
                 type == FPTYPE_STREAMNAME || \
                 type == FPTYPE_FILENAME || \
                 type == FPTYPE_FITSFILENAME || \
                 type == FPTYPE_EXECFILENAME || \
                 type == FPTYPE_DIRNAME || \
                 type == FPTYPE_FPSNAME || \
                 type == FPTYPE_PROCESS) \
            strncpy(vs, (char*)ptr, 63); \
    } while (0)

/**
 * @brief Measure column widths for 6-arg binding.
 */
#define X_HELP_MEASURE_V2(kw, ptr, type, \
                          is_primary, flag, desc) \
    { \
        const char *_kp = \
            (kw[0] == '.') ? &kw[1] : kw; \
        int _kl = (int) strlen(_kp); \
        if (_kl > col_kw_w) col_kw_w = _kl; \
        char _ts[20] = "???"; \
        X_HELP_V2_FILL_TYPESTR_(type, _ts); \
        int _tl = (int) strlen(_ts); \
        if (_tl > col_tp_w) col_tp_w = _tl; \
        char _vs[64] = ""; \
        X_HELP_V2_FILL_VALSTR_(type, ptr, _vs);\
        int _vl = (int) strlen(_vs); \
        if (_vl > col_df_w) col_df_w = _vl; \
    }

/**
 * @brief Help-print for the 6-arg binding format.
 *
 * Uses col_kw_w, col_tp_w, col_df_w (set by
 * X_HELP_MEASURE_V2) for dynamic column widths.
 */
#define X_HELP_PRINT_V2(kw, ptr, type, \
                         is_primary, flag, desc)\
    { \
        char cli_idx_str[8]; \
        char val_str[64] = ""; \
        char type_str[20] = "???"; \
        const char *disp_kw = \
            (kw[0] == '.') ? &kw[1] : kw; \
        if (is_primary) \
            sprintf(cli_idx_str, "%3d", \
                    CLIargcnt); \
        else \
            strcpy(cli_idx_str, " - "); \
        X_HELP_V2_FILL_TYPESTR_(type, type_str);\
        X_HELP_V2_FILL_VALSTR_(type, ptr, \
                               val_str); \
        if (show_help_color) { \
            const char *_clr = is_primary \
                ? COLORPRIMARY : COLORARGnotCLI;\
            printf("  %s %s%-*s%s" \
                   " %-*s %-*s %s\n", \
                   cli_idx_str, \
                   _clr, col_kw_w, disp_kw, \
                   COLORRESET, \
                   col_tp_w, type_str, \
                   col_df_w, val_str, desc); \
        } else { \
            printf("  %s %-*s %-*s" \
                   " %-*s %s\n", \
                   cli_idx_str, \
                   col_kw_w, disp_kw, \
                   col_tp_w, type_str, \
                   col_df_w, val_str, desc); \
        } \
        if (is_primary) CLIargcnt++; \
    }


#define _FPS_MAIN_STANDALONE_V2_IMPL( \
    APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
    CONFCHECK_FN) \
int main(int argc, char *argv[]) { \
    milk_data_init(); \
    extern void milkfps_set_image_array( \
        IMAGE *imarray, long nb_max); \
    milkfps_set_image_array( \
        milk_data.image, milk_data.NB_MAX_IMAGE); \
    fps_cli_set_standalone_args(argc, argv); \
    char fps_name[STRINGMAXLEN_FPS_NAME] = ""; \
    strncpy(fps_name, \
            (APP_INFO).fps_name, \
            STRINGMAXLEN_FPS_NAME - 1); \
    char arg_fps_name[STRINGMAXLEN_FPS_NAME] \
        = ""; \
    int use_tmux = 0; \
    int use_procinfo = 0; \
    int show_help = 0; \
    int show_h1 = 0; \
    int show_help_color = 1; \
    char *command = NULL; \
    char *keywords = NULL; \
    char *description = NULL; \
    char *colon_pos = NULL; \
    FPS_CLI_BINDING my_bindings_[] = { \
        PARAMS_MACRO(FPS_X_BINDING) \
    }; \
    int nb_bindings_ = sizeof(my_bindings_) \
                     / sizeof(FPS_CLI_BINDING); \
    CLICMDARGDEF farg_[] = { \
        PARAMS_MACRO(FPS_X_FARG) \
    }; \
    (void) farg_; \
    for (int i = 1; i < argc; i++) { \
        if (strcmp(argv[i], "-h") == 0 || \
            strcmp(argv[i], "--help") == 0) { \
            show_help = 1; \
        } else if ( \
            strcmp(argv[i], "-h1") == 0 || \
            strcmp(argv[i], \
                   "--help-oneline") == 0) { \
            show_h1 = 1; \
        } else if (strcmp(argv[i], "-hc") == 0 ||\
            strcmp(argv[i], \
                   "--help-color") == 0) { \
            show_help = 1; \
            show_help_color = 1; \
        } else if (strcmp(argv[i], \
                   "-hnc") == 0 || \
            strcmp(argv[i], \
                   "--help-no-color") == 0) { \
            show_help = 1; \
            show_help_color = 0; \
        } else if (strcmp(argv[i], \
                   "-tmux") == 0) { \
            use_tmux = 1; \
        } else if (strcmp(argv[i], \
                   "-procinfo") == 0 || \
            strcmp(argv[i], \
                   "--procinfo") == 0) { \
            use_procinfo = 1; \
        } else if ((strcmp(argv[i], "-k") == 0 ||\
            strcmp(argv[i], \
                   "--keywords") == 0) \
            && i + 1 < argc) { \
            keywords = argv[++i]; \
        } else if ((strcmp(argv[i], "-d") == 0 ||\
            strcmp(argv[i], \
                   "--description") == 0) \
            && i + 1 < argc) { \
            description = argv[++i]; \
        } else if ((strcmp(argv[i], "-n") == 0 ||\
            strcmp(argv[i], \
                   "--name") == 0) \
            && i + 1 < argc) { \
            strncpy(arg_fps_name, argv[++i], \
                    STRINGMAXLEN_FPS_NAME - 1); \
        } else if (command == NULL) { \
            command = argv[i]; \
            if ((colon_pos = strchr(command, \
                                    ':')) \
                != NULL) { \
                *colon_pos = '\0'; \
                strncpy(arg_fps_name, command, \
                    STRINGMAXLEN_FPS_NAME - 1); \
                command = colon_pos + 1; \
            } \
        } \
    } \
    if (command == NULL) { \
        command = "run"; \
    } else { \
        if (strcmp(command, "fpsinit") != 0 && \
            strcmp(command, "fps") != 0 && \
            strcmp(command, "fpslist") != 0 && \
            strcmp(command, "confstart") != 0 && \
            strcmp(command, "confstep") != 0 && \
            strcmp(command, "confstop") != 0 && \
            strcmp(command, "runstart") != 0 && \
            strcmp(command, "runstop") != 0 && \
            strcmp(command, "exec") != 0 && \
            strcmp(command, "set") != 0 && \
            strcmp(command, "run") != 0) { \
            fprintf(stderr, \
                    COLORERROR "Error:" \
                    COLORRESET " '%s' is not a" \
                    " valid command. Run with" \
                    " -h for help.\n", \
                    command); \
            return 1; \
        } \
    } \
    if (strlen(arg_fps_name) > 0) { \
        strncpy(fps_name, arg_fps_name, \
                STRINGMAXLEN_FPS_NAME - 1); \
    } \
    (void)keywords; (void)description; \
    if (show_h1) { \
        printf("%s\n", \
               (APP_INFO).description); \
        return 0; \
    } \
    if (show_help || (argc < 2)) { \
        if (show_help_color) { \
            printf("\n" COLORHEADER "Usage:" \
                   COLORRESET " %s " \
                   COLOROPTION "[fpsname:]" \
                   COLORRESET COLORCOMMAND \
                   "<Command>" COLORRESET " " \
                   COLOROPTION "[Options]" \
                   COLORRESET "\n", argv[0]); \
            printf("  Compiled: %s %s\n\n", \
                   __DATE__, __TIME__); \
            printf(COLORHEADER "Description:" \
                   COLORRESET "\n  %s\n\n", \
                   (APP_INFO).description); \
        } else { \
            printf("\nUsage: %s [fpsname:]" \
                   "<Command> [Options]\n", \
                   argv[0]); \
            printf("  Compiled: %s %s\n\n", \
                   __DATE__, __TIME__); \
            printf("Description:\n  %s\n\n", \
                   (APP_INFO).description); \
        } \
        if (show_help_color) \
            printf(COLORHEADER "Commands:" \
                   COLORRESET "\n"); \
        else printf("Commands:\n"); \
        if (show_help_color) { \
            printf("  " COLORCOMMAND "fpsinit" \
                   COLORRESET \
                   "    Create the FPS.\n"); \
            printf("  " COLORCOMMAND "fps" \
                   COLORRESET \
                   "        Print FPS " \
                   "content.\n"); \
            printf("  " COLORCOMMAND "fpslist" \
                   COLORRESET \
                   "    List matching " \
                   "FPS instances.\n"); \
            printf("  " COLORCOMMAND "confstart" \
                   COLORRESET \
                   "  Configuration " \
                   "loop.\n"); \
            printf("  " COLORCOMMAND "confstep" \
                   COLORRESET \
                   "   Single config " \
                   "step.\n"); \
            printf("  " COLORCOMMAND "confstop" \
                   COLORRESET \
                   "   Stop config " \
                   "loop.\n"); \
            printf("  " COLORCOMMAND "runstart" \
                   COLORRESET \
                   "   Main processing " \
                   "loop.\n"); \
            printf("  " COLORCOMMAND "runstop" \
                   COLORRESET \
                   "    Stop processing " \
                   "loop.\n"); \
            printf("  " COLORCOMMAND "set" \
                   COLORRESET " " \
                   COLOROPTION "[args]" \
                   COLORRESET \
                   "  Set positional args" \
                   " (. to skip).\n"); \
            printf("  " COLORCOMMAND "exec" \
                   COLORRESET " " \
                   COLOROPTION "[args]" \
                   COLORRESET \
                   " Auto-init + set" \
                   " args + run.\n\n"); \
        } else { \
            printf("  fpsinit    Create the " \
                   "FPS.\n"); \
            printf("  fps        Print FPS " \
                   "content.\n"); \
            printf("  fpslist    List matching " \
                   "FPS instances.\n"); \
            printf("  confstart  Configuration " \
                   "loop.\n"); \
            printf("  confstep   Single config " \
                   "step.\n"); \
            printf("  confstop   Stop config " \
                   "loop.\n"); \
            printf("  runstart   Main processing" \
                   " loop.\n"); \
            printf("  runstop    Stop processing" \
                   " loop.\n"); \
            printf("  set [args] Set positional" \
                   " args (. to skip).\n"); \
            printf("  exec [args] Auto-init +" \
                   " set args + run.\n\n"); \
        } \
        int col_kw_w = 7; \
        int col_tp_w = 4; \
        int col_df_w = 7; \
        PARAMS_MACRO(X_HELP_MEASURE_V2) \
        if (show_help_color) \
            printf(COLORHEADER \
                   "CLI arguments:" \
                   COLORRESET "\n"); \
        else printf("CLI arguments:\n"); \
        printf("  %-3s %-*s %-*s %-*s %s\n", \
               "Idx", \
               col_kw_w, "Keyword", \
               col_tp_w, "Type", \
               col_df_w, "Default", \
               "Description"); \
        printf("  %-3s %-*s %-*s %-*s %s\n", \
               "---", \
               col_kw_w, "-------", \
               col_tp_w, "----", \
               col_df_w, "-------", \
               "-----------"); \
        int CLIargcnt = 0; \
        (void) CLIargcnt; \
        PARAMS_MACRO(X_HELP_PRINT_V2) \
        printf("\n"); \
        if (show_help_color) \
            printf(COLOROPTION "Run " \
                   COLORCOMMAND \
                   "milk-fpsexec-help" \
                   COLOROPTION \
                   " for detailed FPS " \
                   "framework info." \
                   COLORRESET "\n\n"); \
        else \
            printf("Run milk-fpsexec-help" \
                   " for detailed FPS " \
                   "framework info." \
                   "\n\n"); \
        return 0; \
    } \
    if (command == NULL) { \
        fprintf(stderr, "Error: Missing " \
                "command argument.\n"); \
        return 1; \
    } \
    if (strcmp(command, "exec") != 0) \
        printf("FPS " COLORCOMMAND "%s" \
               COLORRESET " %s\n", \
               fps_name, command); \
    if (strcmp(command, "fps") == 0) { \
        FUNCTION_PARAMETER_STRUCT fps; \
        if (function_parameter_struct_connect( \
                fps_name, &fps, \
                FPSCONNECT_SIMPLE) == -1) { \
            fprintf(stderr, \
                    "Error: cannot connect to " \
                    "FPS '%s'.\n", fps_name); \
            return 1; \
        } \
        function_parameter_print_info( \
            &fps, 0, 0); \
        function_parameter_struct_disconnect( \
            &fps); \
        return 0; \
    } else if (strcmp(command, \
                      "fpslist") == 0) { \
        FUNCTION_PARAMETER_STRUCT *fpsarray = \
            (FUNCTION_PARAMETER_STRUCT *) \
            calloc(NB_FPS_MAX, \
                   sizeof( \
                   FUNCTION_PARAMETER_STRUCT));\
        if (fpsarray == NULL) return 1; \
        for (int i = 0; i < NB_FPS_MAX; i++) \
            fpsarray[i].SMfd = -1; \
        KEYWORD_TREE_NODE *keywnode = \
            (KEYWORD_TREE_NODE *) \
            calloc(NB_KEYWNODE_MAX, \
                   sizeof(KEYWORD_TREE_NODE)); \
        if (keywnode == NULL) { \
            free(fpsarray); return 1; } \
        int NBkwn = 0, NBfps = 0; \
        long NBpindex = 0; \
        functionparameter_scan_fps(0, "_ALL", \
            fpsarray, keywnode, \
            &NBkwn, &NBfps, &NBpindex, 0); \
        if (NBfps > 0) { \
            char *eb = strrchr(argv[0], '/'); \
            if (eb) eb++; else eb = argv[0]; \
            int found = 0; \
            for (int i = 0; i < NBfps; i++) { \
                char *fb = strrchr( \
                    fpsarray[i].md \
                        ->execfullpath, '/'); \
                if (fb) fb++; \
                else fb = fpsarray[i].md \
                    ->execfullpath; \
                if (strcmp(eb, fb) == 0) { \
                    if (!found) { \
                        printf("%-30s %-10s " \
                               "%s\n", \
                               "FPS Name", \
                               "Status", \
                               "Description"); \
                        printf("----------" \
                               "----------" \
                               "----------" \
                               "----------\n");\
                        found = 1; \
                    } \
                    char ss[32] = "UNKNOWN"; \
                    if (fpsarray[i].md->status \
                        & FUNCTION_PARAMETER_STRUCT_STATUS_CONF) \
                        strcpy(ss, "CONF"); \
                    else if (fpsarray[i].md \
                        ->status \
                        & FUNCTION_PARAMETER_STRUCT_STATUS_RUN) \
                        strcpy(ss, "RUN"); \
                    printf("%-30s %-10s %s\n", \
                           fpsarray[i].md->name,\
                           ss, \
                           fpsarray[i].md \
                               ->description); \
                } \
                function_parameter_struct_disconnect( \
                    &fpsarray[i]); \
            } \
            if (!found) \
                printf("No matching FPS for " \
                       "'%s'.\n", eb); \
        } else { \
            printf("No FPS found.\n"); \
        } \
        free(keywnode); \
        free(fpsarray); \
        return 0; \
    } \
    if (use_tmux) { \
        char path[1024]; \
        if (functionparameter_FPS_get_executable_path( \
                path, sizeof(path)) == NULL) { \
            if (realpath(argv[0], path) == NULL)\
                strncpy(path, argv[0], 1023); \
        } \
        char name_arg[256] = ""; \
        if (strcmp(fps_name, \
                  (APP_INFO).fps_name) != 0) { \
            snprintf(name_arg, \
                     sizeof(name_arg), \
                     " %s:%s", \
                     fps_name, command); \
        } else { \
            snprintf(name_arg, \
                     sizeof(name_arg), \
                     " %s", command); \
        } \
        functionparameter_FPS_tmux_standalone_setup( \
            fps_name); \
        if (strcmp(command, "exec") == 0) { \
            { \
                FUNCTION_PARAMETER_STRUCT fc_; \
                if (fps_name[0] != '_') { \
                    if (function_parameter_struct_connect( \
                            fps_name, &fc_, \
                            FPSCONNECT_SIMPLE) \
                        == -1) \
                    { \
                        printf("FPS " COLORCOMMAND \
                               "%s" COLORRESET \
                               " exec -> " \
                               "\033[33m" "NEW" \
                               COLORRESET "\n", \
                               fps_name); \
                        fps_generic_init(fps_name,\
                            (FPS_APP_INFO *) \
                            &(APP_INFO), \
                            my_bindings_, \
                            nb_bindings_, \
                            use_procinfo); \
                    } else { \
                        printf("FPS " COLORCOMMAND \
                               "%s" COLORRESET \
                               " exec -> " \
                               COLORCOMMAND \
                               "REUSE" \
                               COLORRESET "\n", \
                               fps_name); \
                        function_parameter_struct_disconnect( \
                            &fc_); \
                    } \
                } \
            } \
            char run_arg[512] = ""; \
            if (strcmp(fps_name, \
                      (APP_INFO).fps_name) != 0)\
            { \
                snprintf(run_arg, \
                         sizeof(run_arg), \
                         " %s:runstart", \
                         fps_name); \
            } else { \
                snprintf(run_arg, \
                         sizeof(run_arg), \
                         " runstart"); \
            } \
            if (use_procinfo) { \
                strncat(run_arg, " -procinfo", \
                        sizeof(run_arg) \
                        - strlen(run_arg) - 1); \
            } \
            functionparameter_FPS_tmux_send_dispatch( \
                fps_name, "runstart", path, \
                run_arg); \
            return 0; \
        } \
        if (functionparameter_FPS_tmux_send_dispatch( \
                fps_name, command, path, \
                name_arg) == 0) { \
            return 0; \
        } \
        if (strcmp(command, "fpsinit") == 0) { \
            fps_generic_init(fps_name, \
                (FPS_APP_INFO *)&(APP_INFO), \
                my_bindings_, nb_bindings_, \
                use_procinfo); \
        } \
        return 0; \
    } \
    if (strcmp(command, "fpsinit") == 0) { \
        return fps_generic_init(fps_name, \
            (FPS_APP_INFO *)&(APP_INFO), \
            my_bindings_, nb_bindings_, \
            use_procinfo); \
    } else if (strcmp(command, \
                      "confstart") == 0) { \
        return fps_generic_conf_cb( \
            fps_name, 1, CONFCHECK_FN); \
    } else if (strcmp(command, \
                      "confstep") == 0) { \
        return fps_generic_conf_cb( \
            fps_name, 0, CONFCHECK_FN); \
    } else if (strcmp(command, \
                      "confstop") == 0) { \
        return fps_generic_confstop(fps_name); \
    } else if (strcmp(command, "set") == 0) { \
        FUNCTION_PARAMETER_STRUCT fps; \
        if (function_parameter_struct_connect( \
                fps_name, &fps, \
                FPSCONNECT_SIMPLE) == -1) { \
            fprintf(stderr, \
                    "Error: FPS '%s' not found." \
                    " Run fpsinit first.\n", \
                    fps_name); \
            return 1; \
        } \
        fps_process_cli_and_sync( \
            &fps, farg_, \
            my_bindings_, nb_bindings_); \
        function_parameter_struct_disconnect( \
            &fps); \
        printf("FPS " COLORCOMMAND "%s" \
               COLORRESET " set done\n", \
               fps_name); \
        return 0; \
    } else if (strcmp(command, "exec") == 0) { \
        /* Auto-init if FPS doesn't exist yet, \
         * then run. fps_name goes to shared mem \
         * when name lacks _ prefix. */ \
        { \
            FUNCTION_PARAMETER_STRUCT fps_chk_; \
            if (fps_name[0] != '_') { \
                if (function_parameter_struct_connect( \
                        fps_name, &fps_chk_, \
                        FPSCONNECT_SIMPLE) == -1) \
                { \
                    printf("FPS " COLORCOMMAND \
                           "%s" COLORRESET \
                           " exec -> " \
                           "\033[33m" "NEW" \
                           COLORRESET "\n", \
                           fps_name); \
                    fps_generic_init(fps_name, \
                        (FPS_APP_INFO *)&(APP_INFO), \
                        my_bindings_, nb_bindings_, \
                        use_procinfo); \
                } else { \
                    printf("FPS " COLORCOMMAND \
                           "%s" COLORRESET \
                           " exec -> " \
                           COLORCOMMAND "REUSE" \
                           COLORRESET "\n", \
                           fps_name); \
                    function_parameter_struct_disconnect( \
                        &fps_chk_); \
                } \
            } \
        } \
        return fps_generic_run(fps_name, \
            (FPS_APP_INFO *)&(APP_INFO), \
            farg_, my_bindings_, nb_bindings_, \
            COMPUTE_FN); \
    } else if (strcmp(command, \
                      "runstart") == 0 || \
               strcmp(command, "run") == 0) { \
        return fps_generic_run(fps_name, \
            (FPS_APP_INFO *)&(APP_INFO), \
            farg_, my_bindings_, nb_bindings_, \
            COMPUTE_FN); \
    } else if (strcmp(command, \
                      "runstop") == 0) { \
        return fps_generic_runstop(fps_name); \
    } \
    fprintf(stderr, \
            "Invalid command: %s\n", command); \
    return 1; \
}


/**
 * @brief V2 standalone macro with confcheck.
 *
 * Usage:
 *   FPS_MAIN_STANDALONE_V2_CONFCHECK(
 *       app_info, PARAMS, compute_fn,
 *       customCONFcheck)
 */
#define FPS_MAIN_STANDALONE_V2_CONFCHECK( \
    APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
    CONFCHECK_FN) \
    _FPS_MAIN_STANDALONE_V2_IMPL( \
        APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
        CONFCHECK_FN)


/**
 * @brief V2 standalone macro (no confcheck).
 *
 * For backward compatibility with existing
 * callers.  Passes NULL as confcheck.
 */
#define FPS_MAIN_STANDALONE_V2( \
    APP_INFO, PARAMS_MACRO, COMPUTE_FN) \
    _FPS_MAIN_STANDALONE_V2_IMPL( \
        APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
        NULL)

/**
 * @brief Standard initialization preamble for FPSINIT function
 */
#define FPS_INIT_STD_PREAMBLE(VARfps, VARfps_name, VARkeywords, VARdescription, VARhelptext) \
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
                functionparameter_CheckParametersAll(&fps); \
            } \
            usleep(10000); \
        } \
    } else { \
        printf("Running single configuration step for '%s'\n", VARfps_name); \
        fps = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_FPSINIT); \
        BLOCK_VAR_MAP \
        function_parameter_FPCONFloopstep(&fps); \
        BLOCK_VALIDATE \
        functionparameter_CheckParametersAll(&fps); \
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
    processinfo_waitoninputstream_init(VARprocessinfo, VARinput_image, ((VARinput_image) != NULL) ? PROCESSINFO_TRIGGERMODE_SEMAPHORE : PROCESSINFO_TRIGGERMODE_IMMEDIATE, -1); \
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