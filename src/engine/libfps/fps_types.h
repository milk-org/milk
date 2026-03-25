/**
 * @file fps_types.h
 * @brief FPS type definitions, structs, constants, and flags.
 *
 * Include this header when you only need FPS type
 * definitions (FUNCTION_PARAMETER, FUNCTION_PARAMETER_STRUCT,
 * flags, constants) without pulling in the 35+ API
 * sub-headers and convenience macros from fps.h.
 *
 * This is a subset of fps.h — anything that includes
 * fps.h already has everything in this file.
 */

#ifndef FPS_TYPES_H
#define FPS_TYPES_H

#include <time.h>
#include <sys/types.h>
#include <stdint.h>
#include "cmdsettings.h"

/* ===== String length constants ===== */

#define STRINGMAXLEN_FPS_LOGMSG  1000
#define STRINGMAXLEN_FPS_CMDLINE 1000

#define NB_FPS_MAX 100
#define MAXNBLEVELS 20

/* ===== FPS connect modes ===== */

#define FPSCONNECT_SIMPLE 0
#define FPSCONNECT_CONF   1
#define FPSCONNECT_RUN    2

/* ===== FPS command codes (uint32_t) ===== */

#define FPSCMDCODE_CONFSTART     0x00000001
#define FPSCMDCODE_CONFSTOP      0x00000002
#define FPSCMDCODE_FPSINIT       0x00000004
#define FPSCMDCODE_FPSINITCREATE 0x00000008
#define FPSCMDCODE_RUNSTART      0x00000010
#define FPSCMDCODE_RUNSTOP       0x00000020
#define FPSCMDCODE_TMUXSTART     0x00000100
#define FPSCMDCODE_TMUXSTOP      0x00000200
#define FPSCMDCODE_IGNORE        0x00001000

/* ===== Parameter keyword limits ===== */

#define FUNCTION_PARAMETER_KEYWORD_STRMAXLEN 64
#define FUNCTION_PARAMETER_KEYWORD_MAXLEVEL  20

/* ===== Parameter type codes ===== */

#define FPTYPE_UNDEF   0x00000001
#define FPTYPE_INT32   0x00000002
#define FPTYPE_UINT32  0x00000004
#define FPTYPE_INT64   0x00000008
#define FPTYPE_UINT64  0x00000010
#define FPTYPE_FLOAT32 0x00000020
#define FPTYPE_FLOAT64 0x00000040

#define FPTYPE_PID      0x00000080
#define FPTYPE_TIMESPEC 0x00000100

#define FPTYPE_FILENAME     0x00000200
#define FPTYPE_FITSFILENAME 0x00000400
#define FPTYPE_EXECFILENAME 0x00000800

#define FPTYPE_DIRNAME 0x00001000

#define FPTYPE_STREAMNAME 0x00002000

#define FPTYPE_STRING 0x00004000

#define FPTYPE_ONOFF  0x00008000

#define FPTYPE_PROCESS 0x00010000

#define FPTYPE_FPSNAME 0x00020000

#define FPTYPE_STRING_NOT_STREAM 0x00040000

#define FPTYPE_IS_STRING(type) \
    ((type == FPTYPE_STRING) || \
     (type == FPTYPE_FILENAME) || \
     (type == FPTYPE_FITSFILENAME) || \
     (type == FPTYPE_EXECFILENAME) || \
     (type == FPTYPE_DIRNAME) || \
     (type == FPTYPE_STREAMNAME) || \
     (type == FPTYPE_PROCESS) || \
     (type == FPTYPE_FPSNAME) || \
     (type == FPTYPE_STRING_NOT_STREAM))

#define STRINGMAXLEN_FPSTYPE  20

#define FUNCTION_PARAMETER_DESCR_STRMAXLEN 64
#define FUNCTION_PARAMETER_STRMAXLEN       64

/* ===== Parameter flags (uint64_t) ===== */

#define FPFLAG_ACTIVE       0x0000000000000001
#define FPFLAG_USED         0x0000000000000002
#define FPFLAG_VISIBLE      0x0000000000000004

#define FPFLAG_WRITE        0x0000000000000010
#define FPFLAG_WRITECONF    0x0000000000000020
#define FPFLAG_WRITERUN     0x0000000000000040
#define FPFLAG_WRITESTATUS  0x0000000000000080

#define FPFLAG_LOG          0x0000000000000100
#define FPFLAG_SAVEONCHANGE 0x0000000000000200
#define FPFLAG_SAVEONCLOSE  0x0000000000000400

#define FPFLAG_IMPORTED     0x0000000000001000
#define FPFLAG_FEEDBACK     0x0000000000002000
#define FPFLAG_ONOFF        0x0000000000004000

#define FPFLAG_CHECKINIT    0x0000000000010000
#define FPFLAG_MINLIMIT     0x0000000000020000
#define FPFLAG_MAXLIMIT     0x0000000000040000
#define FPFLAG_ERROR        0x0000000000080000

#define STREAM_LOAD_SOURCE_NOTFOUND        0
#define STREAM_LOAD_SOURCE_NOTFOUND_STRING \
    "STREAM_LOAD_SOURCE_NOTFOUND"

#define STREAM_LOAD_SOURCE_LOCALMEM        1
#define STREAM_LOAD_SOURCE_LOCALMEM_STRING \
    "STREAM_LOAD_SOURCE_LOCALMEM"

#define STREAM_LOAD_SOURCE_SHAREMEM        2
#define STREAM_LOAD_SOURCE_SHAREMEM_STRING \
    "STREAM_LOAD_SOURCE_SHAREMEM"

#define STREAM_LOAD_SOURCE_CONFFITS        3
#define STREAM_LOAD_SOURCE_CONFFITS_STRING \
    "STREAM_LOAD_SOURCE_CONFFITS"

#define STREAM_LOAD_SOURCE_CONFNAME        4
#define STREAM_LOAD_SOURCE_CONFNAME_STRING \
    "STREAM_LOAD_SOURCE_CONFNAME"

#define STREAM_LOAD_SOURCE_NULL        5
#define STREAM_LOAD_SOURCE_NULL_STRING \
    "STREAM_LOAD_SOURCE_NULL"

#define STREAM_LOAD_SOURCE_EXITFAILURE        -1
#define STREAM_LOAD_SOURCE_EXITFAILURE_STRING \
    "STREAM_LOAD_SOURCE_EXITFAILURE"

#define FPFLAG_STREAM_LOAD_FORCE_LOCALMEM \
    0x0000000000100000
#define FPFLAG_STREAM_LOAD_FORCE_SHAREMEM \
    0x0000000000200000
#define FPFLAG_STREAM_LOAD_FORCE_CONFFITS \
    0x0000000000400000
#define FPFLAG_STREAM_LOAD_FORCE_CONFNAME \
    0x0000000000800000

#define FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM \
    0x0000000001000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM \
    0x0000000002000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS \
    0x0000000004000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME \
    0x0000000008000000

#define FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM \
    0x0000000010000000
#define FPFLAG_STREAM_LOAD_UPDATE_CONFFITS \
    0x0000000020000000

#define FPFLAG_FILE_CONF_REQUIRED \
    0x0000000040000000
#define FPFLAG_FILE_RUN_REQUIRED \
    0x0000000080000000

#define FPFLAG_FPS_CONF_REQUIRED \
    0x0000000040000000
#define FPFLAG_FPS_RUN_REQUIRED \
    0x0000000080000000

#define FPFLAG_STREAM_CONF_REQUIRED \
    0x0000000040000000
#define FPFLAG_STREAM_RUN_REQUIRED \
    0x0000000080000000

#define FPFLAG_STREAM_ENFORCE_DATATYPE \
    0x0000000100000000

#define FPFLAG_STREAM_TEST_DATATYPE_UINT8 \
    0x0000000200000000
#define FPFLAG_STREAM_TEST_DATATYPE_INT8 \
    0x0000000400000000
#define FPFLAG_STREAM_TEST_DATATYPE_UINT16 \
    0x0000000800000000
#define FPFLAG_STREAM_TEST_DATATYPE_INT16 \
    0x0000001000000000
#define FPFLAG_STREAM_TEST_DATATYPE_UINT32 \
    0x0000002000000000
#define FPFLAG_STREAM_TEST_DATATYPE_INT32 \
    0x0000004000000000
#define FPFLAG_STREAM_TEST_DATATYPE_UINT64 \
    0x0000008000000000
#define FPFLAG_STREAM_TEST_DATATYPE_INT64 \
    0x0000010000000000

#define FPFLAG_PRIMARY_CLI_INPUT \
    0x0000000000000008
#define FPFLAG_CLI_INPUT \
    0x0000020000000000
#define FPFLAG_STREAM_TEST_DATATYPE_HALF \
    0x0000020000000000
#define FPFLAG_STREAM_TEST_DATATYPE_FLOAT \
    0x0000040000000000
#define FPFLAG_STREAM_TEST_DATATYPE_DOUBLE \
    0x0000080000000000

#define FPFLAG_STREAM_ENFORCE_1D \
    0x0000100000000000
#define FPFLAG_STREAM_ENFORCE_2D \
    0x0000200000000000
#define FPFLAG_STREAM_ENFORCE_3D \
    0x0000400000000000
#define FPFLAG_STREAM_ENFORCE_XSIZE \
    0x0008000000000000
#define FPFLAG_STREAM_ENFORCE_YSIZE \
    0x0010000000000000
#define FPFLAG_STREAM_ENFORCE_ZSIZE \
    0x0020000000000000

#define FPFLAG_CHECKSTREAM \
    0x0040000000000000
#define FPFLAG_STREAM_MEMLOADREPORT \
    0x0080000000000000

/* ===== Convenience flag combos ===== */

#define FPFLAG_DEFAULT_INPUT \
    (FPFLAG_ACTIVE | FPFLAG_USED | FPFLAG_VISIBLE | \
     FPFLAG_WRITE | FPFLAG_WRITECONF | \
     FPFLAG_SAVEONCHANGE | FPFLAG_FEEDBACK | \
     FPFLAG_WRITESTATUS)
#define FPFLAG_DEFAULT_OUTPUT \
    (FPFLAG_ACTIVE | FPFLAG_USED | FPFLAG_VISIBLE)
#define FPFLAG_DEFAULT_INPUT_STREAM \
    (FPFLAG_DEFAULT_INPUT | \
     FPFLAG_STREAM_RUN_REQUIRED | \
     FPFLAG_CHECKSTREAM)
#define FPFLAG_DEFAULT_OUTPUT_STREAM \
    (FPFLAG_DEFAULT_INPUT | FPFLAG_CHECKSTREAM)
#define FPFLAG_DEFAULT_STATUS \
    (FPFLAG_ACTIVE | FPFLAG_USED | FPFLAG_VISIBLE)

#define FUNCTION_PARAMETER_NBPARAM_DEFAULT 100

/* ===== FPS structs ===== */

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

    char keywordfull[
        FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[
        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][
        FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel;

    char keywordfrom[
        FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];

    char description[
        FUNCTION_PARAMETER_DESCR_STRMAXLEN];

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
        char string[2][
            FUNCTION_PARAMETER_STRMAXLEN];
    } val;

    union
    {
        FUNCTION_PARAMETER_SUBINFO_STREAM stream;
        FUNCTION_PARAMETER_SUBINFO_FPS    fps;
    } info;

    long cnt0;
    uint64_t value_cnt;
    int cli_index;
} FUNCTION_PARAMETER;

/* ===== FPS metadata and main struct ===== */

#define STRINGMAXLEN_FPS_NAME 100

#define STRINGMAXLEN_PROCESSINFO_TMUXNAME 100

#define FUNCTION_PARAMETER_STRUCT_MSG_SIZE 500

#define FUNCTION_PARAMETER_STRUCT_STATUS_CONF     0x0001
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUN      0x0002
#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF  0x0010
#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN   0x0020
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP  0x0100
#define FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK  0x0200
#define FUNCTION_PARAMETER_STRUCT_STATUS_CHECKERR 0x0400
#define FUNCTION_PARAMETER_STRUCT_STATUS_SAVE     0x0800

#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF 0x1000
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN  0x2000
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL 0x4000

#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN  0x0001
#define FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE   0x0004
#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED  0x0008

#define FPS_NB_MSG                        100
#define FUNCTION_PARAMETER_STRUCT_MSG_LEN 500

#define FPS_MSG_FLAG_NOTINITIALIZED 0x0001
#define FPS_MSG_FLAG_BELOWMIN       0x0002
#define FPS_MSG_FLAG_ABOVEMAX       0x0004
#define FPS_MSG_FLAG_ERROR          0x0008
#define FPS_MSG_FLAG_INFO           0x0010

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
    char keywordarray[
        FPS_KEYWORDARRAY_STRMAXLEN];
    char workdir[FPS_CWD_STRLENMAX];
    char datadir[FPS_DIR_STRLENMAX];
    char confdir[FPS_DIR_STRLENMAX];
    char sourcefname[FPS_SRCDIR_STRLENMAX];
    int sourceline;
    char pname[FPS_PNAME_STRMAXLEN];
    char callprogname[
        FPS_CALLPROGNAME_STRMAXLEN];
    char callfuncname[
        FPS_CALLFUNCNAME_STRMAXLEN];
    char tmuxname[
        STRINGMAXLEN_PROCESSINFO_TMUXNAME];
    char nameindexW[16][10];
    int  NBnameindex;
    pid_t           confpid;
    struct timespec confpidstarttime;
    pid_t           runpid;
    struct timespec runpidstarttime;
    int  NBmodule;
    char modulename[FPS_MAXNB_MODULE][
        FPS_MODULE_STRMAXLEN];
    uint64_t signal;
    uint64_t confwaitus;
    uint32_t status;
    uint64_t processinfo_change_cnt;
    long NBparamMAX;
    char message[FPS_NB_MSG][
        FUNCTION_PARAMETER_STRUCT_MSG_LEN];
    int msgpindex[FPS_NB_MSG];
    uint32_t msgcode[FPS_NB_MSG];
    long msgcnt;
    uint32_t conferrcnt;
} FUNCTION_PARAMETER_STRUCT_MD;

#define FPS_LOCALSTATUS_CONFLOOP 0x0001

typedef struct FUNCTION_PARAMETER_STRUCT
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

/* ===== Task scheduler types ===== */

#define NB_FPSCTRL_TASK_MAX       5000
#define NB_FPSCTRL_TASK_PURGESIZE 50

#define FPSTASK_STATUS_ACTIVE    0x0000000000000001
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
#define FPSTASK_FLAG_WAIT_FOR_FPS_NORUN \
    0x0000000000000004

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
    char     milkseq_name[64];
    void    *milkseq_state; // cast to MILKSEQ_STATE* where needed
} FPSCTRL_PROCESS_VARS;

#define NB_KEYWNODE_MAX 6000
#define MAX_NB_CHILD    3000

typedef struct
{
    char keywordfull[
        FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[
        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][
        FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel;
    int parent_index;
    int NBchild;
    int child[MAX_NB_CHILD];
    int leaf;
    int fpsindex;
    int pindex;
} KEYWORD_TREE_NODE;

#endif /* FPS_TYPES_H */
