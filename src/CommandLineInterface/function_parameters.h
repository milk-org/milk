/**
 * @file function_parameters.h
 * @brief Tools to help expose and control function parameters
 *
 * @see @ref page_FunctionParameterStructure
 *
 *
 */

#ifndef FUNCTION_PARAMETERS_H

#define FUNCTION_PARAMETERS_H





#define STRINGMAXLEN_FPS_LOGMSG       1000
#define STRINGMAXLEN_FPS_CMDLINE      1000



#define NB_FPS_MAX 100

#define MAXNBLEVELS 20

#define FPSCONNECT_SIMPLE 0
#define FPSCONNECT_CONF   1
#define FPSCONNECT_RUN    2


// CMCODE type is uint32_t
#define FPSCMDCODE_CONFSTART          0x00000001  // start configuration process
#define FPSCMDCODE_CONFSTOP           0x00000002  // stop configuration process
#define FPSCMDCODE_FPSINIT            0x00000004  // initialize FPS if does not exist
#define FPSCMDCODE_FPSINITCREATE      0x00000008  // (re-)create FPS even if it exists
#define FPSCMDCODE_RUNSTART           0x00000010  // start run process
#define FPSCMDCODE_RUNSTOP            0x00000020  // stop run process
#define FPSCMDCODE_TMUXSTART          0x00000100  // start tmux sessions
#define FPSCMDCODE_TMUXSTOP           0x00000200  // stop tmux sessions


// function can use this structure to expose parameters for external control or monitoring
// the structure describes how user can interact with parameter, so it allows for control GUIs to connect to parameters

#define FUNCTION_PARAMETER_KEYWORD_STRMAXLEN   64
#define FUNCTION_PARAMETER_KEYWORD_MAXLEVEL    20

// Note that notation allows parameter to have more than one type
// ... to be used with caution: most of the time, use type exclusively
// type is uint32_t

#define FPTYPE_UNDEF         0x00000001
#define FPTYPE_INT32         0x00000002
#define FPTYPE_UINT32        0x00000004
#define FPTYPE_INT64         0x00000008
#define FPTYPE_UINT64        0x00000010
#define FPTYPE_FLOAT32       0x00000020
#define FPTYPE_FLOAT64       0x00000040

#define FPTYPE_PID           0x00000080
#define FPTYPE_TIMESPEC      0x00000100

#define FPTYPE_FILENAME      0x00000200  // generic filename
#define FPTYPE_FITSFILENAME  0x00000400  // FITS file
#define FPTYPE_EXECFILENAME  0x00000800  // executable file

#define FPTYPE_DIRNAME       0x00001000  // directory name
#define FPTYPE_STREAMNAME    0x00002000  // stream name -> process may load from shm if required. See loading stream section below and associated flags
#define FPTYPE_STRING        0x00004000  // generic string
#define FPTYPE_ONOFF         0x00008000  // uses ONOFF bit flag, string[0] and string[1] for OFF and ON descriptions respectively. setval saves ONOFF as integer
#define FPTYPE_PROCESS       0x00010000


#define FPTYPE_FPSNAME       0x00020000 // connection to another FPS



#define FUNCTION_PARAMETER_DESCR_STRMAXLEN   64
#define FUNCTION_PARAMETER_STRMAXLEN         64



// STATUS FLAGS

// parameter use and visibility
#define FPFLAG_ACTIVE        0x0000000000000001    // is this entry registered ?
#define FPFLAG_USED          0x0000000000000002    // is this entry used ? if not, skip all checks
#define FPFLAG_VISIBLE       0x0000000000000004    // is this entry visible (=displayed) ?

// write permission
#define FPFLAG_WRITE         0x0000000000000010    // is value writable when neither CONF and RUN are active
#define FPFLAG_WRITECONF     0x0000000000000020    // can user change value at configuration time ?
#define FPFLAG_WRITERUN      0x0000000000000040    // can user change value at run time ?
#define FPFLAG_WRITESTATUS   0x0000000000000080    // current write status (computed from above flags)

// logging and saving
#define FPFLAG_LOG           0x0000000000000100    // log on change
#define FPFLAG_SAVEONCHANGE  0x0000000000000200    // save to disk on change
#define FPFLAG_SAVEONCLOSE   0x0000000000000400    // save to disk on close

// special types
#define FPFLAG_IMPORTED      0x0000000000001000    // is this entry imported from another parameter ?
#define FPFLAG_FEEDBACK      0x0000000000002000    // is there a separate current value feedback ?
#define FPFLAG_ONOFF         0x0000000000004000    // bit controlled under TYPE_ONOFF

// parameter testing
#define FPFLAG_CHECKINIT     0x0000000000010000    // should parameter be initialized prior to function start ?
#define FPFLAG_MINLIMIT      0x0000000000020000    // enforce min limit
#define FPFLAG_MAXLIMIT      0x0000000000040000    // enforce max limit
#define FPFLAG_ERROR         0x0000000000080000    // is current parameter value OK ?



// if FPTYPE_STREAMNAME
// STREAM FLAGS: actions and tests related to streams

// The stream location may be in :
// --- convention : this is downstream ---
// [a]-LOCALMEM local process memory
// [b]-SHAREMEM system shared memory .. which may itself be a link to another shared memory
// [c]-CONFFITS fits file in conf: a file ./conf/shmim.<stream>.fits, which may itself be a link to another FITS file
// [d]-CONFNAME name of fits file configuration: a file ./conf/shmim.<stream>.fname.conf contains the name of the disk file to be loaded as the stream, relative to current running directory
// --- convention : this is upstream ---


// what is the source from which a stream was successfully loaded
#define STREAM_LOAD_SOURCE_NOTFOUND   0
#define STREAM_LOAD_SOURCE_NOTFOUND_STRING "STREAM_LOAD_SOURCE_NOTFOUND"

#define STREAM_LOAD_SOURCE_LOCALMEM  1
#define STREAM_LOAD_SOURCE_LOCALMEM_STRING "STREAM_LOAD_SOURCE_LOCALMEM"

#define STREAM_LOAD_SOURCE_SHAREMEM  2
#define STREAM_LOAD_SOURCE_SHAREMEM_STRING "STREAM_LOAD_SOURCE_SHAREMEM"

#define STREAM_LOAD_SOURCE_CONFFITS  3
#define STREAM_LOAD_SOURCE_CONFFITS_STRING "STREAM_LOAD_SOURCE_CONFFITS"

#define STREAM_LOAD_SOURCE_CONFNAME  4
#define STREAM_LOAD_SOURCE_CONFNAME_STRING "STREAM_LOAD_SOURCE_CONFNAME"

#define STREAM_LOAD_SOURCE_NULL  5
#define STREAM_LOAD_SOURCE_NULL_STRING "STREAM_LOAD_SOURCE_NULL"

#define STREAM_LOAD_SOURCE_EXITFAILURE   -1
#define STREAM_LOAD_SOURCE_EXITFAILURE_STRING "STREAM_LOAD_SOURCE_EXITFAILURE"

//
// The default policy is to look for the source location first in [a], then [b], etc..., until [d]
// Once source location is found, the downstream locations are updated. For example: search[a]; search[b], find[c]->update[b]->update[a]
//
//
//
// Important scripts (should be in PATH):
// - milkstreamlink  : build sym link between streams
// - milkFits2shm    : smart loading/updating of FITS to SHM
//
// loading CONF to SHM must use script milkFits2shm
//
//


// STREAM LOADING POLICY FLAGS
// These flags modify the default stream load policy
// Default load policy: FORCE flags = 0, SKIPSEARCH flags = 0, UPDATE flags = 0
//
// FORCE flags will force a location to be used and all downstream locations to be updated
// if the FORCE location does not exist, it will fail
// only one such flag should be specified. If several force flags are specified, the first one ((a) over (b)) will be considered
#define FPFLAG_STREAM_LOAD_FORCE_LOCALMEM        0x0000000000100000
#define FPFLAG_STREAM_LOAD_FORCE_SHAREMEM        0x0000000000200000
#define FPFLAG_STREAM_LOAD_FORCE_CONFFITS        0x0000000000400000
#define FPFLAG_STREAM_LOAD_FORCE_CONFNAME        0x0000000000800000

// SKIPSEARCH flags will skip search location
// multiple such flags can be specified
//
// Note that the FORCE flags have priority over the SKIPSEARCH flags
// If a FORCE flag is active, the SKIPSEARCH flags will be ignored
//
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM   0x0000000001000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM   0x0000000002000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS   0x0000000004000000
#define FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME   0x0000000008000000

// UPDATE flags will update upstream locations
#define FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM       0x0000000010000000
#define FPFLAG_STREAM_LOAD_UPDATE_CONFFITS       0x0000000020000000




// Additionally, the following flags specify what to do if stream properties do not match the required properties
//


#define FPFLAG_FILE_CONF_REQUIRED                0x0000000040000000  // file must exist for CONF process to proceed
#define FPFLAG_FILE_RUN_REQUIRED                 0x0000000080000000  // file must exist for RUN process to proceed
// note: we can reuse same codes

#define FPFLAG_FPS_CONF_REQUIRED                 0x0000000040000000  // file must exist for CONF process to proceed
#define FPFLAG_FPS_RUN_REQUIRED                  0x0000000080000000  // file must exist for RUN process to proceed

#define FPFLAG_STREAM_CONF_REQUIRED              0x0000000040000000  // stream has to be in MEM for CONF process to proceed
#define FPFLAG_STREAM_RUN_REQUIRED               0x0000000080000000  // stream has to be in MEM for RUN process to proceed




// Additional notes on load functions in AOloopControl_IOtools
//
/* AOloopControl_IOtools_2Dloadcreate_shmim( const char *name,
    const char *fname,
    long xsize,
    long ysize,
    float DefaultValue)
*/
//




#define FPFLAG_STREAM_ENFORCE_DATATYPE           0x0000000100000000  // enforce stream datatype
// stream type requirement: one of the following tests must succeed (OR) if FPFLAG_STREAM_ENFORCE_DATATYPE
// If creating image, the first active entry is used
#define FPFLAG_STREAM_TEST_DATATYPE_UINT8        0x0000000200000000  // test if stream of type UINT8   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT8         0x0000000400000000  // test if stream of type INT8    (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_UINT16       0x0000000800000000  // test if stream of type UINT16  (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT16        0x0000001000000000  // test if stream of type INT16   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_UINT32       0x0000002000000000  // test if stream of type UINT32  (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT32        0x0000004000000000  // test if stream of type INT32   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_UINT64       0x0000008000000000  // test if stream of type UINT64  (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_INT64        0x0000010000000000  // test if stream of type INT64   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_HALF         0x0000020000000000  // test if stream of type HALF    (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_FLOAT        0x0000040000000000  // test if stream of type FLOAT   (OR test)
#define FPFLAG_STREAM_TEST_DATATYPE_DOUBLE       0x0000080000000000  // test if stream of type DOUBLE  (OR test)

#define FPFLAG_STREAM_ENFORCE_1D                 0x0000100000000000  // enforce 1D image
#define FPFLAG_STREAM_ENFORCE_2D                 0x0000200000000000  // enforce 2D image
#define FPFLAG_STREAM_ENFORCE_3D                 0x0000400000000000  // enforce 3D image
#define FPFLAG_STREAM_ENFORCE_XSIZE              0x0008000000000000  // enforce X size
#define FPFLAG_STREAM_ENFORCE_YSIZE              0x0010000000000000  // enforce Y size
#define FPFLAG_STREAM_ENFORCE_ZSIZE              0x0020000000000000  // enforce Z size

#define FPFLAG_CHECKSTREAM                       0x0040000000000000  // check and display stream status in GUI
#define FPFLAG_STREAM_MEMLOADREPORT              0x0080000000000000  // Write stream load report (for debugging)








// PRE-ASSEMBLED DEFAULT FLAGS

// input parameter (used as default when adding entry)
#define FPFLAG_DEFAULT_INPUT            FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE|FPFLAG_WRITE|FPFLAG_WRITECONF|FPFLAG_SAVEONCHANGE|FPFLAG_FEEDBACK
#define FPFLAG_DEFAULT_OUTPUT           FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE
#define FPFLAG_DEFAULT_INPUT_STREAM     FPFLAG_DEFAULT_INPUT|FPFLAG_STREAM_RUN_REQUIRED|FPFLAG_CHECKSTREAM
#define FPFLAG_DEFAULT_OUTPUT_STREAM    FPFLAG_DEFAULT_INPUT|FPFLAG_CHECKSTREAM


// status parameters, no logging, read only
#define FPFLAG_DEFAULT_STATUS     FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE





#define FUNCTION_PARAMETER_NBPARAM_DEFAULT    100       // size of dynamically allocated array of parameters



typedef struct
{
    long      streamID; // if type is stream and MASK_CHECKSTREAM. For CONF only
    uint8_t   stream_atype;

    // these have two entries. First is actual/measured, second is required (0 if dimension not active)
    // tests are specified by flags FPFLAG_STREAM_ENFORCE_1D/2D/3D/XSIZE/YSIZE/ZSIZE
    uint32_t  stream_naxis[2];
    uint32_t  stream_xsize[2];        // xsize
    uint32_t  stream_ysize[2];        // ysize
    uint32_t  stream_zsize[2];        // zsize
    uint8_t   stream_sourceLocation;  // where has the stream been loaded from ?
} FUNCTION_PARAMETER_SUBINFO_STREAM;



typedef struct
{
    long FPSNBparamMAX; // to be written by connect function
    long FPSNBparamActive;
    long FPSNBparamUsed;
} FUNCTION_PARAMETER_SUBINFO_FPS;



typedef struct
{
    uint64_t fpflag;// 64 binary flags, see FUNCTION_PARAMETER_MASK_XXXX

    // Parameter name
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel; // number of levels in keyword

    // if this parameter value imported from another parameter, source is:
    char keywordfrom[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];

    char description[FUNCTION_PARAMETER_DESCR_STRMAXLEN];

	// one of FUNCTION_PARAMETER_TYPE_XXXX
    uint32_t type;        

    union
    {
		// value, min (inclusive), max (inclusive), current state (if different from request)
        int64_t l[4];

		// value, min, max, current state (if different from request)
        double f[4];          
        float  s[4];
        
        // first value is set point, second is current state
        pid_t pid[2];
        
        // first value is set point, second is current state
        struct timespec ts[2];

		// first value is set point, second is current state
        char string[2][FUNCTION_PARAMETER_STRMAXLEN]; 
        
        // if TYPE = PROCESS, string[0] is tmux session, string[1] is launch command
    } val;


    union
    {
        FUNCTION_PARAMETER_SUBINFO_STREAM stream;   // if type stream
        FUNCTION_PARAMETER_SUBINFO_FPS    fps;      // if FPTYPE_FPSNAME
    } info;


    long cnt0; // increments when changed

} FUNCTION_PARAMETER;




#define STRINGMAXLEN_FPS_NAME      100

#define FUNCTION_PARAMETER_STRUCT_MSG_SIZE  500



#define FUNCTION_PARAMETER_STRUCT_STATUS_CONF       0x0001   // is configuration running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUN        0x0002   // is process running ?

#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF    0x0010   // should configuration be running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN     0x0020   // should process be running ?

#define FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP    0x0100   // is process loop running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK    0x0200   // Are parameter values OK to run loop process ? (1=OK, 0=not OK)

// are tmux sessions online ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF   0x1000
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN    0x2000
#define FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL   0x4000





#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN    0x0001   // configuration process
//#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFSTOP   0x0002   // stop configuration process
#define FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE     0x0004   // re-run check of parameter

#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED    0x0008   // CheckParametersAll been completed. 
// Toggles to 1 upon update request
// Toggles to 0 when update completed (in function CheckParametersAll)





// ERROR AND WARNING MESSAGES

#define FPS_NB_MSG                          100 // max number of messages
#define FUNCTION_PARAMETER_STRUCT_MSG_LEN   500


#define FPS_MSG_FLAG_NOTINITIALIZED         0x0001
#define FPS_MSG_FLAG_BELOWMIN               0x0002
#define FPS_MSG_FLAG_ABOVEMAX               0x0004

// by default, a message is a warning
#define FPS_MSG_FLAG_ERROR                  0x0008  // if ERROR, then cannot start function
#define FPS_MSG_FLAG_INFO                   0x0010



#define FPS_CWD_STRLENMAX      200
#define FPS_SRCDIR_STRLENMAX   200
#define FPS_PNAME_STRMAXLEN    100
#define FPS_CALLPROGNAME_STRMAXLEN 80
#define FPS_CALLFUNCNAME_STRMAXLEN 100
#define FPS_DESCR_STRMAXLEN    200


// metadata
typedef struct
{
    // process name
    // Name can include numbers in the format -XX-YY to allow for multiple structures be created by the same process function and to pass arguments (XX, YY) to process function
    char                name[STRINGMAXLEN_FPS_NAME];         // example: pname-01-32

    char                description[FPS_DESCR_STRMAXLEN];
    
    
    // where should the parameter values be saved to disk ?
    char                fpsdirectory[FPS_CWD_STRLENMAX];
    
    // source code file name
    char				sourcefname[FPS_SRCDIR_STRLENMAX];
    // souce code line
    int					sourceline;



    // the name and indices are automatically parsed in the following format
    char                pname[FPS_PNAME_STRMAXLEN];          // example: pname
    char                callprogname[FPS_CALLPROGNAME_STRMAXLEN];
    char                callfuncname[FPS_CALLFUNCNAME_STRMAXLEN];
    char                nameindexW[16][10];  // subnames
    int                 NBnameindex;         // example: 2

    // configuration will run in tmux session pname-XX:conf
    // process       will run in tmux session pname-XX:run

    // PID of process owning parameter structure configuration
    pid_t               confpid;
    struct timespec     confpidstarttime;
    
    // PID of process running on this fps
    pid_t               runpid;
	struct timespec     runpidstarttime;


	// Used to send signals to configuration process
    uint64_t            signal;
    
    // configuration wait timer value [us]
    uint64_t            confwaitus;

    uint32_t            status;          // conf and process status



	// size of parameter array (= max number of parameter supported)
    long NBparamMAX;


    char
    message[FPS_NB_MSG][FUNCTION_PARAMETER_STRUCT_MSG_LEN];

	// to which entry does the message refer to ?
    int msgpindex[FPS_NB_MSG];

	// What is the nature of the message/error ?
    uint32_t msgcode[FPS_NB_MSG];

    long                          msgcnt;

    uint32_t                      conferrcnt;

} FUNCTION_PARAMETER_STRUCT_MD;




// localstatus flags
// 
// run configuration loop
#define FPS_LOCALSTATUS_CONFLOOP 0x0001



typedef struct
{
    // these two structures are shared
    FUNCTION_PARAMETER_STRUCT_MD *md;
    FUNCTION_PARAMETER           *parray;   // array of function parameters

    // these variables are local to each process
    uint16_t  localstatus;   // 1 if conf loop should be active
    int       SMfd;
    uint32_t  CMDmode;
    
    long      NBparam;        // number of parameters in array
    long      NBparamActive;  // number of active parameters

} FUNCTION_PARAMETER_STRUCT;







//
// Tasks can be sequenced
// Typically these are read from command fifo
// The structure is meant to provide basic scheduling functionality
//

// max number of entries in queues (total among all queues)
#define NB_FPSCTRL_TASK_MAX             500    
#define NB_FPSCTRL_TASK_PURGESIZE        50  

// flags
#define FPSTASK_STATUS_ACTIVE    0x0000000000000001   // is the task entry in the array used ?
#define FPSTASK_STATUS_SHOW      0x0000000000000002
#define FPSTASK_STATUS_RUNNING   0x0000000000000004
#define FPSTASK_STATUS_COMPLETED 0x0000000000000008

// status (cumulative)
#define FPSTASK_STATUS_WAITING     0x0000000000000010
#define FPSTASK_STATUS_RECEIVED    0x0000000000000020
#define FPSTASK_STATUS_CMDNOTFOUND 0x0000000000000040
#define FPSTASK_STATUS_CMDFAIL     0x0000000000000080
#define FPSTASK_STATUS_CMDOK       0x0000000000000100

// use WAITONRUN to ensure the queue is blocked until the current run process is done
#define FPSTASK_FLAG_WAITONRUN  0x0000000000000001
#define FPSTASK_FLAG_WAITONCONF 0x0000000000000002

// If ON, the task is a wait point, and will only proceed if the FPS pointed to by fpsindex is NOT running
#define FPSTASK_FLAG_WAIT_FOR_FPS_NORUN 0x0000000000000004

#define NB_FPSCTRL_TASKQUEUE_MAX 100 // max number of queues

typedef struct
{
    int priority;
    // high number = high priority
    // 0 = queue not active

} FPSCTRL_TASK_QUEUE;



typedef struct
{

    char cmdstring[STRINGMAXLEN_FPS_CMDLINE];


    uint64_t inputindex;  // order in which tasks are submitted

    // Tasks in separate queues can run in parallel (not waiting for last task to run new one)
    // Tasks within a queue run sequentially
    uint32_t queue;
    // Default queue is 0

    uint64_t status;
    uint64_t flag;

    int fpsindex;  // used to track status

    struct timespec creationtime;
    struct timespec activationtime;
    struct timespec completiontime;

} FPSCTRL_TASK_ENTRY;






// status of control / monitoring process
//
typedef struct
{
	int      exitloop;    // exit control loop if 1
    int      fpsCTRL_DisplayMode; // Display mode
    uint32_t mode;                // GUI mode
    int      NBfps;               // Number of FPS entries
    int      NBkwn;               // Number of keyword nodes
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
} FPSCTRL_PROCESS_VARS;













#define NB_KEYWNODE_MAX 10000
#define MAX_NB_CHILD 500

typedef struct
{
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                          FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int  keywordlevel;

    int parent_index;

    int NBchild;
    int child[MAX_NB_CHILD];

    int leaf; // 1 if this is a leaf (no child)
    int fpsindex;
    int pindex;


} KEYWORD_TREE_NODE;









#include "fps_add_entry.h"
#include "fps_checkparameter.h"
#include "fps_connect.h"
#include "fps_CTRLscreen.h"
#include "fps_disconnect.h"
#include "fps_execFPScmd.h"
#include "fps_FPCONFexit.h"
#include "fps_FPCONFloopstep.h"
#include "fps_FPCONFsetup.h"
#include "fps_getFPSargs.h"
#include "fps_load.h"
#include "fps_outlog.h"
#include "fps_paramvalue.h"
#include "fps_RUNexit.h"
#include "fps_shmdirname.h"






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
 */
#define FPS_SETUP_INIT(VARfpsname,VARCMDmode) FUNCTION_PARAMETER_STRUCT fps; do { \
  fps.SMfd =  -1; \
  fps = function_parameter_FPCONFsetup((VARfpsname), (VARCMDmode)); \
  strncpy(fps.md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX);\
  fps.md->sourceline = __LINE__; \
  { \
    char msgstring[STRINGMAXLEN_FPS_LOGMSG]; \
    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "LOGSTART %s %d %s %d", (VARfpsname), (VARCMDmode), fps.md->sourcefname, fps.md->sourceline); \
    functionparameter_outlog("FPSINIT", msgstring); \
    functionparameter_outlog_namelink(); \
  } \
} while(0)



/** @brief Connect to FPS
 * 
 * 
 */
#define FPS_CONNECT( VARfpsname, VARCMDmode ) FUNCTION_PARAMETER_STRUCT fps; do { \
  fps.SMfd = -1; \
  if(function_parameter_struct_connect( (VARfpsname) , &fps, (VARCMDmode) ) == -1) { \
    printf("ERROR: fps \"%s\" does not exist -> running without FPS interface\n", VARfpsname); \
    return RETURN_FAILURE; \
  }\
} while(0)



/** @brief Start FPS configuration loop
 */
#define FPS_CONFLOOP_START if( ! fps.localstatus & FPS_LOCALSTATUS_CONFLOOP ) { \
  return RETURN_SUCCESS; \
} \
while(fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { \
{ \
  struct timespec treq, trem; \
  treq.tv_sec = 0; \
  treq.tv_nsec = 50000; \
  nanosleep(&treq, &trem); \
  if(data.signal_INT == 1){fps.localstatus &= ~FPS_LOCALSTATUS_CONFLOOP;} \
} \
if(function_parameter_FPCONFloopstep(&fps) == 1) {

/** @brief End FPS configuration loop
 */
#define FPS_CONFLOOP_END  functionparameter_CheckParametersAll(&fps);} \
} \
function_parameter_FPCONFexit(&fps);





/** @brief Add 64-bit float parameter entry
 * 
 * Default setting for input parameter\n
 * Also creates function parameter index (fp_##key), type long
 * 
 * (void) statement suppresses compiler unused parameter warning
 */
#define FPS_ADDPARAM_FLT64_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FLOAT64, FPFLAG_DEFAULT_OUTPUT, (dflt));\
  (void) fp_##key;\
} while(0)


/** @brief Add INT64 input parameter entry
 */
#define FPS_ADDPARAM_INT64_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)


/** @brief Add stream input parameter entry
 */
#define FPS_ADDPARAM_STREAM_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT_STREAM, (dflt));\
  (void) fp_##key;\
} while(0)


/** @brief Add ON/OFF parameter entry
 */
#define FPS_ADDPARAM_ONOFF(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)





/** @brief Add FLT64 output parameter entry
 */
#define FPS_ADDPARAM_FLT64_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FLOAT64, FPFLAG_DEFAULT_OUTPUT, NULL);\
  (void) fp_##key;\
} while(0)


/** @brief Add INT64 output parameter entry
 */
#define FPS_ADDPARAM_INT64_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_INT64, FPFLAG_DEFAULT_OUTPUT, NULL);\
  (void) fp_##key;\
} while(0)


/** @brief Add stream output parameter entry
 */
#define FPS_ADDPARAM_STREAM_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_STREAMNAME, FPFLAG_DEFAULT_OUTPUT_STREAM, NULL);\
  (void) fp_##key;\
} while(0)





/** @} */ // end group fpsmacro


//long fp_stream_inname  = function_parameter_add_entry(&fps, ".in_name",  "input stream",  FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT_STREAM, pNull);






#ifdef __cplusplus
extern "C" {
#endif




/*
errno_t getFPSlogfname(char *logfname);

errno_t function_parameter_struct_shmdirname(char *shmdname);

errno_t function_parameter_getFPSargs_from_CLIfunc(char *fpsname_default);

errno_t function_parameter_execFPScmd();


errno_t function_parameter_struct_create(int NBparamMAX, const char *name);


errno_t functionparameter_scan_fps(
    uint32_t mode,
    char *fpsnamemask,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    int *ptr_NBkwn,
    int *ptr_fpsindex,
    long *ptr_pindex,
    int verbose
);


long    function_parameter_struct_connect(
    const char                *name,
    FUNCTION_PARAMETER_STRUCT *fps,
    int                        fpsconnectmode
);


int     function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT
        *funcparamstruct);


long function_parameter_structure_load(
	char *fpsname
);



int function_parameter_printlist(FUNCTION_PARAMETER *funcparamarray,
                                 long NBparamMAX);

int functionparameter_GetParamIndex(FUNCTION_PARAMETER_STRUCT *fps,
                                    const char *paramname);

long   functionparameter_GetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname);

int    functionparameter_SetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname, long value);

long *functionparameter_GetParamPtr_INT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname);

double   functionparameter_GetParamValue_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname);

int      functionparameter_SetParamValue_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname, double value);

double *functionparameter_GetParamPtr_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname);

float functionparameter_GetParamValue_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname);

int   functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname, float value);

float *functionparameter_GetParamPtr_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname);

char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname);

int    functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname, const char *stringvalue);

int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname);

int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname, int ONOFFvalue);

uint64_t *functionparameter_GetParamPtr_fpflag(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname);


errno_t functionparameter_PrintParameter_ValueString(
    FUNCTION_PARAMETER *fpsentry,
    char *outstring,
    int stringmaxlen
);



imageID functionparameter_LoadStream(
    FUNCTION_PARAMETER_STRUCT *fps,
    int                        pindex,
    int                        fpsconnectmode
);

int function_parameter_add_entry(FUNCTION_PARAMETER_STRUCT *fps,
                                 const char *keywordstring, const char *descriptionstring, uint64_t type,
                                 uint64_t fpflag, void *dataptr);

int functionparameter_CheckParameter(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     int pindex);
int functionparameter_CheckParametersAll(FUNCTION_PARAMETER_STRUCT *fpsentry);


int functionparameter_ConnectExternalFPS(FUNCTION_PARAMETER_STRUCT *FPS,
        int pindex, FUNCTION_PARAMETER_STRUCT *FPSext);
errno_t functionparameter_GetTypeString(uint32_t type, char *typestring);
int functionparameter_PrintParameterInfo(FUNCTION_PARAMETER_STRUCT *fpsentry,
        int pindex);



FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(const char *fpsname,
        uint32_t CMDmode);
uint16_t function_parameter_FPCONFloopstep(FUNCTION_PARAMETER_STRUCT *fps);
uint16_t function_parameter_FPCONFexit(FUNCTION_PARAMETER_STRUCT *fps);
uint16_t function_parameter_RUNexit(FUNCTION_PARAMETER_STRUCT *fps);

int functionparameter_SaveParam2disk(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     const char *paramname);

int functionparameter_GetFileName(
    FUNCTION_PARAMETER_STRUCT *fps,
    FUNCTION_PARAMETER *fparam,
    char *outfname,
    char *tagname
);


int functionparameter_SaveFPS2disk_dir(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    char *dirname);

int functionparameter_SaveFPS2disk(
    FUNCTION_PARAMETER_STRUCT *fpsentry);
    
    
errno_t	functionparameter_write_archivescript(
    FUNCTION_PARAMETER_STRUCT *fps,
    char *archdirname);  
    

int functionparameter_WriteParameterToDisk(FUNCTION_PARAMETER_STRUCT *fpsentry,
        int pindex, char *tagname, char *commentstr);



int functionparameter_UserInputSetParamValue(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex
);

int functionparameter_FPSprocess_cmdline(
    char *FPScmdline,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar,
    FUNCTION_PARAMETER_STRUCT *fps,
    uint64_t *taskstatus
);

*/

#ifdef __cplusplus
}
#endif

#endif  // FUNCTION_PARAMETERS_H
