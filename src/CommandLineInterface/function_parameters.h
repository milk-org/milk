
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

/* ===============================================================================================
 */
/* ===============================================================================================
 */
/*                                      DEFINES, MACROS */
/* ===============================================================================================
 */
/* ===============================================================================================
 */

#define NB_FPS_MAX 100

#define MAXNBLEVELS 20

#define FPSCONNECT_SIMPLE 0
#define FPSCONNECT_CONF   1
#define FPSCONNECT_RUN    2



#define CMDCODE_CONFSTART          0x0001  // run configuration loop
#define CMDCODE_CONFSTOP           0x0002  // stop configuration process
#define CMDCODE_FPSINIT            0x0004  // initialize FPS if does not exist
#define CMDCODE_FPSINITCREATE      0x0008  // (re-)create FPS even if it exists


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
#define STREAM_LOAD_SOURCE_LOCALMEM  1
#define STREAM_LOAD_SOURCE_SHAREMEM  2
#define STREAM_LOAD_SOURCE_CONFFITS  3
#define STREAM_LOAD_SOURCE_CONFNAME  4
#define STREAM_LOAD_SOURCE_EXITFAILURE   -1

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
#define FPFLAG_DEFAULT_INPUT            FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE|FPFLAG_WRITE|FPFLAG_WRITECONF|FPFLAG_SAVEONCHANGE|FPFLAG_FEEDBACK|FPFLAG_CHECKINIT
#define FPFLAG_DEFAULT_OUTPUT           FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE
#define FPFLAG_DEFAULT_INPUT_STREAM     FPFLAG_DEFAULT_INPUT|FPFLAG_STREAM_RUN_REQUIRED|FPFLAG_CHECKSTREAM
#define FPFLAG_DEFAULT_OUTPUT_STREAM    FPFLAG_DEFAULT_INPUT|FPFLAG_CHECKSTREAM


// status parameters, no logging, read only
#define FPFLAG_DEFAULT_STATUS     FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE





#define FUNCTION_PARAMETER_NBPARAM_DEFAULT    100       // size of dynamically allocated array of parameters



typedef struct {
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



typedef struct {
    long FPSNBparam; // to be written by connect function
    long FPSNBparamActive;
    long FPSNBparamUsed;
} FUNCTION_PARAMETER_SUBINFO_FPS;



typedef struct {
	uint64_t fpflag;// 64 binary flags, see FUNCTION_PARAMETER_MASK_XXXX

	// Parameter name
	char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
	char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
	int keywordlevel; // number of levels in keyword
	
	// if this parameter value imported from another parameter, source is:
	char keywordfrom[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
	
	char description[FUNCTION_PARAMETER_DESCR_STRMAXLEN];
	
	uint32_t type;        // one of FUNCTION_PARAMETER_TYPE_XXXX
	
	union
	{
		int64_t         l[4];  // value, min (inclusive), max (inclusive), current state (if different from request)
		double          f[4];  // value, min, max, current state (if different from request)
        float           s[4];
		pid_t           pid[2]; // first value is set point, second is current state
		struct timespec ts[2]; // first value is set point, second is current state
		
		char            string[2][FUNCTION_PARAMETER_STRMAXLEN]; // first value is set point, second is current state
		// if TYPE = PROCESS, string[0] is tmux session, string[1] is launch command
	} val;
	
	
	union
	{
		FUNCTION_PARAMETER_SUBINFO_STREAM stream;   // if type stream
		FUNCTION_PARAMETER_SUBINFO_FPS    fps;      // if FPTYPE_FPSNAME
	} info;
	
	
	long cnt0; // increments when changed

} FUNCTION_PARAMETER;




#define FPS_CWD_MAX 500

#define FUNCTION_PARAMETER_STRUCT_MSG_SIZE  500

#define FUNCTION_PARAMETER_STRUCT_STATUS_CONF       0x0001   // is configuration running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUN        0x0002   // is process running ?

#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF    0x0010   // should configuration be running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN     0x0020   // should process be running ?

#define FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP    0x0100   // is process loop running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK    0x0200   // Are parameter values OK to run loop process ? (1=OK, 0=not OK)



#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN    0x0001   // configuration process
//#define FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFSTOP   0x0002   // stop configuration process
#define FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE     0x0004   // re-run check of parameter






// ERROR AND WARNING MESSAGES

#define FPS_NB_MSG                          100 // max number of messages
#define FUNCTION_PARAMETER_STRUCT_MSG_LEN   500


#define FPS_MSG_FLAG_NOTINITIALIZED         0x0001
#define FPS_MSG_FLAG_BELOWMIN               0x0002
#define FPS_MSG_FLAG_ABOVEMAX               0x0004

// by default, a message is a warning
#define FPS_MSG_FLAG_ERROR                  0x0008  // if ERROR, then cannot start function
#define FPS_MSG_FLAG_INFO                   0x0010






// metadata
typedef struct {
    // process name
    // Name can include numbers in the format -XX-YY to allow for multiple structures be created by the same process function and to pass arguments (XX, YY) to process function
    char                name[200];         // example: pname-01-32
    char                fpsdirectory[FPS_CWD_MAX]; // where should the parameter values be saved to disk ?

    // the name and indices are automatically parsed in the following format
    char                pname[100];      // example: pname
    char                nameindexW[16][10];   // subnames
    int                 NBnameindex;     // example: 2

    // configuration will run in tmux session pname-XX-conf
    // process       will run in tmux session pname-XX-run
    // expected commands to start and stop process :
    //   ./cmdproc/<pname>-conf-start XX YY (in tmux session)
    //   ./cmdproc/<pname>-run-start XX YY  (in tmux session)
    //   ./cmdproc/<pname>-run-stop XX YY
    //

    pid_t               confpid;            // PID of process owning parameter structure configuration
    pid_t               runpid;             // PID of process running on this fps
    int					conf_fifofd;        // File descriptor for configuration fifo

    uint64_t            signal;       // Used to send signals to configuration process
    uint64_t            confwaitus;   // configuration wait timer value [us]

    uint32_t            status;          // conf and process status
   
    int                 NBparam;      // size of parameter array (= max number of parameter supported)

    char                          message[FPS_NB_MSG][FUNCTION_PARAMETER_STRUCT_MSG_LEN];
    int                           msgpindex[FPS_NB_MSG];                                       // to which entry does the message refer to ?
    uint32_t                      msgcode[FPS_NB_MSG];                                         // What is the nature of the message/error ?
    long                          msgcnt;
    uint32_t                      conferrcnt;

} FUNCTION_PARAMETER_STRUCT_MD;




typedef struct {

    FUNCTION_PARAMETER_STRUCT_MD *md;

    FUNCTION_PARAMETER           *parray;   // array of function parameters

} FUNCTION_PARAMETER_STRUCT;




#ifdef __cplusplus
extern "C" {
#endif

errno_t function_parameter_struct_create    (int NBparam, const char *name);
long    function_parameter_struct_connect   (const char *name, FUNCTION_PARAMETER_STRUCT *fps, int fpsconnectmode);
int     function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT *funcparamstruct);


int function_parameter_printlist(FUNCTION_PARAMETER *funcparamarray, int NBparam);

int functionparameter_GetParamIndex(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

long   functionparameter_GetParamValue_INT64   (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int    functionparameter_SetParamValue_INT64   (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, long value);
long * functionparameter_GetParamPtr_INT64     (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

double   functionparameter_GetParamValue_FLOAT64 (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int      functionparameter_SetParamValue_FLOAT64 (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, double value);
double * functionparameter_GetParamPtr_FLOAT64   (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

float functionparameter_GetParamValue_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int   functionparameter_SetParamValue_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, float value);
float * functionparameter_GetParamPtr_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

char * functionparameter_GetParamPtr_STRING   (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int    functionparameter_SetParamValue_STRING (FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, const char *stringvalue);

int functionparameter_GetParamValue_ONOFF(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int functionparameter_SetParamValue_ONOFF(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, int ONOFFvalue);

uint64_t *functionparameter_GetParamPtr_fpflag(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);



long functionparameter_LoadStream(FUNCTION_PARAMETER_STRUCT *fps, int pindex, int fpsconnectmode);

int function_parameter_add_entry(FUNCTION_PARAMETER_STRUCT *fps, const char *keywordstring, const char *descriptionstring, uint64_t type, uint64_t fpflag, void *dataptr);

int functionparameter_CheckParameter(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex);
int functionparameter_CheckParametersAll(FUNCTION_PARAMETER_STRUCT *fpsentry);


int functionparameter_ConnectExternalFPS(FUNCTION_PARAMETER_STRUCT *FPS, int pindex, FUNCTION_PARAMETER_STRUCT *FPSext);
errno_t functionparameter_GetTypeString(uint32_t type, char *typestring);
int functionparameter_PrintParameterInfo(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex);



FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(const char *fpsname, uint32_t CMDmode, uint16_t *loopstatus);
uint16_t function_parameter_FPCONFloopstep( FUNCTION_PARAMETER_STRUCT *fps, uint32_t CMDmode, uint16_t *loopstatus );
uint16_t function_parameter_FPCONFexit( FUNCTION_PARAMETER_STRUCT *fps );


int functionparameter_SaveParam2disk(FUNCTION_PARAMETER_STRUCT *fpsentry, const char *paramname);

int functionparameter_WriteParameterToDisk(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex, char *tagname, char *commentstr);

errno_t functionparameter_CONFstart(FUNCTION_PARAMETER_STRUCT *fps, int fpsindex);
errno_t functionparameter_CONFstop(FUNCTION_PARAMETER_STRUCT *fps, int fpsindex);
errno_t functionparameter_RUNstart(FUNCTION_PARAMETER_STRUCT *fps, int fpsindex);
errno_t functionparameter_RUNstop(FUNCTION_PARAMETER_STRUCT *fps, int fpsindex);


errno_t functionparameter_outlog_file(char *keyw, char *msgstring, FILE *fpout);
errno_t functionparameter_outlog(char* keyw, char *msgstring);

errno_t functionparameter_CTRLscreen(uint32_t mode, char *fpsname, char *fpsCTRLfifoname);

#ifdef __cplusplus
}
#endif

#endif  // FUNCTION_PARAMETERS_H
