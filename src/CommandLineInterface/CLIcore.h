/**
 * @file    CLIcore.h
 * @brief   Command line interface 
 * 
 * Command line interface (CLI) definitions and function prototypes
 * 
 *
 * @bug No known bugs. 
 * 
 */

#define _GNU_SOURCE


#ifndef _CLICORE_H
#define _CLICORE_H

#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <semaphore.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>	// for random numbers
#include <signal.h>


#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"
#include "processtools.h"



#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif



#define PI 3.14159265358979323846264338328

/// Size of array CLICOREVARRAY
#define SZ_CLICOREVARRAY 1000


/// important directories and info
extern pid_t CLIPID;            // command line interface PID
extern char DocDir[200];		// location of documentation
extern char SrcDir[200];		// location of source
extern char BuildFile[200];		// file name for source
extern char BuildDate[200];
extern char BuildTime[200];

extern int C_ERRNO;			// C errno (from errno.h)

/* #define DEBUG */
#define CFITSEXIT  printf("Program abnormally terminated, File \"%s\", line %d\n", __FILE__, __LINE__);exit(0)

#ifdef DEBUG
#define nmalloc(f,type,n) f = (type*) malloc(sizeof(type)*n);if(f==NULL){printf("ERROR: pointer \"" #f "\" allocation failed\n");exit(0);}else{printf("\nMALLOC: \""#f "\" allocated\n");}
#define nfree(f) free(f);printf("\nMALLOC: \""#f"\" freed\n");
#else
#define nmalloc(f,type,n) f = (type*) malloc(sizeof(type)*n);if(f==NULL){printf("ERROR: pointer \"" #f "\" allocation failed\n");exit(0);}
#define nfree(f) free(f);
#endif

#define TEST_ALLOC(f) if(f==NULL){printf("ERROR: pointer \"" #f "\" allocation failed\n");exit(0);}


#define NB_ARG_MAX                 20



// declare a boolean type "BOOL" 
// TRUE and FALSE improve code readability
//
typedef uint_fast8_t BOOL;
#define FALSE 0
#define TRUE 1




#define DATA_NB_MAX_COMMAND 1000
#define DATA_NB_MAX_MODULE 100

// In STATIC allocation mode, IMAGE and VARIABLE arrays are allocated statically

//#define DATA_STATIC_ALLOC // comment if DYNAMIC
#define STATIC_NB_MAX_IMAGE 5020
#define STATIC_NB_MAX_VARIABLE 5030



//Need to install process with setuid.  Then, so you aren't running privileged all the time do this:
extern uid_t euid_real;
extern uid_t euid_called;
extern uid_t suid;









/*^-----------------------------------------------------------------------------
| commands available through the CLI
+-----------------------------------------------------------------------------*/



typedef struct {
    char key[100];            // command keyword
    char module[200];          // module name
    int_fast8_t (* fp) ();    // command function pointer
    char info   [1000];       // short description/help
    char syntax [1000];       // command syntax
    char example[1000];       // command example
    char Ccall[1000];
} CMD;



typedef struct {
    char name[50];    // module name
    char package[50]; // package to which module belongs
    char info[1000];  // short description
} MODULE;




/* ---------------------------------------------------------- */
/*                                                            */
/*                                                            */
/*       COMMAND LINE ARGs / TOKENS                           */
/*                                                            */
/*                                                            */
/* ---------------------------------------------------------- */


// The command line is parsed and

// cmdargtoken type
// 0 : unsolved
// 1 : floating point (double precision)
// 2 : long
// 3 : string
// 4 : existing image
// 5 : command
typedef struct
{
    int type;
    union
    {
        double numf;
        long numl;
        char string[200];
    } val;
} CMDARGTOKEN;



int CLI_checkarg(int argnum, int argtype);
int CLI_checkarg_noerrmsg(int argnum, int argtype);






extern uint8_t TYPESIZE[32];




typedef struct
{
    int used;
    char name[80];
    int type; /** 0: double, 1: long, 2: string */
    union
    {
        double f;
        long l;
        char s[80];
    } value;
    char comment[200];
} VARIABLE;


// THIS IS WHERE EVERYTHING THAT NEEDS TO BE WIDELY ACCESSIBLE GETS STORED
typedef struct
{
	char package_name[100];
	char package_version[100];
	char configdir[100];
	char sourcedir[100];
	
	char shmdir[100];
	char shmsemdirname[100]; // same ad above with .s instead of /s
	
    struct sigaction sigact; 
    // signals toggle flags
    int signal_USR1;
    int signal_USR2;
    int signal_TERM;
    int signal_INT;
    int signal_SEGV;
    int signal_ABRT;
    int signal_BUS;
    int signal_HUP;
    int signal_PIPE;
    
    // can be used to trace program execution for runtime profiling and debugging
    // todo: move to shared mem
    int  execSRCline;
    char execSRCfunc[200];
    char execSRCmessage[500];
    
    
    int progStatus;  // main program status
    // 0: before automatic loading of shared objects
    // 1: after automatic loading of shared objects
    
    uid_t ruid; // Real UID (= user launching process at startup)
	uid_t euid; // Effective UID (= owner of executable at startup)
	uid_t suid; // Saved UID (= owner of executable at startup)
	// system permissions are set by euid
	// at startup, euid = owner of executable (meant to be root)
	// -> we first drop privileges by setting euid to ruid
	// when root privileges needed, we set euid <- suid
	// when reverting to user privileges : euid <- ruid
    
    int Debug;
    int quiet;
    int overwrite;		// automatically overwrite FITS files
    double INVRANDMAX;
    gsl_rng *rndgen;		// random number generator
    int precision;		// default precision: 0 for float, 1 for double

    // logging, process monitoring
    int CLIlogON;
    char CLIlogname[200];  
    int processinfo;       // 1 if processes info is to be logged
    int processinfoActive; // 1 is the process is currently logged
    PROCESSINFO *pinfo;    // pointer to process info structure

    // Command Line Interface (CLI) INPUT
    int fifoON;
    char processname[100];
    char processname0[100];
    int processnameflag;
    char fifoname[100];
    uint_fast16_t NBcmd;
    
    long NB_MAX_COMMAND;
    CMD cmd[1000];
    
    int parseerror; // 1 if error, 0 otherwise
    long cmdNBarg;  // number of arguments in last command line
    CMDARGTOKEN cmdargtoken[NB_ARG_MAX];
    long cmdindex; // when command is found in command line, holds index of command
    long calctmp_imindex; // used to create temporary images
    int CMDexecuted; // 0 if command has not been executed, 1 otherwise
    long NBmodule;
    
    long NB_MAX_MODULE;
    MODULE module[100];

    // shared memory default
    int SHARED_DFT;

    // Number of keyword per iamge default
    int NBKEWORD_DFT;

    // images, variables
    long NB_MAX_IMAGE;
    #ifdef DATA_STATIC_ALLOC
    IMAGE image[STATIC_NB_MAX_IMAGE]; // image static allocation mode
	#else
	IMAGE *image;
	#endif
	
    long NB_MAX_VARIABLE;
    #ifdef DATA_STATIC_ALLOC
    VARIABLE variable[STATIC_NB_MAX_VARIABLE]; // variable static allocation mode
	#else
	VARIABLE *variable;
	#endif
	


    float FLOATARRAY[1000];	// array to store temporary variables
    double DOUBLEARRAY[1000];    // for convenience
    char SAVEDIR[500];

    // status counter (used for profiling)
    int status0;
    int status1;
} DATA;


extern DATA data;




// *************************** FUNCTION RETURN VALUE *********************************************
// For function returning type errno_t (= int) 
//
#define RETURN_SUCCESS        0 
#define RETURN_FAILURE       1   // generic error code
#define RETURN_MISSINGFILE   2  










#define MAX_NB_FRAMENAME_CHAR 500
#define MAX_NB_EXCLUSIONS 40


void sig_handler(int signo);

int_fast8_t RegisterModule(char *FileName, char *PackageName, char *InfoString);

uint_fast16_t RegisterCLIcommand(char *CLIkey, char *CLImodule, int_fast8_t (*CLIfptr)(), char *CLIinfo, char *CLIsyntax, char *CLIexample, char *CLICcall);

int_fast8_t runCLI(int argc, char *argv[], char *promptstring);





PROCESSINFO* processinfo_shm_create(char *pname, int CTRLval);
int processinfo_cleanExit(PROCESSINFO *processinfo);
int processinfo_SIGexit(PROCESSINFO *processinfo, int SignalNumber);
int processinfo_WriteMessage(PROCESSINFO *processinfo, const char* msgstring);

int processinfo_exec_start(PROCESSINFO *processinfo);
int processinfo_exec_end(PROCESSINFO *processinfo);


errno_t processinfo_CTRLscreen();

errno_t streamCTRL_CTRLscreen();









// *************************** FUNCTION PARAMETERS *********************************************


#define CMDCODE_CONFSTART          0x0001  // run configuration loop
#define CMDCODE_CONFSTOP           0x0002  // stop configuration process
#define CMDCODE_CONFINIT           0x0004  // (re-)create FPS even if it exists


// function can use this structure to expose parameters for external control or monitoring
// the structure describes how user can interact with parameter, so it allows for control GUIs to connect to parameters

#define FUNCTION_PARAMETER_KEYWORD_STRMAXLEN   32
#define FUNCTION_PARAMETER_KEYWORD_MAXLEVEL    20

// Note that notation allows parameter to have more than one type
// ... to be used with caution: most of the time, use type exclusively
#define FPTYPE_UNDEF         0x0001
#define FPTYPE_INT64         0x0002
#define FPTYPE_FLOAT64       0x0004
#define FPTYPE_PID           0x0008
#define FPTYPE_TIMESPEC      0x0010
#define FPTYPE_FILENAME      0x0020
#define FPTYPE_DIRNAME       0x0040
#define FPTYPE_STREAMNAME    0x0080
#define FPTYPE_STRING        0x0100
#define FPTYPE_ONOFF         0x0200  // uses ONOFF bit flag, string[0] and string[1] for OFF and ON descriptions respectively. setval saves ONOFF as integer
#define FPTYPE_PROCESS       0x0400

#define FUNCTION_PARAMETER_DESCR_STRMAXLEN   64
#define FUNCTION_PARAMETER_STRMAXLEN         64



// STATUS FLAGS

// parameter use and visibility
#define FPFLAG_ACTIVE        0x00000001    // is this entry registered ?
#define FPFLAG_USED          0x00000002    // is this entry used ?
#define FPFLAG_VISIBLE       0x00000004    // is this entry visible (=displayed) ?

// write permission
#define FPFLAG_WRITE         0x00000010    // is value writable when neither CONF and RUN are active
#define FPFLAG_WRITECONF     0x00000020    // can user change value at configuration time ?
#define FPFLAG_WRITERUN      0x00000040    // can user change value at run time ?
#define FPFLAG_WRITESTATUS   0x00000080    // current write status (computed from above flags)

// logging and saving
#define FPFLAG_LOG           0x00000100    // log on change
#define FPFLAG_SAVEONCHANGE  0x00000200    // save to disk on change
#define FPFLAG_SAVEONCLOSE   0x00000400    // save to disk on close

// special types
#define FPFLAG_IMPORTED      0x00001000    // is this entry imported from another parameter ?
#define FPFLAG_FEEDBACK      0x00002000    // is there a separate current value feedback ?
#define FPFLAG_ONOFF         0x00004000    // bit controlled under TYPE_ONOFF

// parameter testing
#define FPFLAG_CHECKINIT     0x00010000    // should parameter be initialized prior to function start ?
#define FPFLAG_CHECKSTREAM   0x00040000    // check stream, read size and type
#define FPFLAG_MINLIMIT      0x00080000    // enforce min limit
#define FPFLAG_MAXLIMIT      0x00100000    // enforce max limit
#define FPFLAG_ERROR         0x00200000    // is current parameter value OK ?



// PRE-ASSEMBLED DEFAULT FLAGS

// input parameter (used as default when adding entry)
#define FPFLAG_DFT_INPUT      FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE|FPFLAG_WRITE|FPFLAG_WRITECONF|FPFLAG_SAVEONCHANGE|FPFLAG_FEEDBACK|FPFLAG_CHECKINIT

// status parameters, no logging, read only
#define FPFLAG_DFT_STATUS     FPFLAG_ACTIVE|FPFLAG_USED|FPFLAG_VISIBLE



#define FUNCTION_PARAMETER_NBPARAM_DEFAULT    100       // size of dynamically allocated array of parameters


typedef struct {
	uint64_t status;// 64 binary flags, see FUNCTION_PARAMETER_MASK_XXXX

	// Parameter name
	char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
	char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
	int keywordlevel; // number of levels in keyword
	
	// if this parameter value imported from another parameter, source is:
	char keywordfrom[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
	
	char description[FUNCTION_PARAMETER_DESCR_STRMAXLEN];
	
	int type;        // one of FUNCTION_PARAMETER_TYPE_XXXX
	
	union
	{
		int64_t         l[4];  // value, min (inclusive), max (inclusive), current state (if different from request)
		double          f[4];  // value, min, max, current state (if different from request)
		pid_t           pid[2]; // first value is set point, second is current state
		struct timespec ts[2]; // first value is set point, second is current state
		
		char            string[2][FUNCTION_PARAMETER_STRMAXLEN]; // first value is set point, second is current state
		// if TYPE = PROCESS, string[0] is tmux session, string[1] is launch command
	} val;
	
	uint32_t  streamID; // if type is stream and MASK_CHECKSTREAM
	
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
    int                 nameindex[10];   // example: 01 32
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
    uint32_t            status;       // conf and process status
    int                 NBparam;      // size of parameter array (= max number of parameter supported)

    char                          message[FPS_NB_MSG][FUNCTION_PARAMETER_STRUCT_MSG_LEN];
    int                           msgpindex[FPS_NB_MSG];                                       // to which entry does the message refer to ?
    uint32_t                      msgcode[FPS_NB_MSG];                                         // What is the nature of the message/error ?
    long                          msgcnt;
    long                          errcnt;

} FUNCTION_PARAMETER_STRUCT_MD;




typedef struct {

    FUNCTION_PARAMETER_STRUCT_MD *md;

    FUNCTION_PARAMETER           *parray;   // array of function parameters

} FUNCTION_PARAMETER_STRUCT;







errno_t function_parameter_struct_create(int NBparam, const char *name);
long function_parameter_struct_connect(const char *name, FUNCTION_PARAMETER_STRUCT *fps);
int function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT *funcparamstruct);


int function_parameter_printlist(FUNCTION_PARAMETER *funcparamarray, int NBparam);
int function_parameter_add_entry(FUNCTION_PARAMETER_STRUCT *fps, char *keywordstring, char *descriptionstring, uint64_t type, uint64_t status, void *dataptr);

int functionparameter_GetParamIndex(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

long functionparameter_GetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int functionparameter_SetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, long value);
long * functionparameter_GetParamPtr_INT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

double functionparameter_GetParamValue_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int functionparameter_SetParamValue_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, double value);
double * functionparameter_GetParamPtr_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

char * functionparameter_GetParamPtr_STRING(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
char *functionparameter_SetParamValue_STRING(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, const char *stringvalue);

int functionparameter_GetParamValue_ONOFF(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);
int functionparameter_SetParamValue_ONOFF(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, int ONOFFvalue);


int functionparameter_CheckParameter(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex);
int functionparameter_CheckParametersAll(FUNCTION_PARAMETER_STRUCT *fpsentry);



FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(const char *fpsname, uint32_t CMDmode, uint16_t *loopstatus);
uint16_t function_parameter_FPCONFloopstep( FUNCTION_PARAMETER_STRUCT *fps, uint32_t CMDmode, uint16_t *loopstatus );
uint16_t function_parameter_FPCONFexit( FUNCTION_PARAMETER_STRUCT *fps );

int functionparameter_WriteParameterToDisk(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex, char *tagname, char *commentstr);

errno_t functionparameter_CTRLscreen(uint32_t mode, char *fpsname, char *fpsCTRLfifoname);

#endif
