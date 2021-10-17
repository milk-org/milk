/**
 * @file    CLIcore.h
 * @brief   Command line interface
 *
 * Command line interface (CLI) definitions and function prototypes
 *
 * @defgroup errcheckmacro     MACROS: Error checking
 * @defgroup debugmacro        MACROS: Debugging
 * @defgroup procinfomacro     MACROS: Process control
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif




#ifndef _CLICORE_H
#define _CLICORE_H


// include sem_timedwait
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE	200809L
#endif


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
#include <string.h>





typedef long imageID;
typedef long variableID;

#include "config.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"



#include "CommandLineInterface/processtools.h"
#include "CommandLineInterface/streamCTRL_TUI.h"
#include "CommandLineInterface/function_parameters.h"

#include "CommandLineInterface/CLIcore_checkargs.h"
#include "CommandLineInterface/CLIcore_modules.h"
#include "CommandLineInterface/CLIcore_help.h"

#include "CommandLineInterface/milkDebugTools.h"



#define PI 3.14159265358979323846264338328

/// Size of array CLICOREVARRAY
#define SZ_CLICOREVARRAY 1000




/// important directories and info
extern pid_t CLIPID;            // command line interface PID
extern char  DocDir[200];		// location of documentation
extern char  SrcDir[200];		// location of source
extern char  BuildFile[200];		// file name for source
extern char  BuildDate[200];
extern char  BuildTime[200];

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





//
//  ************ lib module init **********************************
//

/** @brief Initialize module
 */
#define INIT_MODULE_LIB(modname) \
static errno_t init_module_CLI(); /* forward declaration */ \
static int INITSTATUS_##modname = 0; \
void __attribute__ ((constructor)) libinit_##modname() \
{ \
if ( INITSTATUS_##modname == 0 )      /* only run once */ \
{ \
strcpy(data.moduleshortname_default, MODULE_SHORTNAME_DEFAULT); \
strcpy(data.moduledatestring, __DATE__); \
strcpy(data.moduletimestring, __TIME__); \
strcpy(data.modulename, (#modname)); \
RegisterModule(__FILE__, PROJECT_NAME, MODULE_DESCRIPTION, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH); \
init_module_CLI(); \
INITSTATUS_##modname = 1; \
strcpy(data.modulename, "");              /* reset after use */ \
strcpy(data.moduleshortname_default, ""); /* reset after use */ \
strcpy(data.moduleshortname, "");         /* reset after use */ \
} \
} \
void __attribute__ ((destructor)) libclose_##modname() \
{ \
if ( INITSTATUS_##modname == 1 ) \
{ \
} \
}







#define MAX_NB_FRAMENAME_CHAR 500
#define MAX_NB_EXCLUSIONS      40




// declare a boolean type "BOOL"
// TRUE and FALSE improve code readability
//
typedef uint_fast8_t BOOL;
#define FALSE 0
#define TRUE 1




#define DATA_NB_MAX_COMMAND 2000
#define DATA_NB_MAX_MODULE 200

// In STATIC allocation mode, IMAGE and VARIABLE arrays are allocated statically

//#define DATA_STATIC_ALLOC // comment if DYNAMIC
#define STATIC_NB_MAX_IMAGE 520
#define STATIC_NB_MAX_VARIABLE 5030



//Need to install process with setuid.  Then, so you aren't running privileged all the time do this:
extern uid_t euid_real;
extern uid_t euid_called;
extern uid_t suid;









/*^-----------------------------------------------------------------------------
| commands available through the CLI
+-----------------------------------------------------------------------------*/




#define STRINGMAXLEN_MODULE_NAME          100
#define STRINGMAXLEN_MODULE_SHORTNAME      50
#define STRINGMAXLEN_MODULE_LOADNAME      500
#define STRINGMAXLEN_MODULE_SOFILENAME   1000
#define STRINGMAXLEN_MODULE_PACKAGENAME    50
#define STRINGMAXLEN_MODULE_INFOSTRING   1000
#define STRINGMAXLEN_MODULE_DATESTRING     20
#define STRINGMAXLEN_MODULE_TIMESTRING     20

#define MODULE_TYPE_UNUSED      0
#define MODULE_TYPE_STARTUP     1
#define MODULE_TYPE_CUSTOMLOAD  2

typedef struct
{
    int type;

    char name[STRINGMAXLEN_MODULE_NAME];        // module name

    char shortname[STRINGMAXLEN_MODULE_SHORTNAME];   // short name. If non-empty, access functions as <shortname>.<functionname>

    char loadname[STRINGMAXLEN_MODULE_LOADNAME];
    char sofilename[STRINGMAXLEN_MODULE_SOFILENAME];

    char package[STRINGMAXLEN_MODULE_PACKAGENAME];     // package to which module belongs
    int versionmajor;	// package version
    int versionminor;
    int versionpatch;

    char info[STRINGMAXLEN_MODULE_INFOSTRING];      // short description

    char datestring[STRINGMAXLEN_MODULE_DATESTRING]; // Compilation date
    char timestring[STRINGMAXLEN_MODULE_TIMESTRING]; // Compilation time

    void *DLib_handle;

} MODULE;



#define STRINGMAXLEN_CMD_KEY        100
#define STRINGMAXLEN_CMD_INFO      1000
#define STRINGMAXLEN_CMD_SYNTAX    1000
#define STRINGMAXLEN_CMD_EXAMPLE   1000
#define STRINGMAXLEN_CMD_CCALL     1000
#define STRINGMAXLEN_CMD_SRCFILE   1000
typedef struct
{
    char     key[STRINGMAXLEN_CMD_KEY];           // command keyword

    // module
    char     module[STRINGMAXLEN_MODULE_NAME];        // module name
    // index of module to which command belongs
    // set to -1 if does not belong to any module
    long     moduleindex;
    char     srcfile[STRINGMAXLEN_CMD_SRCFILE];     // module source filename

    // command function pointer
    errno_t (* fp)();

    char     info[STRINGMAXLEN_CMD_INFO];            // short description/help
    char     syntax[STRINGMAXLEN_CMD_SYNTAX];        // command syntax
    char     example[STRINGMAXLEN_CMD_EXAMPLE];      // command example
    char     Ccall[STRINGMAXLEN_CMD_CCALL];

    // command arguments and parameters
    int nbarg;

    CLICMDARGDATA *argdata; // arguments and parameters to function

    // defines static function capabilities and behavior
    //uint64_t flags;

    // dynamic settings for function
    CMDSETTINGS cmdsettings;
} CMD;






// The command line is parsed and

// cmdargtoken type
// 0 : unsolved
// 1 : floating point (double precision)
// 2 : long
// 3 : string
// 4 : existing image
// 5 : command

#define CMDARGTOKEN_TYPE_UNSOLVED       0
#define CMDARGTOKEN_TYPE_FLOAT          1
#define CMDARGTOKEN_TYPE_LONG           2
#define CMDARGTOKEN_TYPE_STRING         3
#define CMDARGTOKEN_TYPE_EXISTINGIMAGE  4
#define CMDARGTOKEN_TYPE_COMMAND        5
#define CMDARGTOKEN_TYPE_RAWSTRING      6


typedef struct
{
    int type;
    struct
    {
        double numf;
        long numl;
        char string[200];
    } val;
} CMDARGTOKEN;




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




// CODE EXECUTION TRACING

// maximum number of functions in stack
#define MAXNB_FUNCSTACK  100
// maximum string len per function
#define STRINGMAXLEN_FUNCSTAK_FUNCNAME 100

/**
 * @brief Test point structure
 *
 */
typedef struct
{
    uint64_t        loopcnt;
    int             line;
    char            file[STRINGMAXLEN_FULLFILENAME];
    char            func[STRINGMAXLEN_FUNCTIONNAME];

    int             funclevel;
    long            funccallcnt; // how many times has this function been called ?

    char            funcstack[MAXNB_FUNCSTACK][STRINGMAXLEN_FUNCSTAK_FUNCNAME];
    long            fcntstack[MAXNB_FUNCSTACK]; // function call count
    int             linestack[MAXNB_FUNCSTACK]; // caller line number

    char            msg[STRINGMAXLEN_FUNCTIONARGS]; // user message
    struct timespec time;
} CODETESTPOINT;

// number of entries stored in testpoint trace array
#define CODETESTPOINTARRAY_NBCNT  100000

// THIS IS WHERE EVERYTHING THAT NEEDS TO BE WIDELY ACCESSIBLE GETS STORED
typedef struct
{
    char package_name[100];
    int  package_version_major;
    int  package_version_minor;
    int  package_version_patch;
    char package_version[100];
    char configdir[STRINGMAXLEN_DIRNAME];
    char sourcedir[STRINGMAXLEN_DIRNAME];
    char installdir[STRINGMAXLEN_DIRNAME];

    char shmdir[STRINGMAXLEN_DIRNAME];
    char shmsemdirname[STRINGMAXLEN_DIRNAME]; // same ad above with .s instead of /s


    // SIGNALS
    // =================================================

    struct sigaction sigact;

    int signal_USR1;
    int signal_USR2;
    int signal_TERM;
    int signal_INT;
    int signal_SEGV;
    int signal_ABRT;
    int signal_BUS;
    int signal_HUP;
    int signal_PIPE;



    // TEST POINTS
    // =================================================
    // can be used to trace program execution for runtime profiling and debugging

    // current or last test point
    CODETESTPOINT testpoint;

    // code test point array, circular buffer
    CODETESTPOINT *testpointarray;
    int testpointarrayinit; // toggles to 1 when mem allocated

    // Loop counter. Starts at 0, increments when reaching end of circular buffer
    uint64_t testpointloopcnt;
    // Index counter, indicates position of last written testpoint in circ buffer
    uint64_t testpointcnt;


    /*int    testpoint_line;
    char   testpoint_file[STRINGMAXLEN_FULLFILENAME];
    char   testpoint_func[STRINGMAXLEN_FUNCTIONNAME];
    char   testpoint_msg[STRINGMAXLEN_FUNCTIONARGS]; // function arguments
    struct timespec testpoint_time;
    */



    int progStatus;  // main program status
    // 0: before automatic loading of shared objects
    // 1: after automatic loading of shared objects


    // REAL-TIME PRIO
    // =================================================

    uid_t ruid; // Real UID (= user launching process at startup)
    uid_t euid; // Effective UID (= owner of executable at startup)
    uid_t suid; // Saved UID (= owner of executable at startup)
    // system permissions are set by euid
    // at startup, euid = owner of executable (meant to be root)
    // -> we first drop privileges by setting euid to ruid
    // when root privileges needed, we set euid <- suid
    // when reverting to user privileges : euid <- ruid


    // OPERATION MODE
    // =================================================

    int            Debug;
    int            quiet;

    int            errorexit;       // exit on error
    int            exitcode;        // CLI exit code

    int            overwrite;		// automatically overwrite FITS files
    int            rmSHMfile;       // remove shared memory files upon delete
    double         INVRANDMAX;
    gsl_rng       *rndgen;		// random number generator
    int            precision;		// default precision: 0 for float, 1 for double


    // LOGGING, PROCESS MONITORING
    // =================================================

    int            CLIloopON;
    int            CLIlogON;
    char           CLIlogname[200];
    int            processinfo;       // 1 if processes info is to be logged
    int            processinfoActive; // 1 is the process is currently logged
    PROCESSINFO   *pinfo;             // pointer to process info structure



    // COMMAND LINE INTERFACE (CLI)
    // =================================================

    int            fifoON;
    char           processname[100];
    char           processname0[100];
    int            processnameflag;
    char           fifoname[STRINGMAXLEN_FULLFILENAME];
    uint32_t       NBcmd;

    CMD            cmd[DATA_NB_MAX_COMMAND];

    char           CLIcmdline[STRINGMAXLEN_CLICMDLINE];
    int            CLIexecuteCMDready;
    int            CLImatchMode;
    // 1 if error, 0 otherwise
    int            parseerror;
    // number of arguments in last command line
    long           cmdNBarg;
    CMDARGTOKEN    cmdargtoken[NB_ARG_MAX];

    // when command is found in command line, holds index of command
    long           cmdindex;
    // used to create temporary images
    long           calctmp_imindex;
    // 0 if command has not been executed, 1 otherwise
    int            CMDexecuted;
    // 0 if command successfull, 1+ otherwise
    errno_t        CMDerrstatus;


    // MODULES
    // =================================================

    long           NBmodule;
    //long           NB_MAX_MODULE;

    // module info gets sorted into module structure
    MODULE         module[DATA_NB_MAX_MODULE];

    // temporary storage
    long           moduleindex;
    int            moduletype;
    char           modulename[STRINGMAXLEN_MODULE_NAME];
    char           moduleloadname[STRINGMAXLEN_MODULE_LOADNAME];
    char           modulesofilename[STRINGMAXLEN_MODULE_SOFILENAME];
    char           moduleshortname[STRINGMAXLEN_MODULE_SHORTNAME];
    char           moduleshortname_default[STRINGMAXLEN_MODULE_SHORTNAME];
    char           moduledatestring[STRINGMAXLEN_MODULE_DATESTRING];
    char           moduletimestring[STRINGMAXLEN_MODULE_TIMESTRING];


    // FUNCTION PARAMETER STRUCTURES (FPSs)
    // =================================================

    // array of FPSs
    long           NB_MAX_FPS;
    FUNCTION_PARAMETER_STRUCT *fpsarray;


    // Function parameter structure (FPS) CLI integration
    // These entries are set when CLI process links to FPS
    FUNCTION_PARAMETER_STRUCT *fpsptr;
    char           FPS_name[STRINGMAXLEN_FPS_NAME]; // name of FPS if in use
    // Which type of FPS process is the current process ?
    // conf, run, ctrl
    char
    FPS_PROCESS_TYPE[STRINGMAXLEN_FPSPROCESSTYPE]; // included in log file name
    long           FPS_TIMESTAMP;     // included in log file name
    uint32_t       FPS_CMDCODE;       // current FPS mode
    errno_t (*FPS_CONFfunc)();        // pointer to FPS conf function
    errno_t (*FPS_RUNfunc)();         // pointer to FPS run function





    // IMAGES
    // =================================================
    long           NB_MAX_IMAGE;
#ifdef DATA_STATIC_ALLOC
    // image static allocation mode
    IMAGE          image[STATIC_NB_MAX_IMAGE];
#else
    IMAGE         *image;
#endif
    int            MEM_MONITOR; // memory monitor enabled ?

    // shared memory default
    int            SHARED_DFT;

    // Number of keyword per image default
    int            NBKEYWORD_DFT;



    // VARIABLES
    // =================================================

    long           NB_MAX_VARIABLE;
#ifdef DATA_STATIC_ALLOC
    // variable static allocation mode
    VARIABLE variable[STATIC_NB_MAX_VARIABLE];
#else
    VARIABLE      *variable;
#endif





    // CONVENIENCE STORAGE
    // =================================================
    float          FLOATARRAY[1000];	// array to store temporary variables
    double         DOUBLEARRAY[1000];    // for convenience
    char           SAVEDIR[STRINGMAXLEN_DIRNAME];

    // gen purpose return value
    // used for system commands
    int            retvalue;

    // status counter (used for profiling)
    int            status0;
    int            status1;

} DATA;


extern DATA data;

#include "CommandLineInterface/CLIcore_utils.h"




errno_t set_signal_catch();

void sig_handler(int signo);


/*
errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch
);

uint32_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
);
*/

errno_t runCLItest(int argc, char *argv[], char *promptstring);

errno_t runCLI(int argc, char *argv[], char *promptstring);

errno_t CLI_execute_line();





errno_t write_process_log();

#endif
