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






// define (custom) types for function return value

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

typedef long imageID;
typedef long variableID;

#ifndef STANDALONE
#include "config.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"
#include "processtools.h"
#include "streamCTRL.h"
#include "function_parameters.h"
#endif


#define PI 3.14159265358979323846264338328

/// Size of array CLICOREVARRAY
#define SZ_CLICOREVARRAY 1000

#define STRINGMAXLEN_DEFAULT       1000
#define STRINGMAXLEN_ERRORMSG      1000
#define STRINGMAXLEN_CLICMD        1000
#define STRINGMAXLEN_COMMAND       1000
#define STRINGMAXLEN_STREAMNAME     100
#define STRINGMAXLEN_IMGNAME        100
#define STRINGMAXLEN_FILENAME       200  // without directory, includes extension
#define STRINGMAXLEN_DIRNAME        800 
#define STRINGMAXLEN_FULLFILENAME  1000  // includes directory name 
#define STRINGMAXLEN_FUNCTIONNAME   200
#define STRINGMAXLEN_FUNCTIONARGS  1000
#define STRINGMAXLEN_SHMDIRNAME     200



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






//
// ************ ERROR HANDLING **********************************
//

/** @brief Print error (in red) and continue
 *  @ingroup errcheckmacro
 */
#ifndef STANDALONE
#define PRINT_ERROR(...) do { \
sprintf(data.testpoint_msg, __VA_ARGS__); \
printf("ERROR: %c[%d;%dm %s %c[%d;m\n", (char) 27, 1, 31, data.testpoint_msg, (char) 27, 0); \
sprintf(data.testpoint_file, "%s", __FILE__); \
sprintf(data.testpoint_func, "%s", __func__); \
data.testpoint_line = __LINE__; \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time); \
} while(0)
#else
#define PRINT_ERROR(...) printf("ERROR: %c[%d;%dm %s %c[%d;m\n", (char) 27, 1, 31, __VA_ARGS__, (char) 27, 0)
#endif



/**
 * @brief Print warning and continue
 * @ingroup errcheckmacro
 */
#define PRINT_WARNING(...) do { \
char warnmessage[1000]; \
sprintf(warnmessage, __VA_ARGS__); \
fprintf(stderr, \
"%c[%d;%dm WARNING [ FILE: %s   FUNCTION: %s  LINE: %d ]  %c[%d;m\n", \
(char) 27, 1, 35, __FILE__, __func__, __LINE__, (char) 27, 0); \
if(C_ERRNO != 0) \
{ \
char buff[256]; \
if( strerror_r( errno, buff, 256 ) == 0 ) { \
fprintf(stderr,"C Error: %s\n", buff ); \
} else { \
fprintf(stderr,"Unknown C Error\n"); \
} \
} else { \
fprintf(stderr,"No C error (errno = 0)\n"); } \
fprintf(stderr, "%c[%d;%dm ", (char) 27, 1, 35); \
fprintf(stderr, "%s", warnmessage); \
fprintf(stderr, " %c[%d;m\n", (char) 27, 0); \
C_ERRNO = 0; \
} while(0)




/**
 * @ingroup debugmacro
 * @brief register trace point
 */
#if defined NDEBUG || defined STANDALONE
#define DEBUG_TRACEPOINT(...)
#else
#define DEBUG_TRACEPOINT(...) do {                    \
sprintf(data.testpoint_file, "%s", __FILE__);         \
sprintf(data.testpoint_func, "%s", __func__);         \
data.testpoint_line = __LINE__;                       \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time);  \
sprintf(data.testpoint_msg, __VA_ARGS__);             \
} while(0)
#endif


/**
 * @ingroup debugmacro
 * @brief register and log trace point
 */
#if defined NDEBUG || defined STANDALONE
#define DEBUG_TRACEPOINTLOG(...)
#else
#define DEBUG_TRACEPOINTLOG(...) do {                \
sprintf(data.testpoint_file, "%s", __FILE__);        \
sprintf(data.testpoint_func, "%s", __func__);        \
data.testpoint_line = __LINE__;                      \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time); \
sprintf(data.testpoint_msg, __VA_ARGS__);            \
write_process_log();                                 \
} while(0)
#endif



//
// ************ ERROR-CHECKING FUNCTIONS **********************************
//




/**
 * @ingroup errcheckmacro
 * @brief system call with error checking and handling
 *
 */
#define EXECUTE_SYSTEM_COMMAND(...) do {                                   \
char syscommandstring[STRINGMAXLEN_COMMAND];                               \
int slen = snprintf(syscommandstring, STRINGMAXLEN_COMMAND, __VA_ARGS__);  \
if(slen<1) {                                                               \
    PRINT_ERROR("snprintf wrote <1 char");                                 \
    abort();                                                               \
}                                                                          \
if(slen >= STRINGMAXLEN_COMMAND) {                                         \
    PRINT_ERROR("snprintf string truncation");                             \
    abort();                                                               \
}                                                                          \
if(system(syscommandstring) != 0) {                                        \
    PRINT_ERROR("system() returns non-zero value\ncommand \"%s\" failed", syscommandstring); \
}                                                                          \
} while(0)



/**
 * @ingroup errcheckmacro
 * @brief snprintf with error checking and handling
 *
 */
#define SNPRINTF_CHECK(string, maxlen, ...) do { \
int slen = snprintf(string, maxlen, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= maxlen) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)



/**
 * @ingroup errcheckmacro
 * @brief Write image name to string
 *
 * Requires existing image string of len #STRINGMAXLEN_IMGNAME
 *
 * Example use:
 * @code
 * char imname[STRINGMAXLEN_IMGNAME];
 * char name[]="im";
 * int imindex = 34;
 * WRITE_FULLFILENAME(imname, "%s_%04d", name, imindex);
 * @endcode
 *
 *
 */
#define WRITE_IMAGENAME(imname, ...) do { \
int slen = snprintf(imname, STRINGMAXLEN_IMGNAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_IMGNAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)


#define CREATE_IMAGENAME(imname, ...) \
char imname[STRINGMAXLEN_IMGNAME]; \
do { \
int slen = snprintf(imname, STRINGMAXLEN_IMGNAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_IMGNAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)



/**
 * @ingroup errcheckmacro
 * @brief Write filename to string
 *
 * Requires existing image string of len #STRINGMAXLEN_FILENAME
 *
 * Example use:
 * @code
 * char fname[STRINGMAXLEN_FILENAME];
 * char name[]="imlog";
 * WRITE_FULLFILENAME(fname, "%s.txt", name);
 * @endcode
 *
 */
#define WRITE_FILENAME(fname, ...) do { \
int slen = snprintf(fname, STRINGMAXLEN_FILENAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_FILENAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief Write full path filename to string
 *
 * Requires existing image string of len #STRINGMAXLEN_FULLFILENAME
 *
 * Example use:
 * @code
 * char ffname[STRINGMAXLEN_FULLFILENAME];
 * char directory[]="/tmp/";
 * char name[]="imlog";
 * WRITE_FULLFILENAME(ffname, "%s/%s.txt", directory, name);
 * @endcode
 *
 */
#define WRITE_FULLFILENAME(ffname, ...) do { \
int slen = snprintf(ffname, STRINGMAXLEN_FULLFILENAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_FULLFILENAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief Write a string to file
 *
 * Creates file, writes string, and closes file.
 *
 * Example use:
 * @code
 * float piapprox = 3.14;
 * WRITE_STRING_TO_FILE("logfile.txt", "pi is approximately %f\n", piapprox);
 * @endcode
 *
 */
#define WRITE_STRING_TO_FILE(fname, ...) do { \
FILE *fptmp;                                                                \
fptmp = fopen(fname, "w");                                                  \
if (fptmp == NULL) {                                                        \
int errnum = errno;                                                         \
PRINT_ERROR("fopen() returns NULL");                                        \
fprintf(stderr, "Error opening file %s: %s\n", fname, strerror( errnum ));  \
abort();                                                                    \
} else {                                                                    \
fprintf(fptmp, __VA_ARGS__);                                                \
fclose(fptmp);                                                              \
}                                                                           \
} while(0)










// *************************** FUNCTION RETURN VALUE *********************************************
// For function returning type errno_t (= int)
//
#define RETURN_SUCCESS        0
#define RETURN_FAILURE       1   // generic error code
#define RETURN_MISSINGFILE   2


#define MAX_NB_FRAMENAME_CHAR 500
#define MAX_NB_EXCLUSIONS 40

#ifndef STANDALONE



// testing argument type for command line interface
#define CLIARG_FLOAT            1  // floating point number
#define CLIARG_LONG             2  // integer (int or long)
#define CLIARG_STR_NOT_IMG      3  // string, not existing image
#define CLIARG_IMG              4  // existing image
#define CLIARG_STR              5  // string

#define CLICMD_SUCCESS          0
#define CLICMD_INVALID_ARG      1
#define CLICMD_ERROR            2


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
#define STATIC_NB_MAX_IMAGE 520
#define STATIC_NB_MAX_VARIABLE 5030



//Need to install process with setuid.  Then, so you aren't running privileged all the time do this:
extern uid_t euid_real;
extern uid_t euid_called;
extern uid_t suid;









/*^-----------------------------------------------------------------------------
| commands available through the CLI
+-----------------------------------------------------------------------------*/



typedef struct
{
    char     key[100];           // command keyword
    char     module[200];        // module name
    long     moduleindex;        // index of module to which command belongs
    char     modulesrc[200];     // module source filename
    errno_t (* fp)();            // command function pointer
    char     info[1000];         // short description/help
    char     syntax[1000];       // command syntax
    char     example[1000];      // command example
    char     Ccall[1000];
} CMD;



typedef struct
{
    char name[50];        // module name

    char shortname[80];   // short name. If non-empty, access functions as <shortname>.<functionname>

    char package[50];     // package to which module belongs
    int versionmajor;	// package version
    int versionminor;
    int versionpatch;

    char info[1000];      // short description

    char datestring[20]; // Compilation date
    char timestring[20]; // Compilation time

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
    struct
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
    int  package_version_major;
    int  package_version_minor;
    int  package_version_patch;
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
    int    testpoint_line;
    char   testpoint_file[STRINGMAXLEN_FILENAME];
    char   testpoint_func[STRINGMAXLEN_FUNCTIONNAME];
    char   testpoint_msg[STRINGMAXLEN_FUNCTIONARGS]; // function arguments
    struct timespec testpoint_time;


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

    int            Debug;
    int            quiet;
    int            overwrite;		// automatically overwrite FITS files
    int            rmSHMfile;       // remove shared memory files upon delete
    double         INVRANDMAX;
    gsl_rng       *rndgen;		// random number generator
    int            precision;		// default precision: 0 for float, 1 for double

    // logging, process monitoring
    int            CLIloopON;
    int            CLIlogON;
    char           CLIlogname[200];
    int            processinfo;       // 1 if processes info is to be logged
    int            processinfoActive; // 1 is the process is currently logged
    PROCESSINFO   *pinfo;             // pointer to process info structure

    // Command Line Interface (CLI) INPUT
    int            fifoON;
    char           processname[100];
    char           processname0[100];
    int            processnameflag;
    char           fifoname[100];
    uint_fast16_t  NBcmd;

    long           NB_MAX_COMMAND;
    CMD            cmd[1000];

    int            parseerror;         // 1 if error, 0 otherwise
    long           cmdNBarg;           // number of arguments in last command line
    CMDARGTOKEN    cmdargtoken[NB_ARG_MAX];
    long
    cmdindex;           // when command is found in command line, holds index of command
    long           calctmp_imindex;    // used to create temporary images
    int
    CMDexecuted;        // 0 if command has not been executed, 1 otherwise


    // Modules
    long           NBmodule;
    long           NB_MAX_MODULE;
    MODULE         module[100];
    long           moduleindex;
    char           modulename[100];
    char           moduleshortname[80];
    char           moduleshortname_default[80];
    char           moduledatestring[20];
    char           moduletimestring[20];

    // Function parameter structure (FPS) instegration
    // These entries are set when CLI process enters FPS function
    char           FPS_name[STRINGMAXLEN_FPS_NAME]; // name of FPS if in use
    uint32_t       FPS_CMDCODE; // current FPS mode
    errno_t        (*FPS_CONFfunc)(); // pointer to FPS conf function
    errno_t        (*FPS_RUNfunc)(); // pointer to FPS run function
	

    // shared memory default
    int            SHARED_DFT;

    // Number of keyword per iamge default
    int            NBKEWORD_DFT;

    // images, variables
    long           NB_MAX_IMAGE;
#ifdef DATA_STATIC_ALLOC
    IMAGE          image[STATIC_NB_MAX_IMAGE]; // image static allocation mode
#else
    IMAGE         *image;
#endif
	int            MEM_MONITOR; // memory monitor enabled ?

    long           NB_MAX_VARIABLE;
#ifdef DATA_STATIC_ALLOC
    VARIABLE
    variable[STATIC_NB_MAX_VARIABLE]; // variable static allocation mode
#else
    VARIABLE      *variable;
#endif



    float          FLOATARRAY[1000];	// array to store temporary variables
    double         DOUBLEARRAY[1000];    // for convenience
    char           SAVEDIR[500];

    // status counter (used for profiling)
    int            status0;
    int            status1;

} DATA;


extern DATA data;




errno_t set_signal_catch();

void sig_handler(int signo);

errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch    
);

uint_fast16_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
);

errno_t runCLItest(int argc, char *argv[], char *promptstring);
errno_t runCLI(int argc, char *argv[], char *promptstring);

#endif // ifndef STANDALONE

errno_t write_process_log();

#endif
