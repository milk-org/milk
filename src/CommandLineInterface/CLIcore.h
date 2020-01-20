/**
 * @file    CLIcore.h
 * @brief   Command line interface 
 * 
 * Command line interface (CLI) definitions and function prototypes
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
#include "streamCTRL.h"
#include "function_parameters.h"



// define (custom) types for function return value

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

typedef long imageID;
typedef long variableID;






#define PI 3.14159265358979323846264338328

/// Size of array CLICOREVARRAY
#define SZ_CLICOREVARRAY 1000


#define STRINGMAXLEN_FILENAME 1000
#define STRINGMAXLEN_FUNCTIONNAME 200
#define STRINGMAXLEN_FUNCTIONARGS 1000


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







#define TESTPOINT(...) do { \
sprintf(data.testpoint_file, "%s", __FILE__); \
sprintf(data.testpoint_func, "%s", __func__); \
data.testpoint_line = __LINE__; \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time); \
sprintf(data.testpoint_msg, __VA_ARGS__); \
} while(0)





// testing argument type for command line interface
#define CLIARG_FLOAT            1
#define CLIARG_LONG             2
#define CLIARG_STR_NOT_IMG      3  // string, not existing image
#define CLIARG_IMG              4  // existing image
#define CLIARG_STR              5  // string

#define CLICMD_SUCCESS          0
#define CLICMD_INVALID_ARG      1



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
    char     key[100];           // command keyword
    char     module[200];        // module name
    errno_t  (* fp) ();          // command function pointer
    char     info[1000];         // short description/help
    char     syntax[1000];       // command syntax
    char     example[1000];      // command example
    char     Ccall[1000];
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
    int  testpoint_line;
    char testpoint_file[STRINGMAXLEN_FILENAME];
    char testpoint_func[STRINGMAXLEN_FUNCTIONNAME];
    char testpoint_msg[STRINGMAXLEN_FUNCTIONARGS]; // function arguments
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
    
    int       Debug;
    int       quiet;
    int       overwrite;		// automatically overwrite FITS files
    double    INVRANDMAX;
    gsl_rng * rndgen;		// random number generator
    int       precision;		// default precision: 0 for float, 1 for double

    // logging, process monitoring
    int          CLIloopON;
    int          CLIlogON;
    char         CLIlogname[200];  
    int          processinfo;       // 1 if processes info is to be logged
    int          processinfoActive; // 1 is the process is currently logged
    PROCESSINFO *pinfo;             // pointer to process info structure

    // Command Line Interface (CLI) INPUT
    int  fifoON;
    char processname[100];
    char processname0[100];
    int  processnameflag;
    char fifoname[100];
    uint_fast16_t NBcmd;
    
    long  NB_MAX_COMMAND;
    CMD   cmd[1000];
    
    int          parseerror;         // 1 if error, 0 otherwise
    long         cmdNBarg;           // number of arguments in last command line
    CMDARGTOKEN  cmdargtoken[NB_ARG_MAX];
    long         cmdindex;           // when command is found in command line, holds index of command
    long         calctmp_imindex;    // used to create temporary images
    int          CMDexecuted;        // 0 if command has not been executed, 1 otherwise
    long         NBmodule;
    
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

errno_t set_signal_catch();

void sig_handler(int signo);

errno_t RegisterModule(
    const char * restrict FileName,
    const char * restrict PackageName,
    const char * restrict InfoString
);

uint_fast16_t RegisterCLIcommand(
    const char * restrict CLIkey,
    const char * restrict CLImodule,
    errno_t (*CLIfptr)(),
    const char * restrict CLIinfo,
    const char * restrict CLIsyntax,
    const char * restrict CLIexample,
    const char * restrict CLICcall
);

errno_t runCLItest(int argc, char *argv[], char *promptstring);
errno_t runCLI(int argc, char *argv[], char *promptstring);

#endif
