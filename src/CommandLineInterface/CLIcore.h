/**
 * @file    CLIcore.h
 * @brief   Command line interface 
 * 
 * Command line interface (CLI) definitions and function prototypes
 * 
 * @author  O. Guyon
 * @date    9 Jul 2017
 *
 * @bug No known bugs. 
 * 
 */

#define _GNU_SOURCE


#ifndef _CLICORE_H
#define _CLICORE_H

#include <stdint.h>
#include <stdio.h>
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


int_fast8_t processinfo_CTRLscreen();

int_fast8_t streamCTRL_CTRLscreen();







// function can use this structure to expose parameters for external control or monitoring
// the structure describes how user can interact with parameter, so it allows for control GUIs to connect to parameters

#define FUNCTION_PARAMETER_KEYWORD_STRMAXLEN   16
#define FUNCTION_PARAMETER_KEYWORD_MAXLEVEL    20

// Note that notation allows parameter to have more than one type
// ... to be used with caution: most of the time, use type exclusively
#define FUNCTION_PARAMETER_TYPE_UNDEF         0x0001
#define FUNCTION_PARAMETER_TYPE_INT64         0x0002
#define FUNCTION_PARAMETER_TYPE_FLOAT64       0x0004
#define FUNCTION_PARAMETER_TYPE_PID           0x0008
#define FUNCTION_PARAMETER_TYPE_TIMESPEC      0x0010
#define FUNCTION_PARAMETER_TYPE_FILENAME      0x0020
#define FUNCTION_PARAMETER_TYPE_DIRNAME       0x0040
#define FUNCTION_PARAMETER_TYPE_STREAMNAME    0x0080
#define FUNCTION_PARAMETER_TYPE_STRING        0x0100

#define FUNCTION_PARAMETER_DESCR_STRMAXLEN   64
#define FUNCTION_PARAMETER_STRMAXLEN         64

// status flags
#define FUNCTION_PARAMETER_STATUS_ACTIVE        0x0001    // is this entry used ?
#define FUNCTION_PARAMETER_STATUS_VISIBLE       0x0002    // is this entry visible (=displayed) ?
#define FUNCTION_PARAMETER_STATUS_WRITECONF     0x0004    // can user change value at configuration time ?
#define FUNCTION_PARAMETER_STATUS_WRITERUN      0x0008    // can user change value at run time ?
#define FUNCTION_PARAMETER_STATUS_LOG           0x0010    // log on change
#define FUNCTION_PARAMETER_STATUS_SAVEONCHANGE  0x0020    // save to disk on change
#define FUNCTION_PARAMETER_STATUS_SAVEONCLOSE   0x0040    // save to disk on close
#define FUNCTION_PARAMETER_STATUS_MINLIMIT      0x0080    // enforce min limit
#define FUNCTION_PARAMETER_STATUS_MAXLIMIT      0x0100    // enforce max limit
#define FUNCTION_PARAMETER_STATUS_CHECKSTREAM   0x0200    // check stream, read size and type
#define FUNCTION_PARAMETER_STATUS_IMPORTED      0x0400    // is this entry imported from another parameter ?

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
		int64_t         l[3];  // value, min (inclusive), max (inclusive)
		double          f[3];  // value, min, max
		pid_t           pid;
		struct timespec ts;
		char            string[FUNCTION_PARAMETER_STRMAXLEN];
	} val;
	
	uint32_t  streamID; // if type is stream and MASK_CHECKSTREAM
	
	long cnt0; // increments when changed

} FUNCTION_PARAMETER;







#define FUNCTION_PARAMETER_STRUCT_STATUS_CONF       0x0001   // has configuration been done ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUN        0x0002   // is process running ?
#define FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP    0x0004   // is process loop running ?

// metadata
typedef struct {
	char                name[100];
	pid_t               confpid;      // PID of process owning parameter structure configuration
	pid_t               runpid;       // PID of process running on this fps
	uint32_t            pstatus;      // process status
	int                 NBparam;      // size of parameter array (= max number of parameter supported)		
} FUNCTION_PARAMETER_STRUCT_MD;

typedef struct {
	FUNCTION_PARAMETER_STRUCT_MD *md;
	FUNCTION_PARAMETER           *parray;   // array of function parameters
} FUNCTION_PARAMETER_STRUCT;







int function_parameter_struct_create(int NBparam, char *name);
long function_parameter_struct_connect(char *name, FUNCTION_PARAMETER_STRUCT *fps);
int function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT *funcparamstruct, int NBparam);


int function_parameter_printlist(FUNCTION_PARAMETER *funcparamarray, int NBparam);
int function_parameter_add_entry(FUNCTION_PARAMETER *funcparamarray, char *keywordstring, char *descriptionstring, uint64_t type, int NBparam, void *dataptr);


int_fast8_t functionparameter_CTRLscreen(char *fpsname);

#endif
