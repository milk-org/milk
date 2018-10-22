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


#include "ImageStreamIO/ImageStruct.h"


#define PI 3.14159265358979323846264338328

/// Size of array CLICOREVARRAY
#define SZ_CLICOREVARRAY 1000


#define PROCESSINFOLISTSIZE 10000

/// important directories and info
extern pid_t CLIPID;
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


// timing info for real-time loop processes
#define PROCESSINFO_NBtimer  100




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







// --------------------- MANAGING PROCESSES -------------------------------


/**
 * 
 * This structure hold process information and hooks required for basic monitoring and control
 * Unlike the larger DATA structure above, it is meant to be stored in shared memory for fast access by other processes
 * 
 * 
 * File name:  /tmp/proc.PID.shm
 * 
 */
typedef struct 
{
	char   name[200];             // process name (human-readable)

	char   source_FUNCTION[200];  // source code function
	char   source_FILE[200];      // source code file
	int    source_LINE;           // source code line

	pid_t  PID;                   // process ID
	// file name is /tmp/proc.PID.shm
	
	struct timespec createtime;   // time at which pinfo was created


	long   loopcnt;               // counter, useful for loop processes to monitor activity
	int    CTRLval;               // control value to be externally written. 
								  // 0: run                     (default) 
								  // 1: pause
                                  // 2: increment single step (will go back to 1)
                                  // 3: exit loop
								
								
	char   tmuxname[100];         // name of tmux session in which process is running, or "NULL"
	int    loopstat;              // 0: initialization (before loop)
	                              // 1: in loop
	                              // 2: paused
	                              // 3: terminated (clean exit)
	                              // 4: ERROR (typically used when loop can't start, e.g. missing input)

	char   statusmsg[200];        // status message
	int    statuscode;            // status code 

	FILE  *logFile;
	char  logfilename[250];
	
	 // OPTIONAL TIMING MEASUREMENT
	// Used to measure how long loop process takes to complete task
	// Provides means to stop/pause loop process if timing constraints exceeded
	//

	int MeasureTiming;  // 1 if timing is measured, 0 otherwise
	
	// the last PROCESSINFO_NBtimer times are stored in a circular buffer, from which timing stats are derived
	int    timerindex;                                // last written index in circular buffer
	int    timingbuffercnt;                           // increments every cycle of the circular buffer
	struct timespec texecstart[PROCESSINFO_NBtimer];  // task starts
	struct timespec texecend[PROCESSINFO_NBtimer];    // task ends

	long   dtmedian_iter_ns;                      // median time offset between iterations [nanosec]
	long   dtmedian_exec_ns;                      // median compute/busy time [nanosec]
	
	// If enabled=1, pause process if dtiter larger than limit
	int    dtiter_limit_enable;
	long   dtiter_limit_value;
	long   dtiter_limit_cnt;
	
	// If enabled=1, pause process if dtexec larger than limit
	int    dtexec_limit_enable;
	long   dtexec_limit_value;
	long   dtexec_limit_cnt;
	
	char description[200];
	
	
} PROCESSINFO;











//
// This structure maintains a list of active processes
// It is used to quickly build (without scanning directory) an array of PROCESSINFO
//
typedef struct
{
	pid_t         PIDarray[PROCESSINFOLISTSIZE];
	int           active[PROCESSINFOLISTSIZE];
	
} PROCESSINFOLIST;


// ---------------------  -------------------------------


typedef struct
{
	char name[200];
	char description[200];
} STRINGLISTENTRY;






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
int processinfo_WriteMessage(PROCESSINFO *processinfo, char* msgstring);

int processinfo_exec_start(PROCESSINFO *processinfo);
int processinfo_exec_end(PROCESSINFO *processinfo);


int_fast8_t processinfo_CTRLscreen();

#endif
