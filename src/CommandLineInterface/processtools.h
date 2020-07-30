/**
 * @file    processtools.h
 * @brief   Command line interface
 *
 * Command line interface (CLI) definitions and function prototypes
 *
 * @bug No known bugs.
 *
 */

/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

#ifndef _PROCESSTOOLS_H
#define _PROCESSTOOLS_H

#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#define PROCESSINFOLISTSIZE 10000


#define MAXNBSUBPROCESS 50
#define MAXNBCPU 100
// timing info for real-time loop processes
#define PROCESSINFO_NBtimer 100



#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif




// --------------------- MANAGING PROCESSES -------------------------------

#define STRINGMAXLEN_PROCESSINFO_NAME          80
#define STRINGMAXLEN_PROCESSINFO_SRCFUNC      200
#define STRINGMAXLEN_PROCESSINFO_SRCFILE      200
#define STRINGMAXLEN_PROCESSINFO_TMUXNAME     100
#define STRINGMAXLEN_PROCESSINFO_STATUSMSG    200
#define STRINGMAXLEN_PROCESSINFO_LOGFILENAME  250
#define STRINGMAXLEN_PROCESSINFO_DESCRIPTION  200


// input stream triggering mode

// trigger immediately
#define PROCESSINFO_TRIGGERMODE_IMMEDIATE      0

// trigger when cnt0 increments
#define PROCESSINFO_TRIGGERMODE_CNT0           1

// trigger when cnt1 increments
#define PROCESSINFO_TRIGGERMODE_CNT1           2

// trigger when semaphore is posted
#define PROCESSINFO_TRIGGERMODE_SEMAPHORE      3

// trigger after a time delay
#define PROCESSINFO_TRIGGERMODE_DELAY          4


// trigger is currently waiting for input
#define PROCESSINFO_TRIGGERSTATUS_WAITING      1

#define PROCESSINFO_TRIGGERSTATUS_RECEIVED     2
#define PROCESSINFO_TRIGGERSTATUS_TIMEDOUT     3


/**
 *
 * This structure hold process information and hooks required for basic
 * monitoring and control Unlike the larger DATA structure above, it is meant to
 * be stored in shared memory for fast access by other processes
 *
 *
 * File name:  /tmp/proc.PID.shm
 *
 */
typedef struct {
    char name[STRINGMAXLEN_PROCESSINFO_NAME];             /// process name (human-readable)

    char source_FUNCTION[STRINGMAXLEN_PROCESSINFO_SRCFUNC];  /// source code function
    char source_FILE[STRINGMAXLEN_PROCESSINFO_SRCFILE];      /// source code file
    int  source_LINE;            /// source code line

    pid_t PID;                  /// process ID; file name is /tmp/proc.PID.shm

    struct timespec createtime;  // time at which pinfo was created

    long loopcnt;    // counter, useful for loop processes to monitor activity
    long loopcntMax; // exit loop if loopcnt = loopcntMax. Set to -1 for infinite loop
    int  CTRLval;     // control value to be externally written.
    // 0: run                     (default)
    // 1: pause
    // 2: increment single step (will go back to 1)
    // 3: exit loop

    char tmuxname[STRINGMAXLEN_PROCESSINFO_TMUXNAME];  // name of tmux session in which process is running, or
    // "NULL"
    int loopstat;        
    // 0: INIT       Initialization before loop
    // 1: ACTIVE     in loop
    // 2: PAUSED     loop paused (do not iterate)
    // 3: STOPPED    terminated (clean exit following user request to stop process)
    // 4: ERROR      process could not run, typically used when loop can't start, e.g. missing input
    // 5: SPINNING   do not compute (loop iterates, but does not compute. output stream(s) will still be posted/incremented)
	// 6: CRASHED    pid has gone away without proper exit sequence. Will attempt to generate exit log file (using atexit) to identify crash location

    char statusmsg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];  // status message
    int  statuscode;       // status code

    FILE *logFile;
    char  logfilename[STRINGMAXLEN_PROCESSINFO_LOGFILENAME];
    
    
    
    // OPTIONAL INPUT STREAM SETUP
    // Used to specify which stream will trigger the computation and track trigger state
    // Enables use of function processinfo_waitoninputstream()
    // Enables streamproctrace entry
    // Must be inialized by processinfo_waitoninputstream_init()
    int       triggermode;                    // see TRIGGERMODE codes
    imageID   triggerstreamID;                // -1 if not initialized
    ino_t     triggerstreaminode;
    char      triggerstreamname[STRINGMAXLEN_IMAGE_NAME];  
    int       triggersem;                     // semaphore index
    uint64_t  triggerstreamcnt;               // previous value of trigger counter, updates on trigger
	struct timespec triggerdelay;            // for PROCESSINFO_TRIGGERMODE_DELAY
	struct timespec triggertimeout;          // how long to wait until trigger ?
	uint64_t  trigggertimeoutcnt;
	int       triggermissedframe;               // have we missed any frame, if yes how many ?
	//  0  : no missed frame, loop has been waiting for semaphore to be posted
	//  1  : no missed frame, but semaphore was already posted and at 1 when triggering
	//  2+ : frame(s) missed
	uint64_t  triggermissedframe_cumul;      // cumulative missed frames
	int       triggerstatus;   // see TRIGGERSTATUS codes


	
    int RT_priority;    // -1 if unused. 0-99 for higher priority


    // OPTIONAL TIMING MEASUREMENT
    // Used to measure how long loop process takes to complete task
    // Provides means to stop/pause loop process if timing constraints exceeded
    //
    int MeasureTiming;  // 1 if timing is measured, 0 otherwise
    
    // the last PROCESSINFO_NBtimer times are stored in a circular buffer, from
    // which timing stats are derived
    int    timerindex;       // last written index in circular buffer
    int    timingbuffercnt;  // increments every cycle of the circular buffer
    struct timespec texecstart[PROCESSINFO_NBtimer];  // task starts
    struct timespec texecend[PROCESSINFO_NBtimer];    // task ends

    long dtmedian_iter_ns;  // median time offset between iterations [nanosec]
    long dtmedian_exec_ns;  // median compute/busy time [nanosec]

    // If enabled=1, pause process if dtiter larger than limit
    int dtiter_limit_enable;
    long dtiter_limit_value;
    long dtiter_limit_cnt;

    // If enabled=1, pause process if dtexec larger than limit
    int dtexec_limit_enable;
    long dtexec_limit_value;
    long dtexec_limit_cnt;


    char description[STRINGMAXLEN_PROCESSINFO_DESCRIPTION];

} PROCESSINFO;




//
// This structure maintains a list of active processes
// It is used to quickly build (without scanning directory) an array of
// PROCESSINFO
//
typedef struct {
    pid_t PIDarray[PROCESSINFOLISTSIZE];
    int   active[PROCESSINFOLISTSIZE];
    char  pnamearray[PROCESSINFOLISTSIZE][STRINGMAXLEN_PROCESSINFO_NAME];  // short name

} PROCESSINFOLIST;



typedef struct
{
	int           active;
	pid_t         PID;
	char          name[40];
	long          updatecnt;

	long          loopcnt;
	int           loopstat;
	
	int           createtime_hr;
	int           createtime_min;
	int           createtime_sec;
	long          createtime_ns;
	
	char          cpuset[16];       /**< cpuset name  */
	char          cpusallowed[20];
	int           cpuOKarray[MAXNBCPU];
	int           threads;
	
	int           rt_priority;
	float         memload;
		
	char          statusmsg[200];
	char          tmuxname[100];
	
	int           NBsubprocesses;
	int           subprocPIDarray[MAXNBSUBPROCESS];	
	
	double        sampletimearray[MAXNBSUBPROCESS];  // time at which sampling was performed [sec]
	double        sampletimearray_prev[MAXNBSUBPROCESS];
	
	long          ctxtsw_voluntary[MAXNBSUBPROCESS];
	long          ctxtsw_nonvoluntary[MAXNBSUBPROCESS];
	long          ctxtsw_voluntary_prev[MAXNBSUBPROCESS];
	long          ctxtsw_nonvoluntary_prev[MAXNBSUBPROCESS];
	
	long long     cpuloadcntarray[MAXNBSUBPROCESS];
	long long     cpuloadcntarray_prev[MAXNBSUBPROCESS];
	float         subprocCPUloadarray[MAXNBSUBPROCESS];
	float         subprocCPUloadarray_timeaveraged[MAXNBSUBPROCESS];
	
	long          VmRSSarray[MAXNBSUBPROCESS];
	
	int           processorarray[MAXNBSUBPROCESS];

	
} PROCESSINFODISP;





typedef struct
{
	int      loop;   // 1 : loop     0 : exit
	long     loopcnt;
	
	int      twaitus; // sleep time between scans
	double   dtscan; // measured time interval between scans [s]
	pid_t    scanPID;
	int      scandebugline; // for debugging	
	
	
	// ensure list of process and mmap operation blocks display
	int      SCANBLOCK_requested;  // scan thread toggles to 1 to requests blocking
	int      SCANBLOCK_OK;         // display thread toggles to 1 to let scan know it can proceed
		
	PROCESSINFOLIST *pinfolist;  // copy of pointer  static PROCESSINFOLIST *pinfolist

	long NBpinfodisp;
	PROCESSINFODISP *pinfodisp;
	
	int DisplayMode;
	int DisplayDetailedMode;
	
    //
    // these arrays are indexed together
    // the index is different from the displayed order
    // new process takes first available free index
    //
    PROCESSINFO  *pinfoarray[PROCESSINFOLISTSIZE];
    int           pinfommapped[PROCESSINFOLISTSIZE];             // 1 if mmapped, 0 otherwise
    pid_t         PIDarray[PROCESSINFOLISTSIZE];                 // used to track changes
    int           updatearray[PROCESSINFOLISTSIZE];              // 0: don't load, 1: (re)load
    int           fdarray[PROCESSINFOLISTSIZE];                  // file descriptors
    long          loopcntarray[PROCESSINFOLISTSIZE];
    long          loopcntoffsetarray[PROCESSINFOLISTSIZE];
    int           selectedarray[PROCESSINFOLISTSIZE];

    int           sorted_pindex_time[PROCESSINFOLISTSIZE];
		
		
	int NBcpus;
	int NBcpusocket;
	
	float CPUload[MAXNBCPU];
	long long CPUcnt0[MAXNBCPU];
	long long CPUcnt1[MAXNBCPU];
	long long CPUcnt2[MAXNBCPU];
	long long CPUcnt3[MAXNBCPU];
	long long CPUcnt4[MAXNBCPU];
	long long CPUcnt5[MAXNBCPU];
	long long CPUcnt6[MAXNBCPU];
	long long CPUcnt7[MAXNBCPU];
	long long CPUcnt8[MAXNBCPU];

	int CPUids[MAXNBCPU];  // individual cpus (same cores)
	int CPUphys[MAXNBCPU]; // Physical CPU socket

	int CPUpcnt[MAXNBCPU];	
	
	int NBpindexActive;
	int pindexActive[PROCESSINFOLISTSIZE];
	int psysinfostatus[PROCESSINFOLISTSIZE];
	
} PROCINFOPROC;





// ---------------------  -------------------------------

typedef struct {
    char name[200];
    char description[200];
} STRINGLISTENTRY;

#ifdef __cplusplus
extern "C" {
#endif



PROCESSINFO * processinfo_setup(
    char *pinfoname,	
    char descriptionstring[200],
    char msgstring[200],
    const char *functionname,
    const char *filename,
    int   linenumber
);

errno_t processinfo_error(
    PROCESSINFO *processinfo,
    char *errmsgstring
);

errno_t processinfo_loopstart(
    PROCESSINFO *processinfo
);

int processinfo_loopstep(
    PROCESSINFO *processinfo
);

int processinfo_compute_status(
    PROCESSINFO *processinfo
);




PROCESSINFO *processinfo_shm_create(const char *pname, int CTRLval);
PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
int processinfo_shm_close(PROCESSINFO *pinfo, int fd);
int processinfo_cleanExit(PROCESSINFO *processinfo);
int processinfo_SIGexit(PROCESSINFO *processinfo, int SignalNumber);
int processinfo_WriteMessage(PROCESSINFO *processinfo, const char *msgstring);
int processinfo_exec_start(PROCESSINFO *processinfo);
int processinfo_exec_end(PROCESSINFO *processinfo);


int processinfo_CatchSignals();
int processinfo_ProcessSignals(PROCESSINFO *processinfo);

errno_t processinfo_waitoninputstream_init(
	PROCESSINFO *processinfo,
	imageID      trigID,
	int          triggermode,
	int          semindexrequested
);

errno_t processinfo_waitoninputstream(
    PROCESSINFO *processinfo
);

errno_t processinfo_update_output_stream(
    PROCESSINFO *processinfo,
    imageID outstreamID
);



errno_t processinfo_CTRLscreen();

#ifdef __cplusplus
}
#endif

#endif  // _PROCESSTOOLS_H
