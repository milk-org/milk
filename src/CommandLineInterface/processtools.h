/**
 * @file    processtools.h
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

#ifndef SHAREDMEMDIR
#define SHAREDMEMDIR "/tmp" /**< location of file mapped semaphores */
#endif

#define MAXNBSUBPROCESS 50
#define MAXNBCPU 100
// timing info for real-time loop processes
#define PROCESSINFO_NBtimer 100







// --------------------- MANAGING PROCESSES -------------------------------




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
    char name[200];             /// process name (human-readable)

    char source_FUNCTION[200];  /// source code function
    char source_FILE[200];      /// source code file
    int source_LINE;            /// source code line

    pid_t PID;                  /// process ID; file name is /tmp/proc.PID.shm

    struct timespec createtime;  // time at which pinfo was created

    long loopcnt;  // counter, useful for loop processes to monitor activity
    int CTRLval;   // control value to be externally written.
    // 0: run                     (default)
    // 1: pause
    // 2: increment single step (will go back to 1)
    // 3: exit loop

    char tmuxname[100];  // name of tmux session in which process is running, or
    // "NULL"
    int loopstat;        // 0: initialization (before loop)
    // 1: in loop
    // 2: loop paused (do not iterate)
    // 3: terminated (clean exit)
    // 4: ERROR (typically used when loop can't start, e.g. missing input)
    // 5: do not compute (loop iterates, but does not compute. output stream(s) will still be posted/incremented)


    char statusmsg[200];  // status message
    int statuscode;       // status code

    FILE *logFile;
    char logfilename[250];

    // OPTIONAL TIMING MEASUREMENT
    // Used to measure how long loop process takes to complete task
    // Provides means to stop/pause loop process if timing constraints exceeded
    //

    int MeasureTiming;  // 1 if timing is measured, 0 otherwise

    // the last PROCESSINFO_NBtimer times are stored in a circular buffer, from
    // which timing stats are derived
    int timerindex;       // last written index in circular buffer
    int timingbuffercnt;  // increments every cycle of the circular buffer
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

    char description[200];

} PROCESSINFO;




//
// This structure maintains a list of active processes
// It is used to quickly build (without scanning directory) an array of
// PROCESSINFO
//
typedef struct {
    pid_t PIDarray[PROCESSINFOLISTSIZE];
    int active[PROCESSINFOLISTSIZE];

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
	int           rt_priority;
	float         memload;
	
	
	int           NBsubprocesses;
	int           subprocPIDarray[MAXNBSUBPROCESS];
	
	char          statusmsg[200];
	char          tmuxname[100];
	
} PROCESSINFODISP;





typedef struct
{
	int loop;   // 1 : loop     0 : exit
	long loopcnt;
	
	int twaitus; // sleep time between scans
	double dtscan; // measured time interval between scans [s]
		
	PROCESSINFOLIST *pinfolist;  // copy of pointer  static PROCESSINFOLIST *pinfolist

	long NBpinfodisp;
	PROCESSINFODISP *pinfodisp;
	
	int DisplayMode;
	
	
    //
    // these arrays are indexed together
    // the index is different from the displayed order
    // new process takes first available free index
    //
    PROCESSINFO *pinfoarray[PROCESSINFOLISTSIZE];
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



PROCESSINFO *processinfo_shm_create(char *pname, int CTRLval);
int processinfo_cleanExit(PROCESSINFO *processinfo);
int processinfo_SIGexit(PROCESSINFO *processinfo, int SignalNumber);
int processinfo_WriteMessage(PROCESSINFO *processinfo, const char *msgstring);
int processinfo_exec_start(PROCESSINFO *processinfo);
int processinfo_exec_end(PROCESSINFO *processinfo);


int processinfo_CatchSignals();
int processinfo_ProcessSignals(PROCESSINFO *processinfo);


int_fast8_t processinfo_CTRLscreen();

#ifdef __cplusplus
}
#endif

#endif  // _PROCESSTOOLS_H
