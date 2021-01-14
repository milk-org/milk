/**
 * @file    processinfo.h
 *
 *
 *
 */


#ifndef _PROCESSINFO_H
#define _PROCESSINFO_H



#define STRINGMAXLEN_PROCESSINFO_NAME          80
#define STRINGMAXLEN_PROCESSINFO_SRCFUNC      200
#define STRINGMAXLEN_PROCESSINFO_SRCFILE      200
#define STRINGMAXLEN_PROCESSINFO_TMUXNAME     100
#define STRINGMAXLEN_PROCESSINFO_STATUSMSG    200
#define STRINGMAXLEN_PROCESSINFO_LOGFILENAME  250
#define STRINGMAXLEN_PROCESSINFO_DESCRIPTION  200


// timing info for real-time loop processes
#define PROCESSINFO_NBtimer 100

#include "CLIcore.h"

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
typedef struct
{
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
    cpu_set_t CPUmask;


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





#endif
