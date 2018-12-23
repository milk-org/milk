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
    char name[200];  // process name (human-readable)

    char source_FUNCTION[200];  // source code function
    char source_FILE[200];      // source code file
    int source_LINE;            // source code line

    pid_t PID;  // process ID
    // file name is /tmp/proc.PID.shm

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
    // 2: paused
    // 3: terminated (clean exit)
    // 4: ERROR (typically used when loop can't start, e.g. missing
    // input)

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
int processinfo_WriteMessage(PROCESSINFO *processinfo, char *msgstring);


int_fast8_t processinfo_CTRLscreen();

#ifdef __cplusplus
}
#endif

#endif  // _PROCESSTOOLS_H
