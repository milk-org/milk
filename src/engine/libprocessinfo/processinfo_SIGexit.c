/**
 * @file processinfo_SIGexit.c
 * @brief Processinfo sigexit module
 */

#include <signal.h>

#include "processinfo_internal.h"
#include "processinfo_WriteMessage.h"

#ifndef CLOCK_MILK
#define CLOCK_MILK CLOCK_REALTIME
#endif


/**
 * @brief Handle fatal signal exit for a processinfo process.
 *
 * Records the signal number in processinfo metadata,
 * sets loopstat to error, and performs cleanup.
 */
int processinfo_SIGexit(
    PROCESSINFO *processinfo,
    int         SignalNumber)
{
    char            timestring[200];
    struct timespec tstop;
    struct tm      *tstoptm;
    char            msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

    clock_gettime(CLOCK_MILK, &tstop);
    tstoptm = gmtime(&tstop.tv_sec);

    snprintf(timestring,
             200,
             "%02d:%02d:%02d.%03d",
             tstoptm->tm_hour,
             tstoptm->tm_min,
             tstoptm->tm_sec,
             (int)(0.000001 * (tstop.tv_nsec)));
    processinfo->loopstat = 3; // clean exit

    char SIGstr[12];
    int  SIGflag = 0;
    switch(SignalNumber)
    {

    case SIGHUP: // Hangup detected on controlling terminal or death of controlling process
        snprintf(SIGstr, sizeof(SIGstr), "SIGHUP");
        SIGflag = 1;
        break;

    case SIGINT: // Interrupt from keyboard
        snprintf(SIGstr, sizeof(SIGstr), "SIGINT");
        SIGflag = 1;
        break;

    case SIGQUIT: // Quit from keyboard
        snprintf(SIGstr, sizeof(SIGstr), "SIGQUIT");
        SIGflag = 1;
        break;

    case SIGILL: // Illegal Instruction
        snprintf(SIGstr, sizeof(SIGstr), "SIGILL");
        SIGflag = 1;
        break;

    case SIGABRT: // Abort signal from abort
        snprintf(SIGstr, sizeof(SIGstr), "SIGABRT");
        SIGflag = 1;
        break;

    case SIGFPE: // Floating-point exception
        snprintf(SIGstr, sizeof(SIGstr), "SIGFPE");
        SIGflag = 1;
        break;

    case SIGKILL: // Kill signal
        snprintf(SIGstr, sizeof(SIGstr), "SIGKILL");
        SIGflag = 1;
        break;

    case SIGSEGV: // Invalid memory reference
        snprintf(SIGstr, sizeof(SIGstr), "SIGSEGV");
        SIGflag = 1;
        break;

    case SIGPIPE: // Broken pipe: write to pipe with no readers
        snprintf(SIGstr, sizeof(SIGstr), "SIGPIPE");
        SIGflag = 1;
        break;

    case SIGALRM: // Timer signal from alarm
        snprintf(SIGstr, sizeof(SIGstr), "SIGALRM");
        SIGflag = 1;
        break;

    case SIGTERM: // Termination signal
        snprintf(SIGstr, sizeof(SIGstr), "SIGTERM");
        SIGflag = 1;
        break;

    case SIGUSR1: // User-defined signal 1
        snprintf(SIGstr, sizeof(SIGstr), "SIGUSR1");
        SIGflag = 1;
        break;

    case SIGUSR2: // User-defined signal 1
        snprintf(SIGstr, sizeof(SIGstr), "SIGUSR2");
        SIGflag = 1;
        break;

    case SIGCHLD: // Child stopped or terminated
        snprintf(SIGstr, sizeof(SIGstr), "SIGCHLD");
        SIGflag = 1;
        break;

    case SIGCONT: // Continue if stoppedshmimTCPtransmit
        snprintf(SIGstr, sizeof(SIGstr), "SIGCONT");
        SIGflag = 1;
        break;

    case SIGSTOP: // Stop process
        snprintf(SIGstr, sizeof(SIGstr), "SIGSTOP");
        SIGflag = 1;
        break;

    case SIGTSTP: // Stop typed at terminal
        snprintf(SIGstr, sizeof(SIGstr), "SIGTSTP");
        SIGflag = 1;
        break;

    case SIGTTIN: // Terminal input for background process
        snprintf(SIGstr, sizeof(SIGstr), "SIGTTIN");
        SIGflag = 1;
        break;

    case SIGTTOU: // Terminal output for background process
        snprintf(SIGstr, sizeof(SIGstr), "SIGTTOU");
        SIGflag = 1;
        break;

    case SIGBUS: // Bus error (bad memory access)
        snprintf(SIGstr, sizeof(SIGstr), "SIGBUS");
        SIGflag = 1;
        break;

    case SIGPOLL: // Pollable event (Sys V).
        snprintf(SIGstr, sizeof(SIGstr), "SIGPOLL");
        SIGflag = 1;
        break;

    case SIGPROF: // Profiling timer expired
        snprintf(SIGstr, sizeof(SIGstr), "SIGPROF");
        SIGflag = 1;
        break;

    case SIGSYS: // Bad system call (SVr4)
        snprintf(SIGstr, sizeof(SIGstr), "SIGSYS");
        SIGflag = 1;
        break;

    case SIGTRAP: // Trace/breakpoint trap
        snprintf(SIGstr, sizeof(SIGstr), "SIGTRAP");
        SIGflag = 1;
        break;

    case SIGURG: // Urgent condition on socket (4.2BSD)
        snprintf(SIGstr, sizeof(SIGstr), "SIGURG");
        SIGflag = 1;
        break;

    case SIGVTALRM: // Virtual alarm clock (4.2BSD)
        snprintf(SIGstr, sizeof(SIGstr), "SIGVTALRM");
        SIGflag = 1;
        break;

    case SIGXCPU: // CPU time limit exceeded (4.2BSD)
        snprintf(SIGstr, sizeof(SIGstr), "SIGXCPU");
        SIGflag = 1;
        break;

    case SIGXFSZ: // File size limit exceeded (4.2BSD)
        snprintf(SIGstr, sizeof(SIGstr), "SIGXFSZ");
        SIGflag = 1;
        break;
    }

    if(SIGflag == 1)
    {
        int slen = snprintf(msgstring,
                            STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                            "%s at %s",
                            SIGstr,
                            timestring);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }

        processinfo_WriteMessage(processinfo, msgstring);
    }

    return 0;
}
