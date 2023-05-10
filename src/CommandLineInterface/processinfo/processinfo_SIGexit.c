#include "CLIcore.h"
#include <processtools.h>


int processinfo_SIGexit(PROCESSINFO *processinfo, int SignalNumber)
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
            strcpy(SIGstr, "SIGHUP");
            SIGflag = 1;
            break;

        case SIGINT: // Interrupt from keyboard
            strcpy(SIGstr, "SIGINT");
            SIGflag = 1;
            break;

        case SIGQUIT: // Quit from keyboard
            strcpy(SIGstr, "SIGQUIT");
            SIGflag = 1;
            break;

        case SIGILL: // Illegal Instruction
            strcpy(SIGstr, "SIGILL");
            SIGflag = 1;
            break;

        case SIGABRT: // Abort signal from abort
            strcpy(SIGstr, "SIGABRT");
            SIGflag = 1;
            break;

        case SIGFPE: // Floating-point exception
            strcpy(SIGstr, "SIGFPE");
            SIGflag = 1;
            break;

        case SIGKILL: // Kill signal
            strcpy(SIGstr, "SIGKILL");
            SIGflag = 1;
            break;

        case SIGSEGV: // Invalid memory reference
            strcpy(SIGstr, "SIGSEGV");
            SIGflag = 1;
            break;

        case SIGPIPE: // Broken pipe: write to pipe with no readers
            strcpy(SIGstr, "SIGPIPE");
            SIGflag = 1;
            break;

        case SIGALRM: // Timer signal from alarm
            strcpy(SIGstr, "SIGALRM");
            SIGflag = 1;
            break;

        case SIGTERM: // Termination signal
            strcpy(SIGstr, "SIGTERM");
            SIGflag = 1;
            break;

        case SIGUSR1: // User-defined signal 1
            strcpy(SIGstr, "SIGUSR1");
            SIGflag = 1;
            break;

        case SIGUSR2: // User-defined signal 1
            strcpy(SIGstr, "SIGUSR2");
            SIGflag = 1;
            break;

        case SIGCHLD: // Child stopped or terminated
            strcpy(SIGstr, "SIGCHLD");
            SIGflag = 1;
            break;

        case SIGCONT: // Continue if stoppedshmimTCPtransmit
            strcpy(SIGstr, "SIGCONT");
            SIGflag = 1;
            break;

        case SIGSTOP: // Stop process
            strcpy(SIGstr, "SIGSTOP");
            SIGflag = 1;
            break;

        case SIGTSTP: // Stop typed at terminal
            strcpy(SIGstr, "SIGTSTP");
            SIGflag = 1;
            break;

        case SIGTTIN: // Terminal input for background process
            strcpy(SIGstr, "SIGTTIN");
            SIGflag = 1;
            break;

        case SIGTTOU: // Terminal output for background process
            strcpy(SIGstr, "SIGTTOU");
            SIGflag = 1;
            break;

        case SIGBUS: // Bus error (bad memory access)
            strcpy(SIGstr, "SIGBUS");
            SIGflag = 1;
            break;

        case SIGPOLL: // Pollable event (Sys V).
            strcpy(SIGstr, "SIGPOLL");
            SIGflag = 1;
            break;

        case SIGPROF: // Profiling timer expired
            strcpy(SIGstr, "SIGPROF");
            SIGflag = 1;
            break;

        case SIGSYS: // Bad system call (SVr4)
            strcpy(SIGstr, "SIGSYS");
            SIGflag = 1;
            break;

        case SIGTRAP: // Trace/breakpoint trap
            strcpy(SIGstr, "SIGTRAP");
            SIGflag = 1;
            break;

        case SIGURG: // Urgent condition on socket (4.2BSD)
            strcpy(SIGstr, "SIGURG");
            SIGflag = 1;
            break;

        case SIGVTALRM: // Virtual alarm clock (4.2BSD)
            strcpy(SIGstr, "SIGVTALRM");
            SIGflag = 1;
            break;

        case SIGXCPU: // CPU time limit exceeded (4.2BSD)
            strcpy(SIGstr, "SIGXCPU");
            SIGflag = 1;
            break;

        case SIGXFSZ: // File size limit exceeded (4.2BSD)
            strcpy(SIGstr, "SIGXFSZ");
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
