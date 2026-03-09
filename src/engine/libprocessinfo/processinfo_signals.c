#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <string.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_signals.h"
#include "processinfo_SIGexit.h"
#include "processinfo_procdirname.h"
#include "processinfo_WriteMessage.h"

#ifndef CLOCK_MILK
#define CLOCK_MILK CLOCK_REALTIME
#endif

void processinfo_sig_handler(int signo)
{
    switch(signo)
    {
        case SIGUSR1:
            processinfo_signal_USR1 = 1;
            break;
        case SIGUSR2:
            processinfo_signal_USR2 = 1;
            break;
        case SIGINT:
            processinfo_signal_INT = 1;
            break;
        case SIGTERM:
            processinfo_signal_TERM = 1;
            break;
        case SIGSEGV:
            processinfo_signal_SEGV = 1;
            break;
        case SIGABRT:
            processinfo_signal_ABRT = 1;
            break;
        case SIGBUS:
            processinfo_signal_BUS = 1;
            break;
        case SIGHUP:
            processinfo_signal_HUP = 1;
            break;
        case SIGPIPE:
            processinfo_signal_PIPE = 1;
            break;
    }
}

int processinfo_CatchSignals()
{
    struct sigaction sigact;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigact.sa_handler = processinfo_sig_handler;

    if(sigaction(SIGTERM, &sigact, NULL) == -1) printf("\ncan't catch SIGTERM\n");
    if(sigaction(SIGINT, &sigact, NULL) == -1) printf("\ncan't catch SIGINT\n");
    if(sigaction(SIGABRT, &sigact, NULL) == -1) printf("\ncan't catch SIGABRT\n");
    if(sigaction(SIGBUS, &sigact, NULL) == -1) printf("\ncan't catch SIGBUS\n");
    if(sigaction(SIGSEGV, &sigact, NULL) == -1) printf("\ncan't catch SIGSEGV\n");
    if(sigaction(SIGHUP, &sigact, NULL) == -1) printf("\ncan't catch SIGHUP\n");
    if(sigaction(SIGPIPE, &sigact, NULL) == -1) printf("\ncan't catch SIGPIPE\n");

    return 0;
}

int processinfo_ProcessSignals(PROCESSINFO *processinfo)
{
    int loopOK = 1;
    // process signals

    if(processinfo_signal_TERM == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGTERM);
    }

    if(processinfo_signal_INT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGINT);
    }

    if(processinfo_signal_ABRT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGABRT);
    }

    if(processinfo_signal_BUS == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGBUS);
    }

    if(processinfo_signal_SEGV == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGSEGV);
    }

    if(processinfo_signal_HUP == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGHUP);
    }

    if(processinfo_signal_PIPE == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGPIPE);
    }

    return loopOK;
}

int processinfo_cleanExit(PROCESSINFO *processinfo)
{

    if(processinfo->loopstat != 4)
    {
        struct timespec tstop;
        struct tm      *tstoptm;
        char            msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

        clock_gettime(CLOCK_MILK, &tstop);
        tstoptm = gmtime(&tstop.tv_sec);

        if(processinfo->CTRLval == 3)  // loop exit from processinfo control
        {
            snprintf(msgstring,
                     STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                     "CTRLexit %02d:%02d:%02d.%03d",
                     tstoptm->tm_hour,
                     tstoptm->tm_min,
                     tstoptm->tm_sec,
                     (int)(0.000001 * (tstop.tv_nsec)));
            strncpy(processinfo->statusmsg,
                    msgstring,
                    STRINGMAXLEN_PROCESSINFO_STATUSMSG - 1);
        }

        if(processinfo->loopstat == 1)
        {
            snprintf(msgstring,
                     STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                     "Loop exit %02d:%02d:%02d.%03d",
                     tstoptm->tm_hour,
                     tstoptm->tm_min,
                     tstoptm->tm_sec,
                     (int)(0.000001 * (tstop.tv_nsec)));
            strncpy(processinfo->statusmsg,
                    msgstring,
                    STRINGMAXLEN_PROCESSINFO_STATUSMSG - 1);
        }

        processinfo->loopstat = 3; // clean exit
    }

    // Remove processinfo shm file on clean exit
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    char SM_fname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(SM_fname,
                       "%s/proc.%s.%06d.shm",
                       procdname,
                       processinfo->name,
                       processinfo->PID);
    remove(SM_fname);

    return 0;
}