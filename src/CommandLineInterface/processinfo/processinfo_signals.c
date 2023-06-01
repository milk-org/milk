#include "CLIcore.h"
#include <processtools.h>
#include "processinfo_procdirname.h"


int processinfo_CatchSignals()
{
    if(sigaction(SIGTERM, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGTERM\n");
    }

    if(sigaction(SIGINT, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGINT\n");
    }

    if(sigaction(SIGABRT, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGABRT\n");
    }

    if(sigaction(SIGBUS, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGBUS\n");
    }

    if(sigaction(SIGSEGV, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGSEGV\n");
    }

    if(sigaction(SIGHUP, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGHUP\n");
    }

    if(sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGPIPE\n");
    }

    return 0;
}



int processinfo_ProcessSignals(PROCESSINFO *processinfo)
{
    int loopOK = 1;
    // process signals

    if(data.signal_TERM == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGTERM);
    }

    if(data.signal_INT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGINT);
    }

    if(data.signal_ABRT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGABRT);
    }

    if(data.signal_BUS == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGBUS);
    }

    if(data.signal_SEGV == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGSEGV);
    }

    if(data.signal_HUP == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGHUP);
    }

    if(data.signal_PIPE == 1)
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
