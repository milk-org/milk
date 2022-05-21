#include "CLIcore.h"
#include <processtools.h>


int processinfo_WriteMessage(PROCESSINFO *processinfo, const char *msgstring)
{
    struct timespec tnow;
    struct tm      *tmnow;

    DEBUG_TRACEPOINT(" ");

    clock_gettime(CLOCK_REALTIME, &tnow);
    tmnow = gmtime(&tnow.tv_sec);

    strcpy(processinfo->statusmsg, msgstring);

    DEBUG_TRACEPOINT(" ");

    fprintf(processinfo->logFile,
            "%02d:%02d:%02d.%06d  %8ld.%09ld  %06d  %s\n",
            tmnow->tm_hour,
            tmnow->tm_min,
            tmnow->tm_sec,
            (int) (0.001 * (tnow.tv_nsec)),
            tnow.tv_sec,
            tnow.tv_nsec,
            (int) processinfo->PID,
            msgstring);

    DEBUG_TRACEPOINT(" ");
    fflush(processinfo->logFile);
    return 0;
}
