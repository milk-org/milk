/**
 * @file processinfo_WriteMessage.c
 * @brief Processinfo writemessage module
 */

#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_WriteMessage.h"

#ifndef CLOCK_MILK
#define CLOCK_MILK CLOCK_REALTIME
#endif


int processinfo_WriteMessage(
    PROCESSINFO *processinfo,
    const char  *msgstring
)
{
    struct timespec tnow;
    struct tm      *tnowtm;
    clock_gettime(CLOCK_MILK, &tnow);
    tnowtm = gmtime(&tnow.tv_sec);

    snprintf(processinfo->statusmsg,
             STRINGMAXLEN_PROCESSINFO_STATUSMSG,
             "%02d:%02d:%02d.%03d %s",
             tnowtm->tm_hour,
             tnowtm->tm_min,
             tnowtm->tm_sec,
             (int)(0.000001 * (tnow.tv_nsec)),
             msgstring);

    if(processinfo->PID == 0) // not initialized
    {
        strcpy(processinfo->statusmsg, msgstring);
    }

#ifdef PROCESSINFO_LOGFILE
    if(processinfo->logFile != NULL)
    {
        fprintf(processinfo->logFile,
                "%02d:%02d:%02d.%09ld %06d %s\n",
                tnowtm->tm_hour,
                tnowtm->tm_min,
                tnowtm->tm_sec,
                tnow.tv_nsec,
                (int) processinfo->PID,
                msgstring);

        fflush(processinfo->logFile);
    }
#endif

    return 0;
}


int processinfo_WriteMessage_fmt(
    PROCESSINFO *processinfo,
    const char *format,
    ...
)
{
    va_list args;
    char msg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

    va_start(args, format);
    vsnprintf(msg, STRINGMAXLEN_PROCESSINFO_STATUSMSG, format, args);
    va_end(args);

    processinfo_WriteMessage(processinfo, msg);

    return 0;
}