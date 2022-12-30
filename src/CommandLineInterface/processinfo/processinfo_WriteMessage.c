#include <stdarg.h>

#include "CLIcore.h"
#include <processtools.h>


int processinfo_WriteMessage(
    PROCESSINFO *processinfo,
    const char *msgstring
)
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
            (int)(0.001 * (tnow.tv_nsec)),
            tnow.tv_sec,
            tnow.tv_nsec,
            (int) processinfo->PID,
            msgstring);

    DEBUG_TRACEPOINT(" ");
    fflush(processinfo->logFile);

    return EXIT_SUCCESS;
}



int processinfo_WriteMessage_fmt(
    PROCESSINFO *processinfo,
    const char *format,
    ...
)
{
    // determine required buffer size
    va_list args;
    va_start(args, format);
    int len = vsnprintf(NULL, 0, format, args);
    va_end(args);
    if(len < 0)
    {
        return EXIT_FAILURE;
    }

    // format message
    char msg[len +
                 1]; // or use heap allocation if implementation doesn't support VLAs
    va_start(args, format);
    vsnprintf(msg, len + 1, format, args);
    va_end(args);

    // call myFunction
    processinfo_WriteMessage(processinfo, msg);

    return EXIT_SUCCESS;
}
