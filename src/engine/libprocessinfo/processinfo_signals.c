// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_signals.c
 * @brief Processinfo signals module
 */

#include <pthread.h>
#include <signal.h>

#include "processinfo_internal.h"
#include "processinfo_signals.h"
#include "processinfo_SIGexit.h"
#include "processinfo_procdirname.h"

#ifndef CLOCK_MILK
#    define CLOCK_MILK CLOCK_REALTIME
#endif

/* The first PI_NB_STOP_SIGNALS end the loop through flags; the others are faults. */
static const int pi_signals[] = { SIGTERM, SIGINT,  SIGHUP, SIGPIPE, SIGSEGV,
                                  SIGBUS,  SIGABRT, SIGFPE, SIGILL };
#define PI_NB_SIGNALS ((int) (sizeof(pi_signals) / sizeof(pi_signals[0])))
#define PI_NB_STOP_SIGNALS 4

static struct sigaction pi_prev[PI_NB_SIGNALS]; // dispositions before the first CatchSignals()
static int              pi_depth = 0;           // CatchSignals() calls not yet restored
static pthread_mutex_t  pi_mutex = PTHREAD_MUTEX_INITIALIZER;

static volatile sig_atomic_t pi_crash_info_fd = -1; // processinfo_set_crash_record()
static volatile sig_atomic_t pi_crash_out_fd  = -1;

void processinfo_set_crash_record(int info_fd, int out_fd)
{
    pi_crash_info_fd = info_fd;
    pi_crash_out_fd  = out_fd;
}

/* async-signal-safe: write @a @b "\n" to @fd */
static void pi_crash_line(int fd, const char *a, const char *b)
{
    char   line[64];
    size_t n = 0;

    if (fd < 0)
    {
        return;
    }
    for (const char *s = a; *s != '\0' && n < sizeof(line) - 1; s++)
    {
        line[n++] = *s;
    }
    for (const char *s = b; *s != '\0' && n < sizeof(line) - 1; s++)
    {
        line[n++] = *s;
    }
    line[n++] = '\n';
    (void) !write(fd, line, n);
}

/**
 * @brief Signal handler for process info processes.
 */
void processinfo_sig_handler(int signo)
{
    switch (signo)
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

static const char *fatal_signal_name(int signo)
{
    switch (signo)
    {
    case SIGSEGV:
        return "SIGSEGV";
    case SIGBUS:
        return "SIGBUS";
    case SIGABRT:
        return "SIGABRT";
    case SIGFPE:
        return "SIGFPE";
    case SIGILL:
        return "SIGILL";
    }
    return "fatal signal";
}

/* Returning would re-run a faulting instruction forever: record the crash with plain stores (the
 * only async-signal-safe way), then hand the signal to the previous disposition. */
static void MILK_COLD processinfo_fatal_handler(int signo)
{
    PROCESSINFO *pi   = processinfo_self;
    const char  *name = fatal_signal_name(signo);
    pi_crash_line(pi_crash_info_fd, "signal=", name);
    pi_crash_line(pi_crash_out_fd, "=== killed by ", name);
    if (pi != NULL)
    {
        int n        = 0;
        pi->loopstat = PROCESSINFO_LOOPSTAT_CRASHED;
        for (; name[n] != '\0' && n < STRINGMAXLEN_PROCESSINFO_STATUSMSG - 1; n++)
        {
            pi->statusmsg[n] = name[n];
        }
        pi->statusmsg[n] = '\0';
    }

    struct sigaction next;
    sigemptyset(&next.sa_mask);
    next.sa_flags   = 0;
    next.sa_handler = SIG_DFL;
    for (int i = PI_NB_STOP_SIGNALS; i < PI_NB_SIGNALS; i++)
    {
        if (pi_signals[i] == signo && pi_prev[i].sa_handler != SIG_IGN)
        {
            next = pi_prev[i]; // e.g. milk-cli's crash report
        }
    }
    sigaction(signo, &next, NULL);
    raise(signo); // pending until this handler returns
}

int processinfo_CatchSignals()
{
    struct sigaction stop_act;
    struct sigaction fatal_act;

    sigemptyset(&stop_act.sa_mask);
    stop_act.sa_flags   = 0;
    stop_act.sa_handler = processinfo_sig_handler;
    sigemptyset(&fatal_act.sa_mask);
    fatal_act.sa_flags   = SA_ONSTACK; // uses the caller's alternate stack if it set one
    fatal_act.sa_handler = processinfo_fatal_handler;

    pthread_mutex_lock(&pi_mutex);
    if (pi_depth++ == 0)
    {
        // a stop requested during an earlier loop of this process must not end this one
        processinfo_signal_TERM = 0;
        processinfo_signal_INT  = 0;
        processinfo_signal_HUP  = 0;
        processinfo_signal_PIPE = 0;
        for (int i = 0; i < PI_NB_SIGNALS; i++)
        {
            if (sigaction(pi_signals[i], (i < PI_NB_STOP_SIGNALS) ? &stop_act : &fatal_act,
                          &pi_prev[i]) == -1)
            {
                printf("\ncan't catch signal %d\n", pi_signals[i]);
            }
        }
    }
    pthread_mutex_unlock(&pi_mutex);

    return 0;
}

int processinfo_RestoreSignals()
{
    pthread_mutex_lock(&pi_mutex);
    if (pi_depth > 0 && --pi_depth == 0)
    {
        for (int i = 0; i < PI_NB_SIGNALS; i++)
        {
            sigaction(pi_signals[i], &pi_prev[i], NULL);
        }
    }
    pthread_mutex_unlock(&pi_mutex);

    return 0;
}

/**
 * @brief Processes any pending signals for the current process.
 */
int processinfo_ProcessSignals(PROCESSINFO *processinfo)
{
    int loopOK = 1;
    // process signals

    if (processinfo_signal_TERM == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGTERM);
    }

    if (processinfo_signal_INT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGINT);
    }

    if (processinfo_signal_ABRT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGABRT);
    }

    if (processinfo_signal_BUS == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGBUS);
    }

    if (processinfo_signal_SEGV == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGSEGV);
    }

    if (processinfo_signal_HUP == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGHUP);
    }

    if (processinfo_signal_PIPE == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGPIPE);
    }

    return loopOK;
}

/**
 * @brief Performs a clean exit for a process info instance.
 */
int processinfo_cleanExit(PROCESSINFO *processinfo)
{
    if (processinfo->loopstat != 4)
    {
        struct timespec tstop;
        struct tm      *tstoptm;
        char            msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

        clock_gettime(CLOCK_MILK, &tstop);
        tstoptm = gmtime(&tstop.tv_sec);

        if (processinfo->CTRLval == 3) // loop exit from processinfo control
        {
            snprintf(msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "CTRLexit %02d:%02d:%02d.%03d",
                     tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec,
                     (int) (0.000001 * (tstop.tv_nsec)));
            strncpy(processinfo->statusmsg, msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG - 1);
        }

        if (processinfo->loopstat == 1)
        {
            snprintf(msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "Loop exit %02d:%02d:%02d.%03d",
                     tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec,
                     (int) (0.000001 * (tstop.tv_nsec)));
            strncpy(processinfo->statusmsg, msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG - 1);
        }

        processinfo->loopstat = 3; // clean exit
    }

    // Remove processinfo shm file on clean exit
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    char SM_fname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname, processinfo->name,
                       processinfo->PID);
    remove(SM_fname);

    if (processinfo_self == processinfo)
    {
        processinfo_self = NULL; // its file is gone: a later crash has nothing to record
    }

    return 0;
}
