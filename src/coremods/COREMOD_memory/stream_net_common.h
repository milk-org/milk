/**
 * @file    stream_net_common.h
 * @brief   Shared helpers for TCP/UDP stream transport
 *
 * Static inline functions and macros used by both
 * stream_TCP.c and stream_UDP.c to eliminate
 * duplicated boilerplate.
 */

#ifndef STREAM_NET_COMMON_H
#define STREAM_NET_COMMON_H

#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"
#include "processinfo.h"
#include "processinfo_SIGexit.h"
#include "processinfo_WriteMessage.h"
#include "milkDebugTools.h"
#include "milkdata.h"

#ifndef CLOCK_MILK
#define CLOCK_MILK CLOCK_REALTIME
#endif


/* ============================================================
 * Signal dispatch — replaces 7-way if/else if chain
 * calling processinfo_SIGexit() for each dcsig* flag.
 * ========================================================= */

/**
 * DCSIG_PROCESS_EXIT - dispatch processinfo SIGexit
 * for the first set dcsig* flag.
 *
 * Must be called only when DCSIG_ANY_SET() is true
 * and dcprocinfo == 1.
 */
#define DCSIG_PROCESS_EXIT(pinfo) \
    do { \
        if (dcsigTERM) \
            processinfo_SIGexit((pinfo), SIGTERM); \
        else if (dcsigINT) \
            processinfo_SIGexit((pinfo), SIGINT); \
        else if (dcsigABRT) \
            processinfo_SIGexit((pinfo), SIGABRT); \
        else if (dcsigBUS) \
            processinfo_SIGexit((pinfo), SIGBUS); \
        else if (dcsigSEGV) \
            processinfo_SIGexit((pinfo), SIGSEGV); \
        else if (dcsigHUP) \
            processinfo_SIGexit((pinfo), SIGHUP); \
        else if (dcsigPIPE) \
            processinfo_SIGexit((pinfo), SIGPIPE); \
    } while (0)


/* ============================================================
 * Signal handler installation
 * ========================================================= */

#ifdef MILK_NO_CLI
/**
 * stream_net_sig_handler_standalone - set dcsig* flag
 * for the received signal.
 *
 * Used only in MILK_NO_CLI builds where sig_handler()
 * from CLIcore_standalone.h is a no-op and does not
 * populate the dcsig* flags needed by DCSIG_ANY_SET().
 */
static void stream_net_sig_handler_standalone(int signo)
{
    switch (signo)
    {
    case SIGTERM: dcsigTERM = 1; break;
    case SIGINT:  dcsigINT  = 1; break;
    case SIGABRT: dcsigABRT = 1; break;
    case SIGBUS:  dcsigBUS  = 1; break;
    case SIGSEGV: dcsigSEGV = 1; break;
    case SIGHUP:  dcsigHUP  = 1; break;
    case SIGPIPE: dcsigPIPE = 1; break;
    default: break;
    }
}
#endif /* MILK_NO_CLI */

/**
 * stream_net_signal_catch - install signal handlers
 *
 * In CLI builds delegates to set_signal_catch().
 * In standalone (MILK_NO_CLI) builds calls sigaction()
 * directly, since set_signal_catch() is a no-op there.
 */
static inline void stream_net_signal_catch(void)
{
#ifdef MILK_NO_CLI
    dcsigact.sa_handler = stream_net_sig_handler_standalone;
    sigemptyset(&dcsigact.sa_mask);
    dcsigact.sa_flags = 0;

    if (sigaction(SIGTERM, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGTERM");
    if (sigaction(SIGINT, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGINT");
    if (sigaction(SIGABRT, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGABRT");
    if (sigaction(SIGBUS, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGBUS");
    if (sigaction(SIGSEGV, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGSEGV");
    if (sigaction(SIGHUP, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGHUP");
    if (sigaction(SIGPIPE, &dcsigact, NULL) == -1)
        PRINT_ERROR("can't catch SIGPIPE");
#else
    set_signal_catch();
#endif
}


/* ============================================================
 * RT scheduling setup
 * ========================================================= */

/**
 * stream_net_rt_sched_set - set SCHED_FIFO priority
 * @priority: 0-99, higher = more priority
 *
 * Temporarily elevates to euid, sets scheduler,
 * then drops back to ruid.
 */
static inline void stream_net_rt_sched_set(
    int priority)
{
    struct sched_param sp;
    sp.sched_priority = priority;

    if (seteuid(dceuid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO, &sp);
    if (seteuid(dcruid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }
}


/* ============================================================
 * Semaphore drain — drive semaphore to zero on
 * first iteration of transmit loop.
 * ========================================================= */

/**
 * stream_net_sem_drain - drain semaphore on first
 * iteration
 * @img:      IMAGE pointer
 * @semtrig:  semaphore index
 * @iter_p:   pointer to iteration counter
 * @pinfo:    PROCESSINFO pointer
 *
 * On the first call (*iter_p == 0), drains the
 * semaphore and increments *iter_p.
 */
static inline void stream_net_sem_drain(
    IMAGE *img,
    int semtrig,
    long long *iter_p,
    PROCESSINFO *pinfo)
{
    if (*iter_p == 0)
    {
        processinfo_WriteMessage(
            pinfo, "Driving sem to 0");
        printf("Driving semaphore to zero ... ");
        fflush(stdout);

        int semval =
            ImageStreamIO_semvalue(img, semtrig);
        int semvalcnt = semval;
        for (long scnt = 0;
             scnt < semvalcnt; scnt++)
        {
            semval =
                ImageStreamIO_semvalue(
                    img, semtrig);
            printf("sem = %d\n", semval);
            fflush(stdout);
            ImageStreamIO_semtrywait(
                img, semtrig);
        }
        printf("done\n");
        fflush(stdout);

        semval =
            ImageStreamIO_semvalue(img, semtrig);
        printf("-> sem = %d\n", semval);
        fflush(stdout);

        (*iter_p)++;
    }
}


/* ============================================================
 * Slice tracking — clamp slice index to valid range
 * ========================================================= */

/**
 * stream_net_clamp_slice - clamp frame slice index
 * @raw_slice: incoming cnt1 value
 * @old_slice: previous slice
 * @nb_slices: total number of slices
 *
 * Returns the clamped slice index.
 */
static inline int stream_net_clamp_slice(
    int raw_slice,
    int old_slice,
    int nb_slices)
{
    int slice = raw_slice;

    if (slice > old_slice + 1)
    {
        slice = old_slice + 1;
    }
    if (nb_slices > 1)
    {
        if (old_slice == nb_slices - 1)
        {
            slice = 0;
        }
    }
    if (slice > nb_slices - 1)
    {
        slice = 0;
    }
    return slice;
}


/* ============================================================
 * Sync mode decision — sem vs counter
 * ========================================================= */

/**
 * stream_net_decide_sync - decide semaphore vs
 * counter synchronization
 * @sem_count:  number of semaphores on image
 * @force_cnt:  1 to force counter sync (mode param)
 * @semtrig:    semaphore index to use if sem sync
 * @pinfo:      PROCESSINFO pointer
 *
 * Returns 1 for semaphore sync, 0 for counter sync.
 */
static inline int stream_net_decide_sync(
    int sem_count,
    int force_cnt,
    int semtrig,
    PROCESSINFO *pinfo)
{
    if (sem_count == 0 || force_cnt == 1)
    {
        processinfo_WriteMessage(
            pinfo, "sync using counter");
        return 0;
    }

    char msg[200];
    snprintf(msg, 200,
             "sync using semaphore %d", semtrig);
    processinfo_WriteMessage(pinfo, msg);
    return 1;
}


/* ============================================================
 * Semaphore-based wait with timeout
 * ========================================================= */

/**
 * stream_net_sem_wait - timed semaphore wait
 * @img:      IMAGE pointer
 * @semtrig:  semaphore index
 *
 * Returns 0 on success, non-zero on timeout.
 */
static inline int stream_net_sem_wait(
    IMAGE *img,
    int semtrig)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MILK, &ts) == -1)
    {
        PRINT_ERROR("clock_gettime: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }
    ts.tv_sec += 2;
    return ImageStreamIO_semtimedwait(
        img, semtrig, &ts);
}


#endif /* STREAM_NET_COMMON_H */
