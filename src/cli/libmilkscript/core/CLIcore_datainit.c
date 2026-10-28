// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_datainit.c
 *
 * @brief data structure init
 *
 * Delegates core array allocation to milk_data_init()
 * and syncs data.core = milk_data so both globals share
 * the same image/variable/FPS arrays.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_signals.h"
#include "milk_path.h"

#include "COREMOD_memory/COREMOD_memory.h"

/*
 * Initialize the "data" structure.
 *
 * Core fields (image, variable, FPS arrays) are
 * allocated by milk_data_init(), then copied into
 * data.core so both DATA.core and milk_data refer
 * to the same storage.
 *
 * Calls milk_data_init() to allocate core arrays
 * (images, variables, FPS), then creates built-in
 * mathematical constants (_PI, _e, _c, etc.) and
 * seeds the random number generator.
 */
errno_t CLI_data_init()
{
    DEBUG_TRACE_FSTART();

    struct timeval t1;

    /* Allocate core arrays (image, variable, FPS) */
    milk_data_init();

    /* Sync: data.core shares milk_data's arrays */
    data.core = milk_data;

    create_variable_ID("_PI", 3.14159265358979323846264338328);
    create_variable_ID("_e", exp(1));
    create_variable_ID("_gamma", 0.5772156649);
    create_variable_ID("_c", 299792458.0);
    create_variable_ID("_h", 6.626075540e-34);
    create_variable_ID("_k", 1.38065812e-23);
    create_variable_ID("_pc", 3.0856776e16);
    create_variable_ID("_ly", 9.460730472e15);
    create_variable_ID("_AU", 1.4959787066e11);

    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * CLI_runtime_init() - process-level setup shared by milk-cli and milk-script:
 * install/source dirs, RNG, default flags and signal handlers.
 */
void CLI_export_own_installdir(void)
{
    const char *cur = getenv("MILK_INSTALLDIR");
    char        exe[1024];
    ssize_t     n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if ((cur != NULL && cur[0] != '\0') || n <= 0)
    {
        return;
    }
    exe[n]      = '\0';
    char *slash = strrchr(exe, '/');
    if (slash == NULL)
    {
        return;
    }
    *slash = '\0'; /* .../bin */
    slash  = strrchr(exe, '/');
    if (slash == NULL || strcmp(slash + 1, "bin") != 0)
    {
        return;
    }
    *slash = '\0'; /* the install directory */
    char script[1200];
    snprintf(script, sizeof(script), "%s/bin/milk.bash", exe);
    if (access(script, R_OK) == 0)
    {
        setenv("MILK_INSTALLDIR", exe, 0);
    }
}

errno_t CLI_runtime_init(void)
{
    // to take advantage of kernel priority:
    // owner=root mode=4755
    getresuid(&dcruid, &dceuid, &dcsuid);

    // Pure-C xorshift64* (replaces GSL)
    milk_rng_init((uint64_t) time(NULL));

    dcprogstatus = 0;

    CLI_export_own_installdir(); /* modules, milk.bash and units then all come from this install */
    char *installdir_env = getenv("MILK_INSTALLDIR");
    if (installdir_env != NULL)
    {
        milkpath_resolve_realpath(dcinstalldir, STRINGMAXLEN_DIRNAME, installdir_env);
    }
    else
    {
#ifdef INSTALLDIR
        milkpath_resolve_realpath(dcinstalldir, STRINGMAXLEN_DIRNAME, INSTALLDIR);
#else
        milkpath_resolve_realpath(dcinstalldir, STRINGMAXLEN_DIRNAME, "/usr/local/milk");
#endif
    }

    char *sourcedir_env = getenv("MILK_SOURCEDIR");
    if (sourcedir_env != NULL)
    {
        strncpy(dcsourcedir, sourcedir_env, STRINGMAXLEN_DIRNAME - 1);
    }
    else
    {
#ifdef SOURCEDIR
        strncpy(dcsourcedir, SOURCEDIR, STRINGMAXLEN_DIRNAME - 1);
#else
        strncpy(dcsourcedir, "", STRINGMAXLEN_DIRNAME - 1);
#endif
    }

    dcdebug     = 0;
    dcoverwrite = 0;
    dcprecision = 0; // float is default precision
    dcshareddft = 0; // do not allocate shared memory for images
    snprintf(dcsavedir, STRINGMAXLEN_DIRNAME, ".");

    data.CLIlogON = 0; // log every command
    data.fifoON   = 0;
    data.fifofd   = -1;
    dcprocinfo    = 1; // process info for intensive processes
    dcprocinfoact = 0; // toggles to 1 when process is logged

    dcsigact.sa_handler = sig_handler;
    sigemptyset(&dcsigact.sa_mask);
    dcsigact.sa_flags = 0;

    // Request the kernel for a sigint if parent dies
    // This is useful if stdin is a pipe from the parent process
    // and the parent dies suddenly. This confuses libreadline.
    prctl(PR_SET_PDEATHSIG, SIGINT);

    dcsigUSR1 = 0;
    dcsigUSR2 = 0;
    dcsigTERM = 0;
    dcsigINT  = 0;
    dcsigBUS  = 0;
    dcsigSEGV = 0;
    dcsigABRT = 0;
    dcsigHUP  = 0;
    dcsigPIPE = 0;

    if (sigaction(SIGUSR1, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR1\n");
    }
    if (sigaction(SIGUSR2, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR2\n");
    }

    set_signal_catch();

    return RETURN_SUCCESS;
}
