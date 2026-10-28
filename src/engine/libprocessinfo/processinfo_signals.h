// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_signals.h
 * @brief Processinfo signals module
 */

#ifndef _PROCESSINFO_SIGNALS_H
#define _PROCESSINFO_SIGNALS_H

#include <signal.h>

#include "processinfo.h"

extern volatile sig_atomic_t processinfo_signal_USR1;
extern volatile sig_atomic_t processinfo_signal_USR2;
extern volatile sig_atomic_t processinfo_signal_TERM;
extern volatile sig_atomic_t processinfo_signal_INT;
extern volatile sig_atomic_t processinfo_signal_SEGV;
extern volatile sig_atomic_t processinfo_signal_ABRT;
extern volatile sig_atomic_t processinfo_signal_BUS;
extern volatile sig_atomic_t processinfo_signal_HUP;
extern volatile sig_atomic_t processinfo_signal_PIPE;

/**
 * processinfo_CatchSignals - SIGINT/TERM/HUP/PIPE end the loop; faults mark it CRASHED, then the
 * previous handler or the default action takes them. Nests; the first call saves the previous ones.
 */
int processinfo_CatchSignals();

/** processinfo_RestoreSignals - undo one processinfo_CatchSignals(); the last one restores. */
int processinfo_RestoreSignals();

/**
 * processinfo_set_crash_record - a fault also writes "signal=<NAME>" to @info_fd and "=== killed by <NAME>"
 * to @out_fd (-1: not written). The unit lifecycle passes the unit's run.info and output file.
 */
void processinfo_set_crash_record(int info_fd, int out_fd);

int processinfo_ProcessSignals(PROCESSINFO *processinfo);

int processinfo_cleanExit(PROCESSINFO *processinfo);

/** processinfo_exit_requested - nonzero once CTRLval asks for exit or SIGINT/TERM/HUP arrived. */
static inline int processinfo_exit_requested(const PROCESSINFO *processinfo)
{
    return processinfo->CTRLval == PROCESSINFO_CTRLVAL_EXIT || processinfo_signal_INT ||
           processinfo_signal_TERM || processinfo_signal_HUP;
}

#endif
