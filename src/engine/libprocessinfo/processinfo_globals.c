// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    processinfo_globals.c
 * @brief   Global variables for processinfo library
 */

#include "processinfo_internal.h"
#include "processinfo_signals.h"

// List of active processes in shared memory
PROCESSINFOLIST *pinfolist = NULL;

pid_t CLIPID = 0;

// This process's processinfo, for the fatal-signal handler
PROCESSINFO *volatile processinfo_self = NULL;

// Signals
volatile sig_atomic_t processinfo_signal_USR1 = 0;
volatile sig_atomic_t processinfo_signal_USR2 = 0;
volatile sig_atomic_t processinfo_signal_TERM = 0;
volatile sig_atomic_t processinfo_signal_INT  = 0;
volatile sig_atomic_t processinfo_signal_SEGV = 0;
volatile sig_atomic_t processinfo_signal_ABRT = 0;
volatile sig_atomic_t processinfo_signal_BUS  = 0;
volatile sig_atomic_t processinfo_signal_HUP  = 0;
volatile sig_atomic_t processinfo_signal_PIPE = 0;
