// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_labels.c
 * @brief Display names of processinfo loop states, control requests and trigger modes: one table each,
 *        shared by milk-CTRL, procCTRL, fpsCTRL and the command-line tools.
 */

#include "processinfo.h"
#include "processinfo_labels.h"
#include "processtools_trigger.h"

/** processinfo_loopstat_name - PROCESSINFO_LOOPSTAT_* name: short (at most 4 chars, "RUN") or long ("RUNNING"). */
const char *processinfo_loopstat_name(int loopstat, int long_form)
{
    static const char *const sn[] = { "INIT", "RUN", "PAUS", "STOP", "ERR", "SPIN", "CRSH" };
    static const char *const ln[] = { "INIT",  "RUNNING",  "PAUSED", "STOPPED",
                                      "ERROR", "SPINNING", "CRASHED" };
    if (loopstat < PROCESSINFO_LOOPSTAT_INIT || loopstat > PROCESSINFO_LOOPSTAT_CRASHED)
    {
        return long_form ? "UNKNOWN" : "??";
    }
    return long_form ? ln[loopstat] : sn[loopstat];
}

/** processinfo_ctrlval_name - PROCESSINFO_CTRLVAL_* request name: RUN, PAUSE, STEP or EXIT. */
const char *processinfo_ctrlval_name(int ctrlval)
{
    static const char *const n[] = { "RUN", "PAUSE", "STEP", "EXIT" };
    return (ctrlval < PROCESSINFO_CTRLVAL_RUN || ctrlval > PROCESSINFO_CTRLVAL_EXIT) ? "UNKNOWN"
                                                                                     : n[ctrlval];
}

/** processinfo_triggermode_name - PROCESSINFO_TRIGGERMODE_* name: short (3 chars, "CN0") or long ("CNT0"). */
const char *processinfo_triggermode_name(int mode, int long_form)
{
    static const char *const sn[] = { "IMM", "CN0", "CN1", "SEM", "DLY", "SMP", "CN2" };
    static const char *const ln[] = { "IMMEDIATE", "CNT0",         "CNT1", "SEMAPHORE",
                                      "DELAY",     "SEM+TIMEOUTS", "CNT2" };
    if (mode < PROCESSINFO_TRIGGERMODE_IMMEDIATE || mode > PROCESSINFO_TRIGGERMODE_CNT2)
    {
        return long_form ? "UNKNOWN" : " - ";
    }
    return long_form ? ln[mode] : sn[mode];
}
