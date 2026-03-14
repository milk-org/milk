/**
 * @file processinfo_signals.h
 * @brief Processinfo signals module
 */

#ifndef _PROCESSINFO_SIGNALS_H
#define _PROCESSINFO_SIGNALS_H

#include "processinfo.h"

extern int processinfo_signal_USR1;
extern int processinfo_signal_USR2;
extern int processinfo_signal_TERM;
extern int processinfo_signal_INT;
extern int processinfo_signal_SEGV;
extern int processinfo_signal_ABRT;
extern int processinfo_signal_BUS;
extern int processinfo_signal_HUP;
extern int processinfo_signal_PIPE;

int processinfo_CatchSignals();
int processinfo_ProcessSignals(PROCESSINFO *processinfo);

int processinfo_cleanExit(PROCESSINFO *processinfo);

#endif
