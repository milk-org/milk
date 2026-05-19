/**
 * @file    processinfo_globals.c
 * @brief   Global variables for processinfo library
 */

#include "processinfo.h"

// List of active processes in shared memory
PROCESSINFOLIST *pinfolist = NULL;

pid_t CLIPID = 0;

// Signals
int processinfo_signal_USR1 = 0;
int processinfo_signal_USR2 = 0;
int processinfo_signal_TERM = 0;
int processinfo_signal_INT  = 0;
int processinfo_signal_SEGV = 0;
int processinfo_signal_ABRT = 0;
int processinfo_signal_BUS  = 0;
int processinfo_signal_HUP  = 0;
int processinfo_signal_PIPE = 0;
