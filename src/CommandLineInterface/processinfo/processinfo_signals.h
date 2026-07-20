// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _PROCESSINFO_SIGNALS_H
#define _PROCESSINFO_SIGNALS_H


int processinfo_CatchSignals();

int processinfo_ProcessSignals(PROCESSINFO *processinfo);

int processinfo_cleanExit(PROCESSINFO *processinfo);

#endif
