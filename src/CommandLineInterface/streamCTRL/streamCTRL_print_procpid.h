// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _STREAMCTRL_PRINT_PROCPID_H
#define _STREAMCTRL_PRINT_PROCPID_H


int streamCTRL_print_procpid(
    int      DispPID_NBchar,
    pid_t    procpid,
    pid_t   *upstreamproc,
    int      NBupstreamproc,
    uint32_t mode
);

#endif
