// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_RT_H
#define MILK_RT_H

int milkrt_RTPrio(const int rtprio);
int milkrt_Tset(const char *tsetspec);
int milkrt_TsetExt(const int pid, const char *tsetspec);
int milkrt_CPUset(const char *csetname);
int milkrt_CPUsetExt(const int pid, const char *csetname, const int rtprio);

#endif
