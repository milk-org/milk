// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_labels.h
 * @brief Display names of processinfo loop states, control requests and trigger modes
 */

#ifndef _PROCESSINFO_LABELS_H
#define _PROCESSINFO_LABELS_H

#ifdef __cplusplus
extern "C"
{
#endif

const char *processinfo_loopstat_name(int loopstat, int long_form);
const char *processinfo_ctrlval_name(int ctrlval);
const char *processinfo_triggermode_name(int mode, int long_form);

#ifdef __cplusplus
}
#endif

#endif /* _PROCESSINFO_LABELS_H */
