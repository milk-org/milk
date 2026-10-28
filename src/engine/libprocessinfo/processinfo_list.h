// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_list.h
 * @brief Maintenance of the process registry (processinfo.list.shm) by viewers and tools
 */

#ifndef _PROCESSINFO_LIST_H
#define _PROCESSINFO_LIST_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define PROCESSINFO_REAP_FREE_SLOTS \
    0x1 /* free every dead process's slot, not only those without a proc file */
#define PROCESSINFO_REAP_UNLINK 0x2 /* also remove dead processes' proc files (implies freeing) */

int processinfo_list_reap(unsigned flags);
int processinfo_list_clear_slot(long pindex, pid_t pid);
int processinfo_pid_gone(pid_t pid, double createtime);

#ifdef __cplusplus
}
#endif

#endif /* _PROCESSINFO_LIST_H */
