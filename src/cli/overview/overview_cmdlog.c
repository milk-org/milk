// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_cmdlog.c
 * @brief Command log ring buffer for milk-CTRL
 *
 * Provides a simple ring-buffer API for posting
 * timestamped, leveled log entries that the TUI
 * renders at the bottom of the screen.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "overview_layout.h"

/**
 * ov_cmdlog_push - append a log entry.
 * @log:   command log ring buffer
 * @level: severity / status level
 * @fmt:   printf-style format string
 *
 * Writes into the next slot, advancing the head.
 * Old entries are silently overwritten when the
 * ring is full.
 */
void ov_cmdlog_push(OV_CMDLOG *log, ov_cmdlog_level_t level, const char *fmt, ...)
{
    if (log == NULL || fmt == NULL)
    {
        return;
    }

    OV_CMDLOG_ENTRY *e = &log->entries[log->head];
    clock_gettime(CLOCK_REALTIME, &e->ts);
    e->level = level;

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->msg, OV_CMDLOG_MSG, fmt, ap);
    va_end(ap);

    log->head = (log->head + 1) % OV_CMDLOG_MAX;
    if (log->count < OV_CMDLOG_MAX)
    {
        log->count++;
    }
}
