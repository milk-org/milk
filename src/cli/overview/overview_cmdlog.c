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
#include <unistd.h>

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

    for (char *p = e->msg; *p != '\0'; p++)
    {
        if (*p == '\n' || *p == '\r' || *p == '\t')
        {
            *p = ' ';
        }
    }

    log->head = (log->head + 1) % OV_CMDLOG_MAX;
    if (log->count < OV_CMDLOG_MAX)
    {
        log->count++;
    }
}

/**
 * cmdlog_has - Return 1 if @msg is already one of the stored entries.
 */
static int cmdlog_has(const OV_CMDLOG *log, const char *msg)
{
    for (int i = 0; i < log->count; i++)
    {
        if (strncmp(log->entries[i].msg, msg, OV_CMDLOG_MSG) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * ov_cmdlog_drain_fd - push complete lines read from a non-blocking fd (captured library output).
 * Escape sequences and control bytes are dropped; a line already in the ring is not repeated.
 * Return: number of lines pushed.
 */
int ov_cmdlog_drain_fd(OV_CMDLOG *log, int fd)
{
    static char line[OV_CMDLOG_MSG];
    static int  len = 0;
    static int  esc = 0; /* 0 text, 1 after ESC, 2 inside CSI */
    char        buf[1024];
    int         pushed = 0;
    ssize_t     n      = 0;

    if (log == NULL || fd < 0)
    {
        return 0;
    }
    for (int rd = 0; rd < 8 && (n = read(fd, buf, sizeof(buf))) > 0; rd++)
    {
        for (ssize_t i = 0; i < n; i++)
        {
            unsigned char c = (unsigned char) buf[i];
            if (esc == 1)
            {
                esc = (c == '[') ? 2 : 0;
                continue;
            }
            if (esc == 2)
            {
                esc = (c >= 0x40 && c <= 0x7e) ? 0 : 2;
                continue;
            }
            if (c == 0x1b)
            {
                esc = 1;
                continue;
            }
            if (c == '\n')
            {
                line[len] = '\0';
                if (len > 0 && !cmdlog_has(log, line))
                {
                    ov_cmdlog_push(log, OV_CMDLOG_WARN, "%s", line);
                    pushed++;
                }
                len = 0;
                continue;
            }
            if (c == '\t' || c == '\r')
            {
                c = ' ';
            }
            if (c >= 0x20 && len < OV_CMDLOG_MSG - 1)
            {
                line[len++] = (char) c;
            }
        }
    }
    return pushed;
}
