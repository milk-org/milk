// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_notify.h
 * @brief Messages to a service manager (the sd_notify protocol) without libsystemd
 *
 * A service manager that waits for readiness (systemd's Type=notify) names a datagram socket in
 * $NOTIFY_SOCKET; the process sends it newline-separated KEY=VALUE lines: READY=1, STATUS=,
 * ERRNO=, WATCHDOG=1, STOPPING=1. Knows nothing about FPS.
 */

#ifndef MILK_NOTIFY_H
#define MILK_NOTIFY_H

#include <stdint.h>

/**
 * milknotify_open - a socket for $NOTIFY_SOCKET (a path, or @name in the abstract namespace), close-on-exec.
 * Unsets NOTIFY_SOCKET: the programs this process runs are not the service. Return: fd, or -1 if none.
 */
int milknotify_open(void);

/** milknotify_send - send @msg (lines "KEY=VALUE\n") on @fd. Return: 0 or -1. */
int milknotify_send(int fd, const char *msg);

/** milknotify_watchdog_usec - $WATCHDOG_USEC if meant for this process ($WATCHDOG_PID unset or ours), else 0; unsets both. */
uint64_t milknotify_watchdog_usec(void);

#endif
