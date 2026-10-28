// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_notify.c
 * @brief Messages to a service manager (milk_notify.h)
 */

#include "milk_notify.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int milknotify_open(void)
{
    const char        *v = getenv("NOTIFY_SOCKET");
    struct sockaddr_un a;
    size_t             len;
    int                fd;

    if (v == NULL || (v[0] != '/' && v[0] != '@') || (len = strlen(v)) >= sizeof(a.sun_path))
    {
        return -1; // none, or an address this sender does not speak (vsock:)
    }
    memset(&a, 0, sizeof(a));
    a.sun_family = AF_UNIX;
    memcpy(a.sun_path, v, len);
    if (a.sun_path[0] == '@')
    {
        a.sun_path[0] = '\0'; // abstract namespace: the length, not a NUL, ends the name
    }
    unsetenv("NOTIFY_SOCKET");
    if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0)
    {
        return -1;
    }
    // the manager's socket passes credentials: the kernel attaches the sender's pid to each message
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) == -1 ||
        connect(fd, (struct sockaddr *) &a,
                (socklen_t) (offsetof(struct sockaddr_un, sun_path) + len)) != 0)
    {
        int err = errno;
        close(fd);
        errno = err;
        return -1;
    }
    return fd;
}

int milknotify_send(int fd, const char *msg)
{
    ssize_t n;

    if (fd < 0)
    {
        return -1;
    }
    do
    {
        n = send(fd, msg, strlen(msg), MSG_NOSIGNAL);
    } while (n < 0 && errno == EINTR);
    return (n < 0) ? -1 : 0;
}

uint64_t milknotify_watchdog_usec(void)
{
    const char *usec = getenv("WATCHDOG_USEC");
    const char *pid  = getenv("WATCHDOG_PID");
    uint64_t    v    = 0;
    char       *end;

    if (usec != NULL && (pid == NULL || strtol(pid, NULL, 10) == (long) getpid()))
    {
        unsigned long long u = strtoull(usec, &end, 10);
        v                    = (end != usec && *end == '\0') ? (uint64_t) u : 0;
    }
    unsetenv("WATCHDOG_USEC");
    unsetenv("WATCHDOG_PID");
    return v;
}
