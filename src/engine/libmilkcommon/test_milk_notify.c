// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkcommon-notify: messages reach $NOTIFY_SOCKET (a path or an abstract name), the variable is
 * gone once read, and $WATCHDOG_USEC counts only for the process $WATCHDOG_PID names. */

#include "milk_notify.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

/* A manager's socket at @name (@abstract: in the abstract namespace); NOTIFY_SOCKET names it */
static int manager(const char *name, int abstract)
{
    struct sockaddr_un a;
    size_t             len = strlen(name);
    int                fd  = socket(AF_UNIX, SOCK_DGRAM, 0);

    memset(&a, 0, sizeof(a));
    a.sun_family = AF_UNIX;
    memcpy(a.sun_path + (abstract ? 1 : 0), name, len);
    len += abstract ? 1 : 0;
    if (fd < 0 || bind(fd, (struct sockaddr *) &a,
                       (socklen_t) (offsetof(struct sockaddr_un, sun_path) + len)) != 0)
    {
        return -1;
    }
    char env[128];
    snprintf(env, sizeof(env), "%s%s", abstract ? "@" : "", name);
    setenv("NOTIFY_SOCKET", env, 1);
    return fd;
}

static void exchange(int mfd)
{
    char buf[64] = { 0 };
    int  fd      = milknotify_open();

    CHECK(fd >= 0 && getenv("NOTIFY_SOCKET") == NULL && (fcntl(fd, F_GETFD) & FD_CLOEXEC));
    CHECK(milknotify_send(fd, "READY=1\nSTATUS=up\n") == 0);
    CHECK(recv(mfd, buf, sizeof(buf) - 1, 0) == 18 && strcmp(buf, "READY=1\nSTATUS=up\n") == 0);
    close(fd);
}

int main(void)
{
    char path[64];
    char pid[32];

    snprintf(path, sizeof(path), "/tmp/test-notify-%d", (int) getpid());
    int mfd = manager(path, 0);
    CHECK(mfd >= 0);
    exchange(mfd);
    close(mfd);
    unlink(path);

    snprintf(path, sizeof(path), "test-notify-%d", (int) getpid());
    mfd = manager(path, 1);
    CHECK(mfd >= 0);
    exchange(mfd);
    close(mfd);

    unsetenv("NOTIFY_SOCKET");
    CHECK(milknotify_open() == -1 && milknotify_send(-1, "READY=1\n") == -1);
    setenv("NOTIFY_SOCKET", "vsock:2:1", 1); // an address this sender does not speak
    CHECK(milknotify_open() == -1);

    setenv("WATCHDOG_USEC", "3000000", 1);
    CHECK(milknotify_watchdog_usec() == 3000000 && getenv("WATCHDOG_USEC") == NULL);
    snprintf(pid, sizeof(pid), "%d", (int) getpid());
    setenv("WATCHDOG_USEC", "3000000", 1);
    setenv("WATCHDOG_PID", pid, 1);
    CHECK(milknotify_watchdog_usec() == 3000000 && getenv("WATCHDOG_PID") == NULL);
    setenv("WATCHDOG_USEC", "3000000", 1);
    setenv("WATCHDOG_PID", "1", 1); // another process's watchdog
    CHECK(milknotify_watchdog_usec() == 0 && getenv("WATCHDOG_USEC") == NULL);
    setenv("WATCHDOG_USEC", "3s", 1);
    CHECK(milknotify_watchdog_usec() == 0);

    if (failures > 0)
    {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
