// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest: milkspawn_* pass arguments literally, report ENOENT and exit codes, and close fds above 2. */

#include "milk_spawn.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            return 1;                                                    \
        }                                                                \
    } while (0)

int main(void)
{
    const char *lit      = "a;$(touch spawn_pwned) `id` b";
    const char *pf[]     = { "printf", "%s", lit, NULL };
    const char *no[]     = { "milk-no-such-tool-xyz", NULL };
    const char *fl[]     = { "false", NULL };
    const char *ct[]     = { "cat", NULL };
    const char *tf[]     = { "test", "-e", "/proc/self/fd/57", NULL };
    char        buf[128] = { 0 };
    int         p[2];

    CHECK(pipe(p) == 0);
    CHECK(milkspawn_run(pf, p[1], -1) == 0);
    close(p[1]);
    CHECK(read(p[0], buf, sizeof(buf) - 1) == (ssize_t) strlen(lit));
    close(p[0]);
    CHECK(strcmp(buf, lit) == 0);
    CHECK(access("spawn_pwned", F_OK) != 0);

    errno = 0;
    CHECK(milkspawn_start(no, -1, -1) == -1 && errno == ENOENT);
    CHECK(milkspawn_run(fl, -1, -1) == 1);

    CHECK(pipe(p) == 0);
    CHECK(milkspawn_run(ct, p[1], -1) == 0);
    close(p[1]);
    CHECK(read(p[0], buf, sizeof(buf)) == 0);
    close(p[0]);

    CHECK(dup2(STDOUT_FILENO, 57) == 57);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    CHECK(milkspawn_run(tf, -1, -1) == 1);
#endif
    printf("all ok\n");
    return 0;
}
