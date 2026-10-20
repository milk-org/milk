// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    test_ov_opts.c
 * @brief   Unit test of the milk-CTRL command-line parser (ov_opts_parse).
 */

#include <stdio.h>
#include <string.h>

#include "milkCTRL_opts.h"
#include "overview_layout.h"

static int failures;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

#define ARGV(...) ((char *const[]) { "milk-CTRL", __VA_ARGS__, NULL })
#define ARGC(...) ((int) (sizeof((char *[]) { "milk-CTRL", __VA_ARGS__ }) / sizeof(char *)))
#define PARSE(o, ...) ov_opts_parse(ARGC(__VA_ARGS__), ARGV(__VA_ARGS__), (o))

int main(void)
{
    OV_OPTS o;
    char   *none[] = { "milk-CTRL", NULL };

    CHECK(ov_opts_parse(1, none, &o) == 0 && o.view == -1 && o.pattern == NULL && o.interval == 0);
    CHECK(PARSE(&o, "--view", "procs") == 0 && o.view == OV_VIEW_PROCS);
    CHECK(PARSE(&o, "--view=loops", "wfs|dm") == 0 && o.view == OV_VIEW_LOOPS &&
          strcmp(o.pattern, "wfs|dm") == 0);
    CHECK(PARSE(&o, "--view", "bogus") == -1 && strstr(o.err, "unknown view") != NULL);
    CHECK(PARSE(&o, "--interval", "0.1") == 0 && o.interval > 0.09);
    CHECK(PARSE(&o, "--interval", "10") == 0);
    CHECK(PARSE(&o, "--interval", "0.05") == -1);
    CHECK(PARSE(&o, "--interval", "11") == -1);
    CHECK(PARSE(&o, "--interval", "1x") == -1);
    CHECK(PARSE(&o, "--interval") == -1 && strstr(o.err, "requires an argument") != NULL);
    CHECK(PARSE(&o, "--log", "/tmp/x.log", "--debug") == 0 && o.debug &&
          strcmp(o.log, "/tmp/x.log") == 0);
    CHECK(PARSE(&o, "--debug") == -1); /* nowhere to write */
    CHECK(PARSE(&o, "--shm-dir", "/") == 0 && strcmp(o.shm_dir, "/") == 0);
    CHECK(PARSE(&o, "-d", "/nonexistent-dir-xyz") == -1 &&
          strstr(o.err, "not a directory") != NULL);
    CHECK(PARSE(&o, "-T", "no-such-theme") == -1 && strstr(o.err, "unknown theme") != NULL);
    CHECK(PARSE(&o, "a(") == -1 && strstr(o.err, "invalid PATTERN") != NULL);
    CHECK(PARSE(&o, "one", "two") == -1 && strstr(o.err, "unexpected argument") != NULL);
    CHECK(PARSE(&o, "--bogus") == -1 && strstr(o.err, "unknown option") != NULL);
    CHECK(PARSE(&o, "0123456789012345678901234567890123456789012345678901234567890123") == -1);
    CHECK(PARSE(&o, "--view", "fps") == 0 && o.view == OV_VIEW_FPS); /* re-entrant after errors */
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
