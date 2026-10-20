// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkCTRL_opts_parse.c
 * @brief   Pure command-line parser of milk-CTRL (no output, no environment changes).
 */

#include <getopt.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "milkCTRL_opts.h"
#include "overview_layout.h"
#include "overview_theme.h"

/** bad - record usage error "@what: @arg" in @o. Return: -1. */
static int bad(OV_OPTS *o, const char *what, const char *arg)
{
    snprintf(o->err, sizeof(o->err), "%s: %s", what, arg);
    return -1;
}

/**
 * ov_opts_parse - fill @o from the command line (help options are handled before). Re-entrant: getopt
 * is reset. Return: 0, or -1 with the usage error in @o->err.
 */
int ov_opts_parse(int argc, char *const argv[], OV_OPTS *o)
{
    static const struct option lopts[] = {
        { "theme", required_argument, NULL, 'T' },
        { "shm-dir", required_argument, NULL, 'd' },
        { "view", required_argument, NULL, 'v' },
        { "interval", required_argument, NULL, 'i' },
        { "log", required_argument, NULL, 'l' },
        { "debug", no_argument, NULL, 'g' },
        { NULL, 0, NULL, 0 },
    };
    memset(o, 0, sizeof(*o));
    o->view = -1;
    optind  = 0; /* GNU: restart scanning */
    opterr  = 0;
    int c;
    while ((c = getopt_long(argc, argv, ":d:T:", lopts, NULL)) != -1)
    {
        struct stat st;
        char       *end = NULL;
        switch (c)
        {
        case 'd':
            if (stat(optarg, &st) != 0 || !S_ISDIR(st.st_mode))
            {
                return bad(o, "not a directory", optarg);
            }
            o->shm_dir = optarg;
            break;
        case 'T':
            if (ov_theme_find_by_id(optarg) < 0)
            {
                return bad(o, "unknown theme", optarg);
            }
            o->theme = optarg;
            break;
        case 'v':
            o->view = ov_view_by_name(optarg);
            if (o->view < 0)
            {
                return bad(o, "unknown view (dash, streams, procs, fps, conn, loops)", optarg);
            }
            break;
        case 'i':
            o->interval = strtod(optarg, &end);
            if (end == optarg || *end != '\0' ||
                !(o->interval >= OV_INTERVAL_MIN && o->interval <= OV_INTERVAL_MAX))
            {
                return bad(o, "interval must be 0.1 to 10 seconds", optarg);
            }
            break;
        case 'l':
            o->log = optarg;
            break;
        case 'g':
            o->debug = 1;
            break;
        case ':':
            return bad(o, "option requires an argument", argv[optind - 1]);
        default:
            return bad(o, "unknown option", argv[optind - 1]);
        }
    }
    if (optind < argc)
    {
        regex_t re;
        if (regcomp(&re, argv[optind], REG_EXTENDED | REG_ICASE | REG_NOSUB) != 0)
        {
            return bad(o, "invalid PATTERN (extended regex)", argv[optind]);
        }
        regfree(&re);
        if (strlen(argv[optind]) >= sizeof(((OV_PANEL_STATE *) 0)->filter))
        {
            return bad(o, "PATTERN is longer than 63 characters", argv[optind]);
        }
        o->pattern = argv[optind++];
    }
    if (optind < argc)
    {
        return bad(o, "unexpected argument", argv[optind]);
    }
    if (o->debug && o->log == NULL)
    {
        return bad(o, "--debug writes to the --log file", "add --log FILE");
    }
    return 0;
}
