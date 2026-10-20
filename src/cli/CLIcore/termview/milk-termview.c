// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-termview.c
 * @brief Standalone executable for TrueColor terminal image viewer.
 *
 * This is the entry point for the milk-termview command line utility.
 * It does NOT link to CLIcore. It relies solely on ImageStreamIO and
 * standard POSIX libraries.
 */

#include <getopt.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milk_help.h"
#include "termview.h"
#include "termview_internal.h"

#define TV_DESCRIPTION "TrueColor terminal image viewer"
#define TV_DESCRIPTION_LONG                                                             \
    "A low-resource terminal image viewer for milk shared-memory streams. It maps\n"    \
    "the stream read-only and redraws only the cells that changed, in 24-bit, 256 or\n" \
    "16 colours. Menus set the colormap, scaling, range and view; the status bar\n"     \
    "shows the stream state, rate and range. A stream that is removed or recreated\n"   \
    "is followed."

/** print_names() - Comma-separated list of the valid names from column @col, wrapped before
 * column 80 with continuation lines indented to @indent. */
static void print_names(FILE *f, const char *const *names, int n, int col, int indent)
{
    for (int i = 0; i < n; i++)
    {
        int len = (int) strlen(names[i]) + ((i + 1 < n) ? 1 : 0);
        if (i > 0 && col + 1 + len > 79)
        {
            fprintf(f, "\n%*s", indent, "");
            col = indent;
        }
        else if (i > 0)
        {
            fputc(' ', f);
            col++;
        }
        fprintf(f, "%s%s", names[i], (i + 1 < n) ? "," : "");
        col += len;
    }
}

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, TV_DESCRIPTION, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [options] %s<stream>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  A low-resource terminal image viewer for milk shared-memory streams. It maps\n"
           "  the stream read-only and redraws only the cells that changed, in 24-bit, 256\n"
           "  or 16 colours. The menubar (click, or F2-F6) sets the colormap, scaling, range\n"
           "  and view; the status bar shows the stream state (LIVE, STALE, PAUSED, GONE),\n"
           "  rate and range. A stream that is removed or recreated is followed.\n\n");

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<stream>", 22),
           "Stream name, name.im.shm, or path to the .im.shm file");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-c, --cmap NAME", 22), "Initial colormap");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-s, --scale NAME", 22), "Initial scaling");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-r, --range NAME", 22),
           "Initial range: minmax, 0.1, 0.5, 1, 5, 10, 15, 20 (%)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--min V --max V", 22),
           "Lock the display range to [min, max]");
    printf("  " MH_PAD_FMT " %s %s(20)%s\n", MH_PAD(MH_OPT, "-f, --fps N", 22),
           "UI refresh rate, 1-120", mh_color ? MH_DFLT : "", mh_color ? MH_RST : "");
    printf("  " MH_PAD_FMT " %s %s(auto)%s\n", MH_PAD(MH_OPT, "--colors MODE", 22),
           "Colour depth: auto, truecolor, 256, 16", mh_color ? MH_DFLT : "",
           mh_color ? MH_RST : "");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 22), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 22),
           "One-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 22),
           "Verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-hm, --help-mono", 22),
           "Full help, no ANSI color");
    printf("  Colormaps: ");
    print_names(stdout, tv_cmap_names, COLORMAP_NB, 13, 13);
    printf("\n  Scales:    ");
    print_names(stdout, tv_scale_names, SCALE_NB, 13, 13);
    printf("\n\n");

    milk_help_section("Keys", mh_color);
    for (int i = 0; i < tv_help_nkeys; i++)
    {
        printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_CMD, tv_help_keys[i][0], 14), tv_help_keys[i][1]);
    }
    printf("\n");

    milk_help_section("Examples", mh_color);
    printf("  %s$%s %s%s%s %swfsim%s\n", mh_color ? MH_DIM : "", mh_color ? MH_RST : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s$%s %s%s%s -c inferno -s log -r 1 %sdmdisp%s\n\n", mh_color ? MH_DIM : "",
           mh_color ? MH_RST : "", mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Environment", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "MILK_SHM_DIR", 22),
           "Directory holding the .im.shm stream files");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "COLORTERM", 22),
           "truecolor or 24bit: 24-bit colour (--colors auto)");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "TERM", 22),
           "Else 24-bit for kitty, foot..., 256 for xterm-like");

    const char *see_also[] = { "milk-streamCTRL:milk-CTRL on its streams view",
                               "milk-stream-help:stream tools overview" };
    milk_help_see_also(see_also, 2, mh_color);
}

/** arg_error() - Report a command-line error on stderr; returns the exit status. */
static int arg_error(const char *progname, const char *msg, const char *val)
{
    int color = isatty(STDERR_FILENO);
    fprintf(stderr, "%sError:%s %s%s%s%s\n", color ? MH_ERR : "", color ? MH_RST : "", msg,
            val ? " '" : "", val ? val : "", val ? "'" : "");
    fprintf(stderr, "Run '%s -h' for usage.\n", progname);
    return 1;
}

/** parse_name() - Look up @val in @names; lists the valid names on failure. */
static int parse_name(const char        *progname,
                      const char        *what,
                      const char *const *names,
                      int                n,
                      const char        *val,
                      int               *out)
{
    int idx = tv_lookup_name(names, n, val);
    if (idx < 0)
    {
        char msg[64];
        snprintf(msg, sizeof(msg), "unknown %s", what);
        arg_error(progname, msg, val);
        int col = fprintf(stderr, "Valid %s names: ", what);
        print_names(stderr, names, n, col, 2);
        fprintf(stderr, "\n");
        return -1;
    }
    *out = idx;
    return 0;
}

int main(int argc, char **argv)
{
    const char *progname = basename(argv[0]);

    int action = milk_help_init(argc, argv, TV_DESCRIPTION, TV_DESCRIPTION_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, action == MH_ACTION_HELP);
        return 0;
    }

    termview_options_t opt;
    termview_options_init(&opt);

    enum
    {
        OPT_MIN = 0x100,
        OPT_MAX,
        OPT_COLORS
    };
    static const struct option longopts[] = {
        { "cmap", required_argument, NULL, 'c' },
        { "scale", required_argument, NULL, 's' },
        { "range", required_argument, NULL, 'r' },
        { "fps", required_argument, NULL, 'f' },
        { "min", required_argument, NULL, OPT_MIN },
        { "max", required_argument, NULL, OPT_MAX },
        { "colors", required_argument, NULL, OPT_COLORS },
        { NULL, 0, NULL, 0 },
    };

    int have_min = 0, have_max = 0;
    int c;
    opterr = 0;
    while ((c = getopt_long(argc, argv, ":c:s:r:f:", longopts, NULL)) != -1)
    {
        int    idx;
        double v;
        switch (c)
        {
        case 'c':
            if (parse_name(progname, "colormap", tv_cmap_names, COLORMAP_NB, optarg, &idx) != 0)
            {
                return 1;
            }
            opt.colormap = (termview_colormap_t) idx;
            break;
        case 's':
            if (parse_name(progname, "scale", tv_scale_names, SCALE_NB, optarg, &idx) != 0)
            {
                return 1;
            }
            opt.scale = (termview_scale_t) idx;
            break;
        case 'r':
            idx = tv_lookup_name(tv_range_keys, RANGE_NB, optarg);
            if (idx < 0 &&
                parse_name(progname, "range", tv_range_names, RANGE_NB, optarg, &idx) != 0)
            {
                return 1;
            }
            opt.range = (termview_range_t) idx;
            break;
        case 'f':
            if (tv_parse_double(optarg, &v) != 0 || v < 1 || v > 120 || v != (double) (int) v)
            {
                return arg_error(progname, "--fps expects an integer 1-120, got", optarg);
            }
            opt.fps = (int) v;
            break;
        case OPT_MIN:
        case OPT_MAX:
            if (tv_parse_double(optarg, &v) != 0)
            {
                return arg_error(progname, "expected a finite number, got", optarg);
            }
            if (c == OPT_MIN)
            {
                opt.manual_min = v;
                have_min       = 1;
            }
            else
            {
                opt.manual_max = v;
                have_max       = 1;
            }
            break;
        case OPT_COLORS:
        {
            static const char *const depths[] = { "auto", "16", "256", "truecolor", "24bit" };
            idx                               = tv_lookup_name(depths, 5, optarg);
            if (idx < 0)
            {
                return arg_error(progname, "--colors expects auto, truecolor, 256 or 16, got",
                                 optarg);
            }
            opt.colors = (idx == 4) ? 3 : idx;
            break;
        }
        case ':':
            return arg_error(progname, "missing value for option", argv[optind - 1]);
        default:
            return arg_error(progname, "unknown option", argv[optind - 1]);
        }
    }

    if (have_min != have_max)
    {
        return arg_error(progname, "--min and --max must be given together", NULL);
    }
    if (have_min)
    {
        if (!(opt.manual_max > opt.manual_min))
        {
            return arg_error(progname, "--max must be greater than --min", NULL);
        }
        opt.range_locked = true;
    }
    if (optind != argc - 1)
    {
        return arg_error(
            progname, (optind >= argc) ? "missing stream name" : "expected one stream name", NULL);
    }

    return termview_run(argv[optind], &opt);
}
