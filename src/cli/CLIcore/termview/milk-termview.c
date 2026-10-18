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
#define TV_DESCRIPTION_LONG                                                     \
    "A low-resource TrueColor terminal image viewer. It maps a shared memory\n" \
    "stream read-only and renders it with ANSI graphics, redrawing only what changed."

/** print_names() - Comma-separated list of the valid names for an option. */
static void print_names(FILE *f, const char *const *names, int n)
{
    for (int i = 0; i < n; i++)
    {
        fprintf(f, "%s%s", (i > 0) ? ", " : "", names[i]);
    }
}

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, TV_DESCRIPTION, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [options] %s<stream>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  A low-resource TrueColor terminal image viewer. It maps a shared memory\n"
           "  stream read-only and renders it with ANSI graphics, redrawing only the\n"
           "  cells that changed.\n\n");

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<stream>", 22),
           "Stream name, name.im.shm, or path to the .im.shm file");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-c, --cmap NAME", 22), "Initial colormap");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-s, --scale NAME", 22), "Initial scaling");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-r, --range NAME", 22),
           "Initial range: minmax, 0.1, 0.5, 1, 5, 10, 15, 20 (percentile)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--min V --max V", 22),
           "Lock the display range to [min, max]");
    printf("  " MH_PAD_FMT " %s %s(20)%s\n", MH_PAD(MH_OPT, "-f, --fps N", 22),
           "UI refresh rate, 1-120", mh_color ? MH_DFLT : "", mh_color ? MH_RST : "");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 22), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 22),
           "One-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 22),
           "Verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-hm, --help-mono", 22),
           "Full help, no ANSI color");
    printf("  Colormaps: ");
    print_names(stdout, tv_cmap_names, COLORMAP_NB);
    printf("\n  Scales:    ");
    print_names(stdout, tv_scale_names, SCALE_NB);
    printf("\n\n");

    milk_help_section("Keys", mh_color);
    static const char *const keys[][2] = {
        { "Arrows", "Pan the view" },
        { "+ = / - _", "Zoom in / out" },
        { "0", "Reset zoom and pan" },
        { "c / s / r", "Next colormap / scaling / range (Up/Down in the popup)" },
        { "l", "Lock or unlock the current range" },
        { "m / M", "Type a manual min / max (locks the range)" },
        { "h / v", "Flip horizontally / vertically" },
        { "< , / > .", "Decrease / increase the UI refresh rate" },
        { "Ctrl-L", "Repaint the screen" },
        { "Ctrl-Z", "Suspend" },
        { "q, Ctrl-C", "Quit" },
        { "Wheel", "Zoom in / out" },
        { "Left drag", "Pan" },
        { "Right drag", "Zoom to the selected box" },
    };
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++)
    {
        printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_CMD, keys[i][0], 12), keys[i][1]);
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
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "MILK_SHM_DIR", 22),
           "Directory holding the .im.shm stream files");

    const char *see_also[] = { "milk-streamCTRL:stream monitor and control",
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
        fprintf(stderr, "Valid %s names: ", what);
        print_names(stderr, names, n);
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
        OPT_MAX
    };
    static const struct option longopts[] = {
        { "cmap", required_argument, NULL, 'c' },
        { "scale", required_argument, NULL, 's' },
        { "range", required_argument, NULL, 'r' },
        { "fps", required_argument, NULL, 'f' },
        { "min", required_argument, NULL, OPT_MIN },
        { "max", required_argument, NULL, OPT_MAX },
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
