// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILKCTRL_OPTS_H
#define MILKCTRL_OPTS_H

/**
 * @file milkCTRL_opts.h
 * @brief Command-line options of milk-CTRL: a pure parser, the help banner, and the wrapper main() calls.
 */

#define OV_INTERVAL_MIN 0.1
#define OV_INTERVAL_MAX 10.0

typedef struct
{
    const char *shm_dir;  /* -d, --shm-dir: streams and processinfo directory */
    const char *theme;    /* -T, --theme */
    int         view;     /* --view, as ov_view_t; -1 = DASH */
    const char *pattern;  /* positional: initial filter of the view's list */
    double      interval; /* --interval seconds; 0 = default */
    const char *log;      /* --log FILE: command log mirror */
    int         debug;    /* --debug: also key events and scan timing in the --log file */
    char        err[160]; /* usage error when ov_opts_parse() fails */
} OV_OPTS;

int  ov_opts_parse(int argc, char *const argv[], OV_OPTS *o);
void milkctrl_print_help(const char *prog, int mh_color);
int  milkctrl_parse_options(int argc, char *argv[], OV_OPTS *o);

#endif /* MILKCTRL_OPTS_H */
