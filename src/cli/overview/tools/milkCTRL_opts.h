// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILKCTRL_OPTS_H
#define MILKCTRL_OPTS_H

/**
 * @file milkCTRL_opts.h
 * @brief Command-line option parsing and help banner for milk-CTRL.
 */

void milkctrl_print_help(const char *prog, int mh_color);

int milkctrl_parse_options(int argc, char *argv[], const char **out_cli_theme);

#endif /* MILKCTRL_OPTS_H */
