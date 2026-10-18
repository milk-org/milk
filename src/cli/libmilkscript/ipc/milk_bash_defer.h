// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_defer.h
 * @brief `@defer` entries of a bash shell (libmilkbash only)
 */

#ifndef MILK_BASH_DEFER_H
#define MILK_BASH_DEFER_H

/** milk_bash_defer_argv() - the milkdefer builtin: `[--] CMD...`, `--exit CMD...`, `--restore SPEC...`,
 * and for milk.bash `--return U`, `--orphans`, `--exit-run` (status 1: evaluate __MILK_DRUN_SH). */
int milk_bash_defer_argv(int argc, char **argv);

#endif
