// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkscript.h
 * @brief Public API for the milk scripting engine
 *
 * This header exposes the minimal API needed to
 * initialize the milk scripting engine, run a
 * script in the bash co-process, and clean up.
 *
 * The scripting engine has NO dependency on readline
 * or ncurses. It is the foundation on which the
 * interactive milk-cli is built.
 */

#ifndef MILKSCRIPT_H
#define MILKSCRIPT_H

#include <errno.h>

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

/**
 * @brief Initialize the milk scripting engine.
 *
 * Sets up signals, shared memory directory, PID,
 * random number generator, core modules, data
 * structures, and command table. After this call,
 * `milkscript_run()` is ready to use.
 *
 * @param argc  Argument count (from main)
 * @param argv  Argument vector (from main)
 * @return 0 on success, non-zero on failure
 */
errno_t milkscript_init(int argc, char **argv);

/**
 * @brief Run a milk script stream in the bash co-process.
 *
 * Reads @p fp (typically stdin or a script file) to
 * EOF, then runs the whole text as one script.
 *
 * @param fp  Input stream (stdin or a script file)
 * @return 0 if the script succeeded, non-zero otherwise
 */
errno_t milkscript_run(FILE *fp);

/**
 * @brief Clean up and release scripting engine resources.
 *
 * Frees allocated command tables and image arrays.
 * Should be called before process exit.
 */
void milkscript_cleanup(void);

#endif /* MILKSCRIPT_H */
