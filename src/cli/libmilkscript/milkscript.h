// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkscript.h
 * @brief Public API for the milk scripting engine
 *
 * This header exposes the minimal API needed to
 * initialize the milk scripting engine, execute
 * commands, and run a non-interactive REPL loop.
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
 * `milkscript_execute()` is ready to use.
 *
 * @param argc  Argument count (from main)
 * @param argv  Argument vector (from main)
 * @return 0 on success, non-zero on failure
 */
errno_t milkscript_init(int argc, char **argv);

/**
 * @brief Execute a single command line string.
 *
 * Copies @p cmdline into the engine's command buffer
 * and runs the full processing pipeline: variable
 * expansion, flow control, alias lookup, math eval,
 * and native/external command dispatch.
 *
 * @param cmdline  Null-terminated command string
 * @return 0 on success
 */
errno_t milkscript_execute(const char *cmdline);

/**
 * @brief Run a non-interactive REPL on a FILE stream.
 *
 * Reads lines from @p fp (typically stdin or a script
 * file), executing each via `milkscript_execute()`.
 * Exits when EOF is reached or "exitCLI" is entered.
 *
 * @param fp  Input stream (use stdin for a basic REPL)
 * @return 0 on normal exit
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
