// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash.h
 * @brief Supervised bash co-process and `@{...}` inline translator.
 *
 * Role & Goal:
 *   Provides the public interface of milk-cli's bash co-process:
 *   1. `milk_translate_shorthand_to_bash()`: Translates inline `@{...}` entity/calculator
 *      expansions and `\@` escapes into `milk.bash` syntax while preserving 1:1 line numbers.
 *   2. `milk_bash_*()`: Manages the persistent supervised bash co-process
 *      and in-process UNIX domain socket server used by interactive `milk-cli`
 *      and `milk-cli <script.milk>`.
 */

#ifndef MILK_BASH_H
#define MILK_BASH_H

#include <stddef.h>

/**
 * @brief Translate milk-cli shorthands into standard milk.bash syntax.
 *
 * Preserves 1:1 line numbering across multi-line scripts so bash `$LINENO`
 * matches the source file line numbers.
 *
 * @param src          Input script or command line
 * @param allow_legacy Non-zero to also translate un-sigiled `module.command` names
 * @return Newly heap-allocated translated string (caller must `free()`), or NULL
 */
char *milk_translate_shorthand_to_bash(const char *src, int allow_legacy);

/**
 * @brief Initialize the in-process milk socket server and supervised bash co-process.
 *
 * @return 0 on success, -1 on error
 */
int milk_bash_init(void);

/**
 * @brief Translate and execute a command or multi-line block in the bash co-process.
 *
 * While bash executes the command, the parent milk-cli process services all
 * incoming socket requests (`milkcall`, `milkcalc`, `milkget`, `milkset`, etc.)
 * in-process.
 *
 * @param cmdline      Command line or multi-line script text
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the bash command (0 = success)
 */
int milk_bash_exec(const char *cmdline, int *out_exit_req);

/**
 * @brief Read, translate, and execute a script file in the bash co-process.
 *
 * @param filename     Path to script file
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the script (0 = success)
 */
int milk_bash_exec_file(const char *filename, int *out_exit_req);

/** The in-process server's requests to poll (-1 before start), for the REPL's idle select():
 * new connections and kept ones (milk_server_ready_fd()). */
int milk_bash_ready_fd(void);

/** Serve the pending requests of the in-process server while the prompt is idle. */
void milk_bash_serve(void);

/** 1 once a client asked milk-cli's session to end (`@session end` from another shell). */
int milk_bash_ended(void);

/** The session milk-cli's bash starts on (`milk-cli --session NAME`): its socket. */
void milk_bash_set_session(const char *sockpath);

/** The session milk-cli's bash uses, as of its last command: its name, or "" for milk-cli's own. */
const char *milk_bash_session(void);

/**
 * @brief Shut down the supervised bash co-process (running EXIT/defer traps)
 *        and close the in-process socket server.
 */
void milk_bash_shutdown(void);

/**
 * @brief Abort an in-flight bash co-process command after a signal interrupt
 *        or fault-isolation `siglongjmp`.
 */
void milk_bash_abort(void);

/**
 * @brief Build the path of a private command file named @label in this process's own
 *        directory (created on first use, removed at exit).
 *
 * @return 0 on success, -1 if the directory cannot be created
 */
int milk_bash_cmdfile_path(const char *label, char *out, size_t outsz);

/**
 * @brief Return 1 if @path is a file inside this process's private command directory.
 */
int milk_bash_cmdfile_owned(const char *path);

/** milk_bash_cmdfile_cleanup() - remove the private command directory and the files in it. */
void milk_bash_cmdfile_cleanup(void);

/**
 * @brief Return 1 if @src needs more input to be a complete bash construct (open quote,
 *        substitution, here-document, compound command, trailing `\`, `|`, `&&`, `||`).
 */
int milk_shell_incomplete(const char *src);

#endif /* MILK_BASH_H */
