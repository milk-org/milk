// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash.h
 * @brief The bash co-process of milk-cli and milk-script, and the `@{...}` translator.
 *
 * Purpose:     milk-cli and milk-script run every command line in one persistent bash, so shell
 *              state (variables, functions, cwd, traps) lasts from one line to the next; the
 *              front end answers that bash's milk requests itself, as its own session.
 * Does:        milk_translate_shorthand_to_bash() rewrites `@{...}` into milkget and milkcalc
 *              calls, keeping line numbers; milk_bash_exec() and milk_bash_exec_file() run
 *              text or a file in bash, started on first use, and handle calc mode; the text is
 *              sourced from a file in a private directory (milk_bash_cmdfile_*());
 *              milk_bash_serve() answers requests while the prompt is idle; milk_bash_abort()
 *              ends a command after Ctrl-C or a fault, and milk_bash_shutdown() ends bash, which
 *              runs its EXIT trap; milk_shell_incomplete() tells whether a text needs more lines.
 * Used by:     milk-cli (CLIcore.c: the REPL, -c, -s, piped and FIFO input;
 *              CLIcore_UI_completion.c, CLIcore_UI_builtins.c, CLIcore_UI_history.c);
 *              milk-script and the embedding API (milkscript_main.c, milkscript_api.c);
 *              `milk-call translate` (milk_call_main.c); the milk.script unit
 *              (milk_unit_script.c); the milktranslate builtin (milk_bash_hooks.c).
 * Uses:        milk_server.h (the in-process server), milk_shell_lex.h (the shell scanner),
 *              CLIcore_script.h (calc mode, the variables exported to bash), /bin/bash and
 *              scripts/milk.bash.
 * See also:    milk_bash_translate.c, milk_bash_coproc.c, milk_bash_exec.c, milk_bash_cmdfile.c,
 *              milk_bash_internal.h; docs/guide/scripting.md, "The Language at a Glance" and
 *              "Errors and Exit Status".
 * Constraints: the translator, the command files and milk_shell_incomplete() are also compiled
 *              into libmilkbash.so; the co-process is in libmilkscript only. The translator keeps
 *              its scan in static storage: one call at a time.
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
 * @param allow_legacy Unused (callers still pass it)
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
 * While bash executes the command, the front end (milk-cli or milk-script) services
 * all incoming socket requests (`milkcall`, `milkcalc`, `milkget`, `milkset`, etc.)
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
