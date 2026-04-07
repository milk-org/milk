/**
 * @file CLIcore_UI_execute_internal.h
 *
 * @brief Internal prototypes for CLI execute helpers.
 *
 * These functions are NOT part of the public API.
 * They are shared across the execute sub-modules:
 *   - CLIcore_UI_execute_preproc.c  (text transforms)
 *   - CLIcore_UI_execute_redir.c    (I/O redirection)
 *   - CLIcore_UI_execute.c          (main dispatch)
 *   - CLIcore_UI_execute_debug.c    (debug/entry pts)
 *
 * Do not include this header from outside libmilkscript.
 */

#ifndef CLICORE_UI_EXECUTE_INTERNAL_H
#define CLICORE_UI_EXECUTE_INTERNAL_H

#include <errno.h>
#include "CLIcore.h"

/* ---- Pre-processing transforms (preproc.c) ---- */

/**
 * @brief Detect shell meta-characters outside quotes.
 * @return 1 if restricted symbols found, 0 if clean
 */
int cli_check_unquoted_restricted_symbols(
    const char *cmdline
);

/**
 * @brief Split command line at top-level && or ||.
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_split_logical_op(errno_t *retval);

/**
 * @brief Rewrite stream pipeline |> syntax.
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_rewrite_stream_pipe(errno_t *retval);

/**
 * @brief Split command at top-level pipe operator.
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_split_pipe(errno_t *retval);

/**
 * @brief Split at semicolon or chaining ops.
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_split_semicolon(errno_t *retval);

/**
 * @brief Rewrite dot-source syntax to source command.
 * @return Always 0 (not consumed)
 */
int cli_rewrite_dot_source(void);

/* ---- I/O Redirection handlers (redir.c) ---- */

/**
 * @brief Execute an external command with minimal overhead.
 * @param cmd  Fully expanded command string
 * @return Exit status (0 = success)
 */
int cli_run_external(const char *cmd);

/**
 * @brief Handle subshell execution: (cmd1; cmd2)
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_subshell(errno_t *retval);

/**
 * @brief Handle late here-string (pipe-based).
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_herestring_late(errno_t *retval);

/**
 * @brief Handle stderr redirection (2>&1, 2>file).
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_stderr_redir(errno_t *retval);

/**
 * @brief Handle input redirection (cmd < file).
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_input_redir(errno_t *retval);

/**
 * @brief Handle output redirection (> and >>).
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_output_redir(errno_t *retval);

/**
 * @brief Handle here-string syntax (<<<).
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_herestring_early(errno_t *retval);

/**
 * @brief Handle background execution (cmd &).
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_background(errno_t *retval);

#endif /* CLICORE_UI_EXECUTE_INTERNAL_H */
