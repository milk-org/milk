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

/* ---- Shared scanning helpers ---- */

/**
 * @brief Find first unquoted operator character.
 *
 * Scans @line for the first occurrence of @primary
 * outside quotes and parentheses.  When @reject is
 * non-zero, any match followed by @reject is skipped
 * (e.g. '<' with reject='<' skips '<<').  When
 * @accept is non-zero, only matches followed by
 * @accept count (e.g. '|' with accept='>' matches
 * '|>' only).
 *
 * @return Index of the matched character, or -1
 */
int cli_find_unquoted_op(
    const char *line,
    char       primary,
    char       reject,
    char        accept
);

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

/**
 * @brief Handle shell built-in commands.
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_shell_builtins(void);

/**
 * @brief Test whether firstword names an internal
 * milk command, keyword, or assignment.
 * @param firstword    First token of the command line
 * @param check_assign When non-zero, a token containing
 *                     '=' is also treated as internal
 *                     (variable assignment). Pass 0 to
 *                     skip this check (e.g. calc-eval
 *                     path where "a=b+1" is arithmetic).
 * @return 1 if internal, 0 if external
 */
int is_internal_cmd(const char *firstword, int check_assign);

/**
 * @brief Set up pipe-to-shell stdout redirect.
 * Splits cmd at the first unquoted '|', opens
 * popen() on the RHS, and dup2s stdout.  Caller
 * must call cli_pipe_teardown() when done.
 * @param[out] pipe_fp           opened popen handle (or NULL)
 * @param[out] saved_stdout_fd   dup'd original stdout fd
 */
void cli_pipe_setup(
    FILE **pipe_fp,
    int   *saved_stdout_fd
);

/**
 * @brief Restore stdout after pipe and close the pipe.
 * @param pipe_fp          handle returned by cli_pipe_setup
 * @param saved_stdout_fd  fd returned by cli_pipe_setup
 */
void cli_pipe_teardown(
    FILE *pipe_fp,
    int   saved_stdout_fd
);

/**
 * @brief Set up file-redirect stdout (> file).
 * Splits cmd at the first unquoted '>', opens the
 * file, and dup2s stdout.  Caller must call
 * cli_redir_teardown() when done.
 * @param[out] redir_fp           opened file handle (or NULL)
 * @param[out] saved_stdout_fd    dup'd original stdout fd
 */
void cli_redir_setup(
    FILE **redir_fp,
    int   *saved_stdout_fd
);

/**
 * @brief Restore stdout after file redirect.
 * @param redir_fp         handle returned by cli_redir_setup
 * @param saved_stdout_fd  fd returned by cli_redir_setup
 */
void cli_redir_teardown(
    FILE *redir_fp,
    int   saved_stdout_fd
);

/**
 * @brief Print "did you mean?" suggestions for an
 * unknown command using Levenshtein distance.
 * @param input_cmd  The command string that was not found
 */
void handle_did_you_mean(const char *input_cmd);

#endif /* CLICORE_UI_EXECUTE_INTERNAL_H */
