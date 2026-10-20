// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_parser.h
 * @brief Hand-written expression parser for CLI
 *
 * Replaces the bison-generated parser (calc_bison.y).
 * Uses Pratt (precedence-climbing) parsing to evaluate
 * arithmetic expressions on longs, doubles, and images.
 */

#ifndef CLI_CALC_PARSER_H
#define CLI_CALC_PARSER_H

#include <stddef.h>

/**
 * @brief Parse a single CLI token string
 *
 * Tokenizes and parses the input (which must be
 * terminated by '\\n').  Populates the current
 * data.cmdargtoken[data.cmdNBarg] entry with the
 * result type and value, matching the behavior of
 * the previous bison/flex parser.
 *
 * @param input  null-terminated string ending in '\\n'
 */
void cli_parse(const char *input);

/**
 * @brief Evaluate an entire line as a math expression.
 *
 * If the line successfully evaluates as a generic math
 * expression, print the result and return 1.
 * If it contains syntax errors, return 0.
 */
int cli_calc_eval_line(const char *input);

/**
 * @brief Evaluate a string as a pure math expression, returning the result value silently.
 *
 * @param input     Expression string
 * @param out_type  Pointer to receive the parsed type (1=long, 2=double)
 * @param out_lval  Pointer to receive long value
 * @param out_dval  Pointer to receive double value
 * @return 1 on success (pure math), 0 on failure/string
 */
int cli_calc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval);

/**
 * @brief Evaluate an expression for @calc or @test in bash mode.
 *
 * @param input   Expression string
 * @param is_test 1 for @test (boolean exit status), 0 for @calc
 * @param out_rc  Receives bash exit code (0=true/ok, 1=false, 2=error)
 * @return 1 on valid evaluation, 0 on parse/eval error
 */
int cli_calc_eval_bash(const char *input, int is_test, int *out_rc);

int cli_calc_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz);

/** cli_calc_keep_statement() - `<image> = <element-wise expression>` (also += -= *=) compiled for repeated
 * runs by cli_calc_fused_run(), variables read at each run (free() it); NULL for any other statement. */
void *cli_calc_keep_statement(const char *stmt);

/** cli_calc_fused_run() - run a program from cli_calc_keep_statement() and post its destination; -1 if one of
 * its images changed size or type (cli_calc_lasterr names it). */
int cli_calc_fused_run(void *kept);

/* What cli_calc_eval_bash() prints for a @calc statement (set per request by the server) */
enum
{
    CLI_CALC_OUT_VALUES, /* the values of non-assignments */
    CLI_CALC_OUT_BIND, /* one line per statement: its value, an assignment's too (empty for an image) */
    CLI_CALC_OUT_PIXELS, /* as BIND, with an image's pixels on its line */
    CLI_CALC_OUT_REPL    /* interactive calc mode: <val>, <var> <val>, or [<name> ]<dims> <type> */
};
extern int cli_calc_out_mode;
extern int cli_calc_mode;

/**
 * @brief Delete every temporary `_tmpcalcN` image still tracked (e.g. after a command whose
 *        arguments were evaluated as expressions).
 */
void cli_calc_tmp_cleanup(void);

#endif /* CLI_CALC_PARSER_H */
