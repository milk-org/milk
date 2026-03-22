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

#endif /* CLI_CALC_PARSER_H */
