// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_expand_arith.c
 *
 * @brief Arithmetic expression parser and
 *        $(( expr )) expansion.
 *
 * Implements a recursive-descent arithmetic
 * expression evaluator (ArithParser) that
 * supports:
 *   - Integer and floating-point literals
 *   - CLI variables as operands
 *   - Unary minus, bitwise NOT (~)
 *   - *, /, % (multiplicative)
 *   - +, - (additive)
 *   - <<, >> (bit shift)
 *   - <, >, <=, >=, ==, != (comparison)
 *   - & (bitwise AND)
 *   - ^ (bitwise XOR)
 *   - | (bitwise OR)
 *   - Parenthesized sub-expressions
 *
 * cli_expand_arith() scans a command-line buffer
 * for $(( ... )) sequences, evaluates each via the
 * parser, and replaces them with the numeric result.
 *
 * ArithParser and arith_expr() are also used
 * directly by CLIcore_script_expand_fps.c
 * (expand_fpsvar_write calls cli_expand_arith).
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_expand_arith()
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"
#include "CLIcore_script_expand_internal.h"


/* ============================================================
 *  Recursive-descent arithmetic parser
 * ============================================================
 */

/** @brief Skip whitespace in parser. */
void arith_skip_ws(ArithParser *p)
{
    while (p->s[p->pos] == ' ' || p->s[p->pos] == '\t')
    {
        p->pos++;
    }
}

/**
 * @brief Parse an atomic value.
 *
 * Handles: number literal, CLI variable name,
 * unary minus, bitwise NOT (~), and parenthesized
 * sub-expressions.
 */
double arith_atom(ArithParser *p)
{
    arith_skip_ws(p);

    /* Unary minus */
    if (p->s[p->pos] == '-')
    {
        p->pos++;
        return -arith_atom(p);
    }

    /* Bitwise NOT */
    if (p->s[p->pos] == '~')
    {
        p->pos++;
        return (double) (~(long) arith_atom(p));
    }

    /* Parenthesized sub-expression */
    if (p->s[p->pos] == '(')
    {
        p->pos++;
        double v = arith_expr(p);
        arith_skip_ws(p);
        if (p->s[p->pos] == ')')
        {
            p->pos++;
        }
        return v;
    }

    /* Variable name (bare identifier) */
    if (isalpha((unsigned char) p->s[p->pos]) || p->s[p->pos] == '_')
    {
        char vname[256];
        int  vn = 0;
        while (vn < 255 && (isalnum((unsigned char) p->s[p->pos]) || p->s[p->pos] == '_'))
        {
            vname[vn++] = p->s[p->pos++];
        }
        vname[vn]      = '\0';
        const char *vv = cli_var_lookup(vname);
        if (vv != NULL)
        {
            return strtod(vv, NULL);
        }
        return 0.0;
    }

    /* Numeric literal */
    arith_skip_ws(p);
    const char *start = p->s + p->pos;
    char       *end   = NULL;
    double      v     = strtod(start, &end);
    if (end > start)
    {
        p->pos += (int) (end - start);
        return v;
    }

    return 0.0;
}

/** @brief Parse multiplicative operators: * / % */
double arith_factor(ArithParser *p)
{
    double left = arith_atom(p);
    arith_skip_ws(p);

    while (p->s[p->pos] == '*' || p->s[p->pos] == '/' || p->s[p->pos] == '%')
    {
        char op = p->s[p->pos];
        p->pos++;
        double right = arith_atom(p);
        arith_skip_ws(p);
        if (op == '*')
        {
            left *= right;
        }
        else if (op == '/')
        {
            if (right != 0.0)
            {
                left /= right;
            }
        }
        else if (op == '%')
        {
            if (right != 0.0)
            {
                left = fmod(left, right);
            }
        }
    }
    return left;
}

/** @brief Parse additive operators: + - */
double arith_term(ArithParser *p)
{
    double left = arith_factor(p);
    arith_skip_ws(p);

    while (p->s[p->pos] == '+' || p->s[p->pos] == '-')
    {
        char op = p->s[p->pos];
        p->pos++;
        double right = arith_factor(p);
        arith_skip_ws(p);
        if (op == '+')
        {
            left += right;
        }
        else
        {
            left -= right;
        }
    }
    return left;
}

/** @brief Parse bitwise shift operators: << >> */
double arith_shift(ArithParser *p)
{
    double left = arith_term(p);
    arith_skip_ws(p);

    while ((p->s[p->pos] == '<' && p->s[p->pos + 1] == '<') ||
           (p->s[p->pos] == '>' && p->s[p->pos + 1] == '>'))
    {
        char op = p->s[p->pos];
        p->pos += 2;
        double right = arith_term(p);
        arith_skip_ws(p);
        if (op == '<')
        {
            left = (double) ((long) left << (long) right);
        }
        else
        {
            left = (double) ((long) left >> (long) right);
        }
    }
    return left;
}

/** @brief Parse comparison operators: < > <= >= == != */
double arith_compare(ArithParser *p)
{
    double left = arith_shift(p);
    arith_skip_ws(p);

    if (p->s[p->pos] == '<' && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left <= right) ? 1.0 : 0.0;
    }
    if (p->s[p->pos] == '>' && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left >= right) ? 1.0 : 0.0;
    }
    if (p->s[p->pos] == '<')
    {
        p->pos++;
        double right = arith_shift(p);
        return (left < right) ? 1.0 : 0.0;
    }
    if (p->s[p->pos] == '>')
    {
        p->pos++;
        double right = arith_shift(p);
        return (left > right) ? 1.0 : 0.0;
    }
    if (p->s[p->pos] == '=' && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left == right) ? 1.0 : 0.0;
    }
    if (p->s[p->pos] == '!' && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left != right) ? 1.0 : 0.0;
    }
    return left;
}

/** @brief Parse bitwise AND (&). */
double arith_bitwise_and(ArithParser *p)
{
    double left = arith_compare(p);
    arith_skip_ws(p);

    while (p->s[p->pos] == '&')
    {
        p->pos++;
        double right = arith_compare(p);
        arith_skip_ws(p);
        left = (double) ((long) left & (long) right);
    }
    return left;
}

/** @brief Parse bitwise XOR (^). */
double arith_bitwise_xor(ArithParser *p)
{
    double left = arith_bitwise_and(p);
    arith_skip_ws(p);

    while (p->s[p->pos] == '^')
    {
        p->pos++;
        double right = arith_bitwise_and(p);
        arith_skip_ws(p);
        left = (double) ((long) left ^ (long) right);
    }
    return left;
}

/** @brief Parse bitwise OR (|). */
double arith_bitwise_or(ArithParser *p)
{
    double left = arith_bitwise_xor(p);
    arith_skip_ws(p);

    while (p->s[p->pos] == '|')
    {
        p->pos++;
        double right = arith_bitwise_xor(p);
        arith_skip_ws(p);
        left = (double) ((long) left | (long) right);
    }
    return left;
}

/** @brief Top-level expression entry point. */
double arith_expr(ArithParser *p)
{
    return arith_bitwise_or(p);
}


/* ============================================================
 *  cli_expand_arith — $(( )) expansion
 * ============================================================
 */

/**
 * @brief Expand $(( expr )) arithmetic in place
 *
 * Scans the command line for $(( ... )) sequences.
 * Each is replaced by its numeric result — formatted
 * as an integer when the value is a whole number, or
 * as %g otherwise.
 *
 * @param line    Buffer to expand in-place
 * @param maxlen  Buffer size
 */
void cli_expand_arith(char *line, int maxlen)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i    = 0;

    while (line[i] != '\0' && opos < maxlen - 1)
    {
        if (line[i] == '$' && line[i + 1] == '(' && line[i + 2] == '(')
        {
            i += 3;

            char expr[512];
            int  elen  = 0;
            int  depth = 1;
            while (line[i] != '\0' && elen < 511)
            {
                if (line[i] == '(' && line[i + 1] == '(')
                {
                    depth++;
                    expr[elen++] = line[i++];
                    expr[elen++] = line[i++];
                    continue;
                }
                if (line[i] == ')' && line[i + 1] == ')')
                {
                    depth--;
                    if (depth == 0)
                    {
                        i += 2;
                        break;
                    }
                    expr[elen++] = line[i++];
                    expr[elen++] = line[i++];
                    continue;
                }
                expr[elen++] = line[i++];
            }
            expr[elen] = '\0';

            ArithParser parser;
            parser.s      = expr;
            parser.pos    = 0;
            double result = arith_expr(&parser);

            char rbuf[64];
            if (result == floor(result) && fabs(result) < 1e15)
            {
                snprintf(rbuf, sizeof(rbuf), "%ld", (long) result);
            }
            else
            {
                snprintf(rbuf, sizeof(rbuf), "%g", result);
            }

            int rlen  = (int) strlen(rbuf);
            int avail = maxlen - 1 - opos;
            int clen  = rlen < avail ? rlen : avail;
            memcpy(out + opos, rbuf, (size_t) clen);
            opos += clen;
        }
        else
        {
            out[opos++] = line[i++];
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}
