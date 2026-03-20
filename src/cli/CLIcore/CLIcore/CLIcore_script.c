/**
 * @file CLIcore_script.c
 * @brief CLI scripting engine — variables, FPS access,
 *        arithmetic evaluation, flow control
 *
 * This module implements bash-style scripting constructs
 * for the milk CLI: variable assignment (VAR=val),
 * variable expansion ($VAR, ${VAR}, $?), FPS parameter
 * read (@fpsname.param), and arithmetic $(( expr )).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "CLIcore.h"
#include "CLIcore_script.h"

/* ============================================================
 *  CLI Variable Storage
 * ============================================================
 */

CLI_VAR cli_vars[CLI_MAX_VARS];
int     cli_last_retval = 0;

/**
 * @brief Look up a CLI variable by name
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_get(const char *name)
{
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            return cli_vars[i].val;
        }
    }
    return NULL;
}

/**
 * @brief Set a CLI variable (create or update)
 *
 * @param name  Variable name
 * @param val   Value string
 */
void cli_var_set(
    const char *name,
    const char *val
)
{
    /* Update existing */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            strncpy(cli_vars[i].val, val,
                    CLI_VAR_VALLEN - 1);
            cli_vars[i].val[
                CLI_VAR_VALLEN - 1] = '\0';
            return;
        }
    }
    /* Find empty slot */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(!cli_vars[i].used)
        {
            strncpy(cli_vars[i].name, name,
                    CLI_VAR_NAMELEN - 1);
            cli_vars[i].name[
                CLI_VAR_NAMELEN - 1] = '\0';
            strncpy(cli_vars[i].val, val,
                    CLI_VAR_VALLEN - 1);
            cli_vars[i].val[
                CLI_VAR_VALLEN - 1] = '\0';
            cli_vars[i].used = 1;
            return;
        }
    }
    printf("Error: variable table full "
           "(max %d)\n", CLI_MAX_VARS);
}

/**
 * @brief Remove a CLI variable
 *
 * @param name  Variable name
 */
void cli_var_unset(const char *name)
{
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            cli_vars[i].used = 0;
            cli_vars[i].name[0] = '\0';
            cli_vars[i].val[0] = '\0';
            return;
        }
    }
}

/**
 * @brief Unified variable lookup: CLI vars, then
 *        special vars ($?), then env vars
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_lookup(const char *name)
{
    static char retbuf[32];

    /* $? — last return value */
    if(strcmp(name, "?") == 0)
    {
        snprintf(retbuf, sizeof(retbuf),
                 "%d", cli_last_retval);
        return retbuf;
    }

    /* CLI variable */
    const char *v = cli_var_get(name);
    if(v != NULL)
    {
        return v;
    }

    /* Fall through to environment */
    return getenv(name);
}


/* ============================================================
 *  Variable Assignment Detection
 * ============================================================
 *
 * Detect bash-style VAR=val on the command line.
 * Rules: first token must match [A-Za-z_][A-Za-z0-9_]*=
 * with no spaces before '='.
 */

/**
 * @brief Check if line is VAR=val assignment
 *
 * @param line  Command line string
 * @return 1 if handled as assignment, 0 otherwise
 */
int cli_try_var_assign(const char *line)
{
    const char *p = line;

    /* Skip leading whitespace */
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }

    /* Must start with alpha or underscore */
    if(!isalpha((unsigned char) *p)
            && *p != '_')
    {
        return 0;
    }

    /* Scan variable name */
    const char *name_start = p;
    while(isalnum((unsigned char) *p)
            || *p == '_')
    {
        p++;
    }

    /* Must hit '=' immediately */
    if(*p != '=')
    {
        return 0;
    }

    /* Check it's not a command (e.g. "cmd=")
     * by verifying name is not a registered cmd */
    {
        int namelen = (int)(p - name_start);
        char tmpname[CLI_VAR_NAMELEN];
        if(namelen >= CLI_VAR_NAMELEN)
        {
            namelen = CLI_VAR_NAMELEN - 1;
        }
        memcpy(tmpname, name_start,
               (size_t) namelen);
        tmpname[namelen] = '\0';

        /* Extract value (everything after '=') */
        const char *val = p + 1;

        /* Strip trailing whitespace/newline */
        char valbuf[CLI_VAR_VALLEN];
        strncpy(valbuf, val,
                CLI_VAR_VALLEN - 1);
        valbuf[CLI_VAR_VALLEN - 1] = '\0';
        {
            size_t vl = strlen(valbuf);
            while(vl > 0
                    && (valbuf[vl - 1] == ' '
                        || valbuf[vl - 1] == '\t'
                        || valbuf[vl - 1] == '\n'))
            {
                valbuf[--vl] = '\0';
            }
        }

        cli_var_set(tmpname, valbuf);
        return 1;
    }
}


/* ============================================================
 *  CLI Commands: unset, vars, echo
 * ============================================================
 */

/**
 * @brief unset command — remove a variable
 */
errno_t cli_cmd_unset(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: unset <varname>\n");
        return RETURN_FAILURE;
    }
    cli_var_unset(
        data.cmdargtoken[1].val.string);
    return RETURN_SUCCESS;
}

/**
 * @brief vars command — list all CLI variables
 */
errno_t cli_cmd_vars(void)
{
    int count = 0;
    printf("\n  CLI Variables:\n");
    printf("  %-20s  %s\n",
           "NAME", "VALUE");
    printf("  %-20s  %s\n",
           "----", "-----");
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used)
        {
            printf("  %-20s  %s\n",
                   cli_vars[i].name,
                   cli_vars[i].val);
            count++;
        }
    }
    if(count == 0)
    {
        printf("  (none)\n");
    }
    printf("  $? = %d\n\n", cli_last_retval);
    return RETURN_SUCCESS;
}

/**
 * @brief echo command — print arguments
 *
 * Prints all arguments separated by spaces,
 * followed by a newline. Supports -n flag.
 */
errno_t cli_cmd_echo(void)
{
    int start = 1;
    int newline = 1;

    if(data.cmdNBarg >= 2
            && strcmp(
                data.cmdargtoken[1].val.string,
                "-n") == 0)
    {
        newline = 0;
        start = 2;
    }

    for(int i = start; i < data.cmdNBarg; i++)
    {
        if(i > start)
        {
            printf(" ");
        }
        printf("%s",
               data.cmdargtoken[i].val.string);
    }
    if(newline)
    {
        printf("\n");
    }
    return RETURN_SUCCESS;
}


/* ============================================================
 *  FPS Variable Expansion — @fpsname.param
 * ============================================================
 *
 * Scan for @fpsname.paramname tokens and replace
 * with the live FPS parameter value.
 */

#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_printparameter_valuestring.h"
#include "fps_connect.h"

/**
 * @brief Expand @fpsname.param tokens in place
 *
 * @param line   Command line buffer
 * @param maxlen Buffer size
 */
void cli_expand_fpsvar(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
            && opos < maxlen - 1)
    {
        if(line[i] == '@')
        {
            i++; /* skip @ */
            /* Collect fpsname.paramname */
            char token[512];
            int  tlen = 0;
            while(line[i] != '\0'
                    && tlen < 511
                    && (isalnum(
                            (unsigned char)
                            line[i])
                        || line[i] == '_'
                        || line[i] == '.'
                        || line[i] == '-'))
            {
                token[tlen++] = line[i++];
            }
            token[tlen] = '\0';

            /* Split at first '.' into
             * fpsname and paramname */
            char *dot = strchr(token, '.');
            if(dot == NULL)
            {
                /* No param — just copy
                 * @token literally */
                if(opos < maxlen - 1)
                {
                    out[opos++] = '@';
                }
                int clen = tlen;
                if(opos + clen > maxlen - 1)
                {
                    clen = maxlen - 1 - opos;
                }
                memcpy(out + opos, token,
                       (size_t) clen);
                opos += clen;
                continue;
            }

            *dot = '\0';
            const char *fpsname = token;
            const char *pname = dot + 1;

            /* Connect to FPS */
            FUNCTION_PARAMETER_STRUCT fps;
            int fpsconn =
                function_parameter_struct_connect(
                    fpsname, &fps, FPSCONNECT_SIMPLE);

            if(fpsconn == -1
                    || fps.parray == NULL)
            {
                /* Connection failed —
                 * output empty string */
                continue;
            }

            /* Look up parameter */
            int pindex =
                functionparameter_GetParamIndex(
                    &fps, pname);

            if(pindex < 0)
            {
                /* Try with leading dot */
                char dotname[512];
                snprintf(dotname,
                         sizeof(dotname),
                         ".%s", pname);
                pindex =
                    functionparameter_GetParamIndex(
                        &fps, dotname);
            }

            if(pindex >= 0)
            {
                char vstr[512];
                functionparameter_GetParamValueString(
                    &fps.parray[pindex],
                    vstr,
                    (int) sizeof(vstr));

                int vlen = (int) strlen(vstr);
                int avail = maxlen - 1 - opos;
                int clen = vlen < avail
                           ? vlen : avail;
                memcpy(out + opos, vstr,
                       (size_t) clen);
                opos += clen;
            }

            function_parameter_struct_disconnect(
                &fps);
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


/* ============================================================
 *  Arithmetic Expansion — $(( expr ))
 * ============================================================
 *
 * Evaluate simple arithmetic expressions with
 * integer and float support. Operators: + - * / %
 * Parentheses supported.
 */

/* ---- Recursive descent expression parser ---- */

/** Parser state */
typedef struct
{
    const char *s;
    int         pos;
} ArithParser;

static double arith_expr(ArithParser *p);

static void arith_skip_ws(ArithParser *p)
{
    while(p->s[p->pos] == ' '
            || p->s[p->pos] == '\t')
    {
        p->pos++;
    }
}

static double arith_atom(ArithParser *p)
{
    arith_skip_ws(p);

    /* Unary minus */
    if(p->s[p->pos] == '-')
    {
        p->pos++;
        return -arith_atom(p);
    }

    /* Parenthesized sub-expression */
    if(p->s[p->pos] == '(')
    {
        p->pos++;
        double v = arith_expr(p);
        arith_skip_ws(p);
        if(p->s[p->pos] == ')')
        {
            p->pos++;
        }
        return v;
    }

    /* Number */
    arith_skip_ws(p);
    const char *start = p->s + p->pos;
    char *end = NULL;
    double v = strtod(start, &end);
    if(end > start)
    {
        p->pos += (int)(end - start);
        return v;
    }

    /* Unknown — return 0 */
    return 0.0;
}

static double arith_factor(ArithParser *p)
{
    double left = arith_atom(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '*'
            || p->s[p->pos] == '/'
            || p->s[p->pos] == '%')
    {
        char op = p->s[p->pos];
        p->pos++;
        double right = arith_atom(p);
        arith_skip_ws(p);
        if(op == '*')
        {
            left *= right;
        }
        else if(op == '/')
        {
            if(right != 0.0)
            {
                left /= right;
            }
        }
        else if(op == '%')
        {
            if(right != 0.0)
            {
                left = fmod(left, right);
            }
        }
    }
    return left;
}

static double arith_term(ArithParser *p)
{
    double left = arith_factor(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '+'
            || p->s[p->pos] == '-')
    {
        char op = p->s[p->pos];
        p->pos++;
        double right = arith_factor(p);
        arith_skip_ws(p);
        if(op == '+')
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

/** Comparison operators for use in conditions */
static double arith_compare(ArithParser *p)
{
    double left = arith_term(p);
    arith_skip_ws(p);

    if(p->s[p->pos] == '<'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left <= right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '>'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left >= right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '<')
    {
        p->pos++;
        double right = arith_term(p);
        return (left < right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '>')
    {
        p->pos++;
        double right = arith_term(p);
        return (left > right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '='
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left == right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '!'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left != right) ? 1.0 : 0.0;
    }
    return left;
}

static double arith_expr(ArithParser *p)
{
    return arith_compare(p);
}


/**
 * @brief Expand $(( expr )) in place
 *
 * @param line   Command line buffer
 * @param maxlen Buffer size
 */
void cli_expand_arith(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
            && opos < maxlen - 1)
    {
        /* Detect $(( */
        if(line[i] == '$'
                && line[i + 1] == '('
                && line[i + 2] == '(')
        {
            i += 3; /* skip $(( */

            /* Extract expression until )) */
            char expr[512];
            int  elen = 0;
            int  depth = 1;
            while(line[i] != '\0'
                    && elen < 511)
            {
                if(line[i] == '('
                        && line[i + 1] == '(')
                {
                    depth++;
                    expr[elen++] = line[i++];
                    expr[elen++] = line[i++];
                    continue;
                }
                if(line[i] == ')'
                        && line[i + 1] == ')')
                {
                    depth--;
                    if(depth == 0)
                    {
                        i += 2; /* skip )) */
                        break;
                    }
                    expr[elen++] = line[i++];
                    expr[elen++] = line[i++];
                    continue;
                }
                expr[elen++] = line[i++];
            }
            expr[elen] = '\0';

            /* Evaluate */
            ArithParser parser;
            parser.s = expr;
            parser.pos = 0;
            double result = arith_expr(&parser);

            /* Format result: integer if whole,
             * else float */
            char rbuf[64];
            if(result == floor(result)
                    && fabs(result) < 1e15)
            {
                snprintf(rbuf, sizeof(rbuf),
                         "%ld", (long) result);
            }
            else
            {
                snprintf(rbuf, sizeof(rbuf),
                         "%g", result);
            }

            int rlen = (int) strlen(rbuf);
            int avail = maxlen - 1 - opos;
            int clen = rlen < avail
                       ? rlen : avail;
            memcpy(out + opos, rbuf,
                   (size_t) clen);
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
