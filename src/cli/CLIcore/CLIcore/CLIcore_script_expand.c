#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "CLIcore.h"
#include "CLIcore_UI.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"



#include <math.h>
#include <sys/stat.h>
#include <ctype.h>

/* ============================================================
 *  Arithmetic Expansion Helper Functions
 * ============================================================
 */

typedef struct
{
    const char *s;
    int         pos;
} ArithParser;

double arith_expr(ArithParser *p);

void arith_skip_ws(ArithParser *p)
{
    while(p->s[p->pos] == ' '
            || p->s[p->pos] == '\t')
    {
        p->pos++;
    }
}

double arith_atom(ArithParser *p)
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

    /* Variable name (bare identifier) */
    if(isalpha((unsigned char) p->s[p->pos])
       || p->s[p->pos] == '_')
    {
        char vname[256];
        int vn = 0;
        while(vn < 255
              && (isalnum(
                      (unsigned char)
                      p->s[p->pos])
                  || p->s[p->pos] == '_'))
        {
            vname[vn++] = p->s[p->pos++];
        }
        vname[vn] = '\0';
        const char *vv = cli_var_lookup(vname);
        if(vv != NULL)
        {
            return strtod(vv, NULL);
        }
        return 0.0;
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

    return 0.0;
}

double arith_factor(ArithParser *p)
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

double arith_term(ArithParser *p)
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

double arith_compare(ArithParser *p)
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

double arith_expr(ArithParser *p)
{
    return arith_compare(p);
}


/* ============================================================
 *  FPS Variable Expansion — @fpsname.param
 * ============================================================
 */

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

            char *dot = strchr(token, '.');
            if(dot == NULL)
            {
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

            FUNCTION_PARAMETER_STRUCT fps;
            int fpsconn =
                function_parameter_struct_connect(
                    fpsname, &fps,
                    FPSCONNECT_SIMPLE);

            if(fpsconn == -1
                    || fps.parray == NULL)
            {
                continue;
            }

            int pindex =
                functionparameter_GetParamIndex(
                    &fps, pname);

            if(pindex < 0)
            {
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


/**
 * @brief Expand $(( expr )) in place
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
        if(line[i] == '$'
                && line[i + 1] == '('
                && line[i + 2] == '(')
        {
            i += 3;

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
            parser.s = expr;
            parser.pos = 0;
            double result = arith_expr(&parser);

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


/* ============================================================
 *  Test Condition Evaluator — [ expr ]
 * ============================================================
 *
 * Evaluates bash-style test expressions:
 *   [ val1 -eq val2 ]   numeric equal
 *   [ val1 -ne val2 ]   numeric not equal
 *   [ val1 -lt val2 ]   numeric less than
 *   [ val1 -gt val2 ]   numeric greater than
 *   [ val1 -le val2 ]   numeric less or equal
 *   [ val1 -ge val2 ]   numeric greater or equal
 *   [ str1 == str2 ]    string equal
 *   [ str1 != str2 ]    string not equal
 *   [ -n str ]          string not empty
 *   [ -z str ]          string empty
 *   [ -f path ]         regular file exists
 *   [ -d path ]         directory exists
 *   [ -e path ]         path exists
 *   [ -s path ]         file exists, non-empty
 *   [ ! expr ]          logical NOT
 */

/**
 * @brief Evaluate [ ... ] test expression
 *
 * @param expr   The content between [ and ]
 * @return 1 = true, 0 = false
 */
int cli_eval_test(const char *expr)
{
    /* Tokenize expression */
    char buf[512];
    strncpy(buf, expr, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    /* Strip leading/trailing whitespace */
    char *p = buf;
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }
    {
        size_t len = strlen(p);
        while(len > 0
              && (p[len - 1] == ' '
                  || p[len - 1] == '\t'))
        {
            p[--len] = '\0';
        }
    }

    /* Tokenize into words */
    char *tokens[16];
    int ntok = 0;
    {
        char *saveptr = NULL;
        char *tok = strtok_r(p, " \t", &saveptr);
        while(tok != NULL && ntok < 16)
        {
            tokens[ntok++] = tok;
            tok = strtok_r(NULL, " \t", &saveptr);
        }
    }

    if(ntok == 0)
    {
        return 0;
    }

    /* Unary: -n str, -z str */
    if(ntok == 2
       && strcmp(tokens[0], "-n") == 0)
    {
        return strlen(tokens[1]) > 0 ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-z") == 0)
    {
        return strlen(tokens[1]) == 0 ? 1 : 0;
    }

    /* File tests: -f, -d, -e, -s */
    if(ntok == 2
       && strcmp(tokens[0], "-f") == 0)
    {
        struct stat sb;
        return (stat(tokens[1], &sb) == 0
                && S_ISREG(sb.st_mode))
               ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-d") == 0)
    {
        struct stat sb;
        return (stat(tokens[1], &sb) == 0
                && S_ISDIR(sb.st_mode))
               ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-e") == 0)
    {
        struct stat sb;
        return stat(tokens[1], &sb) == 0
               ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-s") == 0)
    {
        struct stat sb;
        return (stat(tokens[1], &sb) == 0
                && sb.st_size > 0)
               ? 1 : 0;
    }

    /* Variable test: -v VAR */
    if(ntok == 2
       && strcmp(tokens[0], "-v") == 0)
    {
        const char *vv =
            cli_var_get(tokens[1]);
        if(vv != NULL)
        {
            return 1;
        }
        const char *ev =
            getenv(tokens[1]);
        return ev != NULL ? 1 : 0;
    }

    /* Logical NOT: ! expr */
    if(ntok >= 2
       && strcmp(tokens[0], "!") == 0)
    {
        /* Rebuild sub-expression */
        char subexpr[512];
        subexpr[0] = '\0';
        for(int i = 1; i < ntok; i++)
        {
            if(i > 1)
            {
                strncat(subexpr, " ",
                        sizeof(subexpr)
                        - strlen(subexpr)
                        - 1);
            }
            strncat(subexpr, tokens[i],
                    sizeof(subexpr)
                    - strlen(subexpr) - 1);
        }
        return cli_eval_test(subexpr)
               ? 0 : 1;
    }

    /* Single value: true if non-empty */
    if(ntok == 1)
    {
        return strlen(tokens[0]) > 0 ? 1 : 0;
    }

    /* Binary: val1 op val2 */
    if(ntok >= 3)
    {
        const char *lhs = tokens[0];
        const char *op = tokens[1];
        const char *rhs = tokens[2];

        double lv = strtod(lhs, NULL);
        double rv = strtod(rhs, NULL);

        if(strcmp(op, "-eq") == 0)
        {
            return (lv == rv) ? 1 : 0;
        }
        if(strcmp(op, "-ne") == 0)
        {
            return (lv != rv) ? 1 : 0;
        }
        if(strcmp(op, "-lt") == 0)
        {
            return (lv < rv) ? 1 : 0;
        }
        if(strcmp(op, "-gt") == 0)
        {
            return (lv > rv) ? 1 : 0;
        }
        if(strcmp(op, "-le") == 0)
        {
            return (lv <= rv) ? 1 : 0;
        }
        if(strcmp(op, "-ge") == 0)
        {
            return (lv >= rv) ? 1 : 0;
        }
        if(strcmp(op, "==") == 0 || strcmp(op, "=") == 0)
        {
            return strcmp(lhs, rhs) == 0
                   ? 1 : 0;
        }
        if(strcmp(op, "!=") == 0)
        {
            return strcmp(lhs, rhs) != 0
                   ? 1 : 0;
        }
    }

    printf("Error: invalid test expression\n");
    return 0;
}

/* ============================================================
 *  Environment Variable Expansion — $VAR
 * ============================================================
 */

static void emit_str_local(
    char *out,
    int  *opos,
    int   maxlen,
    const char *s
)
{
    while(*s != '\0' && *opos < maxlen - 1)
    {
        out[(*opos)++] = *s++;
    }
}

void cli_expand_env(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0' && opos < maxlen - 1)
    {
        if(line[i] == '$')
        {
            /* Skip $(( — arithmetic let wordexp handle it */
            if(line[i + 1] == '(' && line[i + 2] == '(')
            {
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                continue;
            }
            /* Skip $( — command subst let wordexp handle it */
            if(line[i + 1] == '(')
            {
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                continue;
            }

            i++;
            int has_brace = 0;
            if(line[i] == '{')
            {
                has_brace = 1;
                i++;
            }

            char varname[256];
            int  vlen = 0;

            while(line[i] != '\0' && vlen < 255)
            {
                char c = line[i];
                if(!((c >= 'A' && c <= 'Z')
                     || (c >= 'a' && c <= 'z')
                     || (c >= '0' && c <= '9')
                     || c == '_'
                     || c == '?'
                     || c == '.'))
                {
                    break;
                }
                varname[vlen++] = line[i++];
                if(c == '?')
                {
                    break;
                }
            }
            varname[vlen] = '\0';

            /* Check if this is a supported $VAR string, else just copy literal.
             * E.g. ${#foo} is not supported here, handled elsewhere. */
            if(has_brace)
            {
                if(line[i] == '}')
                {
                    i++;
                }
                else
                {
                    /* Something complex like ${var:-def}, let wordexp handle it.
                     * We output ${varname back into out and continue. */
                    out[opos++] = '$';
                    out[opos++] = '{';
                    for(int k=0; k<vlen; k++) {
                        if(opos < maxlen - 1) out[opos++] = varname[k];
                    }
                    continue;
                }
            }

            const char *val = cli_var_lookup(varname);
            if(val != NULL)
            {
                emit_str_local(out, &opos, maxlen, val);
            }
            else
            {
                /* Variable not found or milk doesn't know it. We let wordexp
                 * try to expand it later from environment! Just emit original string */
                out[opos++] = '$';
                if (has_brace) out[opos++] = '{';
                for(int k=0; k<vlen; k++) {
                    if(opos < maxlen - 1) out[opos++] = varname[k];
                }
                if (has_brace && opos < maxlen - 1) out[opos++] = '}';
            }
        }
        else if (line[i] == '\\' && line[i+1] == '$')
        {
            /* Let wordexp handle escaped dollars, just copy both characters verbatim */
            out[opos++] = line[i++];
            out[opos++] = line[i++];
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