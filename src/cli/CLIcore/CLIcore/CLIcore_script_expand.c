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

    /* Bitwise NOT */
    if(p->s[p->pos] == '~')
    {
        p->pos++;
        return (double)(~(long)arith_atom(p));
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

double arith_shift(ArithParser *p)
{
    double left = arith_term(p);
    arith_skip_ws(p);

    while((p->s[p->pos] == '<' && p->s[p->pos + 1] == '<')
            || (p->s[p->pos] == '>' && p->s[p->pos + 1] == '>'))
    {
        char op = p->s[p->pos];
        p->pos += 2;
        double right = arith_term(p);
        arith_skip_ws(p);
        if(op == '<')
        {
            left = (double)((long)left << (long)right);
        }
        else
        {
            left = (double)((long)left >> (long)right);
        }
    }
    return left;
}

double arith_compare(ArithParser *p)
{
    double left = arith_shift(p);
    arith_skip_ws(p);

    if(p->s[p->pos] == '<'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left <= right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '>'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left >= right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '<')
    {
        p->pos++;
        double right = arith_shift(p);
        return (left < right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '>')
    {
        p->pos++;
        double right = arith_shift(p);
        return (left > right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '='
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left == right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '!'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_shift(p);
        return (left != right) ? 1.0 : 0.0;
    }
    return left;
}

double arith_bitwise_and(ArithParser *p)
{
    double left = arith_compare(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '&')
    {
        p->pos++;
        double right = arith_compare(p);
        arith_skip_ws(p);
        left = (double)((long)left & (long)right);
    }
    return left;
}

double arith_bitwise_xor(ArithParser *p)
{
    double left = arith_bitwise_and(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '^')
    {
        p->pos++;
        double right = arith_bitwise_and(p);
        arith_skip_ws(p);
        left = (double)((long)left ^ (long)right);
    }
    return left;
}

double arith_bitwise_or(ArithParser *p)
{
    double left = arith_bitwise_xor(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '|')
    {
        p->pos++;
        double right = arith_bitwise_xor(p);
        arith_skip_ws(p);
        left = (double)((long)left | (long)right);
    }
    return left;
}

double arith_expr(ArithParser *p)
{
    return arith_bitwise_or(p);
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

    /* Logical OR */
    for(int i = 0; i < ntok; i++)
    {
        if(strcmp(tokens[i], "-o") == 0)
        {
            char left[512]  = "";
            char right[512] = "";
            for(int j = 0; j < i; j++)
            {
                if(j > 0)
                {
                    strncat(left, " ",
                            sizeof(left) - strlen(left) - 1);
                }
                strncat(left, tokens[j],
                        sizeof(left) - strlen(left) - 1);
            }
            if(cli_eval_test(left))
            {
                return 1;
            }

            for(int j = i + 1; j < ntok; j++)
            {
                if(j > i + 1)
                {
                    strncat(right, " ",
                            sizeof(right) - strlen(right) - 1);
                }
                strncat(right, tokens[j],
                        sizeof(right) - strlen(right) - 1);
            }
            return cli_eval_test(right);
        }
    }

    /* Logical AND */
    for(int i = 0; i < ntok; i++)
    {
        if(strcmp(tokens[i], "-a") == 0)
        {
            char left[512]  = "";
            char right[512] = "";
            for(int j = 0; j < i; j++)
            {
                if(j > 0)
                {
                    strncat(left, " ",
                            sizeof(left) - strlen(left) - 1);
                }
                strncat(left, tokens[j],
                        sizeof(left) - strlen(left) - 1);
            }
            if(!cli_eval_test(left))
            {
                return 0;
            }

            for(int j = i + 1; j < ntok; j++)
            {
                if(j > i + 1)
                {
                    strncat(right, " ",
                            sizeof(right) - strlen(right) - 1);
                }
                strncat(right, tokens[j],
                        sizeof(right) - strlen(right) - 1);
            }
            return cli_eval_test(right);
        }
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

    /* File tests: -f, -d, -e, -s, -r, -w, -x, -L */
    if(ntok == 2
       && strcmp(tokens[0], "-r") == 0)
    {
        return access(tokens[1], R_OK) == 0 ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-w") == 0)
    {
        return access(tokens[1], W_OK) == 0 ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-x") == 0)
    {
        return access(tokens[1], X_OK) == 0 ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-L") == 0)
    {
        struct stat sb;
        return (lstat(tokens[1], &sb) == 0
                && S_ISLNK(sb.st_mode))
               ? 1 : 0;
    }
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

            int is_length = 0;
            if(has_brace && line[i] == '#')
            {
                is_length = 1;
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
            
            char index_str[256];
            int  has_index = 0;
            if(has_brace && line[i] == '[')
            {
                i++;
                int ilen = 0;
                while(line[i] != '\0' && line[i] != ']' && ilen < 255)
                {
                    index_str[ilen++] = line[i++];
                }
                if(line[i] == ']') i++;
                index_str[ilen] = '\0';
                has_index = 1;
            }

            char mod_op[3] = {0};
            char mod_arg[256] = {0};
            if(has_brace && line[i] == ':')
            {
                i++;
                if(line[i] == '-' || line[i] == '=' || line[i] == '?' || line[i] == '+')
                {
                    mod_op[0] = ':';
                    mod_op[1] = line[i++];
                }
                else
                {
                    mod_op[0] = ':';
                }
                int mlen = 0;
                while(line[i] != '\0' && line[i] != '}' && mlen < 255)
                {
                    mod_arg[mlen++] = line[i++];
                }
                mod_arg[mlen] = '\0';
            }

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
                    /* Something complex ... */
                    out[opos++] = '$';
                    out[opos++] = '{';
                    if (is_length) out[opos++] = '#';
                    for(int k=0; k<vlen; k++) {
                        if(opos < maxlen - 1) out[opos++] = varname[k];
                    }
                    continue;
                }
            }

            const char *val = NULL;
            if(has_index)
            {
                const char *idx_val = cli_var_lookup(index_str);
                if(idx_val == NULL) idx_val = index_str;
                
                int is_found = 0;
                for(int a = 0; a < CLI_MAX_ASSOC; a++)
                {
                    if(cli_assoc[a].used && strcmp(cli_assoc[a].name, varname) == 0)
                    {
                        for(int e = 0; e < cli_assoc[a].nelem; e++)
                        {
                            if(strcmp(cli_assoc[a].keys[e], idx_val) == 0)
                            {
                                val = cli_assoc[a].vals[e];
                                is_found = 1;
                                break;
                            }
                        }
                        break;
                    }
                }
                if(!is_found)
                {
                    int num_idx = atoi(idx_val);
                    for(int a = 0; a < CLI_MAX_ARRAYS; a++)
                    {
                        if(cli_arrays[a].used && strcmp(cli_arrays[a].name, varname) == 0)
                        {
                            if(num_idx >= 0 && num_idx < cli_arrays[a].nelem)
                            {
                                val = cli_arrays[a].elem[num_idx];
                            }
                            break;
                        }
                    }
                }
            }
            else
            {
                val = cli_var_lookup(varname);
            }

            char val_buf[256];
            val_buf[0] = '\0';

            if(mod_op[0] != '\0')
            {
                if(mod_op[1] == '-')
                {
                    if(val == NULL || val[0] == '\0') val = mod_arg;
                }
                else if(mod_op[1] == '=')
                {
                    if(val == NULL || val[0] == '\0')
                    {
                        val = mod_arg;
                        cli_var_set(varname, val);
                    }
                }
                else if(mod_op[1] == '?')
                {
                    if(val == NULL || val[0] == '\0')
                    {
                        printf("CLI expand error: %s: %s\n", varname, mod_arg);
                        val = "";
                    }
                }
                else if(mod_op[1] == '+')
                {
                    if(val != NULL && val[0] != '\0') val = mod_arg;
                    else val = "";
                }
                else if(mod_op[0] == ':')
                {
                    int offset = 0;
                    int length = 255;
                    char *colon = strchr(mod_arg, ':');
                    if(colon != NULL)
                    {
                        *colon = '\0';
                        const char *lval = cli_var_lookup(mod_arg);
                        offset = atoi(lval ? lval : mod_arg);
                        const char *rval = cli_var_lookup(colon + 1);
                        length = atoi(rval ? rval : colon + 1);
                    }
                    else
                    {
                        const char *lval = cli_var_lookup(mod_arg);
                        offset = atoi(lval ? lval : mod_arg);
                    }

                    if(val != NULL)
                    {
                        int v1 = (int) strlen(val);
                        if(offset < 0) offset = v1 + offset;
                        if(offset < 0) offset = 0;
                        if(offset > v1) offset = v1;
                        
                        if(length < 0) length = v1 - offset + length;
                        if(length < 0) length = 0;
                        if(offset + length > v1) length = v1 - offset;

                        strncpy(val_buf, val + offset, (size_t) length);
                        val_buf[length] = '\0';
                        val = val_buf;
                    }
                }
            }

            if(is_length)
            {
                int len = val ? (int) strlen(val) : 0;
                char numstr[32];
                snprintf(numstr, sizeof(numstr), "%d", len);
                emit_str_local(out, &opos, maxlen, numstr);
            }
            else if(val != NULL)
            {
                emit_str_local(out, &opos, maxlen, val);
            }
            else
            {
                /* E.g. var not found */
                out[opos++] = '$';
                if(has_brace) out[opos++] = '{';
                for(int k=0; k<vlen; k++) {
                    if(opos < maxlen - 1) out[opos++] = varname[k];
                }
                if(has_brace && opos < maxlen - 1) out[opos++] = '}';
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