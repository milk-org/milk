/**
 * @file CLIcore_script_flow.c
 *
 * @brief Flow-control block executors
 *
 * Implements the bash-style flow-control constructs
 * for the milk CLI scripting engine:
 *
 * - **if/elif/else/fi**: Conditional branching with
 *   cascading elif support.
 * - **while/do/done**: Loop with condition re-evaluated
 *   each iteration, supporting break/continue.
 * - **select/do/done**: Interactive menu-style selection
 *   presenting numbered options to the user.
 * - **for/do/done**: Iterates over a word list, with
 *   optional brace expansion, command substitution,
 *   and C-style (( ; ; )) arithmetic loops.
 *
 * ## Design approach
 *
 * Each block executor receives a pre-accumulated array
 * of lines (provided by the block accumulator in
 * CLIcore_script.c). The executor parses the structure
 * (finding body boundaries, separating condition from
 * body), then calls cli_exec_lines() to run the body.
 *
 * Break/continue/return are implemented as global flags
 * (cli_break_flag, cli_continue_flag, cli_return_flag)
 * checked after each cli_exec_lines() call.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"

/**
 * @brief Execute an if/elif/else/fi block.
 *
 * Parses the accumulated lines to extract branch
 * conditions and body ranges, then evaluates each
 * condition in order. The first true branch's body
 * is executed; remaining branches are skipped.
 *
 * @param lines   Array of accumulated lines
 * @param nlines  Number of lines
 */
void cli_exec_block_if(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines)
{
    if(nlines < 2)
    {
        return;
    }

    /* Build a list of branches:
     * Each branch has a condition line index
     * and body range [start, end).
     * The final else has cond_idx = -1. */

    typedef struct
    {
        int cond_idx;
        int body_start;
        int body_end;
    } Branch;

    Branch branches[64];
    int nbranch = 0;

    /* First branch: the if line */
    int body_s = 1;
    /* Skip standalone "then" */
    if(body_s < nlines)
    {
        const char *ts = strip_ws(lines[body_s]);
        if(strcmp(ts, "then") == 0)
        {
            body_s++;
        }
    }

    branches[0].cond_idx = 0;
    branches[0].body_start = body_s;
    branches[0].body_end = nlines; /* Will be updated if elif/else found */
    nbranch = 1;

    /* Scan for elif/else at depth 0 */
    int depth = 0;
    for(int i = body_s; i < nlines; i++)
    {
        const char *ln = strip_ws(lines[i]);
        if(starts_with(ln, "if ")
           || starts_with(ln, "if\t"))
        {
            depth++;
            continue;
        }
        if(strcmp(ln, "fi") == 0)
        {
            if(depth > 0)
            {
                depth--;
                continue;
            }
            /* Should not happen since fi is not appended */
            branches[nbranch - 1].body_end = i;
            break;
        }
        if(depth > 0)
        {
            continue;
        }
        if(starts_with(ln, "elif ")
           || starts_with(ln, "elif\t"))
        {
            branches[nbranch - 1].body_end = i;
            int bs = i + 1;
            if(bs < nlines)
            {
                const char *t2 = strip_ws(lines[bs]);
                if(strcmp(t2, "then") == 0)
                {
                    bs++;
                }
            }
            if(nbranch < 64)
            {
                branches[nbranch].cond_idx = i;
                branches[nbranch].body_start = bs;
                branches[nbranch].body_end
                    = nlines; /* Will be updated if else found */
                nbranch++;
            }
        }
        else if(strcmp(ln, "else") == 0)
        {
            branches[nbranch - 1].body_end = i;
            if(nbranch < 64)
            {
                branches[nbranch].cond_idx
                    = -1; /* else */
                branches[nbranch].body_start = i + 1;
                branches[nbranch].body_end = nlines;
                nbranch++;
            }
            break;
        }
    }

    /* Evaluate branches in order */
    for(int b = 0; b < nbranch; b++)
    {
        int run = 0;
        if(branches[b].cond_idx < 0)
        {
            /* else — always true */
            run = 1;
        }
        else
        {
            const char *cl2 = strip_ws(lines[branches[b].cond_idx]);
            int skip = 2; /* "if" */
            if(starts_with(cl2, "elif"))
            {
                skip = 4;
            }
            run = eval_cond_line(lines[branches[b].cond_idx], skip);
        }
        if(run)
        {
            int bs = branches[b].body_start;
            int be = branches[b].body_end;
            if(be > bs)
            {
                cli_exec_lines(lines + bs, be - bs);
            }
            break;
        }
    }
}


/* ---- Parse while/do/done block ---- */

/**
 * @brief Execute a while/do/done block
 *
 * Expected format:
 *   lines[0]: "while [ condition ]; do"
 *   ...body...
 *   "done"
 */
void cli_exec_block_while(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines)
{
    if(nlines < 2)
    {
        return;
    }

    /* Find body start (after "do") */
    int body_start = 1;
    int body_end = nlines;
    int max_iter = 100000;

    /* Skip standalone 'do' line from
     * semicolon-split */
    if(body_start < body_end)
    {
        const char *ds = strip_ws(lines[body_start]);
        if(strcmp(ds, "do") == 0)
        {
            body_start++;
        }
    }

    for(int iter = 0; iter < max_iter; iter++)
    {
        /* Re-expand condition each iteration */
        char condline[STRINGMAXLEN_CLICMDLINE];
        strncpy(condline, lines[0], STRINGMAXLEN_CLICMDLINE - 1);
        condline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

        /* Run expansion on condition */
        cli_expand_fpsvar(condline, STRINGMAXLEN_CLICMDLINE);
        cli_expand_env(condline, STRINGMAXLEN_CLICMDLINE);
        cli_expand_arith(condline, STRINGMAXLEN_CLICMDLINE);

        /* Parse condition */
        const char *cl = strip_ws(condline);
        cl += 5; /* skip "while" */
        cl = strip_ws(cl);

        int cond_result = 0;
        if(*cl == '[')
        {
            cl++;
            const char *end = strrchr(cl, ']');
            if(end != NULL)
            {
                char cs[512];
                int clen = (int)(end - cl);
                if(clen >= (int) sizeof(cs))
                {
                    clen = (int) sizeof(cs) - 1;
                }
                memcpy(cs, cl, (size_t) clen);
                cs[clen] = '\0';
                cond_result = cli_eval_test(cs);
            }
        }
        else
        {
            /* Check command exit status */
            char ccmd[STRINGMAXLEN_CLICMDLINE];
            strncpy(ccmd, cl, sizeof(ccmd) - 1);
            ccmd[sizeof(ccmd) - 1] = '\0';
            
            char *semicolon = strstr(ccmd, ";");
            if(semicolon)
            {
                char *do_ptr = strstr(semicolon, "do");
                if(do_ptr)
                {
                    *semicolon = '\0';
                }
            }
            
            /* Strip trailing whitespace/semicolon */
            int len = (int)strlen(ccmd);
            while(len > 0 && (ccmd[len - 1] == ';' || ccmd[len - 1] == ' ' || ccmd[len - 1] == '\t'))
            {
                ccmd[--len] = '\0';
            }
            CLI_execute_string(ccmd);
            cond_result = (cli_last_retval == 0) ? 1 : 0;
        }

        if(!cond_result)
        {
            break;
        }

        /* Execute body */
        cli_continue_flag = 0;
        cli_exec_lines(lines +    body_start, body_end - body_start);

        if(cli_break_flag)
        {
            cli_break_flag = 0;
            break;
        }
    }
}


/* ---- Parse until/do/done block ---- */

/**
 * @brief Execute an until/do/done block
 *
 * Loops while condition is FALSE.
 * Expected format:
 *   lines[0]: "until [ condition ]; do"
 *   ...body...
 *   "done"
 */
void cli_exec_block_until(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines)
{
    if(nlines < 2)
    {
        return;
    }

    int body_start = 1;
    int body_end = nlines;
    int max_iter = 100000;

    if(body_start < body_end)
    {
        const char *ds = strip_ws(lines[body_start]);
        if(strcmp(ds, "do") == 0)
        {
            body_start++;
        }
    }

    for(int iter = 0;
        iter < max_iter; iter++)
    {
        char condline[STRINGMAXLEN_CLICMDLINE];
        strncpy(condline, lines[0], STRINGMAXLEN_CLICMDLINE - 1);
        condline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

        cli_expand_fpsvar(condline, STRINGMAXLEN_CLICMDLINE);
        cli_expand_env(condline, STRINGMAXLEN_CLICMDLINE);
        cli_expand_arith(condline, STRINGMAXLEN_CLICMDLINE);

        const char *cl = strip_ws(condline);
        cl += 5; /* skip "until" */
        cl = strip_ws(cl);

        int cond_result = 0;
        if(*cl == '[')
        {
            cl++;
            const char *end = strrchr(cl, ']');
            if(end != NULL)
            {
                char cs[512];
                int clen = (int)(end - cl);
                if(clen
                   >= (int) sizeof(cs))
                {
                    clen = (int) sizeof(cs) - 1;
                }
                memcpy(cs, cl, (size_t) clen);
                cs[clen] = '\0';
                cond_result = cli_eval_test(cs);
            }
        }
        else
        {
            char ccmd[STRINGMAXLEN_CLICMDLINE];
            strncpy(ccmd, cl, sizeof(ccmd) - 1);
            ccmd[sizeof(ccmd) - 1] = '\0';
            char *sc = strstr(ccmd, ";");
            if(sc)
            {
                char *dp = strstr(sc, "do");
                if(dp)
                {
                    *sc = '\0';
                }
            }
            int len = (int) strlen(ccmd);
            while(len > 0
                  && (ccmd[len - 1] == ';'
                      || ccmd[len - 1]
                         == ' '
                      || ccmd[len - 1]
                         == '\t'))
            {
                ccmd[--len] = '\0';
            }
            CLI_execute_string(ccmd);
            cond_result = (cli_last_retval == 0) ? 1 : 0;
        }

        /* until: loop while FALSE */
        if(cond_result)
        {
            break;
        }

        cli_continue_flag = 0;
        cli_exec_lines(lines +    body_start, body_end - body_start);

        if(cli_break_flag)
        {
            cli_break_flag = 0;
            break;
        }
    }
}


/* ---- Parse for/do/done block ---- */

/**
 * @brief Execute select/do/done block
 *
 * Syntax:
 *   select VAR in v1 v2 ...; do
 *     body
 *   done
 */
void cli_exec_block_select(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines)
{
    if(nlines < 2)
    {
        return;
    }
    /* Parse: select VAR in v1 v2 ... */
    const char *hdr = strip_ws(lines[0]);
    hdr += 7; /* skip 'select ' */
    while(*hdr == ' '
          || *hdr == '\t')
    {
        hdr++;
    }
    char vn[CLI_VAR_NAMELEN];
    {
        int vi = 0;
        while(*hdr != '\0'
              && *hdr != ' '
              && *hdr != '\t'
              && vi
              < CLI_VAR_NAMELEN - 1)
        {
            vn[vi++] = *hdr++;
        }
        vn[vi] = '\0';
    }
    while(*hdr == ' '
          || *hdr == '\t')
    {
        hdr++;
    }
    if(starts_with(hdr, "in "))
    {
        hdr += 3;
    }
    /* Collect values */
    char sv[256][CLI_VAR_VALLEN];
    int nsv = 0;
    while(*hdr != '\0'
          && nsv < 256)
    {
        while(*hdr == ' '
              || *hdr == '\t')
        {
            hdr++;
        }
        if(*hdr == ';'
           || *hdr == '\0')
        {
            break;
        }
        int vi = 0;
        while(*hdr != '\0'
              && *hdr != ' '
              && *hdr != '\t'
              && *hdr != ';'
              && vi
              < CLI_VAR_VALLEN - 1)
        {
            sv[nsv][vi++] = *hdr++;
        }
        sv[nsv][vi] = '\0';
        nsv++;
    }
    if(nsv == 0)
    {
        return;
    }
    /* Loop: print menu, read, exec */
    for(;;)
    {
        for(int i = 0;
            i < nsv; i++)
        {
            printf("%d) %s\n", i + 1, sv[i]);
        }
        printf("#? ");
        fflush(stdout);
        char rb[64];
        if(fgets(rb, sizeof(rb),
                 stdin) == NULL)
        {
            break;
        }
        int ch = (int) strtol(rb, NULL, 10);
        if(ch >= 1 && ch <= nsv)
        {
            cli_var_set(vn, sv[ch - 1]);
        }
        else
        {
            cli_var_set(vn, "");
        }
        cli_exec_lines(lines +  1, nlines - 1);
    }
}

/**
 * @brief Execute a for/do/done block
 *
 * Expected format:
 *   lines[0]: "for VAR in val1 val2 ...; do"
 *   ...body...
 *   "done"
 */
void cli_exec_block_for(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines)
{
    if(nlines < 2)
    {
        return;
    }

    /* Check for arithmetic for:
     * for ((init; cond; step)); do */
    {
        const char *af = strip_ws(lines[0]);
        af += 3; /* skip "for" */
        af = strip_ws(af);
        if(af[0] == '(' && af[1] == '(')
        {
            af += 2; /* skip "((" */
            /* Find closing )) */
            const char *ce = strstr(af, "))");
            if(ce != NULL)
            {
                char abuf[STRINGMAXLEN_CLICMDLINE];
                int alen = (int)(ce - af);
                memcpy(abuf, af, (size_t) alen);
                abuf[alen] = '\0';
                /* Split on ; */
                char ainit[256] = "";
                char acond[256] = "";
                char astep[256] = "";
                char *s1 = strchr(abuf, ';');
                if(s1 != NULL)
                {
                    *s1 = '\0';
                    strncpy(ainit, abuf, 255);
                    char *s2 = strchr(s1 + 1, ';');
                    if(s2 != NULL)
                    {
                        *s2 = '\0';
                        strncpy(acond, s1 + 1, 255);
                        strncpy(astep, s2 + 1, 255);
                    }
                }
                /* Execute init */
                {
                    char einit[STRINGMAXLEN_CLICMDLINE];
                    snprintf(einit, sizeof(einit), "$((%s))", ainit);
                    cli_expand_arith(einit, STRINGMAXLEN_CLICMDLINE);
                }
                /* Loop: eval cond,
                 * exec body, eval step */
                for(;;)
                {
                    char econd[STRINGMAXLEN_CLICMDLINE];
                    snprintf(econd, sizeof(econd), "$((%s))", acond);
                    cli_expand_arith(econd, STRINGMAXLEN_CLICMDLINE);
                    long cv = strtol(econd, NULL, 10);
                    if(cv == 0)
                    {
                        break;
                    }
                    cli_exec_lines(lines +  1, nlines - 1);
                    /* step */
                    {
                        char estep[STRINGMAXLEN_CLICMDLINE];
                        snprintf(estep, sizeof(estep), "$((%s))", astep);
                        cli_expand_arith(estep, STRINGMAXLEN_CLICMDLINE);
                    }
                }
                return;
            }
        }
    }

    /* Parse: for VAR in val1 val2 ... */
    const char *fl = strip_ws(lines[0]);
    fl += 3; /* skip "for" */
    fl = strip_ws(fl);

    /* Get variable name */
    char varname[CLI_VAR_NAMELEN];
    {
        int vn = 0;
        while(*fl != '\0' && *fl != ' '
              && *fl != '\t'
              && vn < CLI_VAR_NAMELEN - 1)
        {
            varname[vn++] = *fl++;
        }
        varname[vn] = '\0';
    }
    fl = strip_ws(fl);

    /* Skip "in" */
    if(strncmp(fl, "in ", 3) == 0
       || strncmp(fl, "in\t", 3) == 0)
    {
        fl += 2;
        fl = strip_ws(fl);
    }

    /* Collect values (strip trailing ;do) */
    char vallist[STRINGMAXLEN_CLICMDLINE];
    strncpy(vallist, fl, STRINGMAXLEN_CLICMDLINE - 1);
    vallist[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    /* Remove trailing "; do" or ";do" */
    {
        char *semi = strstr(vallist, ";");
        if(semi != NULL)
        {
            *semi = '\0';
        }
    }

    /* Strip trailing whitespace */
    {
        size_t vl = strlen(vallist);
        while(vl > 0
              && (vallist[vl - 1] == ' '
                  || vallist[vl - 1] == '\t'
                  || vallist[vl - 1] == '\n'))
        {
            vallist[--vl] = '\0';
        }
    }

    int body_start = 1;
    int body_end = nlines;

    /* Skip standalone 'do' line */
    if(body_start < body_end)
    {
        const char *ds = strip_ws(lines[body_start]);
        if(strcmp(ds, "do") == 0)
        {
            body_start++;
        }
    }

    /* Iterate over values */
    char *saveptr = NULL;
    char *val = strtok_r(vallist, " \t", &saveptr);
    while(val != NULL)
    {
        cli_var_set(varname, val);

        cli_continue_flag = 0;
        cli_exec_lines(lines +    body_start, body_end - body_start);

        if(cli_break_flag)
        {
            cli_break_flag = 0;
            break;
        }

        val = strtok_r(NULL, " \t", &saveptr);
    }
}


/* ============================================================
 *  User-Defined Functions
 * ============================================================
 */

CLI_FUNC cli_funcs[CLI_MAX_FUNCS];

/**
 * @brief Register a new user function
 */
