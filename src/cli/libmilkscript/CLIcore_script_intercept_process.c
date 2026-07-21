// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stddef.h>
#include <math.h>
#include "CLIcore/cli_calc_parser.h"
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "milkscript.h"
#include "CLIcore_utils.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_checkargs.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>

extern int cli_block_level;
extern int cli_break_flag;
extern int CLI_trap_enable;
extern int cli_cmd_delay_us;


/**
 * @brief Handler: return from a function call.
 */
int cli_intercept_cmd_return(const char *p)
{
    if (strcmp(p, "return") == 0 || starts_with(p, "return ") || starts_with(p, "return\t"))
    {
        const char *rv = p + 6;
        while (*rv == ' ' || *rv == '\t')
        {
            rv++;
        }
        if (*rv != '\0')
        {
            cli_last_retval = (int) strtol(rv, NULL, 0);
        }
        cli_return_flag = 1;
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: exit the interpreter.
 */
int cli_intercept_cmd_exit(const char *p)
{
    if (strcmp(p, "exit") == 0 || starts_with(p, "exit ") || starts_with(p, "exit\t"))
    {
        int exitcode = 0;
        if (strlen(p) > 4)
        {
            const char *ev = p + 4;
            while (*ev == ' ' || *ev == '\t')
            {
                ev++;
            }
            if (*ev != '\0')
            {
                exitcode = (int) strtol(ev, NULL, 0);
            }
        }

        cli_trap_run_exit();
        exit(exitcode);
    }

    /* exitCLI — milk-specific graceful stop.
     * Sets CLIloopON = 0 so milkscript_run()
     * exits its read loop cleanly (no exit()
     * so trap/atexit handlers still fire). */
    if (strcmp(p, "exitCLI") == 0)
    {
        data.CLIloopON = 0;
        return 1;
    }

    return 0;
}

/**
 * @brief Handler: shift positional args left.
 */
int cli_intercept_cmd_shift(const char *p)
{
    if (strcmp(p, "shift") == 0 || starts_with(p, "shift ") || starts_with(p, "shift\t"))
    {
        int n = 1;
        if (strlen(p) > 5)
        {
            const char *sv = p + 5;
            while (*sv == ' ' || *sv == '\t')
            {
                sv++;
            }
            if (*sv != '\0')
            {
                n = (int) strtol(sv, NULL, 0);
            }
        }
        if (n < 1)
        {
            n = 1;
        }
        /* Shift $1..$9 by n positions */
        for (int i = 1; i < CLI_FUNC_MAXARGS; i++)
        {
            char dst[16], src[16];
            snprintf(dst, sizeof(dst), "%d", i);
            snprintf(src, sizeof(src), "%d", i + n);
            if (i + n < CLI_FUNC_MAXARGS)
            {
                const char *sv2 = cli_var_get(src);
                if (sv2 != NULL)
                {
                    cli_var_set(dst, sv2);
                }
                else
                {
                    cli_var_unset(dst);
                }
            }
            else
            {
                cli_var_unset(dst);
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procctl(const char *p)
{
    if (starts_with(p, "procctl ") || starts_with(p, "procctl\t"))
    {
        const char *ap = p + 7;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int  nlen = 0;
        while (*ap && *ap != ' ' && *ap != '\t' && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int ctrlval = -1;
        if (strncmp(ap, "run", 3) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_RUN;
        }
        else if (strncmp(ap, "pause", 5) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_PAUSE;
        }
        else if (strncmp(ap, "step", 4) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_INCR;
        }
        else if (strncmp(ap, "stop", 4) == 0 || strncmp(ap, "exit", 4) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_EXIT;
        }
        if (ctrlval < 0)
        {
            printf("procctl: unknown action "
                   "'%s' (use run|pause|"
                   "step|stop)\n",
                   ap);
            return 1;
        }
        if (pinfolist != NULL)
        {
            pid_t fpid = 0;
            for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
            {
                if (pinfolist->active[pi] && strcmp(pinfolist->pnamearray[pi], pname) == 0)
                {
                    fpid = pinfolist->PIDarray[pi];
                    break;
                }
            }
            if (fpid > 0)
            {
                char pfn[512];
                char pdname[256];
                processinfo_procdirname(pdname);
                snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                int          pfd = -1;
                PROCESSINFO *pi  = processinfo_shm_link(pfn, &pfd);
                if (pi != MAP_FAILED && pi != NULL)
                {
                    pi->CTRLval = ctrlval;
                    munmap(pi, sizeof(PROCESSINFO));
                    close(pfd);
                }
                else if (pfd >= 0)
                {
                    close(pfd);
                }
            }
            else
            {
                printf("procctl: process "
                       "'%s' not found\n",
                       pname);
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procwait(const char *p)
{
    if (starts_with(p, "procwait ") || starts_with(p, "procwait\t"))
    {
        const char *ap = p + 8;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int  nlen = 0;
        while (*ap && *ap != ' ' && *ap != '\t' && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int tgt = -1;
        if (strncasecmp(ap, "INIT", 4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_INIT;
        }
        else if (strncasecmp(ap, "ACTIVE", 6) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ACTIVE;
        }
        else if (strncasecmp(ap, "PAUSE", 5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_PAUSE;
        }
        else if (strncasecmp(ap, "STOP", 4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_STOP;
        }
        else if (strncasecmp(ap, "ERROR", 5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ERROR;
        }
        else
        {
            tgt = (int) strtol(ap, NULL, 0);
        }
        /* Skip state word */
        while (*ap && *ap != ' ' && *ap != '\t')
        {
            ap++;
        }
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        double timeout = 30.0;
        if (*ap != '\0')
        {
            timeout = strtod(ap, NULL);
        }
        struct timespec slp;
        slp.tv_sec      = 0;
        slp.tv_nsec     = 100000000; /* 100ms */
        double elapsed  = 0.0;
        cli_last_retval = 1;
        while (elapsed < timeout)
        {
            if (pinfolist != NULL)
            {
                for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
                {
                    if (pinfolist->active[pi] && strcmp(pinfolist->pnamearray[pi], pname) == 0)
                    {
                        pid_t fpid = pinfolist->PIDarray[pi];
                        char  pfn[512];
                        char  pdname[256];
                        processinfo_procdirname(pdname);
                        snprintf(pfn, sizeof(pfn),
                                 "%s/proc."
                                 "%d.shm",
                                 pdname, (int) fpid);
                        int          pfd = -1;
                        PROCESSINFO *pii = processinfo_shm_link(pfn, &pfd);
                        if (pii != MAP_FAILED && pii != NULL)
                        {
                            if (pii->loopstat == tgt)
                            {
                                cli_last_retval = 0;
                            }
                            munmap(pii, sizeof(PROCESSINFO));
                            close(pfd);
                        }
                        else if (pfd >= 0)
                        {
                            close(pfd);
                        }
                        break;
                    }
                }
            }
            if (cli_last_retval == 0)
            {
                break;
            }
            nanosleep(&slp, NULL);
            elapsed += 0.1;
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procstat(const char *p)
{
    if (strcmp(p, "procstat") == 0 || starts_with(p, "procstat ") || starts_with(p, "procstat\t"))
    {
        const char *ap = p + 8;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char filter[256];
        filter[0] = '\0';
        if (*ap != '\0')
        {
            strncpy(filter, ap, sizeof(filter) - 1);
            filter[sizeof(filter) - 1] = '\0';
        }
        if (pinfolist != NULL)
        {
            char pdname[256];
            processinfo_procdirname(pdname);
            for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
            {
                if (!pinfolist->active[pi])
                {
                    continue;
                }
                if (filter[0] != '\0' && strcmp(pinfolist->pnamearray[pi], filter) != 0)
                {
                    continue;
                }
                pid_t fpid = pinfolist->PIDarray[pi];
                char  pfn[512];
                snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                int          pfd = -1;
                PROCESSINFO *pii = processinfo_shm_link(pfn, &pfd);
                if (pii == MAP_FAILED || pii == NULL)
                {
                    if (pfd >= 0)
                    {
                        close(pfd);
                    }
                    continue;
                }
                const char *stname = "UNKNOWN";
                switch (pii->loopstat)
                {
                case 0:
                    stname = "INIT";
                    break;
                case 1:
                    stname = "ACTIVE";
                    break;
                case 2:
                    stname = "PAUSED";
                    break;
                case 3:
                    stname = "STOPPED";
                    break;
                case 4:
                    stname = "ERROR";
                    break;
                case 5:
                    stname = "SPINNING";
                    break;
                case 6:
                    stname = "CRASHED";
                    break;
                }
                double hz = 0.0;
                if (pii->dtmedian_iter_ns > 0)
                {
                    hz = 1.0e9 / (double) pii->dtmedian_iter_ns;
                }
                double us = (double) pii->dtmedian_exec_ns / 1000.0;
                printf("name=%s\n"
                       "pid=%d\n"
                       "loopstat=%s\n"
                       "loopcnt=%ld\n"
                       "loopfreq_hz=%.1f\n"
                       "exectime_us=%.1f\n"
                       "rtprio=%d\n"
                       "ctrlval=%d\n"
                       "missedframes=%lu\n"
                       "tmux=%s\n",
                       pii->name, (int) pii->PID, stname, pii->loopcnt, hz, us, pii->RT_priority,
                       pii->CTRLval, (unsigned long) pii->triggermissedframe_cumul, pii->tmuxname);
                munmap(pii, sizeof(PROCESSINFO));
                close(pfd);
                if (filter[0] != '\0')
                {
                    break;
                }
                printf("---\n");
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_time(const char *p)
{
    if (starts_with(p, "time ") || starts_with(p, "time\t"))
    {
        const char *cmd = p + 4;
        while (*cmd == ' ' || *cmd == '\t')
        {
            cmd++;
        }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CLI_execute_string(cmd);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed =
            (double) (t1.tv_sec - t0.tv_sec) + (double) (t1.tv_nsec - t0.tv_nsec) / 1.0e9;
        printf("\nreal\t%.3fs\n", elapsed);
        return 1;
    }
    return 0;
}

/**
 * cli_assert_eval - evaluate an expression to double.
 * @expr: expression string (trimmed)
 * @out:  result written here
 *
 * Return: 1 on success, 0 on failure.
 */
static int cli_assert_eval(const char *expr, double *out)
{
    int    type = 0;
    long   lval = 0;
    double dval = 0.0;
    int    ok   = cli_calc_eval_math_to_val(expr, &type, &lval, &dval);
    if (!ok)
    {
        return 0;
    }
    *out = (type == 1) ? (double) lval : dval;
    return 1;
}

/**
 * cli_cmp_ok - evaluate a single comparison.
 * @a:     left value
 * @b:     right value
 * @op_ch: '<' or '>'
 * @op_eq: 1 if operator includes '='
 *
 * Return: 1 if comparison holds, 0 otherwise.
 */
static int cli_cmp_ok(double a, double b, char op_ch, int op_eq)
{
    if (op_ch == '<')
    {
        return op_eq ? (a <= b) : (a < b);
    }
    return op_eq ? (a >= b) : (a > b);
}

/** Operator symbol string for display. */
static const char *cli_cmp_sym(char op_ch, int op_eq)
{
    if (op_ch == '<')
    {
        return op_eq ? "<=" : "<";
    }
    return op_eq ? ">=" : ">";
}

/**
 * cli_assert_trim - copy src into dst, trim
 *      leading and trailing whitespace.
 * @dst:  output buffer
 * @sz:   sizeof(dst)
 * @src:  source string
 * @len:  number of chars to copy from src
 */
static void cli_assert_trim(char *dst, size_t sz, const char *src, int len)
{
    if (len >= (int) sz)
    {
        len = (int) sz - 1;
    }
    /* skip leading ws */
    while (len > 0 && (*src == ' ' || *src == '\t'))
    {
        src++;
        len--;
    }
    memcpy(dst, src, (size_t) len);
    dst[len] = '\0';
    /* strip trailing ws */
    int ri = len - 1;
    while (ri >= 0 && (dst[ri] == ' ' || dst[ri] == '\t'))
    {
        dst[ri--] = '\0';
    }
}

/**
 * cli_assert_cmp - handle comparison assert.
 * @ap: text after "assert " (trimmed)
 *
 * Supports:
 *   assert expr<val   assert expr<=val
 *   assert expr>val   assert expr>=val
 *   assert lo<expr<hi (range, any combo of
 *          <, <=, >, >=)
 *
 * Return: 1 if a comparison was found and
 *         handled, 0 if no comparison operator.
 */
static int cli_assert_cmp(const char *ap)
{
    /* Find first < or > */
    const char *op1 = NULL;
    for (const char *s = ap; *s; s++)
    {
        if (*s == '<' || *s == '>')
        {
            op1 = s;
            break;
        }
    }
    if (op1 == NULL)
    {
        return 0;
    }

    char o1ch  = *op1;
    int  o1eq  = (*(op1 + 1) == '=');
    int  o1len = 1 + o1eq;

    char part1[512];
    cli_assert_trim(part1, sizeof(part1), ap, (int) (op1 - ap));

    const char *after1 = op1 + o1len;

    /* Look for second < or > (range) */
    const char *op2 = NULL;
    for (const char *s = after1; *s; s++)
    {
        if (*s == '<' || *s == '>')
        {
            op2 = s;
            break;
        }
    }

    if (op2 != NULL)
    {
        /* Range: part1 op1 part2 op2 part3 */
        char o2ch  = *op2;
        int  o2eq  = (*(op2 + 1) == '=');
        int  o2len = 1 + o2eq;

        char part2[512];
        cli_assert_trim(part2, sizeof(part2), after1, (int) (op2 - after1));

        char        part3[512];
        const char *after2 = op2 + o2len;
        cli_assert_trim(part3, sizeof(part3), after2, (int) strlen(after2));

        double v1, v2, v3;
        if (!cli_assert_eval(part1, &v1) || !cli_assert_eval(part2, &v2) ||
            !cli_assert_eval(part3, &v3))
        {
            printf("\033[1;31m"
                   "[ASSERT FAIL] "
                   "cannot evaluate: "
                   "%s %s %s %s %s"
                   "\033[0m\n",
                   part1, cli_cmp_sym(o1ch, o1eq), part2, cli_cmp_sym(o2ch, o2eq), part3);
            cli_last_retval = 1;
            return 1;
        }

        int ok = cli_cmp_ok(v1, v2, o1ch, o1eq) && cli_cmp_ok(v2, v3, o2ch, o2eq);
        if (ok)
        {
            printf("\033[1;32m"
                   "[ASSERT PASS] "
                   "%.*g %s %s(=%.*g) "
                   "%s %.*g"
                   "\033[0m\n",
                   cli_float_digits, v1, cli_cmp_sym(o1ch, o1eq), part2, cli_float_digits, v2,
                   cli_cmp_sym(o2ch, o2eq), cli_float_digits, v3);
        }
        else
        {
            printf("\033[1;31m"
                   "[ASSERT FAIL] "
                   "%.*g %s %s(=%.*g) "
                   "%s %.*g"
                   "\033[0m\n",
                   cli_float_digits, v1, cli_cmp_sym(o1ch, o1eq), part2, cli_float_digits, v2,
                   cli_cmp_sym(o2ch, o2eq), cli_float_digits, v3);
            cli_last_retval = 1;
            if (cli_flag_errexit)
            {
                cli_trap_run(-1);
            }
        }
    }
    else
    {
        /* Single: part1 op1 rhs */
        char rhs[512];
        cli_assert_trim(rhs, sizeof(rhs), after1, (int) strlen(after1));

        double v1, v2;
        if (!cli_assert_eval(part1, &v1) || !cli_assert_eval(rhs, &v2))
        {
            printf("\033[1;31m"
                   "[ASSERT FAIL] "
                   "cannot evaluate: "
                   "%s %s %s"
                   "\033[0m\n",
                   part1, cli_cmp_sym(o1ch, o1eq), rhs);
            cli_last_retval = 1;
            return 1;
        }

        if (cli_cmp_ok(v1, v2, o1ch, o1eq))
        {
            printf("\033[1;32m"
                   "[ASSERT PASS] "
                   "%s(=%.*g) %s %.*g"
                   "\033[0m\n",
                   part1, cli_float_digits, v1, cli_cmp_sym(o1ch, o1eq), cli_float_digits, v2);
        }
        else
        {
            printf("\033[1;31m"
                   "[ASSERT FAIL] "
                   "%s(=%.*g) NOT %s "
                   "%.*g"
                   "\033[0m\n",
                   part1, cli_float_digits, v1, cli_cmp_sym(o1ch, o1eq), cli_float_digits, v2);
            cli_last_retval = 1;
            if (cli_flag_errexit)
            {
                cli_trap_run(-1);
            }
        }
    }
    return 1;
}

int cli_intercept_cmd_assert(const char *p)
{
    if (starts_with(p, "assert ") || starts_with(p, "assert\t"))
    {
        const char *ap = p + 6;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }

        if (*ap == '[')
        {
            /* Bracket syntax:
             * assert [ condition ] "msg" */
            ap++;
            const char *end = strrchr(ap, ']');
            if (end != NULL)
            {
                char cs[512];
                int  clen = (int) (end - ap);
                if (clen >= (int) sizeof(cs))
                {
                    clen = (int) sizeof(cs) - 1;
                }
                memcpy(cs, ap, (size_t) clen);
                cs[clen]   = '\0';
                int result = cli_eval_test(cs);
                if (result)
                {
                    printf("\033[1;32m"
                           "[ASSERT PASS]"
                           "\033[0m\n");
                    cli_last_retval = 0;
                }
                else
                {
                    const char *msg = end + 1;
                    while (*msg == ' ' || *msg == '\t')
                    {
                        msg++;
                    }
                    if (*msg == '"' || *msg == '\'')
                    {
                        msg++;
                    }
                    int mlen = (int) strlen(msg);
                    if (mlen > 0 && (msg[mlen - 1] == '"' || msg[mlen - 1] == '\''))
                    {
                        char mb[512];
                        strncpy(mb, msg, sizeof(mb) - 1);
                        mb[sizeof(mb) - 1] = '\0';
                        if (mlen - 1 < (int) sizeof(mb))
                        {
                            mb[mlen - 1] = '\0';
                        }
                        printf("\033[1;31m"
                               "[ASSERT FAIL] "
                               "%s\033[0m\n",
                               mb);
                    }
                    else
                    {
                        printf("\033[1;31m"
                               "[ASSERT FAIL] "
                               "%s\033[0m\n",
                               msg);
                    }
                    cli_last_retval = 1;
                    if (cli_flag_errexit)
                    {
                        cli_trap_run(-1);
                    }
                }
            }
            else
            {
                printf("\033[1;31m"
                       "[ASSERT] missing ']'"
                       "\033[0m\n");
                cli_last_retval = 1;
            }
        }
        else if (cli_assert_cmp(ap))
        {
            /* Comparison handled */
        }
        else
        {
            /* Equality syntax:
             * assert expr=expected [~tol] */
            const char *eq = strchr(ap, '=');
            if (eq == NULL)
            {
                printf("\033[1;31m"
                       "[ASSERT] syntax: "
                       "assert expr=expected "
                       "[~tol] or expr<val"
                       "\033[0m\n");
                cli_last_retval = 1;
            }
            else
            {
                char lhs[512];
                int  llen = (int) (eq - ap);
                if (llen > 511)
                {
                    llen = 511;
                }
                memcpy(lhs, ap, (size_t) llen);
                lhs[llen] = '\0';

                const char *rp = eq + 1;
                char        rhs[512];
                double      tol = 0.0;

                const char *tilde = strchr(rp, '~');
                if (tilde != NULL)
                {
                    int rlen = (int) (tilde - rp);
                    if (rlen > 511)
                    {
                        rlen = 511;
                    }
                    memcpy(rhs, rp, (size_t) rlen);
                    rhs[rlen] = '\0';
                    tol       = fabs(strtod(tilde + 1, NULL));
                }
                else
                {
                    strncpy(rhs, rp, 511);
                    rhs[511] = '\0';
                }

                /* Trim trailing ws */
                {
                    int ri = (int) strlen(rhs) - 1;
                    while (ri >= 0 && (rhs[ri] == ' ' || rhs[ri] == '\t'))
                    {
                        rhs[ri--] = '\0';
                    }
                }

                int    ltype = 0;
                long   llval = 0;
                double ldval = 0.0;
                int    lok   = cli_calc_eval_math_to_val(lhs, &ltype, &llval, &ldval);
                if (lok && ltype == 1)
                {
                    ldval = (double) llval;
                }

                int    rtype = 0;
                long   rlval = 0;
                double rdval = 0.0;
                int    rok   = cli_calc_eval_math_to_val(rhs, &rtype, &rlval, &rdval);
                if (rok && rtype == 1)
                {
                    rdval = (double) rlval;
                }

                if (!lok || !rok)
                {
                    printf("\033[1;31m"
                           "[ASSERT FAIL] "
                           "cannot evaluate: "
                           "%s = %s"
                           "\033[0m\n",
                           lhs, rhs);
                    cli_last_retval = 1;
                }
                else
                {
                    double diff = fabs(ldval - rdval);
                    if (diff <= tol)
                    {
                        printf("\033[1;32m"
                               "[ASSERT PASS] "
                               "%s = %.*g"
                               " (expected "
                               "%.*g ~%.*g)"
                               "\033[0m\n",
                               lhs, cli_float_digits, ldval, cli_float_digits, rdval,
                               cli_float_digits, tol);
                    }
                    else
                    {
                        printf("\033[1;31m"
                               "[ASSERT FAIL] "
                               "%s = %.*g"
                               " (expected "
                               "%.*g ~%.*g,"
                               " diff=%.*g)"
                               "\033[0m\n",
                               lhs, cli_float_digits, ldval, cli_float_digits, rdval,
                               cli_float_digits, tol, cli_float_digits, diff);
                        cli_last_retval = 1;
                        if (cli_flag_errexit)
                        {
                            cli_trap_run(-1);
                        }
                    }
                }
            }
        }
        return 1;
    }
    return 0;
}

/**
 * cli_intercept_cmd_assigncheck - check and assign value
 * @p: full input line
 *
 * Syntax:
 *   assigncheck [-e] <varname> <value> <tolerance>
 *
 * Evaluates <value> and <tolerance>. If <varname> existed,
 * checks if abs(old - new) <= tolerance. Sets <varname> to <value>.
 * If -e is set (or cli_flag_errexit is 1) and check fails, trap/exit.
 *
 * Return: 1 if handled, 0 if not assigncheck.
 */
int cli_intercept_cmd_assigncheck(const char *p)
{
    const char *sp = strip_ws(p);
    if (!starts_with(sp, "assigncheck ") && !starts_with(sp, "assigncheck\t"))
    {
        return 0;
    }

    sp += 11; // skip "assigncheck"
    while (*sp == ' ' || *sp == '\t')
    {
        sp++;
    }

    int errexit_local = 0;
    if (starts_with(sp, "-e ") || starts_with(sp, "-e\t"))
    {
        errexit_local = 1;
        sp += 2;
        while (*sp == ' ' || *sp == '\t')
        {
            sp++;
        }
    }

    char buf[512];
    strncpy(buf, sp, 511);
    buf[511] = '\0';

    char *tol_ptr = NULL;
    {
        int len = (int) strlen(buf);
        while (len > 0 && (buf[len - 1] == ' ' || buf[len - 1] == '\t' || buf[len - 1] == '\n' ||
                           buf[len - 1] == '\r'))
        {
            buf[--len] = '\0';
        }

        for (int i = len - 1; i >= 0; i--)
        {
            if (buf[i] == ' ' || buf[i] == '\t')
            {
                buf[i]  = '\0';
                tol_ptr = &buf[i + 1];
                while (i > 0 && (buf[i - 1] == ' ' || buf[i - 1] == '\t'))
                {
                    i--;
                    buf[i] = '\0';
                }
                break;
            }
        }
    }

    if (!tol_ptr)
    {
        goto usage_err;
    }

    char *var_ptr = buf;
    char *val_ptr = NULL;
    for (int i = 0; buf[i] != '\0'; i++)
    {
        if (buf[i] == ' ' || buf[i] == '\t')
        {
            buf[i]  = '\0';
            val_ptr = &buf[i + 1];
            while (*val_ptr == ' ' || *val_ptr == '\t')
            {
                val_ptr++;
            }
            break;
        }
    }

    if (!val_ptr || val_ptr[0] == '\0')
    {
        goto usage_err;
    }

    double new_val = 0.0;
    {
        int  vtype = 0;
        long vlval = 0;
        if (!cli_calc_eval_math_to_val(val_ptr, &vtype, &vlval, &new_val))
        {
            printf("\033[1;31m[ASSIGNCHECK FAIL] cannot evaluate value: %s\033[0m\n", val_ptr);
            cli_last_retval = 1;
            if (errexit_local || cli_flag_errexit)
            {
                cli_trap_run(-1);
                cli_trap_run_exit();
                exit(cli_last_retval);
            }
            return 1;
        }
        if (vtype == 1)
        {
            new_val = (double) vlval;
        }
    }

    double tol_val = 0.0;
    {
        int  ttype = 0;
        long tlval = 0;
        if (!cli_calc_eval_math_to_val(tol_ptr, &ttype, &tlval, &tol_val))
        {
            printf("\033[1;31m[ASSIGNCHECK FAIL] cannot evaluate tolerance: %s\033[0m\n", tol_ptr);
            cli_last_retval = 1;
            if (errexit_local || cli_flag_errexit)
            {
                cli_trap_run(-1);
                cli_trap_run_exit();
                exit(cli_last_retval);
            }
            return 1;
        }
        if (ttype == 1)
        {
            tol_val = (double) tlval;
        }
    }

    double old_val = 0.0;
    int    has_old = 0;
    {
        const char *oldv_str = cli_var_lookup(var_ptr);
        if (oldv_str)
        {
            has_old    = 1;
            int  otype = 0;
            long olval = 0;
            if (cli_calc_eval_math_to_val(oldv_str, &otype, &olval, &old_val))
            {
                if (otype == 1)
                {
                    old_val = (double) olval;
                }
            }
        }
    }

    {
        char obuf[128];
        snprintf(obuf, sizeof(obuf), "%.*g", cli_float_digits, new_val);
        cli_var_set(var_ptr, obuf);
    }

    if (has_old)
    {
        double diff = fabs(old_val - new_val);
        if (diff <= tol_val)
        {
            printf(
                "\033[1;32m[ASSIGNCHECK PASS] %s: old=%.*g new=%.*g diff=%.*g (tol %.*g)\033[0m\n",
                var_ptr, cli_float_digits, old_val, cli_float_digits, new_val, cli_float_digits,
                diff, cli_float_digits, tol_val);
        }
        else
        {
            printf(
                "\033[1;31m[ASSIGNCHECK FAIL] %s: old=%.*g new=%.*g diff=%.*g (tol %.*g)\033[0m\n",
                var_ptr, cli_float_digits, old_val, cli_float_digits, new_val, cli_float_digits,
                diff, cli_float_digits, tol_val);
            cli_last_retval = 1;
            if (errexit_local || cli_flag_errexit)
            {
                cli_trap_run(-1);
                cli_trap_run_exit();
                exit(cli_last_retval);
            }
        }
    }
    else
    {
        printf("\033[1;32m[ASSIGNCHECK NEW] %s initialized to %.*g\033[0m\n", var_ptr,
               cli_float_digits, new_val);
    }
    return 1;

usage_err:
    printf("\033[1;31m[ASSIGNCHECK FAIL] Usage: assigncheck [-e] <var> <val> <tol>\033[0m\n");
    cli_last_retval = 1;
    if (errexit_local || cli_flag_errexit)
    {
        cli_trap_run(-1);
        cli_trap_run_exit();
        exit(cli_last_retval);
    }
    return 1;
}

/**
 * cli_intercept_cmd_dpdigits - set/query
 *     float display precision.
 * @p: full input line
 *
 * Syntax:
 *   dpdigits       — print current value
 *   dpdigits N     — set to N (1–17)
 *
 * Return: 1 if handled, 0 if not dpdigits.
 */
int cli_intercept_cmd_dpdigits(const char *p)
{
    const char *sp = strip_ws(p);
    if (strcmp(sp, "dpdigits") == 0)
    {
        printf("dpdigits = %d\n", cli_float_digits);
        return 1;
    }
    if (starts_with(sp, "dpdigits ") || starts_with(sp, "dpdigits\t"))
    {
        const char *ap = sp + 9;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int val = (int) strtol(ap, NULL, 10);
        if (val < 1)
        {
            val = 1;
        }
        if (val > 17)
        {
            val = 17;
        }
        cli_float_digits = val;
        printf("dpdigits = %d\n", cli_float_digits);
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_watch(const char *p)
{
    if (starts_with(p, "watch ") || starts_with(p, "watch\t"))
    {
        const char *wp = p + 5;
        while (*wp == ' ' || *wp == '\t')
        {
            wp++;
        }
        if (*wp == '-' && *(wp + 1) == 'n')
        {
            double interval = 2.0;
            wp += 2;
            while (*wp == ' ' || *wp == '\t')
            {
                wp++;
            }
            interval = strtod(wp, NULL);
            while (*wp != ' ' && *wp != '\t' && *wp != '\0')
            {
                wp++;
            }
            while (*wp == ' ' || *wp == '\t')
            {
                wp++;
            }
            struct timespec ts;
            ts.tv_sec  = (time_t) interval;
            ts.tv_nsec = (long) ((interval - (double) ts.tv_sec) * 1.0e9);
            while (!cli_break_flag)
            {
                printf("\033[2J\033[H"
                       "Every %.1fs: %s\n\n",
                       interval, wp);
                CLI_execute_string(wp);
                nanosleep(&ts, NULL);
            }
            cli_break_flag = 0;
            return 1;
        }
    }
    return 0;
}
