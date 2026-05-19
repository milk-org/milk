/**
 * @file CLIcore_script_expand_test.c
 *
 * @brief Bash-style [ test ] expression evaluator.
 *
 * Implements cli_eval_test(), which evaluates
 * conditional expressions in the style of POSIX
 * `test` / `[ ]`:
 *
 * File tests:  -f, -d, -e, -s, -r, -w, -x, -L
 * String/var:  -n, -z, -v
 * SHM tests:   -S (stream), -F (FPS), -P (process)
 * Process:     -R (process running)
 * Numeric:     -eq, -ne, -lt, -gt, -le, -ge
 * String cmp:  ==, =, !=, =~
 * Logical:     !, -a, -o
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_eval_test()
 */

#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"

/* processinfo functions — linked via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(
    const char *pname, int *fd);
extern errno_t processinfo_procdirname(
    char *procdname);


/* ============================================================
 *  Unary file tests
 * ============================================================
 */

/**
 * test_unary_file - evaluate filesystem unary tests
 * @op:     Operator string (e.g. "-f", "-d")
 * @arg:    File path argument
 * @result: Written with 1 (true) or 0 (false)
 *
 * Handles: -f (regular file), -d (directory),
 * -e (exists), -s (non-empty), -r (readable),
 * -w (writable), -x (executable), -L (symlink).
 *
 * Returns 1 if @op was recognised, 0 otherwise.
 */
static int test_unary_file(
    const char *op,
    const char *arg,
    int        *result)
{
    if(strcmp(op, "-r") == 0)
    {
        *result = access(arg, R_OK) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-w") == 0)
    {
        *result = access(arg, W_OK) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-x") == 0)
    {
        *result = access(arg, X_OK) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-L") == 0)
    {
        struct stat sb;
        *result = (lstat(arg, &sb) == 0 && S_ISLNK(sb.st_mode)) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-f") == 0)
    {
        struct stat sb;
        *result = (stat(arg, &sb) == 0 && S_ISREG(sb.st_mode)) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-d") == 0)
    {
        struct stat sb;
        *result = (stat(arg, &sb) == 0 && S_ISDIR(sb.st_mode)) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-e") == 0)
    {
        struct stat sb;
        *result = stat(arg, &sb) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-s") == 0)
    {
        struct stat sb;
        *result = (stat(arg, &sb) == 0 && sb.st_size > 0) ? 1 : 0;
        return 1;
    }
    return 0;
}


/* ============================================================
 *  Unary SHM and variable tests
 * ============================================================
 */

/**
 * test_unary_shm - evaluate SHM / variable unary tests
 * @op:     Operator string
 * @arg:    Argument string
 * @result: Written with 1 (true) or 0 (false)
 *
 * Handles:
 *   -n  non-empty string
 *   -z  empty string
 *   -v  variable is set
 *   -S  SHM stream exists
 *   -F  FPS instance exists
 *   -P  process is registered in pinfolist
 *   -R  process is registered AND running
 *
 * Returns 1 if @op was recognised, 0 otherwise.
 */
static int test_unary_shm(
    const char *op,
    const char *arg,
    int        *result)
{
    if(strcmp(op, "-n") == 0)
    {
        *result = strlen(arg) > 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-z") == 0)
    {
        *result = strlen(arg) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-v") == 0)
    {
        const char *vv = cli_var_get(arg);
        if(vv != NULL)
        {
            *result = 1;
            return 1;
        }
        *result = getenv(arg) != NULL ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-S") == 0)
    {
        char shmpath[512];
        struct stat sb;
        snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", dcshmdir, arg);
        *result = stat(shmpath, &sb) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-F") == 0)
    {
        char shmpath[512];
        struct stat sb;
        snprintf(shmpath, sizeof(shmpath), "%s/fps.%s.shm", dcshmdir, arg);
        *result = stat(shmpath, &sb) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-P") == 0)
    {
        *result = 0;
        if(pinfolist != NULL)
        {
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(pinfolist->active[pi]
                   && strcmp(
                       pinfolist
                           ->pnamearray[pi],
                       arg) == 0)
                {
                    *result = 1;
                    break;
                }
            }
        }
        return 1;
    }
    if(strcmp(op, "-R") == 0)
    {
        /* Process registered AND running */
        *result = 0;
        if(pinfolist != NULL)
        {
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(pinfolist->active[pi]
                   && strcmp(
                       pinfolist
                           ->pnamearray[pi],
                       arg) == 0)
                {
                    pid_t fpid = pinfolist->PIDarray[pi];
                    if(fpid > 0)
                    {
                        char pfn[512];
                        char pdname[256];
                        processinfo_procdirname(pdname);
                        snprintf(pfn, sizeof(pfn), "%s/proc.%d" ".shm", pdname, (int) fpid);
                        int pfd = -1;
                        PROCESSINFO *pi_shm = processinfo_shm_link(pfn, &pfd);
                        if(pi_shm != MAP_FAILED
                           && pi_shm != NULL)
                        {
                            if(pi_shm->CTRLval
                               == PROCESSINFO_CTRLVAL_RUN)
                            {
                                *result = 1;
                            }
                            munmap(pi_shm, sizeof(PROCESSINFO));
                            close(pfd);
                        }
                        else if(pfd >= 0)
                        {
                            close(pfd);
                        }
                    }
                    break;
                }
            }
        }
        return 1;
    }
    return 0;
}


/* ============================================================
 *  Binary comparison operators
 * ============================================================
 */

/**
 * test_binary_op - evaluate binary test operators
 * @lhs:    Left operand string
 * @op:     Operator
 * @rhs:    Right operand string
 * @result: Written with 1 (true) or 0 (false)
 *
 * Numeric: -eq, -ne, -lt, -gt, -le, -ge
 * String:  ==, =, !=
 *
 * Returns 1 if @op was recognised, 0 otherwise.
 */
static int test_binary_op(
    const char *lhs,
    const char *op,
    const char *rhs,
    int        *result)
{
    double lv = strtod(lhs, NULL);
    double rv = strtod(rhs, NULL);

    if(strcmp(op, "-eq") == 0)
    {
        *result = (lv == rv) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-ne") == 0)
    {
        *result = (lv != rv) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-lt") == 0)
    {
        *result = (lv < rv) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-gt") == 0)
    {
        *result = (lv > rv) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-le") == 0)
    {
        *result = (lv <= rv) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "-ge") == 0)
    {
        *result = (lv >= rv) ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "==") == 0
       || strcmp(op, "=") == 0)
    {
        *result = strcmp(lhs, rhs) == 0 ? 1 : 0;
        return 1;
    }
    if(strcmp(op, "!=") == 0)
    {
        *result = strcmp(lhs, rhs) != 0 ? 1 : 0;
        return 1;
    }
    return 0;
}


/* ============================================================
 *  cli_eval_test — top-level evaluator
 * ============================================================
 */

/**
 * @brief Evaluate a bash-style [ ... ] test expression
 *
 * Tokenizes @expr and evaluates:
 *   - Logical OR  (-o): evaluated left-to-right
 *   - Regex match (=~): POSIX extended regex
 *   - Logical AND (-a): short-circuit left-to-right
 *   - Unary tests: SHM, file, variable
 *   - Logical NOT (!): prefix negation
 *   - Single value: non-empty → true
 *   - Binary tests: numeric and string comparisons
 *
 * @param expr  Space-separated test expression
 * @return 1 if expression evaluates to true, else 0
 */
int cli_eval_test(const char *expr)
{
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
                    strncat(left, " ", sizeof(left) - strlen(left) - 1);
                }
                strncat(left, tokens[j], sizeof(left) - strlen(left) - 1);
            }
            if(cli_eval_test(left))
            {
                return 1;
            }

            for(int j = i + 1; j < ntok; j++)
            {
                if(j > i + 1)
                {
                    strncat(right, " ", sizeof(right) - strlen(right) - 1);
                }
                strncat(right, tokens[j], sizeof(right) - strlen(right) - 1);
            }
            return cli_eval_test(right);
        }
    }

    /* POSIX regex match: str =~ pattern */
    if(ntok == 3
       && strcmp(tokens[1], "=~") == 0)
    {
        regex_t regex;
        int reti = regcomp(&regex, tokens[2], REG_EXTENDED);
        if(reti)
        {
            return 0;
        }
        reti = regexec(&regex, tokens[0], 0, NULL, 0);
        regfree(&regex);
        return !reti;
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
                    strncat(left, " ", sizeof(left) - strlen(left) - 1);
                }
                strncat(left, tokens[j], sizeof(left) - strlen(left) - 1);
            }
            if(!cli_eval_test(left))
            {
                return 0;
            }

            for(int j = i + 1; j < ntok; j++)
            {
                if(j > i + 1)
                {
                    strncat(right, " ", sizeof(right) - strlen(right) - 1);
                }
                strncat(right, tokens[j], sizeof(right) - strlen(right) - 1);
            }
            return cli_eval_test(right);
        }
    }

    /* Unary tests */
    if(ntok == 2)
    {
        int result;
        if(test_unary_shm(
               tokens[0], tokens[1],
               &result))
        {
            return result;
        }
        if(test_unary_file(
               tokens[0], tokens[1],
               &result))
        {
            return result;
        }
    }

    /* Logical NOT: ! expr */
    if(ntok >= 2
       && strcmp(tokens[0], "!") == 0)
    {
        char subexpr[512];
        subexpr[0] = '\0';
        for(int i = 1; i < ntok; i++)
        {
            if(i > 1)
            {
                strncat(subexpr, " ", sizeof(subexpr) - strlen(subexpr) - 1);
            }
            strncat(subexpr, tokens[i], sizeof(subexpr) - strlen(subexpr) - 1);
        }
        return cli_eval_test(subexpr) ? 0 : 1;
    }

    /* Single value: true if non-empty */
    if(ntok == 1)
    {
        return strlen(tokens[0]) > 0 ? 1 : 0;
    }

    /* Binary: val1 op val2 */
    if(ntok >= 3)
    {
        int result;
        if(test_binary_op(
               tokens[0], tokens[1],
               tokens[2], &result))
        {
            return result;
        }
    }

    printf("Error: invalid test expression\n");
    return 0;
}
