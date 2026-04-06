#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
int cli_intercept_part_0(const char *p);
int cli_intercept_part_1(const char *p);
int cli_intercept_part_2(const char *p);
int cli_intercept_part_3(const char *p);
int cli_intercept_part_4(const char *p);
int cli_intercept_part_5(const char *p);
int cli_intercept_part_6(const char *p);
int cli_intercept_part_7(const char *p);
int cli_intercept_part_8(const char *p);
int cli_intercept_part_9(const char *p);
int cli_intercept_part_10(const char *p);
int cli_intercept_part_11(const char *p);
int cli_intercept_part_12(const char *p);
int cli_intercept_part_13(const char *p);
int cli_intercept_part_14(const char *p);
int cli_intercept_part_15(const char *p);
int cli_intercept_part_16(const char *p);
int cli_intercept_part_17(const char *p);
int cli_intercept_part_18(const char *p);
int cli_intercept_part_19(const char *p);
int cli_intercept_part_20(const char *p);
int cli_intercept_part_21(const char *p);
int cli_intercept_part_22(const char *p);
int cli_intercept_part_23(const char *p);
int cli_intercept_part_24(const char *p);
int cli_intercept_part_25(const char *p);
int cli_intercept_part_26(const char *p);
int cli_intercept_part_27(const char *p);
int cli_intercept_part_28(const char *p);
int cli_intercept_part_29(const char *p);
int cli_intercept_part_30(const char *p);
int cli_intercept_part_31(const char *p);
int cli_intercept_part_32(const char *p);
int cli_intercept_part_33(const char *p);
int cli_intercept_part_34(const char *p);
int cli_intercept_part_35(const char *p);
int cli_intercept_part_36(const char *p);
int cli_intercept_part_37(const char *p);
int cli_intercept_part_38(const char *p);
int cli_intercept_part_39(const char *p);
int cli_intercept_part_40(const char *p);
int cli_intercept_part_41(const char *p);
int cli_intercept_part_42(const char *p);
int cli_intercept_part_43(const char *p);
int cli_intercept_part_44(const char *p);
int cli_intercept_part_45(const char *p);
int cli_intercept_part_46(const char *p);
int cli_intercept_part_47(const char *p);
int cli_intercept_part_48(const char *p);
int cli_intercept_part_49(const char *p);
int cli_intercept_part_50(const char *p);
int cli_intercept_part_51(const char *p);
/**
 * @file CLIcore_script.c
 * @brief CLI scripting engine — variables, FPS access,
 *        arithmetic, flow control, user functions
 *
 * Implements bash-style scripting constructs for the
 * milk CLI:
 * - Variable assignment (VAR=val), expansion ($VAR)
 * - FPS parameter read (@fpsname.param)
 * - FPS parameter write (fpsset)
 * - Arithmetic $(( expr ))
 * - Flow control: if/then/else/fi,
 *   while/do/done, for/do/done
 * - User-defined functions:
 *   function name { body }
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"

#include <sys/mman.h>

/* processinfo functions — linked via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(
    const char *pname, int *fd);
extern errno_t processinfo_procdirname(
    char *procdname);

/* ============================================================
 *  CLI Variable Storage
 * ============================================================
 */
/* ============================================================
 *  Block Accumulator — flow control engine
 * ============================================================
 *
 * Multi-line constructs (if/while/for/function)
 * are accumulated in a block buffer until the
 * closing keyword is seen, then the complete
 * block is evaluated.
 */

CLI_BLOCK cli_block_stack[CLI_BLOCK_MAXDEPTH];
int       cli_block_level = 0;

/* Break/continue/return flags */
// defined in CLIcore_script.h
int cli_break_flag = 0;
int cli_continue_flag = 0;
int cli_return_flag = 0;

/* Forward declaration */
void cli_exec_block_if(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
void cli_exec_block_while(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
void cli_exec_block_until(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
void cli_exec_block_for(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);


/* ---- Helper: strip whitespace ---- */

const char *strip_ws(const char *s)
{
    while(*s == ' ' || *s == '\t')
    {
        s++;
    }
    return s;
}

/**
 * @brief Check if @line starts with @prefix.
 *
 * @return Non-zero if @line begins with @prefix
 */
int starts_with(
    const char *line,
    const char *prefix
)
{
    return strncmp(line, prefix,
                   strlen(prefix)) == 0;
}


/**
 * @brief Search for an executable by name in PATH.
 *
 * Performs the same lookup as the `which` command
 * using C library calls only, avoiding any fork/exec.
 * If the name contains a slash it is tested directly.
 *
 * @param name       Command name to look up
 * @param pathbuf    Buffer to receive the full path
 * @param pathbuf_sz Size of pathbuf in bytes
 * @return 1 if found (pathbuf is filled), 0 otherwise
 */
int cli_find_in_path(
    const char *name,
    char       *pathbuf,
    size_t      pathbuf_sz
)
{
    /* Absolute or relative path — test directly */
    if(strchr(name, '/') != NULL)
    {
        if(access(name, X_OK) == 0)
        {
            strncpy(pathbuf, name,
                    pathbuf_sz - 1);
            pathbuf[pathbuf_sz - 1] = '\0';
            return 1;
        }
        return 0;
    }

    const char *PATH_env = getenv("PATH");
    if(PATH_env == NULL)
    {
        PATH_env =
            "/usr/local/bin:"
            "/usr/bin:/bin";
    }

    char path_copy[4096];
    strncpy(path_copy, PATH_env,
            sizeof(path_copy) - 1);
    path_copy[sizeof(path_copy) - 1] = '\0';

    char *dir = strtok(path_copy, ":");
    while(dir != NULL)
    {
        char candidate[1024];
        snprintf(candidate, sizeof(candidate),
                 "%s/%s", dir, name);
        if(access(candidate, X_OK) == 0)
        {
            strncpy(pathbuf, candidate,
                    pathbuf_sz - 1);
            pathbuf[pathbuf_sz - 1] = '\0';
            return 1;
        }
        dir = strtok(NULL, ":");
    }
    return 0;
}


/* ---- Parse if/then/elif/else/fi block ---- */

/**
 * @brief Evaluate a condition line
 *
 * Handles "if [ cond ]", "elif [ cond ]",
 * or bare "if val" forms. The keyword
 * (if/elif) is skipped before evaluation.
 *
 * @param raw   Raw condition line
 * @param skip  Chars to skip ("if"=2, "elif"=4)
 * @return 1 = true, 0 = false
 */
int eval_cond_line(
    const char *raw,
    int skip
)
{
    char cl[STRINGMAXLEN_CLICMDLINE];
    strncpy(cl, raw,
            STRINGMAXLEN_CLICMDLINE - 1);
    cl[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    cli_expand_fpsvar(
        cl, STRINGMAXLEN_CLICMDLINE);
    cli_expand_env(
        cl, STRINGMAXLEN_CLICMDLINE);
    cli_expand_arith(
        cl, STRINGMAXLEN_CLICMDLINE);

    const char *p = strip_ws(cl);
    p += skip;
    p = strip_ws(p);

    if(*p == '[')
    {
        p++;
        const char *end = strrchr(p, ']');
        if(end != NULL)
        {
            char cs[512];
            int clen = (int)(end - p);
            if(clen >= (int) sizeof(cs))
            {
                clen = (int) sizeof(cs) - 1;
            }
            memcpy(cs, p, (size_t) clen);
            cs[clen] = '\0';
            return cli_eval_test(cs);
        }
        return 0;
    }
    
    char pcopy[STRINGMAXLEN_CLICMDLINE];
    strncpy(pcopy, p, STRINGMAXLEN_CLICMDLINE - 1);
    pcopy[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    char *sc = strchr(pcopy, ';');
    if (sc != NULL) {
        *sc = '\0';
    }
    int len = (int) strlen(pcopy);
    while(len > 0 && (pcopy[len-1] == ' ' || pcopy[len-1] == '\t')) {
        pcopy[--len] = '\0';
    }

    CLI_execute_string(pcopy);
    return (cli_last_retval == 0) ? 1 : 0;
}

/**
 * @brief Execute an if/then/elif/else/fi block
 *
 * Supports cascading elif:
 *   if [ cond1 ]; then
 *       body1
 *   elif [ cond2 ]; then
 *       body2
 *   else
 *       body3
 *   fi
 */
/**
 * @brief Pre-execution interceptor for flow-control
 *        and scripting constructs.
 *
 * Called before every line reaches CLI_execute_line().
 * Detects and handles:
 *  - Heredoc accumulation (<<EOF)
 *  - Comments (#)
 *  - Trap commands (trap 'cmd' SIGNAL)
 *  - set -e / set -x flags
 *  - If/elif/else/fi blocks
 *  - While/until loops
 *  - For loops (word-list and C-style)
 *  - Select menus
 *  - Case/esac blocks
 *  - Function definitions (function name { })
 *  - Logical operators (&& and ||)
 *  - on_update stream triggers
 *  - getopts option parsing
 *  - break/continue with nesting depth
 *  - return from user-defined functions
 *
 * @param line  Raw input line to evaluate
 * @return 1 if the line was consumed (do not
 *         execute further), 0 if not intercepted
 */
int cli_script_intercept(const char *line)
{
    const char *p = strip_ws(line);

    /* ---- Heredoc accumulation state ---- */
    static int  heredoc_active = 0;
    static char heredoc_var[CLI_VAR_NAMELEN];
    static char heredoc_delim[64];
    static char heredoc_buf[16384];
    static int  heredoc_pos = 0;

    if(heredoc_active)
    {
        if(strcmp(p, heredoc_delim) == 0)
        {
            /* End of heredoc — assign */
            heredoc_buf[heredoc_pos] = '\0';
            cli_var_set(heredoc_var,
                        heredoc_buf);
            heredoc_active = 0;
        }
        else
        {
            /* Append line + newline */
            int llen = (int) strlen(p);
            if(heredoc_pos + llen + 1
               < (int) sizeof(heredoc_buf))
            {
                memcpy(
                    heredoc_buf + heredoc_pos,
                    p, (size_t) llen);
                heredoc_pos += llen;
                heredoc_buf[
                    heredoc_pos++] = '\n';
            }
        }
        return 1;
    }

    /* Check if this line starts a heredoc:
     *   VAR=<<DELIM */
    if(strchr(p, '=') != NULL)
    {
        const char *eq = strchr(p, '=');
        if(eq[1] == '<' && eq[2] == '<')
        {
            int nlen = (int)(eq - p);
            if(nlen > 0
               && nlen < CLI_VAR_NAMELEN)
            {
                memcpy(heredoc_var, p,
                       (size_t) nlen);
                heredoc_var[nlen] = '\0';
                const char *d = eq + 3;
                while(*d == ' '
                      || *d == '\t')
                {
                    d++;
                }
                int dlen = (int) strlen(d);
                if(dlen > 0 && dlen < 64)
                {
                    strncpy(heredoc_delim,
                            d, 63);
                    heredoc_delim[63] = '\0';
                    heredoc_active = 1;
                    heredoc_pos = 0;
                    heredoc_buf[0] = '\0';
                    return 1;
                }
            }
        }
    }

    /* If we're already accumulating a block,
     * buffer the line */
    if(cli_block_level > 0)
    {
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level - 1];

        /* Check for nested openers */
        if(starts_with(p, "if ")
           || starts_with(p, "if\t")
           || starts_with(p, "while ")
           || starts_with(p, "while\t")
           || starts_with(p, "until ")
           || starts_with(p, "until\t")
           || starts_with(p, "for ")
           || starts_with(p, "for\t")
           || starts_with(p, "select ")
           || starts_with(p,
                          "select\t")
           || starts_with(p, "function ")
           || starts_with(p, "function\t")
           || starts_with(p, "case ")
           || starts_with(p, "case\t"))
        {
            blk->depth++;
        }

        /* Check for closers.
         * When depth > 0 (nested block),
         * ANY closer keyword decrements
         * the depth. Only at depth 0 does
         * the closer need to match the
         * outer block type. */
        int is_close = 0;
        int is_any_close =
            (strcmp(p, "fi") == 0
             || strcmp(p, "done") == 0
             || strcmp(p, "}") == 0
             || strcmp(p, "esac") == 0);

        if(is_any_close && blk->depth > 0)
        {
            /* Nested closer — decrement
             * depth and buffer */
            blk->depth--;
            if(blk->nlines
               < CLI_BLOCK_MAXLINES)
            {
                strncpy(
                    blk->lines[
                        blk->nlines],
                    p,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                blk->nlines++;
            }
            return 1;
        }

        /* Check outer block closer */
        if(blk->type == CLI_BLOCK_IF
           && strcmp(p, "fi") == 0)
        {
            is_close = 1;
        }
        if((blk->type == CLI_BLOCK_WHILE
            || blk->type == CLI_BLOCK_FOR
            || blk->type == CLI_BLOCK_UNTIL)
           && strcmp(p, "done") == 0)
        {
            is_close = 1;
        }
        if(blk->type == CLI_BLOCK_FUNC
           && strcmp(p, "}") == 0)
        {
            is_close = 1;
        }
        if(blk->type == CLI_BLOCK_CASE
           && strcmp(p, "esac") == 0)
        {
            is_close = 1;
        }

        if(is_close)
        {
            /* Outer block complete — save
             * data locally because the stack
             * slot may be reused by nested
             * blocks during execution.
             * Use malloc to avoid stack
             * overflow on deep nesting. */
            int saved_type = blk->type;
            int saved_nlines = blk->nlines;
            char (*saved_lines)[
                STRINGMAXLEN_CLICMDLINE] =
                malloc(
                    (size_t) saved_nlines
                    * STRINGMAXLEN_CLICMDLINE);
            if(saved_lines == NULL)
            {
                printf("Error: malloc failed "
                       "for block lines\n");
                cli_block_level--;
                return 1;
            }
            for(int si = 0;
                si < saved_nlines; si++)
            {
                strncpy(
                    saved_lines[si],
                    blk->lines[si],
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                saved_lines[si][
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
            }
            cli_block_level--;

            if(saved_type == CLI_BLOCK_IF)
            {
                cli_exec_block_if(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type == CLI_BLOCK_WHILE)
            {
                cli_exec_block_while(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type
                == CLI_BLOCK_UNTIL)
            {
                cli_exec_block_until(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type == CLI_BLOCK_FOR)
            {
                cli_exec_block_for(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type
                ==
                CLI_BLOCK_SELECT)
            {
                cli_exec_block_select(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type == CLI_BLOCK_FUNC)
            {
                /* Define function from
                 * buffered lines */
                const char *fl =
                    strip_ws(
                        saved_lines[0]);
                fl += 8; /* "function" */
                fl = strip_ws(fl);
                char fname[CLI_FUNC_NAMELEN];
                {
                    int fn = 0;
                    while(*fl != '\0'
                          && *fl != ' '
                          && *fl != '\t'
                          && *fl != '{'
                          && fn
                             < CLI_FUNC_NAMELEN
                               - 1)
                    {
                        fname[fn++] = *fl++;
                    }
                    fname[fn] = '\0';
                }
                /* Body starts at line 1
                 * (skip function header) */
                cli_func_define(
                    fname,
                    saved_lines + 1,
                    saved_nlines - 1);
            }
            else if(
                saved_type == CLI_BLOCK_CASE)
            {
                cli_exec_block_case(
                    saved_lines,
                    saved_nlines);
            }

            free(saved_lines);
            return 1;
        }

        /* Buffer normal line */
        if(blk->nlines < CLI_BLOCK_MAXLINES)
        {
            strncpy(
                blk->lines[blk->nlines],
                p,
                STRINGMAXLEN_CLICMDLINE - 1);
            blk->nlines++;
        }
        return 1;
    }

    /* ---- Not in a block: check openers ---- */

    /* break / continue / return */
    if(cli_intercept_part_0(p)) return 1;
    if(cli_intercept_part_1(p)) return 1;
    if(cli_intercept_part_2(p)) return 1;
    if(cli_intercept_part_3(p)) return 1;
    if(cli_intercept_part_4(p)) return 1;
    if(cli_intercept_part_5(p)) return 1;
    if(cli_intercept_part_6(p)) return 1;
    if(cli_intercept_part_7(p)) return 1;
    if(cli_intercept_part_8(p)) return 1;
    if(cli_intercept_part_9(p)) return 1;
    if(cli_intercept_part_10(p)) return 1;
    if(cli_intercept_part_11(p)) return 1;
    if(cli_intercept_part_12(p)) return 1;
    if(cli_intercept_part_13(p)) return 1;
    if(cli_intercept_part_14(p)) return 1;
    if(cli_intercept_part_15(p)) return 1;
    if(cli_intercept_part_16(p)) return 1;
    if(cli_intercept_part_17(p)) return 1;
    if(cli_intercept_part_18(p)) return 1;
    if(cli_intercept_part_19(p)) return 1;
    if(cli_intercept_part_20(p)) return 1;
    if(cli_intercept_part_21(p)) return 1;
    if(cli_intercept_part_22(p)) return 1;
    if(cli_intercept_part_23(p)) return 1;
    if(cli_intercept_part_24(p)) return 1;
    if(cli_intercept_part_25(p)) return 1;
    if(cli_intercept_part_26(p)) return 1;
    if(cli_intercept_part_27(p)) return 1;
    if(cli_intercept_part_28(p)) return 1;
    if(cli_intercept_part_29(p)) return 1;
    if(cli_intercept_part_30(p)) return 1;
    if(cli_intercept_part_31(p)) return 1;
    if(cli_intercept_part_32(p)) return 1;
    if(cli_intercept_part_33(p)) return 1;
    if(cli_intercept_part_34(p)) return 1;
    if(cli_intercept_part_35(p)) return 1;
    if(cli_intercept_part_36(p)) return 1;
    if(cli_intercept_part_37(p)) return 1;
    if(cli_intercept_part_38(p)) return 1;
    if(cli_intercept_part_39(p)) return 1;
    if(cli_intercept_part_40(p)) return 1;
    if(cli_intercept_part_41(p)) return 1;
    if(cli_intercept_part_42(p)) return 1;
    if(cli_intercept_part_43(p)) return 1;
    if(cli_intercept_part_44(p)) return 1;
    if(cli_intercept_part_45(p)) return 1;
    if(cli_intercept_part_46(p)) return 1;
    if(cli_intercept_part_47(p)) return 1;
    if(cli_intercept_part_48(p)) return 1;
    if(cli_intercept_part_49(p)) return 1;
    if(cli_intercept_part_50(p)) return 1;
    if(cli_intercept_part_51(p)) return 1;

    /* Try alias expansion before
     * user-defined function call */
    {
        char firstword[
            CLI_FUNC_NAMELEN];
        int fw = 0;
        const char *pp = p;
        while(*pp != '\0'
              && *pp != ' '
              && *pp != '\t'
              && fw
              < CLI_FUNC_NAMELEN - 1)
        {
            firstword[fw++] = *pp++;
        }
        firstword[fw] = '\0';

        /* Check aliases first */
        for(int k = 0;
            k < data.NBalias;
            k++)
        {
            if(strcmp(
                   data.alias[k]
                   .name,
                   firstword) == 0)
            {
                /* Build expanded
                 * command */
                char expanded[
                    STRINGMAXLEN_CLICMDLINE
                ];
                snprintf(
                    expanded,
                    sizeof(expanded),
                    "%s%s",
                    data.alias[k].cmd,
                    pp);
                CLI_execute_string(
                    expanded);
                return 1;
            }
        }

        /* Then try user function */
        if(cli_func_find(firstword)
           != NULL)
        {
            p = strip_ws(line);
            cli_try_func_call(p);
            return 1;
        }
    }

    return 0;
}
