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
static int cli_find_in_path(
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
    if(strcmp(p, "break") == 0)
    {
        cli_break_flag = 1;
        return 1;
    }
    if(strcmp(p, "continue") == 0)
    {
        cli_continue_flag = 1;
        return 1;
    }
    if(strcmp(p, "return") == 0
       || starts_with(p, "return ")
       || starts_with(p, "return\t"))
    {
        const char *rv = p + 6;
        while(*rv == ' ' || *rv == '\t')
        {
            rv++;
        }
        if(*rv != '\0')
        {
            cli_last_retval =
                (int) strtol(rv, NULL, 0);
        }
        cli_return_flag = 1;
        return 1;
    }

    /* exit [N] — exit CLI entirely */
    if(strcmp(p, "exit") == 0
       || starts_with(p, "exit ")
       || starts_with(p, "exit\t"))
    {
        int exitcode = 0;
        if(strlen(p) > 4)
        {
            const char *ev = p + 4;
            while(*ev == ' ' || *ev == '\t')
            {
                ev++;
            }
            if(*ev != '\0')
            {
                exitcode =
                    (int) strtol(ev,
                                 NULL, 0);
            }
        }
        
        cli_trap_run_exit();
        exit(exitcode);
    }

    /* shift [N] — shift positional params */
    if(strcmp(p, "shift") == 0
       || starts_with(p, "shift ")
       || starts_with(p, "shift\t"))
    {
        int n = 1;
        if(strlen(p) > 5)
        {
            const char *sv = p + 5;
            while(*sv == ' ' || *sv == '\t')
            {
                sv++;
            }
            if(*sv != '\0')
            {
                n = (int) strtol(sv,
                                 NULL, 0);
            }
        }
        if(n < 1)
        {
            n = 1;
        }
        /* Shift $1..$9 by n positions */
        for(int i = 1;
            i < CLI_FUNC_MAXARGS; i++)
        {
            char dst[16], src[16];
            snprintf(dst, sizeof(dst),
                     "%d", i);
            snprintf(src, sizeof(src),
                     "%d", i + n);
            if(i + n < CLI_FUNC_MAXARGS)
            {
                const char *sv2 =
                    cli_var_get(src);
                if(sv2 != NULL)
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

    /* procctl <name> run|pause|step|stop */
    if(starts_with(p, "procctl ")
       || starts_with(p, "procctl\t"))
    {
        const char *ap = p + 7;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int nlen = 0;
        while(*ap && *ap != ' '
              && *ap != '\t'
              && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int ctrlval = -1;
        if(strncmp(ap, "run", 3) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_RUN;
        }
        else if(strncmp(ap, "pause", 5) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_PAUSE;
        }
        else if(strncmp(ap, "step", 4) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_INCR;
        }
        else if(strncmp(ap, "stop", 4) == 0
                || strncmp(ap, "exit", 4)
                   == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_EXIT;
        }
        if(ctrlval < 0)
        {
            printf(
                "procctl: unknown action "
                "'%s' (use run|pause|"
                "step|stop)\n", ap);
            return 1;
        }
        if(pinfolist != NULL)
        {
            pid_t fpid = 0;
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(pinfolist->active[pi]
                   && strcmp(
                       pinfolist
                           ->pnamearray[pi],
                       pname) == 0)
                {
                    fpid = pinfolist
                        ->PIDarray[pi];
                    break;
                }
            }
            if(fpid > 0)
            {
                char pfn[512];
                char pdname[256];
                processinfo_procdirname(
                    pdname);
                snprintf(pfn, sizeof(pfn),
                         "%s/proc.%d.shm",
                         pdname,
                         (int) fpid);
                int pfd = -1;
                PROCESSINFO *pi =
                    processinfo_shm_link(
                        pfn, &pfd);
                if(pi != MAP_FAILED
                   && pi != NULL)
                {
                    pi->CTRLval = ctrlval;
                    munmap(pi,
                        sizeof(PROCESSINFO));
                    close(pfd);
                }
                else if(pfd >= 0)
                {
                    close(pfd);
                }
            }
            else
            {
                printf(
                    "procctl: process "
                    "'%s' not found\n",
                    pname);
            }
        }
        return 1;
    }

    /* procwait <name> <state> [timeout] */
    if(starts_with(p, "procwait ")
       || starts_with(p, "procwait\t"))
    {
        const char *ap = p + 8;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int nlen = 0;
        while(*ap && *ap != ' '
              && *ap != '\t'
              && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int tgt = -1;
        if(strncasecmp(ap, "INIT", 4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_INIT;
        }
        else if(strncasecmp(ap, "ACTIVE",
                6) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ACTIVE;
        }
        else if(strncasecmp(ap, "PAUSE",
                5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_PAUSE;
        }
        else if(strncasecmp(ap, "STOP",
                4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_STOP;
        }
        else if(strncasecmp(ap, "ERROR",
                5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ERROR;
        }
        else
        {
            tgt = (int) strtol(ap, NULL, 0);
        }
        /* Skip state word */
        while(*ap && *ap != ' '
              && *ap != '\t')
        {
            ap++;
        }
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        double timeout = 30.0;
        if(*ap != '\0')
        {
            timeout = strtod(ap, NULL);
        }
        struct timespec slp;
        slp.tv_sec = 0;
        slp.tv_nsec = 100000000; /* 100ms */
        double elapsed = 0.0;
        cli_last_retval = 1;
        while(elapsed < timeout)
        {
            if(pinfolist != NULL)
            {
                for(int pi = 0;
                    pi < PROCESSINFOLISTSIZE;
                    pi++)
                {
                    if(pinfolist->active[pi]
                       && strcmp(
                           pinfolist
                               ->pnamearray[
                                   pi],
                           pname) == 0)
                    {
                        pid_t fpid =
                            pinfolist
                                ->PIDarray[
                                    pi];
                        char pfn[512];
                        char pdname[256];
                        processinfo_procdirname(
                            pdname);
                        snprintf(
                            pfn,
                            sizeof(pfn),
                            "%s/proc."
                            "%d.shm",
                            pdname,
                            (int) fpid);
                        int pfd = -1;
                        PROCESSINFO *pii =
                            processinfo_shm_link(
                                pfn, &pfd);
                        if(pii
                           != MAP_FAILED
                           && pii != NULL)
                        {
                            if(pii
                               ->loopstat
                               == tgt)
                            {
                                cli_last_retval
                                    = 0;
                            }
                            munmap(pii,
                                sizeof(
                                PROCESSINFO));
                            close(pfd);
                        }
                        else if(pfd >= 0)
                        {
                            close(pfd);
                        }
                        break;
                    }
                }
            }
            if(cli_last_retval == 0)
            {
                break;
            }
            nanosleep(&slp, NULL);
            elapsed += 0.1;
        }
        return 1;
    }

    /* procstat [name] */
    if(strcmp(p, "procstat") == 0
       || starts_with(p, "procstat ")
       || starts_with(p, "procstat\t"))
    {
        const char *ap = p + 8;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char filter[256];
        filter[0] = '\0';
        if(*ap != '\0')
        {
            strncpy(filter, ap,
                    sizeof(filter) - 1);
            filter[sizeof(filter) - 1]
                = '\0';
        }
        if(pinfolist != NULL)
        {
            char pdname[256];
            processinfo_procdirname(pdname);
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(!pinfolist->active[pi])
                {
                    continue;
                }
                if(filter[0] != '\0'
                   && strcmp(
                       pinfolist
                           ->pnamearray[pi],
                       filter) != 0)
                {
                    continue;
                }
                pid_t fpid =
                    pinfolist
                        ->PIDarray[pi];
                char pfn[512];
                snprintf(pfn,
                         sizeof(pfn),
                         "%s/proc.%d.shm",
                         pdname,
                         (int) fpid);
                int pfd = -1;
                PROCESSINFO *pii =
                    processinfo_shm_link(
                        pfn, &pfd);
                if(pii == MAP_FAILED
                   || pii == NULL)
                {
                    if(pfd >= 0)
                    {
                        close(pfd);
                    }
                    continue;
                }
                const char *stname =
                    "UNKNOWN";
                switch(pii->loopstat)
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
                if(pii->dtmedian_iter_ns
                   > 0)
                {
                    hz = 1.0e9
                        / (double)
                          pii
                          ->dtmedian_iter_ns;
                }
                double us =
                    (double)
                    pii->dtmedian_exec_ns
                    / 1000.0;
                printf(
                    "name=%s\n"
                    "pid=%d\n"
                    "loopstat=%s\n"
                    "loopcnt=%ld\n"
                    "loopfreq_hz=%.1f\n"
                    "exectime_us=%.1f\n"
                    "rtprio=%d\n"
                    "ctrlval=%d\n"
                    "missedframes=%lu\n"
                    "tmux=%s\n",
                    pii->name,
                    (int) pii->PID,
                    stname,
                    pii->loopcnt,
                    hz, us,
                    pii->RT_priority,
                    pii->CTRLval,
                    (unsigned long)
                    pii
                    ->triggermissedframe_cumul,
                    pii->tmuxname);
                munmap(pii,
                    sizeof(PROCESSINFO));
                close(pfd);
                if(filter[0] != '\0')
                {
                    break;
                }
                printf("---\n");
            }
        }
        return 1;
    }

    /* time <command> — measure duration */
    if(starts_with(p, "time ")
       || starts_with(p, "time\t"))
    {
        const char *cmd = p + 4;
        while(*cmd == ' ' || *cmd == '\t')
        {
            cmd++;
        }
        struct timespec t0, t1;
        clock_gettime(
            CLOCK_MONOTONIC, &t0);
        CLI_execute_string(cmd);
        clock_gettime(
            CLOCK_MONOTONIC, &t1);
        double elapsed =
            (double)(t1.tv_sec - t0.tv_sec)
            + (double)(t1.tv_nsec
                       - t0.tv_nsec)
              / 1.0e9;
        printf(
            "\nreal\t%.3fs\n",
            elapsed);
        return 1;
    }

    /* assert [ cond ] "message" */
    if(starts_with(p, "assert ")
       || starts_with(p, "assert\t"))
    {
        const char *ap = p + 6;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        if(*ap == '[')
        {
            ap++;
            const char *end =
                strrchr(ap, ']');
            if(end != NULL)
            {
                char cs[512];
                int clen =
                    (int)(end - ap);
                if(clen
                   >= (int) sizeof(cs))
                {
                    clen =
                        (int) sizeof(cs)
                        - 1;
                }
                memcpy(cs, ap,
                       (size_t) clen);
                cs[clen] = '\0';
                int result =
                    cli_eval_test(cs);
                if(!result)
                {
                    const char *msg =
                        end + 1;
                    while(*msg == ' '
                          || *msg == '\t')
                    {
                        msg++;
                    }
                    /* strip quotes */
                    if(*msg == '"'
                       || *msg == '\'')
                    {
                        msg++;
                    }
                    int mlen =
                        (int) strlen(msg);
                    if(mlen > 0
                       && (msg[mlen - 1]
                           == '"'
                           || msg[mlen - 1]
                              == '\''))
                    {
                        char mb[512];
                        strncpy(
                            mb, msg,
                            sizeof(mb) - 1);
                        mb[sizeof(mb) - 1]
                            = '\0';
                        if(mlen - 1
                           < (int)
                             sizeof(mb))
                        {
                            mb[mlen - 1]
                                = '\0';
                        }
                        printf(
                            "ASSERT "
                            "FAILED: "
                            "%s\n", mb);
                    }
                    else
                    {
                        printf(
                            "ASSERT "
                            "FAILED: "
                            "%s\n", msg);
                    }
                    cli_last_retval = 1;
                    if(cli_flag_errexit)
                    {
                        cli_trap_run(-1);
                    }
                }
            }
        }
        return 1;
    }

    /* watch -n <sec> <command> */
    if(starts_with(p, "watch ")
       || starts_with(p, "watch\t"))
    {
        const char *wp = p + 5;
        while(*wp == ' ' || *wp == '\t')
        {
            wp++;
        }
        if(*wp == '-' && *(wp + 1) == 'n')
        {
            double interval = 2.0;
            wp += 2;
            while(*wp == ' '
                  || *wp == '\t')
            {
                wp++;
            }
            interval = strtod(wp, NULL);
            while(*wp != ' '
                  && *wp != '\t'
                  && *wp != '\0')
            {
                wp++;
            }
            while(*wp == ' '
                  || *wp == '\t')
            {
                wp++;
            }
            struct timespec ts;
            ts.tv_sec =
                (time_t) interval;
            ts.tv_nsec =
                (long)((interval
                        - (double) ts.tv_sec)
                       * 1.0e9);
            while(!cli_break_flag)
            {
                printf(
                    "\033[2J\033[H"
                    "Every %.1fs: %s\n\n",
                    interval, wp);
                CLI_execute_string(wp);
                nanosleep(&ts, NULL);
            }
            cli_break_flag = 0;
            return 1;
        }
    }

    /* trap 'cmd' SIGNAL [SIGNAL...]
     * trap -l
     * Engine events: STREAM:name FPS:f.p=v
     *                PROC:name:STATE */
    if(starts_with(p, "trap ")
       || starts_with(p, "trap\t"))
    {
        p += 4;
        p = strip_ws(p);

        /* trap -l — list active traps */
        if(strncmp(p, "-l", 2) == 0
           && (p[2] == '\0'
               || p[2] == ' '
               || p[2] == '\t'))
        {
            printf("POSIX traps:\n");
            for(int i = 0;
                i < CLI_TRAP_MAXSIGS; i++)
            {
                if(cli_traps[i].used)
                {
                    printf("  sig=%d "
                           "cmd='%s'\n",
                           cli_traps[i]
                               .signum,
                           cli_traps[i].cmd);
                }
            }
            printf("Engine traps:\n");
            for(int i = 0;
                i < CLI_ENGINE_TRAP_MAX;
                i++)
            {
                CLI_ENGINE_TRAP *et =
                    &cli_engine_traps[i];
                if(!et->used)
                {
                    continue;
                }
                const char *tstr = "?";
                if(et->type
                   == CLI_ETRAP_STREAM)
                {
                    tstr = "STREAM";
                }
                else if(et->type
                        == CLI_ETRAP_FPS)
                {
                    tstr = "FPS";
                }
                else if(et->type
                        == CLI_ETRAP_PROC)
                {
                    tstr = "PROC";
                }
                printf("  %s:%s",
                       tstr, et->target);
                if(et->type
                   == CLI_ETRAP_FPS)
                {
                    printf(".%s",
                           et->param);
                }
                printf(
                    " ival=%ldms"
                    " n=%d/%d"
                    " cmd='%s'\n",
                    et->min_interval_ms,
                    et->fire_count,
                    et->max_fires,
                    et->cmd);
            }
            return 1;
        }

        /* Parse optional flags before
         * the quoted command */
        long opt_interval_ms =
            CLI_ETRAP_DEFAULT_MS;
        int  opt_max_fires = -1;

        while(*p == '-')
        {
            if(strncmp(p, "-n", 2) == 0)
            {
                p += 2;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                char *endp = NULL;
                long nv =
                    strtol(p, &endp, 10);
                if(endp != p && nv > 0)
                {
                    opt_max_fires =
                        (int) nv;
                    p = endp;
                }
            }
            else if(strncmp(p, "-i", 2)
                    == 0)
            {
                p += 2;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                char *endp = NULL;
                long iv =
                    strtol(p, &endp, 10);
                if(endp != p && iv >= 0)
                {
                    opt_interval_ms = iv;
                    p = endp;
                }
            }
            else
            {
                break;
            }
            while(*p == ' '
                  || *p == '\t')
            {
                p++;
            }
        }

        /* Extract quoted command */
        char tcmd[CLI_TRAP_CMDLEN];
        tcmd[0] = '\0';
        if(*p == '\'' || *p == '"')
        {
            char q = *p++;
            int ti = 0;
            while(*p != '\0' && *p != q
                  && ti
                  < CLI_TRAP_CMDLEN - 1)
            {
                tcmd[ti++] = *p++;
            }
            tcmd[ti] = '\0';
            if(*p == q)
            {
                p++;
            }
        }
        p = strip_ws(p);

        /* Parse signal / event names */
        while(*p != '\0')
        {
            char sname[128];
            int si = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && si < 127)
            {
                sname[si++] = *p++;
            }
            sname[si] = '\0';
            p = strip_ws(p);
            if(si == 0)
            {
                break;
            }

            /* Check for engine event
             * prefix */
            if(strncmp(sname, "STREAM:",
                       7) == 0)
            {
                const char *nm =
                    sname + 7;
                /* Find or alloc slot */
                int slot = -1;
                for(int i = 0;
                    i < CLI_ENGINE_TRAP_MAX;
                    i++)
                {
                    if(cli_engine_traps[i]
                           .used
                       && cli_engine_traps[i]
                              .type
                       == CLI_ETRAP_STREAM
                       && strcmp(
                              cli_engine_traps
                                  [i].target,
                              nm)
                       == 0)
                    {
                        slot = i;
                        break;
                    }
                }
                if(slot < 0)
                {
                    for(int i = 0;
                        i
                        < CLI_ENGINE_TRAP_MAX;
                        i++)
                    {
                        if(!cli_engine_traps
                                [i].used)
                        {
                            slot = i;
                            break;
                        }
                    }
                }
                if(slot >= 0)
                {
                    CLI_ENGINE_TRAP *et =
                        &cli_engine_traps
                             [slot];
                    if(tcmd[0] == '\0')
                    {
                        /* Clear trap */
                        et->used = 0;
                        et->connected = 0;
                    }
                    else
                    {
                        memset(et, 0,
                            sizeof(*et));
                        et->type =
                            CLI_ETRAP_STREAM;
                        strncpy(
                            et->target, nm,
                            sizeof(
                                et->target)
                            - 1);
                        strncpy(
                            et->cmd, tcmd,
                            CLI_TRAP_CMDLEN
                            - 1);
                        et->min_interval_ms =
                            opt_interval_ms;
                        et->max_fires =
                            opt_max_fires;
                        et->used = 1;
                    }
                }
                continue;
            }

            if(strncmp(sname, "FPS:",
                       4) == 0)
            {
                const char *fp =
                    sname + 4;
                /* Split fpsname.param
                 * and optional op+val */
                char fpsn[128];
                char parn[64];
                int eop = CLI_ETRAP_OP_EQ;
                double eval = 0.0;
                int has_cmp = 0;
                {
                    char tmp[128];
                    strncpy(tmp, fp,
                            sizeof(tmp) - 1);
                    tmp[sizeof(tmp) - 1] =
                        '\0';

                    /* Find operator */
                    char *opp = NULL;
                    char *p_ne =
                        strstr(tmp, "!=");
                    char *p_ge =
                        strstr(tmp, ">=");
                    char *p_le =
                        strstr(tmp, "<=");
                    char *p_eq =
                        strchr(tmp, '=');

                    if(p_ne)
                    {
                        opp = p_ne;
                        eop = CLI_ETRAP_OP_NE;
                        *opp = '\0';
                        eval = strtod(
                            opp + 2, NULL);
                        has_cmp = 1;
                    }
                    else if(p_ge)
                    {
                        opp = p_ge;
                        eop = CLI_ETRAP_OP_GE;
                        *opp = '\0';
                        eval = strtod(
                            opp + 2, NULL);
                        has_cmp = 1;
                    }
                    else if(p_le)
                    {
                        opp = p_le;
                        eop = CLI_ETRAP_OP_LE;
                        *opp = '\0';
                        eval = strtod(
                            opp + 2, NULL);
                        has_cmp = 1;
                    }
                    else if(p_eq)
                    {
                        opp = p_eq;
                        eop = CLI_ETRAP_OP_EQ;
                        *opp = '\0';
                        eval = strtod(
                            opp + 1, NULL);
                        has_cmp = 1;
                    }

                    /* Split at dot */
                    char *dot =
                        strchr(tmp, '.');
                    if(dot)
                    {
                        *dot = '\0';
                        strncpy(fpsn, tmp,
                            sizeof(fpsn) - 1);
                        fpsn[sizeof(fpsn)
                             - 1] = '\0';
                        strncpy(parn,
                            dot + 1,
                            sizeof(parn) - 1);
                        parn[sizeof(parn)
                             - 1] = '\0';
                    }
                    else
                    {
                        strncpy(fpsn, tmp,
                            sizeof(fpsn) - 1);
                        fpsn[sizeof(fpsn)
                             - 1] = '\0';
                        parn[0] = '\0';
                    }
                }

                int slot = -1;
                for(int i = 0;
                    i < CLI_ENGINE_TRAP_MAX;
                    i++)
                {
                    if(!cli_engine_traps
                            [i].used)
                    {
                        slot = i;
                        break;
                    }
                }
                if(slot >= 0)
                {
                    CLI_ENGINE_TRAP *et =
                        &cli_engine_traps
                             [slot];
                    if(tcmd[0] == '\0')
                    {
                        et->used = 0;
                        et->connected = 0;
                    }
                    else
                    {
                        memset(et, 0,
                            sizeof(*et));
                        et->type =
                            CLI_ETRAP_FPS;
                        strncpy(
                            et->target, fpsn,
                            sizeof(
                                et->target)
                            - 1);
                        strncpy(
                            et->param, parn,
                            sizeof(et->param)
                            - 1);
                        et->op = eop;
                        et->cmpval = eval;
                        strncpy(
                            et->cmd, tcmd,
                            CLI_TRAP_CMDLEN
                            - 1);
                        et->min_interval_ms =
                            opt_interval_ms;
                        et->max_fires =
                            opt_max_fires;
                        et->used = 1;
                        (void) has_cmp;
                    }
                }
                continue;
            }

            if(strncmp(sname, "PROC:",
                       5) == 0)
            {
                const char *pp =
                    sname + 5;
                char pname[128];
                int pstate = 0;
                {
                    char *col =
                        strchr(pp, ':');
                    if(col)
                    {
                        size_t len =
                            (size_t)(col
                                     - pp);
                        if(len
                           >= sizeof(pname))
                        {
                            len = sizeof(
                                      pname)
                                  - 1;
                        }
                        strncpy(pname, pp,
                                len);
                        pname[len] = '\0';
                        const char *ss =
                            col + 1;
                        if(strcasecmp(
                               ss, "ACTIVE")
                           == 0)
                        {
                            pstate =
                                PROCESSINFO_LOOPSTAT_ACTIVE;
                        }
                        else if(strcasecmp(
                                    ss,
                                    "STOP")
                                == 0)
                        {
                            pstate =
                                PROCESSINFO_LOOPSTAT_STOP;
                        }
                        else if(strcasecmp(
                                    ss,
                                    "PAUSE")
                                == 0)
                        {
                            pstate =
                                PROCESSINFO_LOOPSTAT_PAUSE;
                        }
                        else if(strcasecmp(
                                    ss,
                                    "CRASHED")
                                == 0)
                        {
                            pstate =
                                PROCESSINFO_LOOPSTAT_CRASHED;
                        }
                        else if(strcasecmp(
                                    ss,
                                    "ERROR")
                                == 0)
                        {
                            pstate =
                                PROCESSINFO_LOOPSTAT_ERROR;
                        }
                    }
                    else
                    {
                        strncpy(pname, pp,
                            sizeof(pname)
                            - 1);
                        pname[sizeof(pname)
                              - 1] = '\0';
                    }
                }

                int slot = -1;
                for(int i = 0;
                    i < CLI_ENGINE_TRAP_MAX;
                    i++)
                {
                    if(!cli_engine_traps
                            [i].used)
                    {
                        slot = i;
                        break;
                    }
                }
                if(slot >= 0)
                {
                    CLI_ENGINE_TRAP *et =
                        &cli_engine_traps
                             [slot];
                    if(tcmd[0] == '\0')
                    {
                        et->used = 0;
                        et->connected = 0;
                    }
                    else
                    {
                        memset(et, 0,
                            sizeof(*et));
                        et->type =
                            CLI_ETRAP_PROC;
                        strncpy(
                            et->target,
                            pname,
                            sizeof(
                                et->target)
                            - 1);
                        et->proc_state =
                            pstate;
                        strncpy(
                            et->cmd, tcmd,
                            CLI_TRAP_CMDLEN
                            - 1);
                        et->min_interval_ms =
                            opt_interval_ms;
                        et->max_fires =
                            opt_max_fires;
                        et->used = 1;
                    }
                }
                continue;
            }

            /* POSIX signal name */
            int sn =
                cli_trap_signum(sname);
            int slot = -1;
            for(int i = 0;
                i < CLI_TRAP_MAXSIGS;
                i++)
            {
                if(cli_traps[i].used
                   && cli_traps[i].signum
                   == sn)
                {
                    slot = i;
                    break;
                }
            }
            if(slot < 0)
            {
                for(int i = 0;
                    i < CLI_TRAP_MAXSIGS;
                    i++)
                {
                    if(!cli_traps[i]
                        .used)
                    {
                        slot = i;
                        break;
                    }
                }
            }
            if(slot >= 0)
            {
                cli_traps[slot].signum =
                    sn;
                strncpy(
                    cli_traps[slot].cmd,
                    tcmd,
                    CLI_TRAP_CMDLEN - 1);
                cli_traps[slot].used = 1;
            }
        }
        return 1;
    }

    /* set -e / set -x / set +e / set +x */
    if(starts_with(p, "set ")
       || starts_with(p, "set\t"))
    {
        p += 3;
        p = strip_ws(p);
        while(*p != '\0')
        {
            if(*p == '-' || *p == '+')
            {
                int on = (*p == '-');
                p++;
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t')
                {
                    if(*p == 'e')
                    {
                        cli_flag_errexit =
                            on;
                    }
                    else if(*p == 'x')
                    {
                        cli_flag_xtrace =
                            on;
                    }
                    p++;
                }
            }
            else
            {
                p++;
            }
            p = strip_ws(p);
        }
        return 1;
    }

    /* export VAR=val — set env variable */
    if(starts_with(p, "export ")
       || starts_with(p, "export\t"))
    {
        p += 6;
        p = strip_ws(p);
        const char *eq = strchr(p, '=');
        if(eq != NULL)
        {
            char ename[CLI_VAR_NAMELEN];
            int elen = (int)(eq - p);
            if(elen >= CLI_VAR_NAMELEN)
            {
                elen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(ename, p,
                   (size_t) elen);
            ename[elen] = '\0';
            const char *eval = eq + 1;
            /* Strip quotes */
            int evlen =
                (int) strlen(eval);
            if(evlen >= 2
               && ((eval[0] == '"'
                    && eval[evlen - 1]
                    == '"')
                   || (eval[0] == '\''
                       && eval[
                           evlen - 1]
                       == '\'')))
            {
                char ebuf[
                    CLI_VAR_VALLEN];
                memcpy(ebuf,
                       eval + 1,
                       (size_t)
                       (evlen - 2));
                ebuf[evlen - 2] = '\0';
                setenv(ename,
                       ebuf, 1);
                cli_var_set(ename,
                            ebuf);
            }
            else
            {
                setenv(ename,
                       eval, 1);
                cli_var_set(ename,
                            eval);
            }
        }
        else
        {
            /* export VAR (no =val):
             * push current value */
            const char *eval =
                cli_var_get(p);
            if(eval != NULL)
            {
                setenv(p, eval, 1);
            }
        }
        return 1;
    }

    /* source file  or  . file */
    if(starts_with(p, "source ")
       || starts_with(p, "source\t")
       || (p[0] == '.'
           && (p[1] == ' '
               || p[1] == '\t')))
    {
        const char *fn = p;
        if(p[0] == '.')
        {
            fn = p + 1;
        }
        else
        {
            fn = p + 6;
        }
        fn = strip_ws(fn);
        FILE *sf = fopen(fn, "r");
        if(sf == NULL)
        {
            fprintf(stderr,
                    "source: %s: "
                    "No such file\n",
                    fn);
        }
        else
        {
            char sline[
                STRINGMAXLEN_CLICMDLINE];
            while(fgets(
                      sline,
                      (int) sizeof(
                          sline),
                      sf) != NULL)
            {
                /* Strip newline */
                int sl =
                    (int) strlen(sline);
                if(sl > 0
                   && sline[sl - 1]
                   == '\n')
                {
                    sline[sl - 1] =
                        '\0';
                }
                CLI_execute_string(sline);
            }
            fclose(sf);
        }
        return 1;
    }

    /* readonly VAR=val */
    if(starts_with(p, "readonly ")
       || starts_with(p,
                      "readonly\t"))
    {
        p += 8;
        p = strip_ws(p);
        const char *eq =
            strchr(p, '=');
        if(eq != NULL)
        {
            char rn[CLI_VAR_NAMELEN];
            int rl = (int)(eq - p);
            if(rl >= CLI_VAR_NAMELEN)
            {
                rl =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(rn, p,
                   (size_t) rl);
            rn[rl] = '\0';
            cli_var_set(rn, eq + 1);
        }
        /* Mark as readonly via env */
        return 1;
    }

    /* break [N] */
    if(starts_with(p, "break")
       && (p[5] == '\0'
           || p[5] == ' '
           || p[5] == '\t'))
    {
        /* Set break level */
        int n = 1;
        if(p[5] != '\0')
        {
            n = (int) strtol(
                p + 5, NULL, 10);
            if(n < 1)
            {
                n = 1;
            }
        }
        cli_last_retval = n;
        return 1;
    }

    /* continue [N] */
    if(starts_with(p, "continue")
       && (p[8] == '\0'
           || p[8] == ' '
           || p[8] == '\t'))
    {
        int n = 1;
        if(p[8] != '\0')
        {
            n = (int) strtol(
                p + 8, NULL, 10);
            if(n < 1)
            {
                n = 1;
            }
        }
        cli_last_retval = n;
        return 1;
    }

    /* printf "fmt" args... */
    if(starts_with(p, "printf ")
       || starts_with(p, "printf\t"))
    {
        p += 6;
        p = strip_ws(p);
        /* Parse format string */
        char fmt[
            STRINGMAXLEN_CLICMDLINE];
        int fi = 0;
        char delim = ' ';
        if(*p == '"' || *p == '\'')
        {
            delim = *p;
            p++;
        }
        while(*p != '\0'
              && *p != delim
              && fi
              < STRINGMAXLEN_CLICMDLINE
              - 1)
        {
            if(*p == '\\'
               && p[1] != '\0')
            {
                switch(p[1])
                {
                case 'n':
                    fmt[fi++] = '\n';
                    break;
                case 't':
                    fmt[fi++] = '\t';
                    break;
                case '\\':
                    fmt[fi++] = '\\';
                    break;
                default:
                    fmt[fi++] = p[1];
                    break;
                }
                p += 2;
            }
            else
            {
                fmt[fi++] = *p++;
            }
        }
        fmt[fi] = '\0';
        if(*p == delim)
        {
            p++;
        }
        /* Collect remaining args */
        char args[32][256];
        int nargs = 0;
        p = strip_ws(p);
        while(*p != '\0'
              && nargs < 32)
        {
            int ai = 0;
            if(*p == '"'
               || *p == '\'')
            {
                char qc = *p++;
                while(*p != '\0'
                      && *p != qc
                      && ai < 255)
                {
                    args[nargs][ai++] =
                        *p++;
                }
                if(*p == qc)
                {
                    p++;
                }
            }
            else
            {
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t'
                      && ai < 255)
                {
                    args[nargs][ai++] =
                        *p++;
                }
            }
            args[nargs][ai] = '\0';
            nargs++;
            p = strip_ws(p);
        }
        /* Simple printf: scan fmt for %s/%d */
        int ai = 0;
        const char *f = fmt;
        while(*f != '\0')
        {
            if(*f == '%'
               && f[1] != '\0')
            {
                if(f[1] == 's')
                {
                    if(ai < nargs)
                    {
                        printf("%s",
                               args[
                                   ai++]);
                    }
                    f += 2;
                }
                else if(f[1] == 'd')
                {
                    if(ai < nargs)
                    {
                        printf(
                            "%d",
                            (int) strtol(
                                args[
                                    ai++],
                                NULL,
                                10));
                    }
                    f += 2;
                }
                else if(f[1] == 'f')
                {
                    if(ai < nargs)
                    {
                        printf(
                            "%f",
                            strtod(
                                args[
                                    ai++],
                                NULL));
                    }
                    f += 2;
                }
                else if(f[1] == '%')
                {
                    putchar('%');
                    f += 2;
                }
                else
                {
                    putchar(*f);
                    f++;
                }
            }
            else
            {
                putchar(*f);
                f++;
            }
        }
        fflush(stdout);
        return 1;
    }

    /* getopts optstring var */
    if(starts_with(p, "getopts ")
       || starts_with(p,
                      "getopts\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* Parse optstring */
        char optstr[128];
        {
            int oi = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && oi < 127)
            {
                optstr[oi++] = *p++;
            }
            optstr[oi] = '\0';
        }
        p = strip_ws(p);
        /* Parse varname */
        char gvar[CLI_VAR_NAMELEN];
        {
            int gi = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && gi
                  < CLI_VAR_NAMELEN - 1)
            {
                gvar[gi++] = *p++;
            }
            gvar[gi] = '\0';
        }
        /* Get OPTIND */
        const char *oidx =
            cli_var_get("OPTIND");
        int optind_val =
            oidx ? (int) strtol(
                       oidx, NULL, 10)
            : 1;
        /* Get current positional arg */
        char pname[32];
        snprintf(pname, sizeof(pname),
                 "%d", optind_val);
        const char *arg =
            cli_var_get(pname);
        if(arg == NULL
           || arg[0] != '-'
           || arg[1] == '\0')
        {
            cli_var_set(gvar, "?");
            cli_last_retval = 1;
            return 1;
        }
        char optch = arg[1];
        /* Check if valid */
        const char *found =
            strchr(optstr, optch);
        if(found == NULL)
        {
            cli_var_set(gvar, "?");
        }
        else
        {
            char ov[2];
            ov[0] = optch;
            ov[1] = '\0';
            cli_var_set(gvar, ov);
            if(found[1] == ':')
            {
                /* Next arg is OPTARG */
                optind_val++;
                char pn2[32];
                snprintf(
                    pn2, sizeof(pn2),
                    "%d", optind_val);
                const char *oa =
                    cli_var_get(pn2);
                if(oa != NULL)
                {
                    cli_var_set(
                        "OPTARG", oa);
                }
            }
        }
        optind_val++;
        {
            char oib[32];
            snprintf(oib,
                     sizeof(oib),
                     "%d",
                     optind_val);
            cli_var_set("OPTIND", oib);
        }
        cli_last_retval = 0;
        return 1;
    }

    /* local VAR=val — set variable in
     * current scope (true shadowing) */
    if(starts_with(p, "local ")
       || starts_with(p, "local\t"))
    {
        p += 5;
        p = strip_ws(p);
        
        char vn[CLI_VAR_NAMELEN];
        const char *eq = strchr(p, '=');
        if(eq != NULL)
        {
            int nl = (int)(eq - p);
            if(nl >= CLI_VAR_NAMELEN) nl = CLI_VAR_NAMELEN - 1;
            memcpy(vn, p, (size_t) nl);
            vn[nl] = '\0';
        }
        else
        {
            strncpy(vn, p, CLI_VAR_NAMELEN - 1);
            vn[CLI_VAR_NAMELEN - 1] = '\0';
        }
        
        /* Save shadow if in function scope and not already shadowed */
        if(cli_local_depth > 0)
        {
            int scount = cli_local_shadow_count[cli_local_depth];
            int already_shadowed = 0;
            for(int i = 0; i < scount; i++)
            {
                if(strcmp(cli_local_shadows[cli_local_depth][i].name, vn) == 0)
                {
                    already_shadowed = 1;
                    break;
                }
            }
            if(!already_shadowed && scount < CLI_MAX_LOCALS_PER_FUNC)
            {
                CLI_LOCAL_SHADOW *sh = &cli_local_shadows[cli_local_depth][scount];
                strncpy(sh->name, vn, CLI_VAR_NAMELEN - 1);
                sh->name[CLI_VAR_NAMELEN - 1] = '\0';
                const char *ov = cli_var_get(vn);
                sh->was_used = (ov != NULL) ? 1 : 0;
                if(ov != NULL)
                {
                    strncpy(sh->val, ov, CLI_VAR_VALLEN - 1);
                    sh->val[CLI_VAR_VALLEN - 1] = '\0';
                }
                cli_local_shadow_count[cli_local_depth]++;
            }
        }

        if(eq != NULL)
        {
            cli_var_set(vn, eq + 1);
        }
        else
        {
            if(cli_var_get(vn) == NULL)
            {
                cli_var_set(vn, "");
            }
        }
        return 1;
    }

    /* declare [-i|-a|-r|-x] VAR[=val] */
    if(starts_with(p, "declare ")
       || starts_with(p,
                      "declare\t")
       || starts_with(p, "typeset ")
       || starts_with(p,
                      "typeset\t"))
    {
        p += 7;
        if(p[0] == ' ' || p[0] == '\t')
        {
            p++;
        }
        p = strip_ws(p);
        /* Parse flags */
        int fl_int = 0;
        int fl_arr = 0;
        int fl_ro = 0;
        int fl_exp = 0;
        while(p[0] == '-')
        {
            p++;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t')
            {
                if(*p == 'i')
                {
                    fl_int = 1;
                }
                else if(*p == 'a')
                {
                    fl_arr = 1;
                }
                else if(*p == 'r')
                {
                    fl_ro = 1;
                }
                else if(*p == 'x')
                {
                    fl_exp = 1;
                }
                p++;
            }
            p = strip_ws(p);
        }
        /* Parse VAR=val */
        const char *eq =
            strchr(p, '=');
        char vn[CLI_VAR_NAMELEN];
        if(eq != NULL)
        {
            int nl = (int)(eq - p);
            if(nl >= CLI_VAR_NAMELEN)
            {
                nl =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, p,
                   (size_t) nl);
            vn[nl] = '\0';
            if(fl_arr)
            {
                /* declare -a arr */
                for(int k = 0;
                    k < CLI_MAX_ARRAYS;
                    k++)
                {
                    if(!cli_arrays[k]
                        .used)
                    {
                        cli_arrays[k]
                            .used = 1;
                        strncpy(
                            cli_arrays[k]
                            .name,
                            vn,
                            CLI_VAR_NAMELEN
                            - 1);
                        cli_arrays[k]
                            .nelem = 0;
                        break;
                    }
                }
            }
            else if(fl_int)
            {
                /* Integer eval */
                long iv = strtol(
                    eq + 1, NULL, 0);
                char ib[32];
                snprintf(ib,
                         sizeof(ib),
                         "%ld", iv);
                cli_var_set(vn, ib);
            }
            else
            {
                cli_var_set(
                    vn, eq + 1);
            }
            if(fl_exp)
            {
                const char *v =
                    cli_var_get(vn);
                if(v != NULL)
                {
                    setenv(vn, v, 1);
                }
            }
        }
        else
        {
            strncpy(vn, p,
                    CLI_VAR_NAMELEN
                    - 1);
            vn[CLI_VAR_NAMELEN - 1] =
                '\0';
            if(cli_var_get(vn) == NULL)
            {
                cli_var_set(vn, "");
            }
        }
        (void) fl_ro; /* TODO: track */
        return 1;
    }

    /* let "expr" or let expr */
    if(starts_with(p, "let ")
       || starts_with(p, "let\t"))
    {
        p += 3;
        p = strip_ws(p);
        /* Strip optional quotes */
        char lexpr[
            STRINGMAXLEN_CLICMDLINE];
        strncpy(lexpr, p,
                STRINGMAXLEN_CLICMDLINE
                - 1);
        lexpr[STRINGMAXLEN_CLICMDLINE
              - 1] = '\0';
        int ll = (int) strlen(lexpr);
        if(ll >= 2
           && ((lexpr[0] == '"'
                && lexpr[ll - 1]
                == '"')
               || (lexpr[0] == '\''
                   && lexpr[ll - 1]
                   == '\'')))
        {
            lexpr[ll - 1] = '\0';
            memmove(lexpr,
                    lexpr + 1,
                    (size_t)(ll - 1));
        }
        /* Build $(( )) expression */
        char ecmd[
            STRINGMAXLEN_CLICMDLINE + 64];
        snprintf(ecmd, sizeof(ecmd),
                 "$((%s))", lexpr);
        /* Find assignment target */
        char *aeq =
            strchr(lexpr, '=');
        if(aeq != NULL
           && aeq != lexpr
           && aeq[-1] != '!'
           && aeq[-1] != '<'
           && aeq[-1] != '>')
        {
            /* Has assignment, e.g.
             * let "x = 1 + 2" */
            *aeq = '\0';
            /* Trim target var */
            char tvar[
                CLI_VAR_NAMELEN];
            {
                const char *ts =
                    lexpr;
                while(*ts == ' '
                      || *ts == '\t')
                {
                    ts++;
                }
                int ti = 0;
                while(*ts != '\0'
                      && *ts != ' '
                      && *ts != '\t'
                      && ti
                      < CLI_VAR_NAMELEN
                      - 1)
                {
                    tvar[ti++] =
                        *ts++;
                }
                tvar[ti] = '\0';
            }
            /* Eval RHS */
            const char *rhs =
                aeq + 1;
            while(*rhs == ' '
                  || *rhs == '\t')
            {
                rhs++;
            }
            char arith[
                STRINGMAXLEN_CLICMDLINE
            ];
            snprintf(arith,
                     sizeof(arith),
                     "$((%s))", rhs);
            cli_expand_env(
                arith,
                STRINGMAXLEN_CLICMDLINE
            );
            cli_var_set(
                tvar, arith);
        }
        else
        {
            /* No assignment, just
             * evaluate */
            cli_expand_env(
                ecmd,
                STRINGMAXLEN_CLICMDLINE
            );
            cli_last_retval =
                (strtol(ecmd, NULL,
                        10) == 0) ? 1
                : 0;
        }
        return 1;
    }

    /* eval "cmd" — execute string */
    if(starts_with(p, "eval ")
       || starts_with(p, "eval\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Strip outer quotes */
        char ecmd[
            STRINGMAXLEN_CLICMDLINE];
        strncpy(ecmd, p,
                STRINGMAXLEN_CLICMDLINE
                - 1);
        ecmd[STRINGMAXLEN_CLICMDLINE
             - 1] = '\0';
        int el = (int) strlen(ecmd);
        if(el >= 2
           && ((ecmd[0] == '"'
                && ecmd[el - 1]
                == '"')
               || (ecmd[0] == '\''
                   && ecmd[el - 1]
                   == '\'')))
        {
            ecmd[el - 1] = '\0';
            memmove(ecmd, ecmd + 1,
                    (size_t)(el - 1));
        }
        CLI_execute_string(ecmd);
        return 1;
    }

    /* type / command -v — check cmd */
    if(starts_with(p, "type ")
       || starts_with(p, "type\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Search registered aliases */
        int found = 0;
        for(int i = 0; i < data.NBalias; i++)
        {
            if(strcmp(data.alias[i].name, p) == 0)
            {
                printf("%s is aliased to `%s`\n", p, data.alias[i].cmd);
                found = 1;
                break;
            }
        }

        /* Search user functions */
        if(!found)
        {
            CLI_FUNC *f = cli_func_find(p);
            if(f != NULL)
            {
                printf("%s is a function\n", p);
                found = 1;
            }
        }

        /* Search registered CLI commands */
        if(!found)
        {
            for(int ci = 0; ci < data.NBcmd; ci++)
            {
                if(strcmp(data.cmd[ci].key, p) == 0)
                {
                    printf("%s is a CLI command\n", p);
                    found = 1;
                    break;
                }
            }
        }

        /* Search external executables in PATH */
        if(!found)
        {
            char path_found[1024];
            if(cli_find_in_path(
                   p,
                   path_found,
                   sizeof(path_found)))
            {
                printf("%s\n", path_found);
                found = 1;
            }
        }

        if(!found)
        {
            printf("milk: type: %s: not found\n", p);
            cli_last_retval = 1;
        }
        else
        {
            cli_last_retval = 0;
        }
        return 1;
    }
    if(starts_with(p, "command ")
       || starts_with(p,
                      "command\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* command -v cmd */
        if(starts_with(p, "-v "))
        {
            p += 3;
            p = strip_ws(p);
            int found = 0;
            for(int ci = 0;
                ci < data.NBcmd; ci++)
            {
                if(strcmp(
                       data.cmd[ci]
                       .key,
                       p) == 0)
                {
                    printf("%s\n", p);
                    found = 1;
                    break;
                }
            }
            cli_last_retval =
                found ? 0 : 1;
            return 1;
        }
        /* command cmd — run directly */
        CLI_execute_string((char *) p);
        return 1;
    }

    /* timeout N cmd */
    if(starts_with(p, "timeout ")
       || starts_with(p,
                      "timeout\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* Parse timeout seconds */
        char *endp;
        double tsec =
            strtod(p, &endp);
        if(endp == p)
        {
            fprintf(stderr,
                    "timeout: "
                    "invalid time\n");
            cli_last_retval = 1;
            return 1;
        }
        const char *cmd_start =
            endp;
        while(*cmd_start == ' '
              || *cmd_start == '\t')
        {
            cmd_start++;
        }
        pid_t tpid = fork();
        if(tpid == 0)
        {
            /* Child: run cmd */
            CLI_execute_string(
                (char *) cmd_start);
            _exit(cli_last_retval);
        }
        else if(tpid > 0)
        {
            /* Parent: wait with
             * timeout */
            struct timespec ts;
            ts.tv_sec =
                (time_t) tsec;
            ts.tv_nsec =
                (long)((tsec
                        - (double)
                        ts.tv_sec)
                       * 1e9);
            int wst = 0;
            struct timespec start;
            clock_gettime(
                CLOCK_MONOTONIC,
                &start);
            while(1)
            {
                int wr =
                    waitpid(tpid,
                            &wst,
                            WNOHANG);
                if(wr > 0)
                {
                    cli_last_retval =
                        WEXITSTATUS(
                            wst);
                    break;
                }
                struct timespec now;
                clock_gettime(
                    CLOCK_MONOTONIC,
                    &now);
                double elapsed =
                    (double)(
                        now.tv_sec
                        - start
                        .tv_sec)
                    + (double)(
                        now.tv_nsec
                        - start
                        .tv_nsec)
                    / 1e9;
                if(elapsed >= tsec)
                {
                    kill(tpid,
                         SIGTERM);
                    usleep(100000);
                    kill(tpid,
                         SIGKILL);
                    waitpid(tpid,
                            &wst, 0);
                    cli_last_retval =
                        124;
                    break;
                }
                usleep(10000);
            }
        }
        return 1;
    }

    /* mapfile / readarray -t arr < file */
    if(starts_with(p, "mapfile ")
       || starts_with(p, "mapfile\t")
       || starts_with(p,
                      "readarray ")
       || starts_with(p,
                      "readarray\t"))
    {
        /* Skip command name */
        if(p[0] == 'm')
        {
            p += 7;
        }
        else
        {
            p += 9;
        }
        p = strip_ws(p);
        /* Parse optional -t flag */
        int strip_nl = 0;
        if(p[0] == '-'
           && p[1] == 't')
        {
            strip_nl = 1;
            p += 2;
            p = strip_ws(p);
        }
        /* Array name */
        char aname[CLI_VAR_NAMELEN];
        {
            int ai = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && *p != '<'
                  && ai
                  < CLI_VAR_NAMELEN - 1)
            {
                aname[ai++] = *p++;
            }
            aname[ai] = '\0';
        }
        p = strip_ws(p);
        /* Check for < file */
        FILE *mf = stdin;
        int should_close = 0;
        if(*p == '<')
        {
            p++;
            p = strip_ws(p);
            mf = fopen(p, "r");
            if(mf == NULL)
            {
                fprintf(stderr,
                        "mapfile: "
                        "%s: "
                        "cannot open\n",
                        p);
                return 1;
            }
            should_close = 1;
        }
        /* Find or create array */
        int slot = -1;
        for(int k = 0;
            k < CLI_MAX_ARRAYS; k++)
        {
            if(cli_arrays[k].used
               && strcmp(
                      cli_arrays[k]
                      .name,
                      aname) == 0)
            {
                slot = k;
                cli_arrays[k].nelem =
                    0;
                break;
            }
        }
        if(slot < 0)
        {
            for(int k = 0;
                k < CLI_MAX_ARRAYS;
                k++)
            {
                if(!cli_arrays[k].used)
                {
                    slot = k;
                    cli_arrays[k]
                        .used = 1;
                    strncpy(
                        cli_arrays[k]
                        .name,
                        aname,
                        CLI_VAR_NAMELEN
                        - 1);
                    cli_arrays[k]
                        .nelem = 0;
                    break;
                }
            }
        }
        if(slot >= 0)
        {
            char mline[
                CLI_VAR_VALLEN];
            while(
                fgets(
                    mline,
                    CLI_VAR_VALLEN,
                    mf) != NULL
                && cli_arrays[slot]
                   .nelem
                < CLI_ARRAY_MAXELEM)
            {
                if(strip_nl)
                {
                    int ml =
                        (int) strlen(
                            mline);
                    if(ml > 0
                       && mline[ml - 1]
                       == '\n')
                    {
                        mline[ml - 1] =
                            '\0';
                    }
                }
                strncpy(
                    cli_arrays[slot]
                    .elem[
                        cli_arrays[slot]
                            .nelem],
                    mline,
                    CLI_VAR_VALLEN
                    - 1);
                cli_arrays[slot]
                    .nelem++;
            }
        }
        if(should_close)
        {
            fclose(mf);
        }
        return 1;
    }

    /* wait — wait for bg children or streams/fps */
    if(strcmp(p, "wait") == 0
       || starts_with(p, "wait ")
       || starts_with(p, "wait\t"))
    {
        char argbuf[STRINGMAXLEN_CLICMDLINE];
        strncpy(argbuf, p, STRINGMAXLEN_CLICMDLINE - 1);
        argbuf[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        
        char *ptr_save = NULL;
        char *tok = strtok_r(argbuf, " \t", &ptr_save); /* "wait" */
        tok = strtok_r(NULL, " \t", &ptr_save);
        
        if (tok != NULL && strcmp(tok, "-S") == 0) {
            char *sname = strtok_r(NULL, " \t", &ptr_save);
            char *tmstr = strtok_r(NULL, " \t", &ptr_save);
            if (!sname) {
                printf("wait: missing stream name\n");
                cli_last_retval = 1;
                return 1;
            }
            double wait_timeout = tmstr ? atof(tmstr) : -1.0;
            
            IMAGE img;
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(sname, &img) == IMAGESTREAMIO_SUCCESS) {
                uint64_t start_cnt0 = img.md->cnt0;
                struct timespec ts_start, ts_now;
                clock_gettime(CLOCK_MONOTONIC, &ts_start);
                cli_last_retval = 1;
                
                while (!cli_break_flag) {
                    if (img.md->cnt0 != start_cnt0) {
                        cli_last_retval = 0;
                        break;
                    }
                    if (wait_timeout >= 0.0) {
                        clock_gettime(CLOCK_MONOTONIC, &ts_now);
                        double elapsed = (double)(ts_now.tv_sec - ts_start.tv_sec) + 
                                         1e-9 * (double)(ts_now.tv_nsec - ts_start.tv_nsec);
                        if (elapsed >= wait_timeout) {
                            break;
                        }
                    }
                    usleep(1000);
                }
                ImageStreamIO_closeIm(&img);
            } else {
                printf("wait: stream %s not found\n", sname);
                cli_last_retval = 1;
            }
            return 1;
        }
        else if (tok != NULL && strcmp(tok, "-F") == 0) {
            char *fname = strtok_r(NULL, " \t", &ptr_save);
            char *pval  = strtok_r(NULL, " \t", &ptr_save);
            char *tmstr = strtok_r(NULL, " \t", &ptr_save);
            
            if (!fname || !pval) {
                printf("wait: missing fps name or param=value\n");
                cli_last_retval = 1;
                return 1;
            }
            
            char *eq = strchr(pval, '=');
            if (!eq) {
                printf("wait: require param=value format\n");
                cli_last_retval = 1;
                return 1;
            }
            *eq = '\0';
            const char *param = pval;
            const char *value = eq + 1;
            double wait_timeout = tmstr ? atof(tmstr) : -1.0;
            
            FUNCTION_PARAMETER_STRUCT fps;
            if (function_parameter_struct_connect(fname, &fps, FPSCONNECT_SIMPLE) != -1 && fps.parray != NULL) {
                int pindex = functionparameter_GetParamIndex(&fps, param);
                if (pindex < 0) {
                    char dotname[512];
                    snprintf(dotname, sizeof(dotname), ".%s", param);
                    pindex = functionparameter_GetParamIndex(&fps, dotname);
                }
                
                if (pindex >= 0) {
                    struct timespec ts_start, ts_now;
                    clock_gettime(CLOCK_MONOTONIC, &ts_start);
                    cli_last_retval = 1;
                    
                    while (!cli_break_flag) {
                        char vstr[512];
                        functionparameter_GetParamValueString(&fps.parray[pindex], vstr, sizeof(vstr));

                        /* First try exact string match (original behavior) */
                        if (strcmp(vstr, value) == 0)
                        {
                            cli_last_retval = 0;
                            break;
                        }

                        /* If not equal as strings, try numeric comparison when both look numeric.
                         * This allows matches such as "1" vs "1.000000" or "1.000000000". */
                        {
                            char  *end_vstr  = NULL;
                            char  *end_value = NULL;
                            double dvstr     = strtod(vstr, &end_vstr);
                            double dvalue    = strtod(value, &end_value);

                            if (end_vstr != vstr && *end_vstr == '\0' &&
                                end_value != value && *end_value == '\0' &&
                                dvstr == dvalue)
                            {
                                cli_last_retval = 0;
                                break;
                            }
                        }
                        if (wait_timeout >= 0.0) {
                            clock_gettime(CLOCK_MONOTONIC, &ts_now);
                            double elapsed = (double)(ts_now.tv_sec - ts_start.tv_sec) + 
                                             1e-9 * (double)(ts_now.tv_nsec - ts_start.tv_nsec);
                            if (elapsed >= wait_timeout) {
                                break;
                            }
                        }
                        usleep(1000);
                    }
                } else {
                    printf("wait: param %s not found in %s\n", param, fname);
                    cli_last_retval = 1;
                }
                function_parameter_struct_disconnect(&fps);
            } else {
                printf("wait: fps %s not found\n", fname);
                cli_last_retval = 1;
            }
            return 1;
        }
        else {
            /* Standard wait for children */
            int wstatus;
            while(waitpid(-1, &wstatus, 0) > 0) {}
            cli_last_retval = 0;
            return 1;
        }
    }

    /* ==============================
     * wait_any — unified event mux
     * ============================== */

    if(starts_with(p, "wait_any ")
       || starts_with(p, "wait_any\t")
       || strcmp(p, "wait_any") == 0)
    {
        /* --- local types --- */
        enum
        {
            WA_STREAM,
            WA_FPS_PARAM,
            WA_PROC_STATE
        };
        enum
        {
            CMP_EQ,
            CMP_NE,
            CMP_GE,
            CMP_LE
        };
        enum
        {
            WA_MAX_EVENTS = 16
        };

        struct wa_event
        {
            int  type;
            char name[256];
            /* FPS */
            char param[256];
            char target_val[256];
            int  cmp_op;
            /* Process */
            int target_state;
            /* Runtime handles */
            IMAGE img;
            uint64_t start_cnt0;
            int img_open;
            FUNCTION_PARAMETER_STRUCT fps;
            int fps_pindex;
            int fps_open;
        };

        struct wa_event events[WA_MAX_EVENTS];
        int nevents = 0;
        double timeout = 30.0;

        /* --- parse arguments --- */
        char argbuf[STRINGMAXLEN_CLICMDLINE];
        strncpy(argbuf, p,
                STRINGMAXLEN_CLICMDLINE - 1);
        argbuf[STRINGMAXLEN_CLICMDLINE - 1]
            = '\0';

        char *sav = NULL;
        char *tok = strtok_r(
            argbuf, " \t", &sav);
        /* skip "wait_any" */
        tok = strtok_r(NULL, " \t", &sav);

        while(tok != NULL
              && nevents < WA_MAX_EVENTS)
        {
            /* -t timeout */
            if(strcmp(tok, "-t") == 0)
            {
                tok = strtok_r(
                    NULL, " \t", &sav);
                if(tok == NULL)
                {
                    fprintf(stderr,
                            "ERROR: wait_any: "
                            "missing timeout "
                            "value after -t\n");
                    fprintf(stderr,
                            "USAGE: wait_any "
                            "[-t timeout] "
                            "<events...>\n");
                    return 255;
                }

                timeout =
                    strtod(tok, NULL);
                tok = strtok_r(
                    NULL, " \t", &sav);
                continue;
            }

            struct wa_event *ev =
                &events[nevents];
            memset(ev, 0, sizeof(*ev));

            if(strncmp(tok, "S:", 2) == 0)
            {
                /* S:<streamname> */
                ev->type = WA_STREAM;
                strncpy(ev->name,
                        tok + 2,
                        sizeof(ev->name)
                        - 1);
                nevents++;
            }
            else if(strncmp(tok, "F:", 2)
                    == 0)
            {
                /* F:<fps>.<param><op><val>
                 * Find first dot after
                 * prefix for fps.param
                 * split */
                ev->type = WA_FPS_PARAM;
                const char *body =
                    tok + 2;
                const char *dot =
                    strchr(body, '.');
                if(dot == NULL)
                {
                    fprintf(stderr,
                            "wait_any: "
                            "bad F: "
                            "token: %s\n",
                            tok);
                    cli_last_retval = 255;
                    return 1;
                }
                /* fps name */
                int nlen =
                    (int)(dot - body);
                if(nlen
                   >= (int) sizeof(
                       ev->name))
                {
                    nlen =
                        (int) sizeof(
                            ev->name)
                        - 1;
                }
                memcpy(ev->name, body,
                       (size_t) nlen);
                ev->name[nlen] = '\0';

                /* param<op>val */
                const char *rest =
                    dot + 1;

                /* Scan for operator:
                 * check 2-char first */
                const char *op_pos = NULL;
                int op_len = 0;
                ev->cmp_op = CMP_EQ;

                op_pos =
                    strstr(rest, ">=");
                if(op_pos != NULL)
                {
                    ev->cmp_op = CMP_GE;
                    op_len = 2;
                }
                if(op_pos == NULL)
                {
                    op_pos = strstr(
                        rest, "<=");
                    if(op_pos != NULL)
                    {
                        ev->cmp_op =
                            CMP_LE;
                        op_len = 2;
                    }
                }
                if(op_pos == NULL)
                {
                    op_pos = strstr(
                        rest, "!=");
                    if(op_pos != NULL)
                    {
                        ev->cmp_op =
                            CMP_NE;
                        op_len = 2;
                    }
                }
                if(op_pos == NULL)
                {
                    op_pos =
                        strchr(rest, '=');
                    if(op_pos != NULL)
                    {
                        ev->cmp_op =
                            CMP_EQ;
                        op_len = 1;
                    }
                }

                if(op_pos == NULL)
                {
                    fprintf(stderr,
                            "wait_any: "
                            "no operator "
                            "in F: "
                            "token: %s\n",
                            tok);
                    cli_last_retval = 255;
                    return 1;
                }

                /* param name */
                int plen =
                    (int)(op_pos - rest);
                if(plen
                   >= (int) sizeof(
                       ev->param))
                {
                    plen =
                        (int) sizeof(
                            ev->param)
                        - 1;
                }
                memcpy(ev->param, rest,
                       (size_t) plen);
                ev->param[plen] = '\0';

                /* target value */
                strncpy(
                    ev->target_val,
                    op_pos + op_len,
                    sizeof(
                        ev->target_val)
                    - 1);

                nevents++;
            }
            else if(strncmp(tok, "P:", 2)
                    == 0)
            {
                /* P:<name>:<STATE> */
                ev->type = WA_PROC_STATE;
                const char *body =
                    tok + 2;
                const char *colon =
                    strchr(body, ':');
                if(colon == NULL)
                {
                    fprintf(stderr,
                            "wait_any: "
                            "bad P: "
                            "token: %s\n",
                            tok);
                    cli_last_retval = 255;
                    return 1;
                }
                int nlen =
                    (int)(colon - body);
                if(nlen
                   >= (int) sizeof(
                       ev->name))
                {
                    nlen =
                        (int) sizeof(
                            ev->name)
                        - 1;
                }
                memcpy(ev->name, body,
                       (size_t) nlen);
                ev->name[nlen] = '\0';

                const char *st =
                    colon + 1;
                if(strcasecmp(st,
                              "INIT")
                   == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_INIT;
                }
                else if(strcasecmp(
                            st,
                            "ACTIVE")
                        == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_ACTIVE;
                }
                else if(strcasecmp(
                            st, "PAUSE")
                        == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_PAUSE;
                }
                else if(strcasecmp(
                            st, "STOP")
                        == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_STOP;
                }
                else if(strcasecmp(
                            st, "ERROR")
                        == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_ERROR;
                }
                else if(strcasecmp(
                            st, "SPIN")
                        == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_SPIN;
                }
                else if(strcasecmp(
                            st,
                            "CRASHED")
                        == 0)
                {
                    ev->target_state =
                        PROCESSINFO_LOOPSTAT_CRASHED;
                }
                else
                {
                    ev->target_state =
                        (int) strtol(
                            st,
                            NULL,
                            0);
                }
                nevents++;
            }
            else
            {
                fprintf(stderr,
                        "wait_any: "
                        "unknown event "
                        "prefix: %s\n",
                        tok);
                cli_last_retval = 255;
                return 1;
            }
            tok = strtok_r(
                NULL, " \t", &sav);
        }

        /* Detect overflow: loop exited
         * because nevents hit the cap
         * but tokens remain */
        if(tok != NULL)
        {
            fprintf(stderr,
                    "ERROR: wait_any: "
                    "too many events "
                    "(max %d)\n",
                    WA_MAX_EVENTS);
            cli_last_retval = 255;
            return 1;
        }

        if(nevents == 0)
        {
            printf("Usage: wait_any "
                   "[-t timeout] "
                   "S:stream "
                   "[F:fps.p=v] "
                   "[P:proc:STATE]\n");
            cli_last_retval = 255;
            return 1;
        }

        /* --- open event handles --- */
        int any_open = 0;
        for(int i = 0; i < nevents; i++)
        {
            struct wa_event *ev =
                &events[i];
            ev->img_open = 0;
            ev->fps_open = 0;

            if(ev->type == WA_STREAM)
            {
                if(ImageStreamIO_read_sharedmem_image_toIMAGE(
                       ev->name,
                       &ev->img)
                   == IMAGESTREAMIO_SUCCESS)
                {
                    ev->start_cnt0 =
                        ev->img.md
                            ->cnt0;
                    ev->img_open = 1;
                    any_open = 1;
                }
            }
            else if(ev->type
                    == WA_FPS_PARAM)
            {
                if(function_parameter_struct_connect(
                       ev->name,
                       &ev->fps,
                       FPSCONNECT_SIMPLE)
                       != -1
                   && ev->fps.parray
                       != NULL)
                {
                    ev->fps_pindex =
                        functionparameter_GetParamIndex(
                            &ev->fps,
                            ev->param);
                    if(ev->fps_pindex
                       < 0)
                    {
                        /* Try with
                         * leading dot */
                        char dname[512];
                        snprintf(
                            dname,
                            sizeof(dname),
                            ".%s",
                            ev->param);
                        ev->fps_pindex =
                            functionparameter_GetParamIndex(
                                &ev->fps,
                                dname);
                    }
                    if(ev->fps_pindex
                       >= 0)
                    {
                        ev->fps_open = 1;
                        any_open = 1;
                    }
                    else
                    {
                        function_parameter_struct_disconnect(
                            &ev->fps);
                    }
                }
            }
            else if(ev->type
                    == WA_PROC_STATE)
            {
                /* procstate uses
                 * global pinfolist;
                 * always "open" */
                any_open = 1;
            }
        }

        if(!any_open)
        {
            fprintf(stderr,
                    "wait_any: "
                    "no events could "
                    "be opened\n");
            cli_last_retval = 255;
            goto wa_cleanup;
        }

        /* --- poll loop --- */
        {
            struct timespec ts_start;
            clock_gettime(
                CLOCK_MONOTONIC,
                &ts_start);
            cli_last_retval = 254;

            while(!cli_break_flag)
            {
                for(int i = 0;
                    i < nevents; i++)
                {
                    struct wa_event *ev =
                        &events[i];
                    int fired = 0;

                    if(ev->type
                           == WA_STREAM
                       && ev->img_open)
                    {
                        if(ev->img.md
                               ->cnt0
                           != ev
                               ->start_cnt0)
                        {
                            fired = 1;
                        }
                    }
                    else if(
                        ev->type
                            == WA_FPS_PARAM
                        && ev->fps_open)
                    {
                        char vstr[512];
                        functionparameter_GetParamValueString(
                            &ev->fps
                                 .parray
                                     [ev->fps_pindex],
                            vstr,
                            sizeof(vstr));

                        switch(
                            ev->cmp_op)
                        {
                        case CMP_EQ:
                        {
                            if(strcmp(
                                   vstr,
                                   ev->target_val)
                               == 0)
                            {
                                fired = 1;
                                break;
                            }
                            char *e1 =
                                NULL;
                            char *e2 =
                                NULL;
                            double d1 =
                                strtod(
                                    vstr,
                                    &e1);
                            double d2 =
                                strtod(
                                    ev->target_val,
                                    &e2);
                            if(e1 != vstr
                               && *e1
                                   == '\0'
                               && e2
                                   != ev
                                       ->target_val
                               && *e2
                                   == '\0'
                               && d1
                                   == d2)
                            {
                                fired = 1;
                            }
                        }
                        break;
                        case CMP_NE:
                        {
                            int eq = 0;
                            if(strcmp(
                                   vstr,
                                   ev->target_val)
                               == 0)
                            {
                                eq = 1;
                            }
                            else
                            {
                                char *e1 =
                                    NULL;
                                char *e2 =
                                    NULL;
                                double d1 =
                                    strtod(
                                        vstr, &e1);
                                double d2 =
                                    strtod(
                                        ev->target_val,
                                        &e2);
                                if(e1
                                       != vstr
                                   && *e1
                                       == '\0'
                                   && e2
                                       != ev
                                           ->target_val
                                   && *e2
                                       == '\0'
                                   && d1
                                       == d2)
                                {
                                    eq = 1;
                                }
                            }
                            if(!eq)
                            {
                                fired = 1;
                            }
                        }
                        break;
                        case CMP_GE:
                        case CMP_LE:
                        {
                            char *e1 =
                                NULL;
                            char *e2 =
                                NULL;
                            double d1 =
                                strtod(
                                    vstr,
                                    &e1);
                            double d2 =
                                strtod(
                                    ev->target_val,
                                    &e2);
                            if(e1 != vstr
                               && *e1
                                   == '\0'
                               && e2
                                   != ev
                                       ->target_val
                               && *e2
                                   == '\0')
                            {
                                if(ev->cmp_op
                                       == CMP_GE
                                   && d1
                                       >= d2)
                                {
                                    fired
                                        = 1;
                                }
                                if(ev->cmp_op
                                       == CMP_LE
                                   && d1
                                       <= d2)
                                {
                                    fired
                                        = 1;
                                }
                            }
                        }
                        break;
                        }
                    }
                    else if(
                        ev->type
                        == WA_PROC_STATE)
                    {
                        if(pinfolist
                           != NULL)
                        {
                            for(int pi =
                                    0;
                                pi
                                < PROCESSINFOLISTSIZE;
                                pi++)
                            {
                                if(!pinfolist
                                    ->active
                                        [pi])
                                {
                                    continue;
                                }
                                if(strcmp(
                                       pinfolist
                                           ->pnamearray
                                               [pi],
                                       ev->name)
                                   != 0)
                                {
                                    continue;
                                }
                                pid_t fpid =
                                    pinfolist
                                        ->PIDarray
                                            [pi];
                                char pfn
                                    [512];
                                char pdname
                                    [256];
                                processinfo_procdirname(
                                    pdname);
                                snprintf(
                                    pfn,
                                    sizeof(
                                        pfn),
                                    "%s/"
                                    "proc."
                                    "%d"
                                    ".shm",
                                    pdname,
                                    (int)
                                        fpid);
                                int pfd =
                                    -1;
                                PROCESSINFO
                                    *pii =
                                    processinfo_shm_link(
                                        pfn,
                                        &pfd);
                                if(pii
                                       != MAP_FAILED
                                   && pii
                                       != NULL)
                                {
                                    if(pii
                                        ->loopstat
                                       == ev
                                           ->target_state)
                                    {
                                        fired
                                            = 1;
                                    }
                                    munmap(
                                        pii,
                                        sizeof(
                                            PROCESSINFO));
                                    close(
                                        pfd);
                                }
                                else if(
                                    pfd
                                    >= 0)
                                {
                                    close(
                                        pfd);
                                }
                                break;
                            }
                        }
                    }

                    if(fired)
                    {
                        cli_last_retval
                            = i;
                        goto wa_cleanup;
                    }
                }

                /* Timeout check */
                if(timeout >= 0.0)
                {
                    struct timespec
                        ts_now;
                    clock_gettime(
                        CLOCK_MONOTONIC,
                        &ts_now);
                    double elapsed =
                        (double)(
                            ts_now.tv_sec
                            - ts_start
                                .tv_sec)
                        + 1e-9
                        * (double)(
                            ts_now
                                .tv_nsec
                            - ts_start
                                .tv_nsec);
                    if(elapsed
                       >= timeout)
                    {
                        cli_last_retval
                            = 254;
                        goto wa_cleanup;
                    }
                }
                usleep(1000);
            }
        }

wa_cleanup:
        /* --- close handles --- */
        for(int i = 0; i < nevents; i++)
        {
            if(events[i].img_open)
            {
                ImageStreamIO_closeIm(
                    &events[i].img);
            }
            if(events[i].fps_open)
            {
                function_parameter_struct_disconnect(
                    &events[i].fps);
            }
        }
        return 1;
    }

    /* [[ expr ]] — extended test */
    if(starts_with(p, "[[ "))
    {
        int plen = (int) strlen(p);
        if(plen >= 5
           && p[plen - 1] == ']'
           && p[plen - 2] == ']')
        {
            /* Extract inner expr */
            char texpr[
                STRINGMAXLEN_CLICMDLINE];
            memcpy(texpr, p + 3,
                   (size_t)(plen - 5));
            texpr[plen - 5] = '\0';
            /* Trim whitespace */
            int tlen =
                (int) strlen(texpr);
            while(tlen > 0
                  && (texpr[tlen - 1]
                      == ' '
                      || texpr[tlen - 1]
                      == '\t'))
            {
                texpr[--tlen] = '\0';
            }
            int result =
                cli_eval_test(texpr);
            cli_last_retval =
                result ? 0 : 1;
        }
        return 1;
    }

    /* local VAR=val — set variable (only
     * meaningful inside function, but works
     * anywhere) */
    if(starts_with(p, "local ")
       || starts_with(p, "local\t"))
    {
        p += 5;
        p = strip_ws(p);
        cli_try_var_assign(p);
        return 1;
    }

    /* if ... */
    if(starts_with(p, "if ")
       || starts_with(p, "if\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_IF;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* while ... */
    if(starts_with(p, "while ")
       || starts_with(p, "while\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_WHILE;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* until ... */
    if(starts_with(p, "until ")
       || starts_with(p, "until\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_UNTIL;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* for ... */
    if(starts_with(p, "for ")
       || starts_with(p, "for\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_FOR;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* select VAR in val1 val2; do ... */
    if(starts_with(p, "select ")
       || starts_with(p, "select\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type =
            CLI_BLOCK_SELECT;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE
                - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* function name { ... } */
    if(starts_with(p, "function ")
       || starts_with(p, "function\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_FUNC;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* case <word> in ... esac */
    if(starts_with(p, "case ")
       || starts_with(p, "case\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_CASE;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* ==============================
     * Tier 9: true / false
     * ============================== */

    if(strcmp(p, "true") == 0)
    {
        cli_last_retval = 0;
        return 1;
    }
    if(strcmp(p, "false") == 0)
    {
        cli_last_retval = 1;
        return 1;
    }

    /* ==============================
     * Tier 9: (( expr )) conditional
     * ============================== */

    if(starts_with(p, "((")
       && strlen(p) >= 5)
    {
        int plen = (int) strlen(p);
        if(p[plen - 1] == ')'
           && p[plen - 2] == ')')
        {
            char aexpr[
                STRINGMAXLEN_CLICMDLINE
            ];
            int elen = plen - 4;
            if(elen
               >= STRINGMAXLEN_CLICMDLINE)
            {
                elen =
                    STRINGMAXLEN_CLICMDLINE
                    - 1;
            }
            memcpy(aexpr, p + 2,
                   (size_t) elen);
            aexpr[elen] = '\0';
            /* Wrap in $(( )) and
             * expand */
            char wrap[
                STRINGMAXLEN_CLICMDLINE + 64
            ];
            snprintf(wrap,
                     sizeof(wrap),
                     "$((%s))",
                     aexpr);
            cli_expand_env(
                wrap,
                STRINGMAXLEN_CLICMDLINE
            );
            long val =
                strtol(wrap, NULL, 10);
            cli_last_retval =
                (val != 0) ? 0 : 1;
            return 1;
        }
    }

    /* ==============================
     * Tier 9: alias / unalias
     * ============================== */

    if(starts_with(p, "alias ")
       || starts_with(p, "alias\t")
       || strcmp(p, "alias") == 0)
    {
        p += 5;
        p = strip_ws(p);
        if(*p == '\0')
        {
            /* List all aliases */
            for(int k = 0;
                k < data.NBalias;
                k++)
            {
                printf("alias %s="
                       "'%s'\n",
                       data.alias[
                           k].name,
                       data.alias[
                           k].cmd);
            }
        }
        else
        {
            /* alias name='cmd' or
             * alias name=cmd */
            char *eq = strchr(p, '=');
            if(eq != NULL)
            {
                char aname[
                    CLI_ALIAS_NAMELEN
                ];
                int nl =
                    (int)(eq - p);
                if(nl
                   >= CLI_ALIAS_NAMELEN)
                {
                    nl =
                        CLI_ALIAS_NAMELEN
                        - 1;
                }
                memcpy(aname, p,
                       (size_t) nl);
                aname[nl] = '\0';
                const char *av =
                    eq + 1;
                /* Strip quotes */
                int avl =
                    (int) strlen(av);
                if(avl >= 2
                   && ((av[0] == '\''
                        && av[avl - 1]
                        == '\'')
                       || (av[0] == '"'
                           && av[
                               avl - 1]
                           == '"')))
                {
                    av++;
                    avl -= 2;
                }
                /* Update existing? */
                int slot = -1;
                for(int k = 0;
                    k < data.NBalias;
                    k++)
                {
                    if(strcmp(
                        data.alias[k]
                        .name,
                        aname) == 0)
                    {
                        slot = k;
                        break;
                    }
                }
                if(slot < 0
                   && data.NBalias
                   < CLI_MAX_ALIASES)
                {
                    slot =
                        data.NBalias++;
                }
                if(slot >= 0)
                {
                    strncpy(
                        data.alias[
                            slot].name,
                        aname,
                        CLI_ALIAS_NAMELEN
                        - 1);
                    data.alias[slot]
                        .name[
                        CLI_ALIAS_NAMELEN
                        - 1] = '\0';
                    int cl =
                        avl
                        < CLI_ALIAS_CMDLEN
                        - 1
                        ? avl
                        : CLI_ALIAS_CMDLEN
                        - 1;
                    memcpy(
                        data.alias[
                            slot].cmd,
                        av,
                        (size_t) cl);
                    data.alias[slot]
                        .cmd[cl] = '\0';
                }
            }
        }
        return 1;
    }

    if(starts_with(p, "unalias ")
       || starts_with(p, "unalias\t"))
    {
        p += 7;
        p = strip_ws(p);
        for(int k = 0;
            k < data.NBalias; k++)
        {
            if(strcmp(
                data.alias[k].name,
                p) == 0)
            {
                /* Shift remaining */
                for(int j = k;
                    j < data.NBalias
                    - 1; j++)
                {
                    data.alias[j] =
                        data.alias[
                            j + 1];
                }
                data.NBalias--;
                break;
            }
        }
        return 1;
    }

    /* ==============================
     * Tier 9: assoc array map[k]=v
     * ============================== */

    {
        const char *br =
            strchr(p, '[');
        if(br != NULL)
        {
            const char *brend =
                strchr(br, ']');
            if(brend != NULL
               && *(brend + 1) == '=')
            {
                char aname[
                    CLI_VAR_NAMELEN];
                int nl =
                    (int)(br - p);
                if(nl
                   >= CLI_VAR_NAMELEN)
                {
                    nl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(aname, p,
                       (size_t) nl);
                aname[nl] = '\0';
                char key[
                    CLI_VAR_NAMELEN];
                int kl =
                    (int)(brend
                          - br - 1);
                if(kl
                   >= CLI_VAR_NAMELEN)
                {
                    kl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(key, br + 1,
                       (size_t) kl);
                key[kl] = '\0';
                const char *val =
                    brend + 2;
                /* Find or create
                 * assoc array */
                int slot = -1;
                for(int k = 0;
                    k < CLI_MAX_ASSOC;
                    k++)
                {
                    if(cli_assoc[k]
                        .used
                       && strcmp(
                           cli_assoc[k]
                           .name,
                           aname)
                       == 0)
                    {
                        slot = k;
                        break;
                    }
                }
                if(slot < 0)
                {
                    for(int k = 0;
                        k
                        < CLI_MAX_ASSOC;
                        k++)
                    {
                        if(!cli_assoc[
                            k].used)
                        {
                            slot = k;
                            cli_assoc[k]
                                .used
                                = 1;
                            strncpy(
                                cli_assoc[
                                    k]
                                .name,
                                aname,
                                CLI_VAR_NAMELEN
                                - 1);
                            cli_assoc[k]
                                .nelem
                                = 0;
                            break;
                        }
                    }
                }
                if(slot >= 0)
                {
                    /* Find existing
                     * key or add */
                    int ki = -1;
                    for(int k = 0;
                        k
                        < cli_assoc[
                            slot]
                        .nelem;
                        k++)
                    {
                        if(strcmp(
                            cli_assoc[
                                slot]
                            .keys[k],
                            key) == 0)
                        {
                            ki = k;
                            break;
                        }
                    }
                    if(ki < 0
                       && cli_assoc[
                           slot]
                       .nelem
                       < CLI_ASSOC_MAXELEM)
                    {
                        ki =
                            cli_assoc[
                                slot]
                            .nelem++;
                        strncpy(
                            cli_assoc[
                                slot]
                            .keys[ki],
                            key,
                            CLI_VAR_NAMELEN
                            - 1);
                    }
                    if(ki >= 0)
                    {
                        strncpy(
                            cli_assoc[
                                slot]
                            .vals[ki],
                            val,
                            CLI_VAR_VALLEN
                            - 1);
                    }
                }
                return 1;
            }
        }
    }

    /* ==============================
     * Tier 10: basename / dirname
     * ============================== */

    if(starts_with(p, "basename "))
    {
        p += 8;
        p = strip_ws(p);
        /* Find last / */
        const char *sl =
            strrchr(p, '/');
        if(sl != NULL)
        {
            printf("%s\n", sl + 1);
        }
        else
        {
            printf("%s\n", p);
        }
        cli_last_retval = 0;
        return 1;
    }

    if(starts_with(p, "dirname "))
    {
        p += 7;
        p = strip_ws(p);
        const char *sl =
            strrchr(p, '/');
        if(sl != NULL && sl != p)
        {
            printf("%.*s\n",
                   (int)(sl - p), p);
        }
        else if(sl == p)
        {
            printf("/\n");
        }
        else
        {
            printf(".\n");
        }
        cli_last_retval = 0;
        return 1;
    }

    /* ==============================
     * Tier 10: pushd / popd / dirs
     * ============================== */

    if(starts_with(p, "pushd ")
       || starts_with(p, "pushd\t"))
    {
        p += 5;
        p = strip_ws(p);
        char cwd[1024];
        if(getcwd(cwd, sizeof(cwd))
           != NULL)
        {
            /* Push current dir as
             * cli var */
            char idx[32];
            /* Count existing
             * _dirstack entries */
            int dcnt = 0;
            for(int k = 0;
                k < CLI_MAX_VARS; k++)
            {
                if(cli_vars[k].used
                   && strncmp(
                       cli_vars[k]
                       .name,
                       "_ds_",
                       4) == 0)
                {
                    dcnt++;
                }
            }
            snprintf(idx,
                     sizeof(idx),
                     "_ds_%d", dcnt);
            cli_var_set(idx, cwd);
        }
        if(chdir(p) != 0)
        {
            printf("pushd: %s: %s\n",
                   p,
                   strerror(errno));
            cli_last_retval = 1;
        }
        else
        {
            cli_last_retval = 0;
        }
        return 1;
    }

    if(strcmp(p, "popd") == 0)
    {
        /* Find highest _ds_N */
        int maxn = -1;
        int maxk = -1;
        for(int k = 0;
            k < CLI_MAX_VARS; k++)
        {
            if(cli_vars[k].used
               && strncmp(
                   cli_vars[k].name,
                   "_ds_", 4) == 0)
            {
                int n = atoi(
                    cli_vars[k].name
                    + 4);
                if(n > maxn)
                {
                    maxn = n;
                    maxk = k;
                }
            }
        }
        if(maxk >= 0)
        {
            if(chdir(
                cli_vars[maxk].val)
               != 0)
            {
                printf("popd: %s\n",
                       strerror(
                           errno));
            }
            cli_vars[maxk].used = 0;
        }
        else
        {
            printf("popd: directory "
                   "stack empty\n");
        }
        cli_last_retval = 0;
        return 1;
    }

    if(strcmp(p, "dirs") == 0)
    {
        char cwd[1024];
        if(getcwd(cwd, sizeof(cwd))
           != NULL)
        {
            printf("%s", cwd);
        }
        for(int n = 0;
            n < CLI_MAX_VARS; n++)
        {
            char idx[32];
            snprintf(idx,
                     sizeof(idx),
                     "_ds_%d", n);
            const char *dv =
                cli_var_get(idx);
            if(dv == NULL)
            {
                break;
            }
            printf(" %s", dv);
        }
        printf("\n");
        cli_last_retval = 0;
        return 1;
    }

    /* ==============================
     * Tier 10: seq START [STEP] END
     * ============================== */

    if(starts_with(p, "seq "))
    {
        p += 3;
        p = strip_ws(p);
        double s1 = 0.0;
        double step = 1.0;
        double s2 = 0.0;
        /* Parse up to 3 numbers */
        char *end1 = NULL;
        s1 = strtod(p, &end1);
        if(end1 != NULL
           && *end1 != '\0')
        {
            const char *p2 =
                strip_ws(end1);
            char *end2 = NULL;
            double v2 =
                strtod(p2, &end2);
            if(end2 != NULL
               && *end2 != '\0')
            {
                const char *p3 =
                    strip_ws(end2);
                double v3 =
                    strtod(p3, NULL);
                /* 3-arg: s1 step s2 */
                step = v2;
                s2 = v3;
            }
            else
            {
                /* 2-arg: s1 s2 */
                s2 = v2;
            }
        }
        else
        {
            /* 1-arg: 1..s1 */
            s2 = s1;
            s1 = 1.0;
        }
        if(step > 0.0)
        {
            for(double v = s1;
                v <= s2 + 1e-12;
                v += step)
            {
                printf("%g\n", v);
            }
        }
        else if(step < 0.0)
        {
            for(double v = s1;
                v >= s2 - 1e-12;
                v += step)
            {
                printf("%g\n", v);
            }
        }
        cli_last_retval = 0;
        return 1;
    }

    /* ==============================
     * Tier 11: waitfor_stream
     * ============================== */

    if(starts_with(p, "waitfor_stream "))
    {
        p += 14;
        p = strip_ws(p);
        char sname[CLI_VAR_NAMELEN];
        int si = 0;
        while(*p != '\0'
              && *p != ' '
              && *p != '\t'
              && si
              < CLI_VAR_NAMELEN - 1)
        {
            sname[si++] = *p++;
        }
        sname[si] = '\0';
        p = strip_ws(p);
        double tout = 10.0;
        if(*p != '\0')
        {
            tout = strtod(p, NULL);
        }
        struct timespec wstart;
        clock_gettime(
            CLOCK_MONOTONIC,
            &wstart);
        int found = 0;
        while(1)
        {
            /* Check if SHM exists */
            char shmpath[256];
            snprintf(shmpath,
                     sizeof(shmpath),
                     "%s/%s"
                     ".im.shm",
                     dcshmdir,
                     sname);
            if(access(shmpath,
                      F_OK) == 0)
            {
                found = 1;
                break;
            }
            struct timespec wnow;
            clock_gettime(
                CLOCK_MONOTONIC,
                &wnow);
            double elapsed =
                (double)(wnow.tv_sec
                    - wstart.tv_sec)
                + (double)(
                    wnow.tv_nsec
                    - wstart.tv_nsec)
                / 1e9;
            if(elapsed >= tout)
            {
                break;
            }
            usleep(50000);
        }
        cli_last_retval =
            found ? 0 : 1;
        return 1;
    }

    /* ==============================
     * Tier 11: waitfor_fps
     * ============================== */

    if(starts_with(p, "waitfor_fps "))
    {
        p += 11;
        p = strip_ws(p);
        char fname[CLI_VAR_NAMELEN];
        int fi = 0;
        while(*p != '\0'
              && *p != ' '
              && *p != '\t'
              && fi
              < CLI_VAR_NAMELEN - 1)
        {
            fname[fi++] = *p++;
        }
        fname[fi] = '\0';
        p = strip_ws(p);
        double tout = 10.0;
        if(*p != '\0')
        {
            tout = strtod(p, NULL);
        }
        struct timespec wstart;
        clock_gettime(
            CLOCK_MONOTONIC,
            &wstart);
        int found = 0;
        while(1)
        {
            char fpath[256];
            snprintf(fpath,
                     sizeof(fpath),
                     "%s/"
                     "fps.%s.shm",
                     dcshmdir,
                     fname);
            if(access(fpath,
                      F_OK) == 0)
            {
                found = 1;
                break;
            }
            struct timespec wnow;
            clock_gettime(
                CLOCK_MONOTONIC,
                &wnow);
            double elapsed =
                (double)(wnow.tv_sec
                    - wstart.tv_sec)
                + (double)(
                    wnow.tv_nsec
                    - wstart.tv_nsec)
                / 1e9;
            if(elapsed >= tout)
            {
                break;
            }
            usleep(50000);
        }
        cli_last_retval =
            found ? 0 : 1;
        return 1;
    }

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
