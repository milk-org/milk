#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
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
 * @brief Handler: evaluate an arithmetic expression.
 */
int cli_intercept_cmd_let(const char *p)
{
    if (starts_with(p, "let ") || starts_with(p, "let\t"))
    {
        p += 3;
        p = strip_ws(p);
        /* Strip optional quotes */
        char lexpr[STRINGMAXLEN_CLICMDLINE];
        strncpy(lexpr, p, STRINGMAXLEN_CLICMDLINE - 1);
        lexpr[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        int ll                             = (int) strlen(lexpr);
        if (ll >= 2 && ((lexpr[0] == '"' && lexpr[ll - 1] == '"') ||
                        (lexpr[0] == '\'' && lexpr[ll - 1] == '\'')))
        {
            lexpr[ll - 1] = '\0';
            memmove(lexpr, lexpr + 1, (size_t) (ll - 1));
        }
        /* Build $(( )) expression */
        char ecmd[STRINGMAXLEN_CLICMDLINE + 64];
        snprintf(ecmd, sizeof(ecmd), "$((%s))", lexpr);
        /* Find assignment target */
        char *aeq = strchr(lexpr, '=');
        if (aeq != NULL && aeq != lexpr && aeq[-1] != '!' && aeq[-1] != '<' && aeq[-1] != '>')
        {
            /* Has assignment, e.g.
             * let "x = 1 + 2" */
            *aeq = '\0';
            /* Trim target var */
            char tvar[CLI_VAR_NAMELEN];
            {
                const char *ts = lexpr;
                while (*ts == ' ' || *ts == '\t')
                {
                    ts++;
                }
                int ti = 0;
                while (*ts != '\0' && *ts != ' ' && *ts != '\t' && ti < CLI_VAR_NAMELEN - 1)
                {
                    tvar[ti++] = *ts++;
                }
                tvar[ti] = '\0';
            }
            /* Eval RHS */
            const char *rhs = aeq + 1;
            while (*rhs == ' ' || *rhs == '\t')
            {
                rhs++;
            }
            char arith[STRINGMAXLEN_CLICMDLINE];
            snprintf(arith, sizeof(arith), "$((%s))", rhs);
            cli_expand_env(arith, STRINGMAXLEN_CLICMDLINE);
            cli_var_set(tvar, arith);
        }
        else
        {
            /* No assignment, just
             * evaluate */
            cli_expand_env(ecmd, STRINGMAXLEN_CLICMDLINE);
            cli_last_retval = (strtol(ecmd, NULL, 10) == 0) ? 1 : 0;
        }
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: evaluate a string as a command.
 */
int cli_intercept_cmd_eval(const char *p)
{
    if (starts_with(p, "eval ") || starts_with(p, "eval\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Strip outer quotes */
        char ecmd[STRINGMAXLEN_CLICMDLINE];
        strncpy(ecmd, p, STRINGMAXLEN_CLICMDLINE - 1);
        ecmd[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        int el                            = (int) strlen(ecmd);
        if (el >= 2 &&
            ((ecmd[0] == '"' && ecmd[el - 1] == '"') || (ecmd[0] == '\'' && ecmd[el - 1] == '\'')))
        {
            ecmd[el - 1] = '\0';
            memmove(ecmd, ecmd + 1, (size_t) (el - 1));
        }
        CLI_execute_string(ecmd);
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: display the type of a variable.
 */
int cli_intercept_cmd_type(const char *p)
{
    if (starts_with(p, "type ") || starts_with(p, "type\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Search registered aliases */
        int found = 0;
        for (int i = 0; i < data.NBalias; i++)
        {
            if (strcmp(data.alias[i].name, p) == 0)
            {
                printf("%s is aliased to `%s`\n", p, data.alias[i].cmd);
                found = 1;
                break;
            }
        }

        /* Search user functions */
        if (!found)
        {
            CLI_FUNC *f = cli_func_find(p);
            if (f != NULL)
            {
                printf("%s is a function\n", p);
                found = 1;
            }
        }

        /* Search registered CLI commands */
        if (!found)
        {
            for (int ci = 0; ci < data.NBcmd; ci++)
            {
                if (strcmp(data.cmd[ci].key, p) == 0)
                {
                    printf("%s is a CLI command\n", p);
                    found = 1;
                    break;
                }
            }
        }

        /* Search external executables in PATH */
        if (!found)
        {
            char path_found[1024];
            if (cli_find_in_path(p, path_found, sizeof(path_found)))
            {
                printf("%s\n", path_found);
                found = 1;
            }
        }

        if (!found)
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
    return 0;
}

int cli_intercept_cmd_command(const char *p)
{
    if (starts_with(p, "command ") || starts_with(p, "command\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* command -v cmd */
        if (starts_with(p, "-v "))
        {
            p += 3;
            p         = strip_ws(p);
            int found = 0;
            for (int ci = 0; ci < data.NBcmd; ci++)
            {
                if (strcmp(data.cmd[ci].key, p) == 0)
                {
                    printf("%s\n", p);
                    found = 1;
                    break;
                }
            }
            cli_last_retval = found ? 0 : 1;
            return 1;
        }
        /* command cmd — run directly */
        CLI_execute_string((char *) p);
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_timeout(const char *p)
{
    if (starts_with(p, "timeout ") || starts_with(p, "timeout\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* Parse timeout seconds */
        char  *endp;
        double tsec = strtod(p, &endp);
        if (endp == p)
        {
            fprintf(stderr, "timeout: "
                            "invalid time\n");
            cli_last_retval = 1;
            return 1;
        }
        const char *cmd_start = endp;
        while (*cmd_start == ' ' || *cmd_start == '\t')
        {
            cmd_start++;
        }
        pid_t tpid = fork();
        if (tpid == 0)
        {
            /* Child: run cmd */
            CLI_execute_string((char *) cmd_start);
            _exit(cli_last_retval);
        }
        else if (tpid > 0)
        {
            /* Parent: wait with
             * timeout */
            struct timespec ts;
            ts.tv_sec           = (time_t) tsec;
            ts.tv_nsec          = (long) ((tsec - (double) ts.tv_sec) * 1e9);
            int             wst = 0;
            struct timespec start;
            clock_gettime(CLOCK_MONOTONIC, &start);
            while (1)
            {
                int wr = waitpid(tpid, &wst, WNOHANG);
                if (wr > 0)
                {
                    cli_last_retval = WEXITSTATUS(wst);
                    break;
                }
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (double) (now.tv_sec - start.tv_sec) +
                                 (double) (now.tv_nsec - start.tv_nsec) / 1e9;
                if (elapsed >= tsec)
                {
                    kill(tpid, SIGTERM);
                    usleep(100000);
                    kill(tpid, SIGKILL);
                    waitpid(tpid, &wst, 0);
                    cli_last_retval = 124;
                    break;
                }
                usleep(10000);
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_mapfile(const char *p)
{
    if (starts_with(p, "mapfile ") || starts_with(p, "mapfile\t") || starts_with(p, "readarray ") ||
        starts_with(p, "readarray\t"))
    {
        /* Skip command name */
        if (p[0] == 'm')
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
        if (p[0] == '-' && p[1] == 't')
        {
            strip_nl = 1;
            p += 2;
            p = strip_ws(p);
        }
        /* Array name */
        char aname[CLI_VAR_NAMELEN];
        {
            int ai = 0;
            while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '<' && ai < CLI_VAR_NAMELEN - 1)
            {
                aname[ai++] = *p++;
            }
            aname[ai] = '\0';
        }
        p = strip_ws(p);
        /* Check for < file */
        FILE *mf           = stdin;
        int   should_close = 0;
        if (*p == '<')
        {
            p++;
            p  = strip_ws(p);
            mf = fopen(p, "r");
            if (mf == NULL)
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
        for (int k = 0; k < CLI_MAX_ARRAYS; k++)
        {
            if (cli_arrays[k].used && strcmp(cli_arrays[k].name, aname) == 0)
            {
                slot                = k;
                cli_arrays[k].nelem = 0;
                break;
            }
        }
        if (slot < 0)
        {
            for (int k = 0; k < CLI_MAX_ARRAYS; k++)
            {
                if (!cli_arrays[k].used)
                {
                    slot               = k;
                    cli_arrays[k].used = 1;
                    strncpy(cli_arrays[k].name, aname, CLI_VAR_NAMELEN - 1);
                    cli_arrays[k].nelem = 0;
                    break;
                }
            }
        }
        if (slot >= 0)
        {
            char mline[CLI_VAR_VALLEN];
            while (fgets(mline, CLI_VAR_VALLEN, mf) != NULL &&
                   cli_arrays[slot].nelem < CLI_ARRAY_MAXELEM)
            {
                if (strip_nl)
                {
                    int ml = (int) strlen(mline);
                    if (ml > 0 && mline[ml - 1] == '\n')
                    {
                        mline[ml - 1] = '\0';
                    }
                }
                strncpy(cli_arrays[slot].elem[cli_arrays[slot].nelem], mline, CLI_VAR_VALLEN - 1);
                cli_arrays[slot].nelem++;
            }
        }
        if (should_close)
        {
            fclose(mf);
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_wait(const char *p)
{
    if (strcmp(p, "wait") == 0 || starts_with(p, "wait ") || starts_with(p, "wait\t"))
    {
        char argbuf[STRINGMAXLEN_CLICMDLINE];
        strncpy(argbuf, p, STRINGMAXLEN_CLICMDLINE - 1);
        argbuf[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

        char *ptr_save = NULL;
        char *tok      = strtok_r(argbuf, " \t", &ptr_save); /* "wait" */
        tok            = strtok_r(NULL, " \t", &ptr_save);

        if (tok != NULL && strcmp(tok, "-S") == 0)
        {
            char *sname = strtok_r(NULL, " \t", &ptr_save);
            char *tmstr = strtok_r(NULL, " \t", &ptr_save);
            if (!sname)
            {
                printf("wait: missing stream name\n");
                cli_last_retval = 1;
                return 1;
            }
            double wait_timeout = tmstr ? atof(tmstr) : -1.0;

            IMAGE img;
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(sname, &img) == IMAGESTREAMIO_SUCCESS)
            {
                uint64_t        start_cnt0 = img.md->cnt0;
                struct timespec ts_start, ts_now;
                clock_gettime(CLOCK_MONOTONIC, &ts_start);
                cli_last_retval = 1;

                while (!cli_break_flag)
                {
                    if (img.md->cnt0 != start_cnt0)
                    {
                        cli_last_retval = 0;
                        break;
                    }
                    if (wait_timeout >= 0.0)
                    {
                        clock_gettime(CLOCK_MONOTONIC, &ts_now);
                        double elapsed = (double) (ts_now.tv_sec - ts_start.tv_sec) +
                                         1e-9 * (double) (ts_now.tv_nsec - ts_start.tv_nsec);
                        if (elapsed >= wait_timeout)
                        {
                            break;
                        }
                    }
                    usleep(1000);
                }
                ImageStreamIO_closeIm(&img);
            }
            else
            {
                printf("wait: stream %s not found\n", sname);
                cli_last_retval = 1;
            }
            return 1;
        }
        else if (tok != NULL && strcmp(tok, "-F") == 0)
        {
            char *fname = strtok_r(NULL, " \t", &ptr_save);
            char *pval  = strtok_r(NULL, " \t", &ptr_save);
            char *tmstr = strtok_r(NULL, " \t", &ptr_save);

            if (!fname || !pval)
            {
                printf("wait: missing fps name or param=value\n");
                cli_last_retval = 1;
                return 1;
            }

            char *eq = strchr(pval, '=');
            if (!eq)
            {
                printf("wait: require param=value format\n");
                cli_last_retval = 1;
                return 1;
            }
            *eq                      = '\0';
            const char *param        = pval;
            const char *value        = eq + 1;
            double      wait_timeout = tmstr ? atof(tmstr) : -1.0;

            FPS fps;
            if (fps_connect(fname, &fps, FPSCONNECT_SIMPLE) != -1 && fps.parray != NULL)
            {
                int pindex = functionparameter_GetParamIndex(&fps, param);
                if (pindex < 0)
                {
                    char dotname[512];
                    snprintf(dotname, sizeof(dotname), ".%s", param);
                    pindex = functionparameter_GetParamIndex(&fps, dotname);
                }

                if (pindex >= 0)
                {
                    struct timespec ts_start, ts_now;
                    clock_gettime(CLOCK_MONOTONIC, &ts_start);
                    cli_last_retval = 1;

                    while (!cli_break_flag)
                    {
                        char vstr[512];
                        functionparameter_GetParamValueString(&fps.parray[pindex], vstr,
                                                              sizeof(vstr));

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

                            if (end_vstr != vstr && *end_vstr == '\0' && end_value != value &&
                                *end_value == '\0' && dvstr == dvalue)
                            {
                                cli_last_retval = 0;
                                break;
                            }
                        }
                        if (wait_timeout >= 0.0)
                        {
                            clock_gettime(CLOCK_MONOTONIC, &ts_now);
                            double elapsed = (double) (ts_now.tv_sec - ts_start.tv_sec) +
                                             1e-9 * (double) (ts_now.tv_nsec - ts_start.tv_nsec);
                            if (elapsed >= wait_timeout)
                            {
                                break;
                            }
                        }
                        usleep(1000);
                    }
                }
                else
                {
                    printf("wait: param %s not found in %s\n", param, fname);
                    cli_last_retval = 1;
                }
                fps_disconnect(&fps);
            }
            else
            {
                printf("wait: fps %s not found\n", fname);
                cli_last_retval = 1;
            }
            return 1;
        }
        else
        {
            /* Standard wait for children */
            int wstatus;
            while (waitpid(-1, &wstatus, 0) > 0)
            {
            }
            cli_last_retval = 0;
            return 1;
        }
    }
    return 0;
}


int cli_intercept_cmd_double_bracket(const char *p)
{
    if (starts_with(p, "[[ "))
    {
        int plen = (int) strlen(p);
        if (plen >= 5 && p[plen - 1] == ']' && p[plen - 2] == ']')
        {
            /* Extract inner expr */
            char texpr[STRINGMAXLEN_CLICMDLINE];
            memcpy(texpr, p + 3, (size_t) (plen - 5));
            texpr[plen - 5] = '\0';
            /* Trim whitespace */
            int tlen = (int) strlen(texpr);
            while (tlen > 0 && (texpr[tlen - 1] == ' ' || texpr[tlen - 1] == '\t'))
            {
                texpr[--tlen] = '\0';
            }
            int result      = cli_eval_test(texpr);
            cli_last_retval = result ? 0 : 1;
        }
        return 1;
    }
    return 0;
}


int cli_intercept_cmd_if(const char *p)
{
    if (starts_with(p, "if ") || starts_with(p, "if\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_IF;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}
