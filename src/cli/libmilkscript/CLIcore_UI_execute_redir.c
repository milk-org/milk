/**
 * @file CLIcore_UI_execute_redir.c
 *
 * @brief CLI I/O redirection and process control handlers.
 *
 * Contains handlers for all shell-like redirection and
 * process control constructs supported by the milk CLI:
 *
 *  - cli_run_external()          External command runner
 *  - cli_handle_subshell()       (cmd1; cmd2) subshell
 *  - cli_handle_herestring_late() <<< pipe-based
 *  - cli_handle_stderr_redir()   2>&1, 2>file
 *  - cli_handle_input_redir()    < file / @S:stream
 *  - cli_handle_output_redir()   > / >> / @F: / @S:
 *  - cli_handle_herestring_early() <<< tmpfile-based
 *  - cli_handle_background()     cmd &
 *
 * Each handler returns 1 if it consumed the command
 * line, or 0 to pass control to the next stage.
 * When consumed, the [out] retval parameter is set.
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_UI_execute_internal.h"

extern int cli_last_retval;

#include "ImageStreamIO.h"
#include "fps_connect.h"
#include "fps_paramvalue.h"


/**
 * @brief Execute an external command with minimal overhead.
 *
 * Prefers a direct fork+exec via posix_spawnp() for simple
 * commands (no shell metacharacters), saving the extra
 * /bin/sh layer that system() would otherwise spawn.
 * Falls back to "/bin/sh -c cmd" only when metacharacters
 * are detected (pipes, redirects, glob, etc.).
 *
 * @param cmd  Fully expanded command string
 * @return  Exit status (0 = success), 127 if not found,
 *          -1 on spawn error
 */
int cli_run_external(
    const char *cmd
)
{
    extern char **environ;

    /* Detect characters that require /bin/sh parsing */
    static const char sh_meta[] =
        "|&;<>`()\\\\"
        "\"'\n*?[{$";
    int needs_shell = 0;
    for(const char *cp = cmd; *cp; cp++)
    {
        if(strchr(sh_meta, *cp))
        {
            needs_shell = 1;
            break;
        }
    }

    pid_t pid  = -1;
    int   ret  = -1;
    int   status = 0;

    if(needs_shell)
    {
        /* Shell metacharacters present:
         * spawn /bin/sh -c cmd — equivalent to
         * system() but using posix_spawn for
         * cleaner signal handling */
        char *const sh_args[] = {
            "/bin/sh", "-c",
            (char *) cmd, NULL
        };
        ret = posix_spawn(
            &pid, "/bin/sh",
            NULL, NULL,
            sh_args, environ);
    }
    else
    {
        /* Simple command: tokenize on whitespace
         * and exec directly, skipping the shell
         * layer entirely */
        char buf[STRINGMAXLEN_CLICMDLINE];
        strncpy(buf, cmd, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';

        char *argv[256];
        int   argc = 0;
        char *tok  = strtok(buf, " \t");
        while(tok != NULL && argc < 255)
        {
            argv[argc++] = tok;
            tok = strtok(NULL, " \t");
        }
        argv[argc] = NULL;

        if(argc == 0)
        {
            return 0;
        }

        ret = posix_spawnp(
            &pid, argv[0],
            NULL, NULL,
            argv, environ);
        if(ret != 0)
        {
            if(ret == ENOENT)
            {
                fprintf(stderr,
                        "milk-cli: %s: not found\n",
                        argv[0]);
                return 127; /* command not found */
            }
            else if(ret == EACCES)
            {
                fprintf(stderr,
                        "milk-cli: %s: permission"
                        " denied\n",
                        argv[0]);
                return 126; /* permission denied */
            }
            else
            {
                fprintf(stderr,
                        "milk-cli: %s: %s\n",
                        argv[0], strerror(ret));
                return 1;
            }
        }
    }

    if(ret != 0)
    {
        return -1;
    }

    if(waitpid(pid, &status, 0) == -1)
    {
        return -1;
    }

    if(WIFEXITED(status))
    {
        return WEXITSTATUS(status);
    }
    if(WIFSIGNALED(status))
    {
        return 128 + WTERMSIG(status);
    }
    return -1;
}


/**
 * @brief Handle subshell execution: (cmd1; cmd2)
 *
 * If the command line is entirely wrapped in parens,
 * forks a child that sequentially executes each
 * semicolon-separated sub-command, then waits for
 * completion.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_subshell(
    errno_t *retval
)
{
    const char *sp = data.CLIcmdline;
    int spl = (int) strlen(sp);
    if (spl < 3
        || sp[0] != '('
        || sp[spl - 1] != ')')
    {
        return 0;
    }
    char sbuf[STRINGMAXLEN_CLICMDLINE];
    memcpy(sbuf, sp + 1,
           (size_t)(spl - 2));
    sbuf[spl - 2] = '\0';
    pid_t spid = fork();
    if (spid == 0)
    {
        char *tok = strtok(sbuf, ";");
        while (tok != NULL)
        {
            const char *st = tok;
            while (*st == ' '
                   || *st == '\t')
            {
                st++;
            }
            if (*st != '\0')
            {
                CLI_execute_string(
                    (char *) st);
            }
            tok = strtok(NULL, ";");
        }
        _exit(0);
    }
    else if (spid > 0)
    {
        int wst;
        waitpid(spid, &wst, 0);
        cli_last_retval =
            WEXITSTATUS(wst);
    }
    *retval = 0;
    return 1;
}


/**
 * @brief Handle late here-string (pipe-based).
 *
 * Alternative here-string handler that uses pipe()
 * instead of tmpfile() for feeding text to stdin.
 * Called after output redirection has been processed.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_herestring_late(
    errno_t *retval
)
{
    const char *hs = strstr(
        data.CLIcmdline, "<<<");
    if (hs == NULL)
    {
        return 0;
    }

    char hcmd[STRINGMAXLEN_CLICMDLINE];
    int hcl = (int)(hs - data.CLIcmdline);
    if (hcl >= STRINGMAXLEN_CLICMDLINE)
    {
        hcl = STRINGMAXLEN_CLICMDLINE - 1;
    }
    memcpy(hcmd, data.CLIcmdline,
           (size_t) hcl);
    hcmd[hcl] = '\0';
    while (hcl > 0
           && (hcmd[hcl - 1] == ' '
               || hcmd[hcl - 1] == '\t'))
    {
        hcmd[--hcl] = '\0';
    }
    const char *tp = hs + 3;
    while (*tp == ' ' || *tp == '\t')
    {
        tp++;
    }
    char htxt[1024];
    strncpy(htxt, tp, sizeof(htxt) - 1);
    htxt[sizeof(htxt) - 1] = '\0';
    int htl = (int) strlen(htxt);
    if (htl >= 2
        && ((htxt[0] == '"'
             && htxt[htl - 1] == '"')
            || (htxt[0] == '\''
                && htxt[htl - 1] == '\'')))
    {
        htxt[htl - 1] = '\0';
        memmove(htxt, htxt + 1,
                (size_t)(htl - 1));
    }
    int pfd[2];
    if (pipe(pfd) == 0)
    {
        ssize_t wr_ignore;
        wr_ignore = write(pfd[1], htxt,
                          strlen(htxt));
        wr_ignore = write(pfd[1], "\n", 1);
        (void) wr_ignore;
        close(pfd[1]);
        int sv = dup(STDIN_FILENO);
        dup2(pfd[0], STDIN_FILENO);
        close(pfd[0]);
        CLI_execute_string(hcmd);
        dup2(sv, STDIN_FILENO);
        close(sv);
    }
    *retval = 0;
    return 1;
}


/**
 * @brief Handle stderr redirection (2>&1, 2>file).
 *
 * Scans for "2>" in the command line, extracts the
 * target (either "&1" to merge to stdout or a filename),
 * sets up the redirect, executes the command, then
 * restores stderr.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_stderr_redir(
    errno_t *retval
)
{
    const char *se = strstr(
        data.CLIcmdline, "2>");
    if (se == NULL)
    {
        return 0;
    }

    char scmd[STRINGMAXLEN_CLICMDLINE];
    int scl = (int)(se - data.CLIcmdline);
    if (scl >= STRINGMAXLEN_CLICMDLINE)
    {
        scl = STRINGMAXLEN_CLICMDLINE - 1;
    }
    memcpy(scmd, data.CLIcmdline,
           (size_t) scl);
    scmd[scl] = '\0';
    while (scl > 0
           && (scmd[scl - 1] == ' '
               || scmd[scl - 1] == '\t'))
    {
        scmd[--scl] = '\0';
    }
    const char *target = se + 2;
    while (*target == ' '
           || *target == '\t')
    {
        target++;
    }
    int sv_err = dup(STDERR_FILENO);
    if (strncmp(target, "&1", 2) == 0)
    {
        dup2(STDOUT_FILENO, STDERR_FILENO);
    }
    else
    {
        char fname[256];
        int fi = 0;
        while (target[fi] != '\0'
               && target[fi] != ' '
               && target[fi] != '\t'
               && fi < 254)
        {
            fname[fi] = target[fi];
            fi++;
        }
        fname[fi] = '\0';
        FILE *ef = fopen(fname, "w");
        if (ef != NULL)
        {
            dup2(fileno(ef),
                 STDERR_FILENO);
            fclose(ef);
        }
    }
    CLI_execute_string(scmd);
    dup2(sv_err, STDERR_FILENO);
    close(sv_err);
    *retval = 0;
    return 1;
}


/**
 * @brief Handle input redirection (cmd < file).
 *
 * Scans for unquoted < (not << or <<<), redirects
 * stdin from the named file, and re-executes.
 * Supports @S:streamname to read from a SHM stream.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_input_redir(
    errno_t *retval
)
{
    const char *cl4 = data.CLIcmdline;
    int in_sq4 = 0, in_dq4 = 0, depth4 = 0;
    int inr_pos = -1;

    for (int ri = 0; cl4[ri] != '\0'; ri++)
    {
        if (cl4[ri] == '\'' && !in_dq4)
        {
            in_sq4 = !in_sq4;
        }
        else if (cl4[ri] == '"' && !in_sq4)
        {
            in_dq4 = !in_dq4;
        }
        else if (!in_sq4 && !in_dq4)
        {
            if (cl4[ri] == '(')
            {
                depth4++;
            }
            else if (cl4[ri] == ')'
                     && depth4 > 0)
            {
                depth4--;
            }
            else if (depth4 == 0
                     && cl4[ri] == '<'
                     && cl4[ri + 1] != '<')
            {
                inr_pos = ri;
                break;
            }
        }
    }
    if (inr_pos < 0)
    {
        return 0;
    }

    int fst = inr_pos + 1;
    while (data.CLIcmdline[fst] == ' '
           || data.CLIcmdline[fst] == '\t')
    {
        fst++;
    }
    char infile[512];
    int ifi = 0;
    while (data.CLIcmdline[fst] != '\0'
           && data.CLIcmdline[fst] != ' '
           && data.CLIcmdline[fst] != '\t'
           && ifi < 511)
    {
        infile[ifi++] =
            data.CLIcmdline[fst++];
    }
    infile[ifi] = '\0';

    data.CLIcmdline[inr_pos] = '\0';
    {
        int cl5 = inr_pos - 1;
        while (cl5 >= 0
               && (data.CLIcmdline[cl5] == ' '
                   || data.CLIcmdline[cl5]
                   == '\t'))
        {
            data.CLIcmdline[cl5--] = '\0';
        }
    }

    FILE *ifp = NULL;
    int is_stream = 0;
    char tempname[256] = "";

    if (strncmp(infile, "@S:", 3) == 0)
    {
        is_stream = 1;
        char *sname = infile + 3;
        IMAGE *img =
            (IMAGE *) malloc(sizeof(IMAGE));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(
                sname, img) == 0)
        {
            snprintf(tempname,
                     sizeof(tempname),
                     "/tmp/milk_cli_inredir_XXXXXX");
            int fd = mkstemp(tempname);
            if (fd >= 0)
            {
                FILE *tf = fdopen(fd, "w");
                if (tf)
                {
                    int typesize =
                        ImageStreamIO_typesize(
                            img->md->datatype);
                    if (typesize > 0)
                    {
                        size_t bytes =
                            (size_t) typesize
                            * img->md->nelement;
                        fwrite(img->array.raw,
                               1, bytes, tf);
                        fclose(tf);
                        ifp = fopen(tempname, "r");
                        unlink(tempname);
                    }
                    else
                    {
                        fprintf(stderr,
                                "stream redirection: "
                                "invalid datatype for "
                                "stream %s\n",
                                sname);
                        fclose(tf);
                        unlink(tempname);
                    }
                }
                else
                {
                    close(fd);
                }
            }
            ImageStreamIO_closeIm(img);
        }
        else
        {
            printf(
                "stream redirection: "
                "stream %s not found\n",
                sname);
        }
        free(img);
    }
    else
    {
        ifp = fopen(infile, "r");
    }

    if (ifp != NULL)
    {
        int sv_in = dup(STDIN_FILENO);
        dup2(fileno(ifp), STDIN_FILENO);

        *retval = CLI_execute_line();

        dup2(sv_in, STDIN_FILENO);
        close(sv_in);
        fclose(ifp);

        return 1;
    }
    else
    {
        if (is_stream && tempname[0] != '\0')
        {
            unlink(tempname);
        }
        /* Redirect was detected but file open
         * failed. Command line was modified —
         * treat as consumed and signal failure. */
        *retval = RETURN_FAILURE;
        return 1;
    }
}


/**
 * @brief Handle output redirection (> and >>).
 *
 * Scans for unquoted > or >> outside parens/quotes,
 * extracts the filename, redirects stdout, executes
 * the truncated command, then restores stdout.
 *
 * Supports stream targets (@S:name) and FPS parameter
 * targets (@F:fpsname.param) in addition to regular
 * file paths.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_output_redir(
    errno_t *retval
)
{
    int redir_mode = 0; /* 1=trunc 2=append */
    int redir_pos = -1;
    int in_sq2 = 0, in_dq2 = 0, depth2 = 0;
    const char *cl2 = data.CLIcmdline;

    for (int ri = 0; cl2[ri] != '\0'; ri++)
    {
        if (cl2[ri] == '\'' && !in_dq2)
        {
            in_sq2 = !in_sq2;
        }
        else if (cl2[ri] == '"' && !in_sq2)
        {
            in_dq2 = !in_dq2;
        }
        else if (!in_sq2 && !in_dq2)
        {
            if (cl2[ri] == '(')
            {
                depth2++;
            }
            else if (cl2[ri] == ')'
                     && depth2 > 0)
            {
                depth2--;
            }
            else if (depth2 == 0
                     && cl2[ri] == '>')
            {
                if (cl2[ri + 1] == '>')
                {
                    redir_mode = 2;
                }
                else
                {
                    redir_mode = 1;
                }
                redir_pos = ri;
            }
        }
    }
    if (redir_pos < 0)
    {
        return 0;
    }

    int fstart = redir_pos
                 + ((redir_mode == 2) ? 2 : 1);
    while (data.CLIcmdline[fstart] == ' '
           || data.CLIcmdline[fstart] == '\t')
    {
        fstart++;
    }
    char rfile[512];
    int fi = 0;
    while (data.CLIcmdline[fstart] != '\0'
           && data.CLIcmdline[fstart] != ' '
           && data.CLIcmdline[fstart] != '\t'
           && fi < 511)
    {
        rfile[fi++] =
            data.CLIcmdline[fstart++];
    }
    rfile[fi] = '\0';

    /* Truncate cmd at redir position */
    data.CLIcmdline[redir_pos] = '\0';
    {
        int cl3 = redir_pos - 1;
        while (cl3 >= 0
               && (data.CLIcmdline[cl3] == ' '
                   || data.CLIcmdline[cl3]
                   == '\t'))
        {
            data.CLIcmdline[cl3--] = '\0';
        }
    }

    FILE *rfp = NULL;
    int is_fps = 0;
    int is_stream = 0;
    char tempname[256];

    if (strncmp(rfile, "@F:", 3) == 0)
    {
        is_fps = 1;
        snprintf(tempname, sizeof(tempname),
                 "/tmp/milk_cli_fredir_XXXXXX");
        int fd = mkstemp(tempname);
        if (fd >= 0)
        {
            rfp = fdopen(fd,
                         (redir_mode == 2)
                         ? "a" : "w");
            if (!rfp) close(fd);
        }
    }
    else if (strncmp(rfile, "@S:", 3) == 0)
    {
        is_stream = 1;
        snprintf(tempname, sizeof(tempname),
                 "/tmp/milk_cli_sredir_XXXXXX");
        int fd = mkstemp(tempname);
        if (fd >= 0)
        {
            rfp = fdopen(fd,
                         (redir_mode == 2)
                         ? "a" : "w");
            if (!rfp) close(fd);
        }
    }
    else
    {
        rfp = fopen(rfile,
                    (redir_mode == 2)
                    ? "a" : "w");
    }

    if (rfp != NULL)
    {
        fflush(stdout);
        int sv_out = dup(STDOUT_FILENO);
        dup2(fileno(rfp), STDOUT_FILENO);

        *retval = CLI_execute_line();

        fflush(stdout);
        dup2(sv_out, STDOUT_FILENO);
        close(sv_out);
        fclose(rfp);

        if (is_fps)
        {
            /* Write captured value to FPS param */
            char *fpspath = rfile + 3;
            char *dot = strchr(fpspath, '.');
            if (dot != NULL)
            {
                *dot = '\0';
                char *fpsname = fpspath;
                char *param = dot + 1;

                FILE *tf = fopen(tempname, "r");
                if (tf)
                {
                    char valbuf[2048] = {0};
                    size_t rn = fread(
                        valbuf, 1,
                        sizeof(valbuf) - 1, tf);
                    valbuf[rn] = '\0';
                    fclose(tf);

                    while(rn > 0
                          && (valbuf[rn - 1] == '\n'
                              || valbuf[rn - 1]
                              == '\r'))
                    {
                        valbuf[--rn] = '\0';
                    }

                    FUNCTION_PARAMETER_STRUCT fps_s;
                    if (function_parameter_struct_connect(
                            fpsname, &fps_s,
                            FPSCONNECT_SIMPLE) != -1
                        && fps_s.parray != NULL)
                    {
                        int pidx =
                            functionparameter_GetParamIndex(
                                &fps_s, param);
                        if (pidx < 0)
                        {
                            char dotname[512];
                            snprintf(dotname,
                                     sizeof(dotname),
                                     ".%s", param);
                            pidx =
                                functionparameter_GetParamIndex(
                                    &fps_s, dotname);
                        }
                        if (pidx >= 0)
                        {
                            const char *kw =
                                fps_s.parray[pidx]
                                .keyword[0];
                            switch (fps_s.parray[pidx].type)
                            {
                                case FPTYPE_INT32:
                                    functionparameter_SetParamValue_INT32(
                                        &fps_s, kw,
                                        strtol(valbuf, NULL, 10));
                                    break;
                                case FPTYPE_UINT32:
                                    functionparameter_SetParamValue_UINT32(
                                        &fps_s, kw,
                                        strtoul(valbuf, NULL, 10));
                                    break;
                                case FPTYPE_INT64:
                                case FPTYPE_PID:
                                    functionparameter_SetParamValue_INT64(
                                        &fps_s, kw,
                                        strtoll(valbuf, NULL, 10));
                                    break;
                                case FPTYPE_UINT64:
                                    functionparameter_SetParamValue_UINT64(
                                        &fps_s, kw,
                                        strtoull(valbuf, NULL, 10));
                                    break;
                                case FPTYPE_FLOAT32:
                                    functionparameter_SetParamValue_FLOAT32(
                                        &fps_s, kw,
                                        strtof(valbuf, NULL));
                                    break;
                                case FPTYPE_FLOAT64:
                                    functionparameter_SetParamValue_FLOAT64(
                                        &fps_s, kw,
                                        strtod(valbuf, NULL));
                                    break;
                                case FPTYPE_TIMESPEC:
                                    functionparameter_SetParamValue_TIMESPEC(
                                        &fps_s, kw,
                                        strtof(valbuf, NULL));
                                    break;
                                case FPTYPE_ONOFF:
                                    if (strcasecmp(valbuf, "ON") == 0
                                        || strcmp(valbuf, "1") == 0)
                                    {
                                        functionparameter_SetParamValue_ONOFF(
                                            &fps_s, kw, 1);
                                    }
                                    else if (
                                        strcasecmp(valbuf, "OFF") == 0
                                        || strcmp(valbuf, "0") == 0)
                                    {
                                        functionparameter_SetParamValue_ONOFF(
                                            &fps_s, kw, 0);
                                    }
                                    break;
                                default:
                                    functionparameter_SetParamValue_STRING(
                                        &fps_s, kw, valbuf);
                                    break;
                            }
                        }
                        else
                        {
                            printf(
                                "fps redirection: "
                                "param %s not found"
                                " in %s\n",
                                param, fpsname);
                        }
                        function_parameter_struct_disconnect(
                            &fps_s);
                    }
                    else
                    {
                        printf(
                            "fps redirection: "
                            "could not connect"
                            " to %s\n",
                            fpsname);
                    }
                }
            }
            else
            {
                printf(
                    "fps redirection: format"
                    " must be @F:fpsname.param\n");
            }
            unlink(tempname);
        }
        else if (is_stream)
        {
            /* Write captured bytes to SHM stream */
            char *sname = rfile + 3;
            IMAGE *img =
                (IMAGE *) malloc(sizeof(IMAGE));
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(
                    sname, img) == 0)
            {
                int do_update = 0;
                FILE *tf = fopen(tempname, "r");
                if (tf)
                {
                    int typesize =
                        ImageStreamIO_typesize(
                            img->md->datatype);
                    if (typesize > 0)
                    {
                        size_t bytes =
                            (size_t) typesize
                            * img->md->nelement;
                        size_t bread = fread(
                            img->array.raw, 1,
                            bytes, tf);
                        (void)bread;
                        do_update = 1;
                    }
                    else
                    {
                        printf(
                            "stream redirection: "
                            "invalid datatype for "
                            "stream %s\n",
                            sname);
                    }
                    fclose(tf);
                }
                if (do_update)
                {
                    ImageStreamIO_UpdateIm(img);
                }
                ImageStreamIO_closeIm(img);
            }
            else
            {
                printf(
                    "stream redirection: "
                    "stream %s not found\n",
                    sname);
            }
            free(img);
            unlink(tempname);
        }
        return 1;
    }
    else
    {
        if ((is_fps || is_stream)
            && tempname[0] != '\0')
        {
            unlink(tempname);
        }
    }
    return 0;
}


/**
 * @brief Handle here-string syntax (<<<).
 *
 * Parses "cmd <<< text", strips quotes from text,
 * writes it to a tmpfile, redirects stdin, and
 * executes the command. This is the "early" handler
 * that runs before background and subshell processing.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_herestring_early(
    errno_t *retval
)
{
    char *hs = strstr(
        data.CLIcmdline, "<<<");
    if (hs == NULL)
    {
        return 0;
    }

    *hs = '\0';
    const char *hsval = hs + 3;
    while (*hsval == ' ' || *hsval == '\t')
    {
        hsval++;
    }
    int hvlen = (int) strlen(hsval);
    char hvbuf[STRINGMAXLEN_CLICMDLINE];
    if (hvlen >= 2
        && ((hsval[0] == '"'
             && hsval[hvlen - 1] == '"')
            || (hsval[0] == '\''
                && hsval[hvlen - 1] == '\'')))
    {
        memcpy(hvbuf, hsval + 1,
               (size_t)(hvlen - 2));
        hvbuf[hvlen - 2] = '\0';
    }
    else
    {
        strncpy(hvbuf, hsval,
                STRINGMAXLEN_CLICMDLINE - 1);
        hvbuf[STRINGMAXLEN_CLICMDLINE - 1] =
            '\0';
    }

    FILE *hsfp = tmpfile();
    if (hsfp != NULL)
    {
        fprintf(hsfp, "%s\n", hvbuf);
        rewind(hsfp);
        int sv_in = dup(STDIN_FILENO);
        dup2(fileno(hsfp), STDIN_FILENO);

        *retval = CLI_execute_line();

        dup2(sv_in, STDIN_FILENO);
        close(sv_in);
        fclose(hsfp);
        return 1;
    }
    return 0;
}


/**
 * @brief Handle background execution (cmd &).
 *
 * If the command ends with an unquoted trailing &
 * (not &&), forks a child process to execute the
 * command and returns immediately. Sets the $!
 * variable to the child's PID.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_handle_background(
    errno_t *retval
)
{
    int ll = (int) strlen(data.CLIcmdline);
    int bi = ll - 1;
    while (bi >= 0
           && (data.CLIcmdline[bi] == ' '
               || data.CLIcmdline[bi] == '\t'))
    {
        bi--;
    }
    if (bi < 0
        || data.CLIcmdline[bi] != '&'
        || (bi > 0
            && data.CLIcmdline[bi - 1] == '&'))
    {
        return 0;
    }

    data.CLIcmdline[bi] = '\0';
    pid_t cpid = fork();
    if (cpid == 0)
    {
        CLI_execute_string(data.CLIcmdline);
        _exit(0);
    }
    else if (cpid > 0)
    {
        printf("[bg] %d\n", (int) cpid);
        char pb[32];
        snprintf(pb, sizeof(pb),
                 "%d", (int) cpid);
        cli_var_set("!", pb);
    }
    *retval = 0;
    return 1;
}
