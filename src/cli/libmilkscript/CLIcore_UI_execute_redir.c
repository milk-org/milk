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

/* Local color macros (plain escapes — no readline wrap) */
#ifndef COLORDIMYELLOW
#    define COLORDIMYELLOW "\033[2;33m"
#endif
#ifndef COLORRST
#    define COLORRST "\033[0m"
#endif

extern int cli_last_retval;

#include "ImageStreamIO.h"
#include "fps_connect.h"
#include "fps_paramvalue.h"


/**
 * cli_find_unquoted_op - find the first unquoted,
 *      un-nested occurrence of an operator character.
 * @line:      NUL-terminated command string to scan
 * @primary:   character to match (e.g. '<', '>', '|')
 * @reject:    if non-zero, skip when line[i+1] == reject
 *             (e.g. reject='<' skips '<<')
 * @accept:    if non-zero, only match when
 *             line[i+1] == accept (e.g. accept='>'
 *             matches '|>' but not '|' alone).
 *             Mutually exclusive with @reject.
 *
 * Tracks single-quote, double-quote, and parenthesis
 * depth so that operators inside quotes or subshells
 * are ignored.
 *
 * Returns the index of the matched character, or -1
 * if no unquoted match was found.
 */
int cli_find_unquoted_op(const char *line, char primary, char reject, char accept)
{
    int in_sq = 0;
    int in_dq = 0;
    int depth = 0;

    for (int i = 0; line[i] != '\0'; i++)
    {
        char c = line[i];
        if (c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
        }
        else if (c == '"' && !in_sq)
        {
            in_dq = !in_dq;
        }
        else if (!in_sq && !in_dq)
        {
            if (c == '(')
            {
                depth++;
            }
            else if (c == ')' && depth > 0)
            {
                depth--;
            }
            else if (depth == 0 && c == primary)
            {
                if (reject != 0 && line[i + 1] == reject)
                {
                    continue;
                }
                if (accept != 0 && line[i + 1] != accept)
                {
                    continue;
                }
                return i;
            }
        }
    }
    return -1;
}


/**
 * @brief Helper to safely create and open a temporary file
 */
static FILE *cli_mkstemp_open(char *namebuf, size_t sz, const char *prefix, const char *mode)
{
    snprintf(namebuf, sz, "%s_XXXXXX", prefix);
    int fd = mkstemp(namebuf);
    if (fd >= 0)
    {
        FILE *tf = fdopen(fd, mode);
        if (tf)
        {
            return tf;
        }
        close(fd);
        unlink(namebuf);
    }
    return NULL;
}

/**
 * @brief Helper to safely redirect a file descriptor and save it
 */
static void cli_fd_redirect(int target_fd, int new_fd, int *saved_fd)
{
    *saved_fd = dup(target_fd);
    dup2(new_fd, target_fd);
}

/**
 * @brief Helper to restore a saved file descriptor
 */
static void cli_fd_restore(int target_fd, int saved_fd)
{
    dup2(saved_fd, target_fd);
    close(saved_fd);
}


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
int cli_run_external(const char *cmd)
{
    extern char **environ;

    /* Detect characters that require /bin/sh parsing */
    static const char sh_meta[]   = "|&;<>`()\\\\"
                                    "\"'\n*?[{$";
    int               needs_shell = 0;
    for (const char *cp = cmd; *cp; cp++)
    {
        if (strchr(sh_meta, *cp))
        {
            needs_shell = 1;
            break;
        }
    }

    pid_t pid    = -1;
    int   ret    = -1;
    int   status = 0;

    if (needs_shell)
    {
        /* Shell metacharacters present:
         * spawn /bin/sh -c cmd — equivalent to
         * system() but using posix_spawn for
         * cleaner signal handling */
        char *const sh_args[] = { "/bin/sh", "-c", (char *) cmd, NULL };
        ret                   = posix_spawn(&pid, "/bin/sh", NULL, NULL, sh_args, environ);
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
        while (tok != NULL && argc < 255)
        {
            argv[argc++] = tok;
            tok          = strtok(NULL, " \t");
        }
        argv[argc] = NULL;

        if (argc == 0)
        {
            return 0;
        }

        ret = posix_spawnp(&pid, argv[0], NULL, NULL, argv, environ);
        if (ret != 0)
        {
            if (ret == ENOENT)
            {
                fprintf(stderr, "milk-cli: %s: not found\n", argv[0]);
                return 127; /* command not found */
            }
            else if (ret == EACCES)
            {
                fprintf(stderr,
                        "milk-cli: %s: permission"
                        " denied\n",
                        argv[0]);
                return 126; /* permission denied */
            }
            else
            {
                fprintf(stderr, "milk-cli: %s: %s\n", argv[0], strerror(ret));
                return 1;
            }
        }
    }

    if (ret != 0)
    {
        return -1;
    }

    if (waitpid(pid, &status, 0) == -1)
    {
        return -1;
    }

    if (WIFEXITED(status))
    {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status))
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
int cli_handle_subshell(errno_t *retval)
{
    const char *sp  = data.CLIcmdline;
    int         spl = (int) strlen(sp);
    if (spl < 3 || sp[0] != '(' || sp[spl - 1] != ')')
    {
        return 0;
    }
    char sbuf[STRINGMAXLEN_CLICMDLINE];
    memcpy(sbuf, sp + 1, (size_t) (spl - 2));
    sbuf[spl - 2] = '\0';
    pid_t spid    = fork();
    if (spid == 0)
    {
        char *tok = strtok(sbuf, ";");
        while (tok != NULL)
        {
            const char *st = tok;
            while (*st == ' ' || *st == '\t')
            {
                st++;
            }
            if (*st != '\0')
            {
                CLI_execute_string((char *) st);
            }
            tok = strtok(NULL, ";");
        }
        _exit(0);
    }
    else if (spid > 0)
    {
        int wst;
        waitpid(spid, &wst, 0);
        cli_last_retval = WEXITSTATUS(wst);
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
int cli_handle_herestring_late(errno_t *retval)
{
    const char *hs = strstr(data.CLIcmdline, "<<<");
    if (hs == NULL)
    {
        return 0;
    }

    char hcmd[STRINGMAXLEN_CLICMDLINE];
    int  hcl = (int) (hs - data.CLIcmdline);
    if (hcl >= STRINGMAXLEN_CLICMDLINE)
    {
        hcl = STRINGMAXLEN_CLICMDLINE - 1;
    }
    memcpy(hcmd, data.CLIcmdline, (size_t) hcl);
    hcmd[hcl] = '\0';
    while (hcl > 0 && (hcmd[hcl - 1] == ' ' || hcmd[hcl - 1] == '\t'))
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
    int htl                = (int) strlen(htxt);
    if (htl >= 2 &&
        ((htxt[0] == '"' && htxt[htl - 1] == '"') || (htxt[0] == '\'' && htxt[htl - 1] == '\'')))
    {
        htxt[htl - 1] = '\0';
        memmove(htxt, htxt + 1, (size_t) (htl - 1));
    }
    int pfd[2];
    if (pipe(pfd) == 0)
    {
        ssize_t wr_ignore;
        wr_ignore = write(pfd[1], htxt, strlen(htxt));
        wr_ignore = write(pfd[1], "\n", 1);
        (void) wr_ignore;
        close(pfd[1]);
        int sv;
        cli_fd_redirect(STDIN_FILENO, pfd[0], &sv);
        close(pfd[0]);
        CLI_execute_string(hcmd);
        cli_fd_restore(STDIN_FILENO, sv);
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
int cli_handle_stderr_redir(errno_t *retval)
{
    const char *se = strstr(data.CLIcmdline, "2>");
    if (se == NULL)
    {
        return 0;
    }

    char scmd[STRINGMAXLEN_CLICMDLINE];
    int  scl = (int) (se - data.CLIcmdline);
    if (scl >= STRINGMAXLEN_CLICMDLINE)
    {
        scl = STRINGMAXLEN_CLICMDLINE - 1;
    }
    memcpy(scmd, data.CLIcmdline, (size_t) scl);
    scmd[scl] = '\0';
    while (scl > 0 && (scmd[scl - 1] == ' ' || scmd[scl - 1] == '\t'))
    {
        scmd[--scl] = '\0';
    }
    const char *target = se + 2;
    while (*target == ' ' || *target == '\t')
    {
        target++;
    }
    int sv_err = -1;
    if (strncmp(target, "&1", 2) == 0)
    {
        cli_fd_redirect(STDERR_FILENO, STDOUT_FILENO, &sv_err);
    }
    else
    {
        char fname[256];
        int  fi = 0;
        while (target[fi] != '\0' && target[fi] != ' ' && target[fi] != '\t' && fi < 254)
        {
            fname[fi] = target[fi];
            fi++;
        }
        fname[fi] = '\0';
        FILE *ef  = fopen(fname, "w");
        if (ef != NULL)
        {
            cli_fd_redirect(STDERR_FILENO, fileno(ef), &sv_err);
            fclose(ef);
        }
    }
    CLI_execute_string(scmd);
    if (sv_err != -1)
    {
        cli_fd_restore(STDERR_FILENO, sv_err);
    }
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
int cli_handle_input_redir(errno_t *retval)
{
    int inr_pos = cli_find_unquoted_op(data.CLIcmdline, '<', '<', 0);
    if (inr_pos < 0)
    {
        return 0;
    }

    int fst = inr_pos + 1;
    while (data.CLIcmdline[fst] == ' ' || data.CLIcmdline[fst] == '\t')
    {
        fst++;
    }
    char infile[512];
    int  ifi = 0;
    while (data.CLIcmdline[fst] != '\0' && data.CLIcmdline[fst] != ' ' &&
           data.CLIcmdline[fst] != '\t' && ifi < 511)
    {
        infile[ifi++] = data.CLIcmdline[fst++];
    }
    infile[ifi] = '\0';

    data.CLIcmdline[inr_pos] = '\0';
    {
        int cl5 = inr_pos - 1;
        while (cl5 >= 0 && (data.CLIcmdline[cl5] == ' ' || data.CLIcmdline[cl5] == '\t'))
        {
            data.CLIcmdline[cl5--] = '\0';
        }
    }

    FILE *ifp           = NULL;
    int   is_stream     = 0;
    char  tempname[256] = "";

    if (strncmp(infile, "@S:", 3) == 0)
    {
        is_stream    = 1;
        char  *sname = infile + 3;
        IMAGE *img   = (IMAGE *) malloc(sizeof(IMAGE));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(sname, img) == 0)
        {
            FILE *tf = cli_mkstemp_open(tempname, sizeof(tempname), "/tmp/milk_cli_inredir", "w");
            if (tf)
            {
                int typesize = ImageStreamIO_typesize(img->md->datatype);
                if (typesize > 0)
                {
                    size_t bytes = (size_t) typesize * img->md->nelement;
                    fwrite(img->array.raw, 1, bytes, tf);
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
            ImageStreamIO_closeIm(img);
        }
        else
        {
            printf("stream redirection: "
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
        int sv_in;
        cli_fd_redirect(STDIN_FILENO, fileno(ifp), &sv_in);

        *retval = CLI_execute_line();

        cli_fd_restore(STDIN_FILENO, sv_in);
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
 * cli_redir_fps_writeback - write captured stdout
 *      text to an FPS parameter.
 * @rfile:    redirect target (e.g. "@F:fps.param")
 * @tempname: path to the captured-output tmpfile
 *
 * Reads the tmpfile, strips trailing newlines,
 * connects to the FPS, and sets the parameter
 * from the string value.
 */
static void cli_redir_fps_writeback(char *rfile, const char *tempname)
{
    char *fpspath = rfile + 3;
    char *dot     = strchr(fpspath, '.');
    if (dot == NULL)
    {
        printf("fps redirection: format"
               " must be @F:fpsname.param\n");
        unlink(tempname);
        return;
    }

    *dot          = '\0';
    char *fpsname = fpspath;
    char *param   = dot + 1;

    FILE *tf = fopen(tempname, "r");
    if (tf == NULL)
    {
        unlink(tempname);
        return;
    }

    char   valbuf[2048] = { 0 };
    size_t rn           = fread(valbuf, 1, sizeof(valbuf) - 1, tf);
    valbuf[rn]          = '\0';
    fclose(tf);

    while (rn > 0 && (valbuf[rn - 1] == '\n' || valbuf[rn - 1] == '\r'))
    {
        valbuf[--rn] = '\0';
    }

    FPS fps_s;
    if (fps_connect(fpsname, &fps_s, FPSCONNECT_SIMPLE) == -1 || fps_s.parray == NULL)
    {
        printf("fps redirection: "
               "could not connect to %s\n",
               fpsname);
        unlink(tempname);
        return;
    }

    int pidx = functionparameter_GetParamIndex(&fps_s, param);
    if (pidx < 0)
    {
        char dotname[512];
        snprintf(dotname, sizeof(dotname), ".%s", param);
        pidx = functionparameter_GetParamIndex(&fps_s, dotname);
    }
    if (pidx >= 0)
    {
        functionparameter_SetParamValue_fromString(&fps_s, pidx, valbuf);
    }
    else
    {
        printf("fps redirection: "
               "param %s not found"
               " in %s\n",
               param, fpsname);
    }
    fps_disconnect(&fps_s);
    unlink(tempname);
}


/**
 * cli_redir_stream_writeback - write captured stdout
 *      bytes into a SHM stream.
 * @rfile:    redirect target (e.g. "@S:stream")
 * @tempname: path to the captured-output tmpfile
 *
 * Opens the tmpfile, connects to the SHM stream,
 * reads raw bytes into the image array, and posts
 * a semaphore update.
 */
static void cli_redir_stream_writeback(const char *rfile, const char *tempname)
{
    char  *sname = (char *) (rfile + 3);
    IMAGE *img   = (IMAGE *) malloc(sizeof(IMAGE));
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(sname, img) == 0)
    {
        int   do_update = 0;
        FILE *tf        = fopen(tempname, "r");
        if (tf)
        {
            int typesize = ImageStreamIO_typesize(img->md->datatype);
            if (typesize > 0)
            {
                size_t bytes = (size_t) typesize * img->md->nelement;
                size_t bread = fread(img->array.raw, 1, bytes, tf);
                (void) bread;
                do_update = 1;
            }
            else
            {
                printf("stream redirection: "
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
        printf("stream redirection: "
               "stream %s not found\n",
               sname);
    }
    free(img);
    unlink(tempname);
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
int cli_handle_output_redir(errno_t *retval)
{
    int redir_pos = cli_find_unquoted_op(data.CLIcmdline, '>', 0, 0);
    if (redir_pos < 0)
    {
        return 0;
    }

    /* 1=truncate, 2=append (>>) */
    int redir_mode = (data.CLIcmdline[redir_pos + 1] == '>') ? 2 : 1;

    int fstart = redir_pos + ((redir_mode == 2) ? 2 : 1);
    while (data.CLIcmdline[fstart] == ' ' || data.CLIcmdline[fstart] == '\t')
    {
        fstart++;
    }
    char rfile[512];
    int  fi = 0;
    while (data.CLIcmdline[fstart] != '\0' && data.CLIcmdline[fstart] != ' ' &&
           data.CLIcmdline[fstart] != '\t' && fi < 511)
    {
        rfile[fi++] = data.CLIcmdline[fstart++];
    }
    rfile[fi] = '\0';

    /* Truncate cmd at redir position */
    data.CLIcmdline[redir_pos] = '\0';
    {
        int cl3 = redir_pos - 1;
        while (cl3 >= 0 && (data.CLIcmdline[cl3] == ' ' || data.CLIcmdline[cl3] == '\t'))
        {
            data.CLIcmdline[cl3--] = '\0';
        }
    }

    FILE       *rfp           = NULL;
    int         is_fps        = 0;
    int         is_stream     = 0;
    char        tempname[256] = "";
    const char *fmode         = (redir_mode == 2) ? "a" : "w";

    if (strncmp(rfile, "@F:", 3) == 0)
    {
        is_fps = 1;
        rfp    = cli_mkstemp_open(tempname, sizeof(tempname), "/tmp/milk_cli_fredir", fmode);
    }
    else if (strncmp(rfile, "@S:", 3) == 0)
    {
        is_stream = 1;
        rfp       = cli_mkstemp_open(tempname, sizeof(tempname), "/tmp/milk_cli_sredir", fmode);
    }
    else
    {
        rfp = fopen(rfile, fmode);
    }

    if (rfp != NULL)
    {
        fflush(stdout);
        int sv_out;
        cli_fd_redirect(STDOUT_FILENO, fileno(rfp), &sv_out);

        *retval = CLI_execute_line();

        fflush(stdout);
        cli_fd_restore(STDOUT_FILENO, sv_out);
        fclose(rfp);

        if (is_fps)
        {
            cli_redir_fps_writeback(rfile, tempname);
        }
        else if (is_stream)
        {
            cli_redir_stream_writeback(rfile, tempname);
        }
        return 1;
    }
    else
    {
        if ((is_fps || is_stream) && tempname[0] != '\0')
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
int cli_handle_herestring_early(errno_t *retval)
{
    char *hs = strstr(data.CLIcmdline, "<<<");
    if (hs == NULL)
    {
        return 0;
    }

    *hs               = '\0';
    const char *hsval = hs + 3;
    while (*hsval == ' ' || *hsval == '\t')
    {
        hsval++;
    }
    int  hvlen = (int) strlen(hsval);
    char hvbuf[STRINGMAXLEN_CLICMDLINE];
    if (hvlen >= 2 && ((hsval[0] == '"' && hsval[hvlen - 1] == '"') ||
                       (hsval[0] == '\'' && hsval[hvlen - 1] == '\'')))
    {
        memcpy(hvbuf, hsval + 1, (size_t) (hvlen - 2));
        hvbuf[hvlen - 2] = '\0';
    }
    else
    {
        strncpy(hvbuf, hsval, STRINGMAXLEN_CLICMDLINE - 1);
        hvbuf[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    }

    FILE *hsfp = tmpfile();
    if (hsfp != NULL)
    {
        fprintf(hsfp, "%s\n", hvbuf);
        rewind(hsfp);
        int sv_in;
        cli_fd_redirect(STDIN_FILENO, fileno(hsfp), &sv_in);

        *retval = CLI_execute_line();

        cli_fd_restore(STDIN_FILENO, sv_in);
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
int cli_handle_background(errno_t *retval)
{
    int ll = (int) strlen(data.CLIcmdline);
    int bi = ll - 1;
    while (bi >= 0 && (data.CLIcmdline[bi] == ' ' || data.CLIcmdline[bi] == '\t'))
    {
        bi--;
    }
    if (bi < 0 || data.CLIcmdline[bi] != '&' || (bi > 0 && data.CLIcmdline[bi - 1] == '&'))
    {
        return 0;
    }

    data.CLIcmdline[bi] = '\0';
    pid_t cpid          = fork();
    if (cpid == 0)
    {
        CLI_execute_string(data.CLIcmdline);
        _exit(0);
    }
    else if (cpid > 0)
    {
        printf("[bg] %d\n", (int) cpid);
        char pb[32];
        snprintf(pb, sizeof(pb), "%d", (int) cpid);
        cli_var_set("!", pb);
    }
    *retval = 0;
    return 1;
}


/* ============================================================
 *  is_internal_cmd
 * ============================================================ */

/**
 * is_internal_cmd - return 1 if firstword is a built-in
 *      keyword, variable assignment, or registered CLI
 *      command; 0 if it looks like an external command.
 * @firstword:    first token of the command line
 * @check_assign: when non-zero, a token containing '='
 *                is treated as an internal assignment.
 *                Pass 0 on the calc-eval path so that
 *                no-space arithmetic like "a=b+1" is
 *                still evaluated by the calc engine.
 */
int is_internal_cmd(const char *firstword, int check_assign)
{
    static const char *keywords[] = { "if",          "elif",    "else",   "fi",
                                      "for",         "while",   "do",     "done",
                                      ".",           "source",  "assert", "dpdigits",
                                      "assigncheck", "exitCLI", "exit",   NULL };

    for (int k = 0; keywords[k] != NULL; k++)
    {
        if (strcmp(firstword, keywords[k]) == 0)
        {
            return 1;
        }
    }

    if (check_assign && strchr(firstword, '=') != NULL)
    {
        return 1;
    }

    for (long i = 0; i < (long) data.NBcmd; i++)
    {
        size_t cmdlen = strlen(data.cmd[i].key);
        if (strncmp(firstword, data.cmd[i].key, cmdlen) == 0 &&
            (firstword[cmdlen] == '\0' || firstword[cmdlen] == ':' || firstword[cmdlen] == ' '))
        {
            return 1;
        }
    }

    return 0;
}


/* ============================================================
 *  cli_pipe_setup / cli_pipe_teardown
 * ============================================================ */

/**
 * cli_pipe_setup - detect unquoted '|' in CLIcmdline,
 *      split at it, and redirect stdout into a popen() pipe.
 * @pipe_fp:         set to the opened pipe, or NULL
 * @saved_stdout_fd: set to dup'd original stdout fd, or -1
 *
 * Replaces '|' with NUL so the LHS can execute normally.
 * Caller must restore with cli_pipe_teardown().
 */
void cli_pipe_setup(FILE **pipe_fp, int *saved_stdout_fd)
{
    *pipe_fp         = NULL;
    *saved_stdout_fd = -1;

    int   pipe_idx = cli_find_unquoted_op(data.CLIcmdline, '|', 0, 0);
    char *pipe_pos = (pipe_idx >= 0) ? data.CLIcmdline + pipe_idx : NULL;

    if (pipe_pos == NULL)
    {
        return;
    }

    *pipe_pos       = '\0';
    const char *rhs = pipe_pos + 1;
    while (*rhs == ' ' || *rhs == '\t')
    {
        rhs++;
    }
    if (*rhs == '\0')
    {
        return;
    }

    printf(COLORDIMYELLOW "[shell pipe] %s" COLORRST "\n", rhs);
    cli_export_vars_to_env();
    *pipe_fp = popen(rhs, "w");
    if (*pipe_fp != NULL)
    {
        cli_fd_redirect(STDOUT_FILENO, fileno(*pipe_fp), saved_stdout_fd);
    }
}

/**
 * cli_pipe_teardown - restore stdout after a pipe and close it.
 * @pipe_fp:         handle from cli_pipe_setup
 * @saved_stdout_fd: fd from cli_pipe_setup
 */
void cli_pipe_teardown(FILE *pipe_fp, int saved_stdout_fd)
{
    if (pipe_fp == NULL)
    {
        return;
    }
    fflush(stdout);
    cli_fd_restore(STDOUT_FILENO, saved_stdout_fd);
    pclose(pipe_fp);
}


/* ============================================================
 *  cli_redir_setup / cli_redir_teardown
 * ============================================================ */

/**
 * cli_redir_setup - detect unquoted '>' in CLIcmdline,
 *      split at it, and redirect stdout to a file.
 * @redir_fp:        set to the opened file, or NULL
 * @saved_stdout_fd: set to dup'd original stdout fd, or -1
 *
 * Replaces '>' with NUL so the LHS can execute normally.
 * Caller must restore with cli_redir_teardown().
 */
void cli_redir_setup(FILE **redir_fp, int *saved_stdout_fd)
{
    *redir_fp        = NULL;
    *saved_stdout_fd = -1;

    int   redir_idx = cli_find_unquoted_op(data.CLIcmdline, '>', 0, 0);
    char *redir_pos = (redir_idx >= 0) ? data.CLIcmdline + redir_idx : NULL;

    if (redir_pos == NULL)
    {
        return;
    }

    *redir_pos        = '\0';
    const char *fname = redir_pos + 1;
    while (*fname == ' ' || *fname == '\t')
    {
        fname++;
    }
    if (*fname == '\0')
    {
        return;
    }

    char fpath[500];
    strncpy(fpath, fname, 499);
    fpath[499] = '\0';
    {
        size_t fl = strlen(fpath);
        while (fl > 0 && (fpath[fl - 1] == ' ' || fpath[fl - 1] == '\t' || fpath[fl - 1] == '\n'))
        {
            fpath[--fl] = '\0';
        }
    }

    *redir_fp = fopen(fpath, "w");
    if (*redir_fp != NULL)
    {
        cli_fd_redirect(STDOUT_FILENO, fileno(*redir_fp), saved_stdout_fd);
    }
}

/**
 * cli_redir_teardown - restore stdout after a file redirect.
 * @redir_fp:        handle from cli_redir_setup
 * @saved_stdout_fd: fd from cli_redir_setup
 */
void cli_redir_teardown(FILE *redir_fp, int saved_stdout_fd)
{
    if (redir_fp == NULL)
    {
        return;
    }
    fflush(stdout);
    cli_fd_restore(STDOUT_FILENO, saved_stdout_fd);
    fclose(redir_fp);
}


/* ============================================================
 *  handle_did_you_mean
 * ============================================================ */

/**
 * handle_did_you_mean - print typo suggestions for an
 *      unknown command using Levenshtein distance.
 * @input_cmd: first token that was not resolved
 *
 * Scans the registered command table for the 3 closest
 * matches (distance ≤ 4) and prints them as suggestions.
 */
void handle_did_you_mean(const char *input_cmd)
{
#ifdef USE_READLINE
    if (input_cmd != NULL && input_cmd[0] != '\0')
    {
        struct distmatch
        {
            int         dist;
            const char *cmd;
        };
        struct distmatch matches[3] = { { 9999, NULL }, { 9999, NULL }, { 9999, NULL } };

        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            int d = levenshtein_distance(input_cmd, data.cmd[i].key);
            if (d < matches[2].dist)
            {
                matches[2].dist = d;
                matches[2].cmd  = data.cmd[i].key;
                if (matches[2].dist < matches[1].dist)
                {
                    struct distmatch tmp = matches[1];
                    matches[1]           = matches[2];
                    matches[2]           = tmp;
                }
                if (matches[1].dist < matches[0].dist)
                {
                    struct distmatch tmp = matches[0];
                    matches[0]           = matches[1];
                    matches[1]           = tmp;
                }
            }
        }

        if (matches[0].dist <= 4 && matches[0].cmd != NULL)
        {
            printf("\033[31mCommand '%s' not found. \033[0m"
                   "Did you mean:\n",
                   input_cmd);
            for (int m = 0; m < 3; m++)
            {
                if (matches[m].cmd && matches[m].dist <= 4 && matches[m].dist < 9999)
                {
                    printf("  - \033[0;96m%s"
                           "\033[0m\n",
                           matches[m].cmd);
                }
            }
            return;
        }
    }
#else
    (void) input_cmd;
#endif
    printf("\033[31mCommand not found, "
           "or command with no effect\n\033[0m");
}
