/**
 * @file CLIcore_UI_completion.c
 *
 * @brief Readline tab-completion and prompt construction
 *
 * Provides the tab-completion engine for the milk CLI.
 * Completion matches against registered commands, shared-
 * memory image streams, FPS names, command arguments (dot-
 * prefixed FPS tags), and filesystem paths.
 *
 * Also provides the prompt builder (including PS1 support)
 * and the readline callback that hands accepted input to
 * the command execution pipeline.
 *
 * ## Key design choices
 *
 * - **Two-pass completion**: The generator first tries
 *   prefix matching, then falls back to substring (fuzzy)
 *   matching if nothing was found and fuzzy mode is on.
 *
 * - **Argument-type-aware completion**: When the cursor is
 *   on a positional argument of a known command, the
 *   completion mode switches to match the expected argument
 *   type (image stream, filename, FPS name, etc.).
 *
 * - **Levenshtein distance**: Used by the "did you mean?"
 *   suggestions when a command is not found.
 */

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif

#include "CLIcore.h"

#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"

#include <fnmatch.h>
#include <glob.h>
#include <sys/wait.h>


#include "timeutils.h"

#define CLICOMPLETIONMODE_COMMANDS  0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2
#define CLICOMPLETIONMODE_FILES    3
#define CLICOMPLETIONMODE_FPSPARAMS 4
#define CLICOMPLETIONMODE_VARS_FPS    5
#define CLICOMPLETIONMODE_VARS_SEQ    6
#define CLICOMPLETIONMODE_VARS_STREAM 7

#define COLORRED       "\001\033[31m\002"
#define COLORHBOLDCYAN "\001\e[0;96m\002"
#define COLORDIMYELLOW "\033[2;33m"
#define COLORRST       "\033[0m"
#define RL_COLORRESET  "\001\033[0m\002"


/* ---- String utilities ---- */

void *xmalloc(int size)
{
    void *buf;

    buf = malloc(size);
    if(!buf)
    {
        fprintf(stderr,
                COLORRED
                "Error: Out of memory. Exiting.'n"
                COLORRESET);
        exit(1);
    }

    return buf;
}

/**
 * @brief Duplicate a string using xmalloc.
 *
 * Allocates memory for a copy of @s and returns
 * the copy. The caller must free() the result.
 *
 * @param s  String to duplicate
 * @return Newly allocated copy of @s
 */
char *dupstr(char *s)
{
    char *r;

    size_t len = strlen(s) + 1;
    r = (char *) xmalloc(len);
    memcpy(r, s, len);
    return (r);
}


/* ---- Readline callback and prompt ---- */

#ifdef USE_READLINE

/**
 * Number of ghost chars rendered on current line.
 * Set by print_ghost(), read by cli_accept_line().
 */
int ghost_chars_on_line = 0;

/**
 * @brief Custom accept-line handler for readline
 *
 * Bound to Enter key. Overwrites ghost suggestion
 * text with spaces before accepting the line, so
 * the terminal scrollback entry is clean.
 */
int cli_accept_line(
    int count,
    int key)
{
    if(ghost_chars_on_line > 0)
    {
        int n = ghost_chars_on_line;
        for(int i = 0; i < n; i++)
        {
            putchar(' ');
        }
        for(int i = 0; i < n; i++)
        {
            putchar('\b');
        }
        fflush(stdout);
        ghost_chars_on_line = 0;
    }

    return rl_newline(count, key);
}

/**
 * @brief Readline callback handler — processes a
 *        completed input line.
 *
 * Invoked by rl_callback_read_char() when the user
 * presses Enter. Copies the input into
 * data.CLIcmdline, handles backslash line
 * continuation (reading extra lines until no
 * trailing backslash), then dispatches the
 * assembled command via CLI_execute_line().
 *
 * If linein is NULL (Ctrl-D / EOF), sets
 * data.CLIloopON=0 to exit the main loop.
 *
 * @param linein  Line text from readline
 *                (caller-allocated, freed here)
 */
void rl_cb_linehandler(char *linein)
{
    if(NULL == linein)
    {
        data.CLIloopON = 0;
        return;
    }

    data.CLIexecuteCMDready = 1;

    // copy input into data.CLIcmdline
    strncpy(data.CLIcmdline, linein,
            STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    /* We will add to history AFTER backslash continuation and history expansion */

    /* Handle backslash line continuation:
     * temporarily switch to blocking readline
     * to read additional lines */
    {
        size_t len = strlen(data.CLIcmdline);
        while(len > 0
                && data.CLIcmdline[len - 1]
                == '\\')
        {
            data.CLIcmdline[len - 1] = ' ';
            /* Remove callback handler to avoid
             * interference, use direct readline
             * for continuation */
            rl_callback_handler_remove();
            char *cont = readline("> ");
            /* Re-install with dummy prompt;
             * the main loop will re-install
             * with the proper prompt after this
             * handler returns */
            rl_callback_handler_install(
                "",
                (rl_vcpfunc_t *)
                &rl_cb_linehandler);
            if(cont == NULL)
            {
                break;
            }
            int avail =
                STRINGMAXLEN_CLICMDLINE
                - (int) strlen(data.CLIcmdline)
                - 1;
            if(avail > 0)
            {
                strncat(data.CLIcmdline, cont,
                        (size_t) avail);
            }
            free(cont);
            len = strlen(data.CLIcmdline);
        }
    }

    /* Expand history (!! and !$) now that the full line is assembled */
    cli_history_expand();

    if(data.CLIcmdline[0] == '\0')
    {
        /* Expansion error. Exit loop and prevent execution. */
        free(linein);
        return;
    }

    /* Record expanded prompt in readline history
     * and structured log BEFORE alias resolution.
     * This ensures up-arrow recalls the expanded command,
     * consistent with native bash behavior. */
    if(data.CLIcmdline[0] != '\0')
    {
        add_history(data.CLIcmdline);
        cli_history_log_prompt(data.CLIcmdline);
        if(data.autocomplete_history)
        {
            append_history(
                1, CLI_history_file());
            history_truncate_file(
                CLI_history_file(), 10000);
        }
    }

    if(data.echo_input)
    {
        printf("\033[32m[echo]\033[0m \u2190 \"%s\"\n", data.CLIcmdline);
    }
    CLI_execute_line();

    free(linein);
}
#endif

/**
 * @brief Build the prompt string for the CLI
 *
 * Checks for a PS1 variable in CLI vars or the
 * environment. Falls back to the default colored
 * prompt with the process name.
 */
errno_t runCLI_prompt(
    char *promptstring,
    char *prompt)
{
    /* Use PS1 only from CLI vars (set inside
     * milk-cli).  Do NOT fall back to
     * getenv("PS1") — the bash PS1 contains
     * shell-specific escapes like $(cmd) that
     * cli_expand_env cannot evaluate, which
     * would corrupt the prompt. */
    const char *ps1_val =
        cli_var_get("PS1");

    if(ps1_val != NULL && strlen(ps1_val) > 0)
    {
        char expanded_ps1[FPS_DIR_STRLENMAX];
        strncpy(expanded_ps1, ps1_val,
                FPS_DIR_STRLENMAX - 1);
        expanded_ps1[FPS_DIR_STRLENMAX - 1] = '\0';
        cli_expand_env(expanded_ps1,
                       FPS_DIR_STRLENMAX);
        strncpy(prompt, expanded_ps1,
                FPS_DIR_STRLENMAX - 1);
        prompt[FPS_DIR_STRLENMAX - 1] = '\0';
        return RETURN_SUCCESS;
    }

    if(strlen(promptstring) > 0)
    {
        if(data.processnameflag == 0)
        {
            snprintf(prompt, FPS_DIR_STRLENMAX,
                     COLORHBOLDCYAN
                     "%s > " RL_COLORRESET,
                     promptstring);
        }
        else
        {
            snprintf(prompt,
                     FPS_DIR_STRLENMAX,
                     COLORHBOLDCYAN
                     "%s-%s > " RL_COLORRESET,
                     promptstring,
                     data.processname);
        }
    }
    else
    {
        snprintf(prompt, FPS_DIR_STRLENMAX,
                 COLORHBOLDCYAN
                 "%s > " RL_COLORRESET,
                 data.processname);
    }

    return RETURN_SUCCESS;
}


/* ---- Levenshtein distance (fuzzy matching) ---- */

#ifdef USE_READLINE

/**
 * @brief Compute Levenshtein edit distance
 *
 * Used to suggest similar commands when a typed
 * command is not found ("did you mean?").
 */
int levenshtein_distance(
    const char *s1,
    const char *s2)
{
    unsigned int len1 = strlen(s1);
    unsigned int len2 = strlen(s2);
    unsigned int *d = (unsigned int *)
                      xmalloc((len1 + 1) * (len2 + 1)
                              * sizeof(unsigned int));

    for(unsigned int i = 0; i <= len1; i++)
    {
        d[i * (len2 + 1)] = i;
    }
    for(unsigned int j = 0; j <= len2; j++)
    {
        d[j] = j;
    }

    for(unsigned int i = 1; i <= len1; i++)
    {
        for(unsigned int j = 1; j <= len2; j++)
        {
            unsigned int cost =
                (s1[i - 1] == s2[j - 1])
                ? 0 : 1;
            unsigned int min1 =
                d[(i - 1) * (len2 + 1) + j] + 1;
            unsigned int min2 =
                d[i * (len2 + 1) + j - 1] + 1;
            unsigned int min3 =
                d[(i - 1) * (len2 + 1)
                          + j - 1] + cost;
            unsigned int m =
                (min1 < min2) ? min1 : min2;
            d[i * (len2 + 1) + j] =
                (m < min3) ? m : min3;
        }
    }
    int dist = d[len1 * (len2 + 1) + len2];
    free(d);
    return dist;
}

#endif
