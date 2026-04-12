/**
 * @file CLIcore_UI_highlight.c
 *
 * @brief Syntax highlighting for readline input
 *
 * Provides real-time syntax coloring of the input line
 * while the user types. The first word is colored green
 * if it matches a registered CLI command, or red if it
 * does not. This gives immediate visual feedback about
 * whether a command will be recognized.
 *
 * ## How it works
 *
 * After readline's normal redisplay, the first word is
 * overwritten in-place using ANSI escape codes. Cursor
 * save/restore ensures the cursor returns to its original
 * position after the colorization pass.
 *
 * The `cli_is_command()` helper is also used by the
 * pipe dispatcher to decide whether to route a pipe
 * target to a native milk command or to a shell process.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#ifdef USE_READLINE
#include <readline/readline.h>
#endif

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"


/**
 * @brief Check if a word is a registered CLI command
 *
 * Scans the command table for an exact match.
 *
 * @param word  The word to look up
 * @return 1 if found, 0 otherwise
 */
int cli_is_command(const char *word)
{
    for(uint32_t i = 0; i < data.NBcmd; i++)
    {
        if(strcmp(data.cmd[i].key, word) == 0)
        {
            return 1;
        }
    }
    return 0;
}

#ifdef USE_READLINE
/**
 * @brief Redisplay with syntax highlighting
 *
 * Called as the readline redisplay function when
 * syntax_highlight is enabled. It:
 *
 * 1. Calls normal rl_redisplay() to render the line
 *    (keeping readline's internal state consistent).
 * 2. Finds the first word boundaries.
 * 3. Checks whether the first word is a registered
 *    command.
 * 4. Overwrites the first word with green (known) or
 *    red (unknown) ANSI color codes.
 * 5. Restores the cursor position.
 */
void cli_highlight_redisplay(void)
{
    if(!data.syntax_highlight)
    {
        rl_redisplay();
        return;
    }

    /*
     * Let readline draw normally first so its
     * internal cursor state stays consistent.
     * Then overwrite just the first word in color.
     */
    rl_redisplay();

    /* Find the first word boundaries */
    int ws = 0;
    while(rl_line_buffer[ws] == ' '
            || rl_line_buffer[ws] == '\t')
    {
        ws++;
    }
    int we = ws;
    while(rl_line_buffer[we] != '\0'
            && rl_line_buffer[we] != ' '
            && rl_line_buffer[we] != '\t')
    {
        we++;
    }
    if(we == ws)
    {
        fflush(stdout);
        return;
    }

    /* Comment lines: color entire line dim green */
    if(rl_line_buffer[ws] == '#')
    {
        fprintf(rl_outstream, "\033[s");
        {
            int back = rl_point;
            if(back > 0)
            {
                fprintf(rl_outstream,
                        "\033[%dD", back);
            }
        }
        fprintf(rl_outstream,
                "\033[2;32m%s\033[0m",
                rl_line_buffer);
        fprintf(rl_outstream, "\033[u");
        fflush(rl_outstream);
        return;
    }

    /* Extract first word */
    char firstword[200];
    int fwlen = we - ws;
    if(fwlen > 199)
    {
        fwlen = 199;
    }
    memcpy(firstword, rl_line_buffer + ws,
           (size_t) fwlen);
    firstword[fwlen] = '\0';

    /* Pick color */
    const char *col;
    if(cli_is_command(firstword))
    {
        col = "\033[32m"; /* green */
    }
    else if(strchr(firstword, '=')
            || strchr(firstword, '+')
            || strchr(firstword, '*')
            || strchr(firstword, '/')
            || strchr(firstword, '('))
    {
        /* Math expression or assignment —
         * leave in default color */
        fflush(rl_outstream);
        return;
    }
    else
    {
        col = "\033[31m"; /* red */
    }

    /*
     * After rl_redisplay(), cursor is at rl_point.
     * Move back to the first word, overwrite with
     * color, then restore cursor position.
     * Use cursor-relative movement (not absolute
     * column) to avoid prompt-width errors from
     * invisible escape sequences in the prompt.
     */
    fprintf(rl_outstream, "\033[s");  /* save */
    {
        int back = rl_point - ws;
        if(back > 0)
        {
            fprintf(rl_outstream,
                    "\033[%dD", back);
        }
    }
    fprintf(rl_outstream, "%s%s\033[0m",
            col, firstword);
    fprintf(rl_outstream, "\033[u");  /* restore */
    fflush(rl_outstream);
}
#endif
