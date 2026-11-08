// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_highlight.c
 *
 * @brief Syntax highlighting for readline input
 *
 * After readline's normal redisplay, the tree-sitter highlighter (cli_ts_highlight_line()) redraws
 * the line in color, in the colors the terminal supports; cursor save/restore returns the cursor to
 * its position. `@config synhl on|off` turns it on or off; calc mode lines are not highlighted.
 */

#include <stdio.h>
#include <string.h>

#ifdef USE_READLINE
#    include <readline/readline.h>
#endif

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "treesitter/cli_treesitter.h"

#ifdef USE_READLINE
/**
 * @brief Redisplay with syntax highlighting
 *
 * Called as the readline redisplay function. Lets readline draw the line first, so its cursor
 * state stays consistent, then overwrites it with the highlighted line.
 */
void cli_highlight_redisplay(void)
{
    rl_redisplay();
#    ifdef USE_TREESITTER
    if (data.syntax_highlight && !cli_calc_mode && strchr(rl_line_buffer, '\n') == NULL)
    {
        fprintf(rl_outstream, "\033[s"); // Save cursor
        if (rl_point > 0)                // to the start of the line
        {
            fprintf(rl_outstream, "\033[%dD", rl_point);
        }
        cli_ts_highlight_line(rl_line_buffer, (int) strlen(rl_line_buffer), rl_point, rl_outstream);
        fprintf(rl_outstream, "\033[u"); // Restore cursor
        fflush(rl_outstream);
    }
#    endif
}
#endif
