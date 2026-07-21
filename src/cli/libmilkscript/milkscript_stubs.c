// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkscript_stubs.c
 * @brief Weak-symbol stubs for interactive-only
 *        functions
 *
 * When libmilkscript is used without the interactive
 * CLIcore layer (e.g. in the milk-script binary),
 * these weak stubs provide safe no-op defaults for
 * functions that only make sense in an interactive
 * terminal.
 *
 * When CLIcore links against libmilkscript, it
 * provides strong definitions that override these.
 */

#include "CLIcore.h"


/**
 * CLI_cleanup_scroll_region - No-op in script mode.
 *
 * In the interactive CLI this clears the hint area
 * and restores the terminal scroll region. In script
 * mode there is no hint area.
 */
__attribute__((weak))
/**
 * @brief Stub: cleanup scroll region in script mode.
 */
void CLI_cleanup_scroll_region(void)
{
    /* no-op: no scroll region in script mode */
}

/**
 * cli_history_save - No-op in script mode.
 *
 * Persistent readline history is an interactive-only
 * feature.
 */
__attribute__((weak))
/**
 * @brief Stub: save command history in script mode.
 */
void cli_history_save(void)
{
    /* no-op: no readline history in script mode */
}

/*
 * Provide a weak definition of the global data structure if not
 * provided by CLIcore.c
 */
__attribute__((weak)) DATA data;

__attribute__((weak)) void cli_expand_braces(char *line, int maxlen)
{
    (void) line;
    (void) maxlen;
}

__attribute__((weak)) int find_streams(void *streaminfo, int filter, const char *namefilter)
{
    (void) streaminfo;
    (void) filter;
    (void) namefilter;
    return 0;
}

__attribute__((weak)) int cli_is_command(const char *word)
{
    (void) word;
    return 0;
}

__attribute__((weak)) int cli_savescript(void)
{
    return 0;
}

__attribute__((weak)) void cli_history_log_cmd(const char *cmd)
{
    (void) cmd;
}

__attribute__((weak)) void cli_history_log_shell(const char *cmd)
{
    (void) cmd;
}

__attribute__((weak)) void cli_session_log_cmd(const char *cmd)
{
    (void) cmd;
}

__attribute__((weak)) void cli_history_expand(void)
{
}

__attribute__((weak)) int cli_savehistory(void)
{
    return 0;
}

__attribute__((weak)) int cli_source(void)
{
    return 0;
}

__attribute__((weak)) void cli_save_last_argument(void)
{
}
