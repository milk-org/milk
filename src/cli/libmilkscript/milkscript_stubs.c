// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkscript_stubs.c
 * @brief Weak-symbol stubs for interactive-only functions
 */

#include "CLIcore.h"

__attribute__((weak)) void CLI_cleanup_scroll_region(void)
{
}

__attribute__((weak)) void cli_history_save(void)
{
}

__attribute__((weak)) DATA data;

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

__attribute__((weak)) void cli_save_last_argument(void)
{
}
