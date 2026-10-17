// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_lexicon.c
 * @brief Bash words and `@` primitives (see milk_lexicon.h).
 */

#include <stddef.h>

#include "milk_lexicon.h"

static const milk_primitive_t primitives[] = {
    { "calc", MILK_PRIM_CALL },
    { "test", MILK_PRIM_CALL },
    { "get", MILK_PRIM_CALL },
    { "set", MILK_PRIM_CALL },
    { "ls", MILK_PRIM_CALL },
    { "exists", MILK_PRIM_CALL },
    { "running", MILK_PRIM_CALL },
    { "waitfor", MILK_PRIM_CALL },
    { "waitany", MILK_PRIM_CALL },
    { "grab", MILK_PRIM_CALL },
    { "replay", MILK_PRIM_CALL },
    { "snapshot", MILK_PRIM_CALL },
    { "diff", MILK_PRIM_CALL },
    { "restore", MILK_PRIM_CALL },
    { "assert", MILK_PRIM_CALL },
    { "prompt", MILK_PRIM_CALL },
    { "batch", MILK_PRIM_CALL },
    { "pipe", MILK_PRIM_CALL },
    { "server", MILK_PRIM_CALL },
    { "log", MILK_PRIM_CALL },
    { "h", MILK_PRIM_CALL },
    { "hb", MILK_PRIM_CALL },
    { "hc", MILK_PRIM_CALL },
    { "hm", MILK_PRIM_CALL },
    { "hf", MILK_PRIM_CALL },
    { "streamcat", MILK_PRIM_CALL },
    { "streamwrite", MILK_PRIM_CALL },
    { "unload", MILK_PRIM_CALL },
    { "defer", 0 },
    { "on", 0 },
    { "source", 0 },
    { "procstat", 0 },
    { "procctl", 0 },
    { "procwait", 0 },
    { NULL, 0 } /* @time exists only in the milk-cli engine */
};

static const char *const repl_commands[] = { "history",  "searchhist",  "fhist",     "ghistory",
                                             "lhistory", "savehistory", "setprompt", "sessionlog",
                                             "synhl",    "syndiag",     "cliindent", "showmatch",
                                             "helprl",   "fhelp",       "fparam",    "help",
                                             "dpdigits", "fpsset",      NULL };

static const char *const bash_words[] = {
    "if",     "elif",  "else",     "fi",           "for",       "while",    "until",
    "do",     "done",  "case",     "esac",         "select",    "function", ".",
    "source", "break", "continue", "return",       "true",      "false",    "exit",
    "shift",  "set",   "export",   "readonly",     "local",     "declare",  "let",
    "eval",   "type",  "command",  "trap",         "time",      "timeout",  "wait",
    "printf", "echo",  "getopts",  "mapfile",      "alias",     "unalias",  "pushd",
    "popd",   "dirs",  "[[",       "include_once", "milksleep", NULL
};

const milk_primitive_t *milk_lexicon_primitives(void)
{
    return primitives;
}

const char *const *milk_lexicon_repl_commands(void)
{
    return repl_commands;
}

const char *const *milk_lexicon_bash_words(void)
{
    return bash_words;
}
