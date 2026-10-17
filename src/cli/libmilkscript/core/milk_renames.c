// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_renames.c
 * @brief Removed spellings and their replacements (see milk_renames.h).
 */

#include <string.h>

#include "milk_renames.h"

typedef struct
{
    const char    *old;
    const char    *new_;
    milk_ren_how_t how;
} rename_t;

#define W MILK_REN_WORD
#define A MILK_REN_ARGS
#define H MILK_REN_HINT

/* Command words: the bare engine names moved to the module `milk`, the removed aliases, and the words
 * the 27 keywords replace */
static const rename_t cmd_renames[] = {
    { "@ci", "milk.ci", W },
    { "@name", "milk.name", W },
    { "@fifo", "milk.fifo", W },
    { "@fpsload", "milk.fpsload", W },
    { "@fpsdump", "milk.fpsdump", W },
    { "@timing", "milk.timing", W },
    { "@cli.timing", "milk.timing", W },
    { "cli.timing", "milk.timing", W },
    { "@savefits", "iofits.saveFITS", W },
    { "@iofits.savefits", "iofits.saveFITS", W },
    { "iofits.savefits", "iofits.saveFITS", W },
    { "@imgstat", "info.imstats", W },
    { "@info.imgstat", "info.imstats", W },
    { "info.imgstat", "info.imstats", W },
    { "@mk2Drnd", "imgen.mkrndim", W },
    { "@imgen.mk2Drnd", "imgen.mkrndim", W },
    { "imgen.mk2Drnd", "imgen.mkrndim", W },
    /* help: one keyword, plus -h/--help on every command and keyword */
    { "@h", "@help", W },
    { "@?", "@help", W },
    { "@hc", "@help", W },
    { "@cmd?", "@help", W },
    { "@hm", "@help", W },
    { "@m?", "@help", W },
    { "@hb", "@help bash", W },
    { "@hf", "@help -k", W },
    { "@cmdinfo?", "@help -k", W },
    { "@helprl", "@help readline", W },
    { "@fhelp", "@help --tui", W },
    { "@help-bash", "@help bash", W },
    { "@help-cmdopts", "@help cmdopts", W },
    { "@help-syntax", "@help syntax", W },
    { "@help-commands", "@help commands", W },
    { "@help-variables", "@help variables", W },
    { "@help-flowcontrol", "@help flowcontrol", W },
    { "@help-scripting", "@help scripting", W },
    { "@help-milk", "@help milk", W },
    /* values */
    { "@", "@calc \"<expr>\"", A },
    { "@fpsset", "@set f.<fps>.<param>=<value>", A },
    { "@procstat", "@get p.<proc>.list, or @ls -l 'p.*'", A },
    { "@procctl", "@set p.<proc>.ctrl=<action>", A },
    { "@streamlist", "@ls", W },
    { "@fpslist", "@ls", W },
    { "@proclist", "@ls", W },
    { "@milkquery", "@ls --json", W },
    { "list-streams", "@ls", W },
    { "@list-streams", "@ls", W },
    { "list-fps", "@ls", W },
    { "@list-fps", "@ls", W },
    /* events */
    { "@usleep", "@sleep <seconds>", A },
    { "@procwait", "@waitfor p.<proc>.status=<status> -t <sec>", A },
    /* data */
    { "@batch", "@pipe", W },
    { "@streamcat", "@grab --stdout", W },
    { "@streamwrite", "@replay --stdin", W },
    /* program */
    { "include_once", "@source --once", W },
    { "@mload", "@module load", W },
    { "@soload", "@module load", W },
    { "@mloadas", "@module load <name> --as <short>", A },
    { "@lm?", "@module list", W },
    { "@dpdigits", "@config digits", W },
    { "@dpsingle", "@config precision single", W },
    { "@dpdouble", "@config precision double", W },
    { "@setprocinfoON", "@config procinfo on", W },
    { "@setprocinfoOFF", "@config procinfo off", W },
    { "@prompt", "@config prompt", W },
    { "@setprompt", "@config prompt", W },
    { "@sessionlog", "@config sessionlog", W },
    { "@synhl", "@config synhl", W },
    { "@syndiag", "@config syndiag", W },
    { "@cliindent", "@config indent", W },
    { "@showmatch", "@config showmatch", W },
    { "@cliformat", "milk-cli --format <file>", H },
    { "@clifold", "milk-cli --outline <file>", H },
    /* introspection */
    { "@searchhist", "@history search", W },
    { "@fhist", "@history --tui", W },
    { "@ghistory", "@history --global", W },
    { "@lhistory", "@history --local", W },
    { "@savehistory", "@history save", W },
    { "@cmdstats", "@history --stats", W },
    { "@fparam", "@set --tui", W },
    /* dropped: plain bash does it */
    { "quit", "exit", W },
    { "exitCLI", "exit", W },
    { "@time", "time", W },
    { "bookmark", "a bash function", H },
    { NULL, NULL, W }
};

#undef W
#undef A
#undef H

/* Entity namespaces */
static const rename_t ns_renames[] = { { "seq.", "q." }, { "procinfo.", "p." }, { NULL, NULL } };

const char *milk_rename_cmd(const char *word, milk_ren_how_t *how)
{
    for (const rename_t *r = cmd_renames; word != NULL && r->old != NULL; r++)
    {
        if (strcmp(word, r->old) == 0)
        {
            if (how != NULL)
            {
                *how = r->how;
            }
            return r->new_;
        }
    }
    return NULL;
}

const char *milk_rename_ns(const char *path, size_t *oldlen)
{
    for (const rename_t *r = ns_renames; path != NULL && r->old != NULL; r++)
    {
        size_t n = strlen(r->old);
        if (strncmp(path, r->old, n) == 0)
        {
            if (oldlen != NULL)
            {
                *oldlen = n;
            }
            return r->new_;
        }
    }
    return NULL;
}
