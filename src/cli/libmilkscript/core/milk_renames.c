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
#define SEQ_GONE "a milk script (the sequencer is gone: @help seq)"
#define UNIT_GONE \
    "a milk script running each unit's fpsexec (<prefix>-fpsexec-<id> -n NAME fpsinit)"
#define TMUX_GONE \
    "milk-fps-runstop / milk-fps-rm -f (units have no tmux session; milk-fps-log NAME)"
#define SERVER_GONE \
    "@session (start -> new or use, list -> ls, local -> leave, stop -> end, attach -> log -f)"

/* Command words: bare engine names moved to module `milk`, removed aliases, and words replaced
 * by the 27 keywords */
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
    /* the sequencer was removed: workflows are milk scripts */
    { "milk-seq", SEQ_GONE, H },
    { "milk-seq-help", SEQ_GONE, H },
    { "seq.list", SEQ_GONE, H },
    { "seq.start", SEQ_GONE, H },
    { "seq.stop", SEQ_GONE, H },
    { "seq.status", SEQ_GONE, H },
    { "seq.submit", SEQ_GONE, H },
    { "seq.tasks", SEQ_GONE, H },
    { "seq.log", SEQ_GONE, H },
    /* the fpslist.txt deployment and the tmux unit hosts were removed (unit lifecycle, U2) */
    { "milk-fpsinit", UNIT_GONE, H },
    { "milk-fpsmkcmd", UNIT_GONE, H },
    { "milk-fpslist-addentry", UNIT_GONE, H },
    { "tmuxsessionname", "$MILK_UNIT_NAME in a unit's process (units have no tmux session)", H },
    { "tmuxkillall", TMUX_GONE, H },
    /* servers are sessions (D33-D35) */
    { "@server", SERVER_GONE, H },
    { NULL, NULL, W }
};

#undef W
#undef A
#undef H
#undef SEQ_GONE
#undef UNIT_GONE
#undef TMUX_GONE
#undef SERVER_GONE

/**
 * milk_rename_cmd - Look up the replacement for a removed or renamed command word
 * @word: Command token as typed by the user (e.g. `"@fpslist"` or `"@savefits"`)
 * @how:  Optional output pointer receiving how the replacement should be presented
 *
 * Searches `cmd_renames[]` so the CLI can guide users from retired command spellings
 * and legacy `@` aliases to their current keyword or `<module>.<cmd>` equivalents.
 *
 * Return: Replacement command or syntax hint string, or NULL if @word is not in the table
 */
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
