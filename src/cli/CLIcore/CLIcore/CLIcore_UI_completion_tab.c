// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_completion_tab.c
 * @brief Readline tab-completion for the milk-cli prompt, over the shared completer.
 *
 * Role & Goal:
 *   Implements `CLI_generator()` and `CLI_completion()` for the `milk-cli` interactive prompt
 *   (TAB and the hint line). The context of the word (cli_ts_determine_completion_mode()) picks
 *   what to ask milk_complete(), which bash uses too: command names, arguments, keyword
 *   options and words, calculator text. Shell variables (`$name`) and file names are completed
 *   here; a command that is not milk's completes file names, as in bash. When nothing matches, a
 *   fuzzy pass keeps the candidates that contain the word.
 *
 * @see CLIcore_UI_completion_menu.c for the interactive Zsh-style match grid.
 * @see CLIcore_UI_completion.c for prompt rendering.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include "milk_keywords.h"
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "milk_server.h"
#include "treesitter/cli_treesitter.h"

extern char **environ;

#ifdef USE_READLINE

int cli_comp_start       = 0;
int cli_comp_end         = 0;
int generator_fuzzy_pass = 0;

/**
 * @brief Return 1 if @candidate matches @query under the current prefix/fuzzy pass.
 */
static inline int match_prefix_or_fuzzy(const char *candidate, const char *query, size_t qlen)
{
    if (generator_fuzzy_pass == 0)
    {
        return (strncmp(candidate, query, qlen) == 0) ? 1 : 0;
    }
    return (strstr(candidate, query) != NULL) ? 1 : 0;
}

/**
 * format_env_var_candidate - Format a variable name as a tab-completion candidate
 * @vname:      Base variable identifier
 * @has_dollar: Non-zero if the token starts with '$'
 * @has_brace:  Non-zero if the token starts with '${'
 * @is_array:   Non-zero to append '[@]}' when completing a braced array variable
 *
 * Wraps @vname in the sigil and brace syntax matching the user's current input
 * prefix so readline replaces the token seamlessly.
 *
 * Return: Newly allocated candidate string via dupstr()
 */
static char *format_env_var_candidate(const char *vname,
                                      int         has_dollar,
                                      int         has_brace,
                                      int         is_array)
{
    char buf[512];
    if (has_brace)
    {
        snprintf(buf, sizeof(buf), is_array ? "${%s[@]}" : "${%s}", vname);
    }
    else if (has_dollar)
    {
        snprintf(buf, sizeof(buf), "$%s", vname);
    }
    else
    {
        snprintf(buf, sizeof(buf), "%s", vname);
    }
    return dupstr(buf);
}

/**
 * @brief Generate shell, script, array, and environment variable candidates (`CLICOMPLETIONMODE_VARS_ENV`).
 */
static char *gen_vars_env_match(const char *text, int state)
{
    static int  var_phase;
    static int  var_idx;
    static int  has_dollar;
    static int  has_brace;
    static char vprefix[256];
    static int  vpreflen;

    if (!state)
    {
        var_phase      = 0;
        var_idx        = 0;
        has_dollar     = (text[0] == '$');
        has_brace      = (has_dollar && text[1] == '{');
        const char *vp = has_brace ? (text + 2) : (has_dollar ? (text + 1) : text);
        strncpy(vprefix, vp, sizeof(vprefix) - 1);
        vprefix[sizeof(vprefix) - 1] = '\0';
        vpreflen                     = (int) strlen(vprefix);
    }

    static const char *const special_vars[] = {
        "?", "$", "!", "#", "0", "MCLIFIFO", "PROCINFO_NCPU", "PROCINFO_NPROC", NULL
    };

    while (var_phase == 0)
    {
        const char *sname = special_vars[var_idx++];
        if (sname == NULL)
        {
            var_phase = 1;
            var_idx   = 0;
            break;
        }
        if (match_prefix_or_fuzzy(sname, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(sname, has_dollar, has_brace, 0);
        }
    }

    while (var_phase == 1)
    {
        if (var_idx >= CLI_MAX_VARS)
        {
            var_phase = 3;
            var_idx   = 0;
            break;
        }
        int i = var_idx++;
        if (cli_vars[i].used && match_prefix_or_fuzzy(cli_vars[i].name, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(cli_vars[i].name, has_dollar, has_brace, 0);
        }
    }

    while (var_phase == 3)
    {
        if (environ == NULL || environ[var_idx] == NULL)
        {
            var_phase = 4;
            var_idx   = 0;
            break;
        }
        const char *entry = environ[var_idx++];
        const char *eq    = strchr(entry, '=');
        if (eq == NULL)
        {
            continue;
        }
        char   ename[256];
        size_t nlen = (size_t) (eq - entry);
        if (nlen >= sizeof(ename))
        {
            nlen = sizeof(ename) - 1;
        }
        strncpy(ename, entry, nlen);
        ename[nlen] = '\0';

        int already_in_cli = 0;
        for (int k = 0; k < CLI_MAX_VARS; k++)
        {
            if (cli_vars[k].used && strcmp(cli_vars[k].name, ename) == 0)
            {
                already_in_cli = 1;
                break;
            }
        }
        if (!already_in_cli && match_prefix_or_fuzzy(ename, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(ename, has_dollar, has_brace, 0);
        }
    }
    return NULL;
}

/* What CLI_generator() gives for the word: candidates, file names, or shell variables */
enum
{
    COMP_LIST,
    COMP_FILES,
    COMP_SHELLVARS
};

/**
 * cli_complete_list - The completion candidates of word @full, at @start in @line, one per line
 * @line:  The prompt line
 * @start: Offset of the word in @line
 * @full:  The word (the quoted text so far inside an open quote: its last word is completed)
 * @out:   Output buffer (the candidates for COMP_LIST)
 * @outsz: Its size
 *
 * Return: COMP_LIST, COMP_FILES (readline completes file names) or COMP_SHELLVARS
 */
static int g_comp_in_quote = 0; /* the candidates go on inside an open quote: do not close it */

static int cli_complete_list(const char *line, int start, const char *full, char *out, size_t outsz)
{
    /* Inside an open quote readline gives the quoted text so far: complete its last word */
    size_t k = strlen(full);
    while (k > 0 && strchr(" \t\n<>=;|&(", full[k - 1]) == NULL)
    {
        k--;
    }
    const char *text         = full + k;
    char        cmdname[128] = "";
    char        ctype[512];
    int         argidx = 0;
    int         inl    = (strncmp(text, "@{", 2) == 0); /* inline calculator text: @{expr */
    int         mode   = cli_ts_determine_completion_mode(line, start + (int) k, text, cmdname,
                                                          sizeof(cmdname), &argidx);
    g_comp_in_quote    = (k > 0);
    start += (int) k;

    out[0] = '\0';
    if (inl || (cli_calc_mode && line[strspn(line, " \t")] != '@'))
    {
        snprintf(ctype, sizeof(ctype), "calc");
    }
    else if (mode == CLICOMPLETIONMODE_FILES)
    {
        return COMP_FILES;
    }
    else if (mode == CLICOMPLETIONMODE_VARS_ENV)
    {
        return COMP_SHELLVARS;
    }
    else if (mode == CLICOMPLETIONMODE_COMMANDS)
    {
        snprintf(ctype, sizeof(ctype), "cmd");
    }
    else if (mode >= 0) /* an entity path or a modifier */
    {
        snprintf(ctype, sizeof(ctype), "arg");
    }
    else if (strcmp(cmdname, "export") == 0 || strcmp(cmdname, "readonly") == 0 ||
             strcmp(cmdname, "unset") == 0 || strcmp(cmdname, "local") == 0 ||
             strcmp(cmdname, "declare") == 0)
    {
        return COMP_SHELLVARS;
    }
    else if (find_command_match(cmdname) < 0 && milk_lexicon_keyword(cmdname) == NULL)
    {
        return COMP_FILES; /* a command that is not milk's, as in bash */
    }
    else
    {
        int p = start; /* the word before this one */
        while (p > 0 && isspace((unsigned char) line[p - 1]))
        {
            p--;
        }
        int b = p;
        while (b > 0 && !isspace((unsigned char) line[b - 1]))
        {
            b--;
        }
        snprintf(ctype, sizeof(ctype), "arg:%s:%d:%.*s", cmdname, argidx + 1, p - b, line + b);
    }

    const char *w    = text + (inl ? 2 : 0);
    size_t      keep = 0;
    char        buf[32768];
    size_t      n     = milk_complete(w, ctype, buf, sizeof(buf));
    int         fuzzy = (n == 0 && data.autocomplete_fuzzy);
    if (fuzzy) /* the candidates after the word's last '.' that contain the rest of it */
    {
        const char *dot = strrchr(w, '.');
        char        cut[512];
        keep = (dot != NULL) ? (size_t) (dot - w + 1) : 0;
        snprintf(cut, sizeof(cut), "%.*s", (int) keep, w);
        n = milk_complete(cut, ctype, buf, sizeof(buf));
    }
    generator_fuzzy_pass = fuzzy;
    if (n > 0 && strncmp(buf, "__MILK_COMPLETE_F", 17) == 0) /* FILES or FITS */
    {
        return COMP_FILES;
    }
    size_t pos = 0;
    for (char *c = buf; n > 0 && *c != '\0';)
    {
        size_t l    = strcspn(c, "\n");
        char   save = c[l];
        c[l]        = '\0';
        if (l > 0 && (!fuzzy || (l >= keep && strstr(c + keep, w + keep) != NULL)))
        {
            int m = snprintf(out + pos, outsz - pos, "%.*s%s%s\n", (int) (text - full), full,
                             inl ? "@{" : "", c);
            pos += (m > 0 && pos + (size_t) m < outsz) ? (size_t) m : 0;
        }
        c[l] = save;
        c += l + (save != '\0');
    }
    return COMP_LIST;
}

/**
 * CLI_generator - Readline candidate generator: the next completion of @text
 * @text:  The word being completed (it starts at `cli_comp_start` in `rl_line_buffer`)
 * @state: 0 for the first call
 *
 * Return: a malloc()ed candidate, or NULL after the last one
 */
char *CLI_generator(const char *text, int state)
{
    static char  list[32768];
    static char *next = NULL;
    static int   kind = COMP_LIST;

    if (!state)
    {
        generator_fuzzy_pass = 0;
        kind = cli_complete_list(rl_line_buffer, cli_comp_start, text, list, sizeof(list));
        next = list;
    }
    if (kind == COMP_FILES)
    {
        return rl_filename_completion_function(text, state);
    }
    if (kind == COMP_SHELLVARS)
    {
        char *m = gen_vars_env_match(text, state);
        if (m == NULL && state == 0 && data.autocomplete_fuzzy) /* nothing starts with it */
        {
            generator_fuzzy_pass = 1;
            m                    = gen_vars_env_match(text, 0);
        }
        return m;
    }
    while (next != NULL && *next != '\0')
    {
        char  *nl = strchr(next, '\n');
        size_t n  = (nl != NULL) ? (size_t) (nl - next) : strlen(next);
        char  *c  = strndup(next, n);
        next      = (nl != NULL) ? nl + 1 : NULL;
        if (c != NULL && n > 0)
        {
            /* no space or closing quote after a path or a call: it goes on */
            rl_completion_suppress_append = (c[n - 1] == '.' || c[n - 1] == '/' || c[n - 1] == '(');
            rl_completion_suppress_quote  = rl_completion_suppress_append || g_comp_in_quote;
            return c;
        }
        free(c);
    }
    return NULL;
}

/**
 * @brief Readline custom completion dispatcher invoked on TAB.
 */
char **CLI_completion(const char *text, int start, int end)
{
    cli_comp_start                 = start;
    cli_comp_end                   = end;
    rl_completion_append_character = ' ';
    char **matches                 = rl_completion_matches((char *) text, &CLI_generator);
    rl_attempted_completion_over   = 1;
    return matches;
}

#endif /* USE_READLINE */
