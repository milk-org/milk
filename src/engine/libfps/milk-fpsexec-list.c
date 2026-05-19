/**
 * @file    milk-fpsexec-list.c
 * @brief   List installed milk-fpsexec-* compute units
 *
 * Replaces the bash script milk-fpsexec-list.
 * Discovers commands by scanning PATH directories with glob(3).
 * Fetches descriptions via popen("-h1").
 * Supports regex (default) and fuzzy subsequence filtering.
 */

#define _GNU_SOURCE

#include <ctype.h>
#include <glob.h>
#include <regex.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "milk_help.h"

#define FEL_DESC \
    "list installed milk-fpsexec-* standalone compute units"

#define FEL_DESC_LONG \
    "List all installed milk-fpsexec-* standalone executables\n" \
    "with their one-line descriptions.\n" \
    "\n" \
    "These executables are compute units that can run\n" \
    "independently or be managed by the FPS framework.\n" \
    "Supports regex and fuzzy subsequence search to filter\n" \
    "results. Optional JSON output for tooling integration."

/* Column width for command name */
#define COL_NAME 45
/* Maximum number of discovered executables */
#define FEL_MAX_CMDS 1024
/* Max description length */
#define FEL_DESC_MAX 256

/* Fuzzy scoring bonuses */
#define FZ_CHAR_SCORE   10
#define FZ_CONSEC_BONUS  5
#define FZ_BOUND_BONUS   8
#define FZ_START_BONUS   3

/** Entry: one discovered milk-fpsexec-* command */
struct felentry
{
    char  name[128];
    char  desc[FEL_DESC_MAX];
    int   score;        /* fuzzy score (0 = not set) */
    /* highlight positions (fuzzy mode) */
    int   hl_pos[FEL_DESC_MAX + 128];
    int   hl_n;
};

static struct felentry g_entries[FEL_MAX_CMDS];
static int             g_n_entries;

/* ----------------------------------------------------------------
 * print_help
 * -------------------------------------------------------------- */
static void print_help(
    const char *progname,
    int mh_color)
{
    milk_help_banner(progname, FEL_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sSEARCH_TERM%s ...]\n\n",
           mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FEL_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-f, --fuzzy",
           mh_color ? MH_RST : "",
           "Fuzzy subsequence match (chars in order, not adjacent)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-j, --json",
           mh_color ? MH_RST : "",
           "Output as JSON array [{name, description}]");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "",
           "One-line description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Arguments", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_ARG : "", "SEARCH_TERM",
           mh_color ? MH_RST : "",
           "Filter terms (regex by default, fuzzy with -f)");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_ARG : "", "",
           mh_color ? MH_RST : "",
           "Multiple terms in fuzzy mode: all must match");
    milk_help_section("Fuzzy Score", mh_color);
    printf("  +%-2d per matched char, +%-2d consecutive, "
           "+%-2d word boundary, +%d start\n\n",
           FZ_CHAR_SCORE, FZ_CONSEC_BONUS,
           FZ_BOUND_BONUS, FZ_START_BONUS);
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-fpsexec-list%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fpsexec-list%s %sim%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fpsexec-list%s %s-f gauss 2d%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fpsexec-list%s %s-j%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");
    const char *see_also[] =
    {
        "milk-fpsexec-help:print detailed FPS framework info",
        "milk-fpsCTRL:launch the FPS dashboard TUI",
        "milk-fps-set:set an FPS parameter value"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

/* ----------------------------------------------------------------
 * fetch_description() - Run <cmd> -h1 and capture one line
 * Returns 1 on success, 0 on failure.
 * -------------------------------------------------------------- */
static int fetch_description(
    const char *cmd,
    char       *desc,
    size_t     descsz)
{
    char cmdline[256];
    snprintf(cmdline, sizeof(cmdline), "%s -h1 2>/dev/null", cmd);

    FILE *fp = popen(cmdline, "r");
    if(fp == NULL)
    {
        return 0;
    }

    int ok = 0;
    if(fgets(desc, (int)descsz, fp) != NULL)
    {
        /* Strip trailing newline */
        size_t n = strlen(desc);
        if(n > 0 && desc[n - 1] == '\n')
        {
            desc[n - 1] = '\0';
        }
        ok = (strlen(desc) > 0);
    }
    pclose(fp);
    return ok;
}

/* ----------------------------------------------------------------
 * discover_commands() - Find all milk-fpsexec-* in PATH
 * Fills g_entries[], sets g_n_entries.
 * -------------------------------------------------------------- */
static const char *g_skip[] =
{
    "milk-fpsexec-help",
    "milk-fpsexec-list",
    NULL
};

/**
 * @brief Checks if a given fps name should be skipped from listing.
 */
static int should_skip(const char *name)
{
    for(int skpi = 0; g_skip[skpi] != NULL; skpi++)
    {
        if(strcmp(name, g_skip[skpi]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Discovers and prints installed fps executables by scanning the PATH.
 */
static void discover_commands(void)
{
    const char *path_env = getenv("PATH");
    if(path_env == NULL)
    {
        return;
    }

    char path_copy[8192];
    strncpy(path_copy, path_env, sizeof(path_copy) - 1);
    path_copy[sizeof(path_copy) - 1] = '\0';

    /* Track seen names to deduplicate across PATH dirs */
    static char seen[FEL_MAX_CMDS][128];
    int n_seen = 0;

    char *dir = strtok(path_copy, ":");
    while(dir != NULL && g_n_entries < FEL_MAX_CMDS)
    {
        char pattern[512];
        snprintf(pattern, sizeof(pattern),
                 "%s/milk-fpsexec-*", dir);

        glob_t gl;
        if(glob(pattern, GLOB_NOSORT, NULL, &gl) == 0)
        {
            for(size_t path_idx = 0;
                    path_idx < gl.gl_pathc && g_n_entries < FEL_MAX_CMDS;
                    path_idx++)
            {
                /* Extract basename */
                const char *bname =
                    strrchr(gl.gl_pathv[path_idx], '/');
                bname = bname ? bname + 1 : gl.gl_pathv[path_idx];

                if(should_skip(bname))
                {
                    continue;
                }

                /* Dedup check */
                int dup = 0;
                for(int seen_idx = 0; seen_idx < n_seen; seen_idx++)
                {
                    if(strcmp(seen[seen_idx], bname) == 0)
                    {
                        dup = 1;
                        break;
                    }
                }
                if(dup)
                {
                    continue;
                }

                /* Only include executables */
                struct stat st;
                if(stat(gl.gl_pathv[path_idx], &st) != 0 ||
                        !(st.st_mode & S_IXUSR))
                {
                    continue;
                }

                struct felentry *e = &g_entries[g_n_entries];
                strncpy(e->name, bname, sizeof(e->name) - 1);
                e->name[sizeof(e->name) - 1] = '\0';

                if(!fetch_description(
                            gl.gl_pathv[path_idx], e->desc,
                            sizeof(e->desc)))
                {
                    strncpy(e->desc, "(no description)",
                            sizeof(e->desc) - 1);
                }

                e->score = 0;
                e->hl_n  = 0;

                /* Record in seen list */
                if(n_seen < FEL_MAX_CMDS)
                {
                    strncpy(seen[n_seen], bname,
                            sizeof(seen[0]) - 1);
                    n_seen++;
                }

                g_n_entries++;
            } // for path_idx
        } // if glob ok
        globfree(&gl);
        dir = strtok(NULL, ":");
    } // while dir
}

/* ----------------------------------------------------------------
 * Comparison functions for qsort
 * -------------------------------------------------------------- */
static int cmp_name(
    const void *a,
    const void *b)
{
    const struct felentry *ea = (const struct felentry *)a;
    const struct felentry *eb = (const struct felentry *)b;
    return strcmp(ea->name, eb->name);
}

/**
 * @brief Compare two entries by search score (descending).
 *
 * Used by qsort for fuzzy-match result ranking.
 */
static int cmp_score_desc(
    const void *a,
    const void *b)
{
    const struct felentry *ea = (const struct felentry *)a;
    const struct felentry *eb = (const struct felentry *)b;
    return eb->score - ea->score; /* higher score first */
}

/* ----------------------------------------------------------------
 * fuzzy_score() - Subsequence match with scoring
 *
 * Fills positions[] (up to posmax) with matched char indices in
 * the combined "name  desc" string. Returns score (0 = no match).
 *
 * Scoring:
 *   +10 per matched char
 *   +5  consecutive match bonus
 *   +8  word boundary bonus (prev char is - _ space)
 *   +3  start-of-string bonus
 * -------------------------------------------------------------- */
static int fuzzy_score(
    const char *query,
    const char *text,
    int        *positions,
    int        posmax,
    int        *n_pos_out)
{
    size_t qlen = strlen(query);
    size_t tlen = strlen(text);

    int score     = 0;
    int qi        = 0;
    int prev_ti   = -2;
    int n_pos     = 0;

    for(size_t ti = 0;
            ti < tlen && (size_t)qi < qlen;
            ti++)
    {
        char tc = (char)tolower((unsigned char)text[ti]);
        char qc = (char)tolower((unsigned char)query[qi]);

        if(tc != qc)
        {
            continue;
        }

        /* Record position */
        if(n_pos < posmax)
        {
            positions[n_pos++] = (int)ti;
        }

        score += FZ_CHAR_SCORE;

        if(prev_ti + 1 == (int)ti)
        {
            score += FZ_CONSEC_BONUS;
        }

        if(ti == 0)
        {
            score += FZ_START_BONUS;
        }
        else
        {
            char prev_c = text[ti - 1];
            if(prev_c == '-' || prev_c == '_' ||
                    prev_c == ' ')
            {
                score += FZ_BOUND_BONUS;
            }
        }

        prev_ti = (int)ti;
        qi++;
    } // for ti

    if((size_t)qi < qlen)
    {
        /* Full query not consumed -- no match */
        if(n_pos_out)
        {
            *n_pos_out = 0;
        }
        return 0;
    }

    if(n_pos_out)
    {
        *n_pos_out = n_pos;
    }
    return score;
}

/* ----------------------------------------------------------------
 * print_highlighted() - Print a string with ANSI highlights
 *
 * Characters at positions in hl_pos[] are printed in bold yellow.
 * -------------------------------------------------------------- */
static void print_highlighted(
    const char *s,
    const int  *hl_pos,
    int        hl_n)
{
    /* Build a quick lookup set */
    int slen = (int)strlen(s);
    /* Use a bitset -- max COL_NAME+FEL_DESC_MAX chars */
    static char is_hl[COL_NAME + FEL_DESC_MAX + 4];
    memset(is_hl, 0, sizeof(is_hl));

    for(int hl_idx = 0; hl_idx < hl_n; hl_idx++)
    {
        if(hl_pos[hl_idx] < (int)sizeof(is_hl))
        {
            is_hl[hl_pos[hl_idx]] = 1;
        }
    }

    int in_hl = 0;
    for(int ch_idx = 0; ch_idx < slen; ch_idx++)
    {
        if(is_hl[ch_idx] && !in_hl)
        {
            printf("\033[1;33m");
            in_hl = 1;
        }
        else if(!is_hl[ch_idx] && in_hl)
        {
            printf("\033[0m");
            in_hl = 0;
        }
        putchar(s[ch_idx]);
    }
    if(in_hl)
    {
        printf("\033[0m");
    }
}

/* ----------------------------------------------------------------
 * print_header() - Print table header
 * -------------------------------------------------------------- */
static void print_header(void)
{
    printf("\033[1;34m%-*s %s\033[0m\n",
           COL_NAME, "COMMAND", "DESCRIPTION");

    for(int ch_idx = 0; ch_idx < COL_NAME; ch_idx++)
    {
        putchar('-');
    }
    printf(" ");
    for(int ch_idx = 0; ch_idx < 40; ch_idx++)
    {
        putchar('-');
    }
    putchar('\n');
}

/* ----------------------------------------------------------------
 * output_json() - Emit JSON array of matching entries
 * -------------------------------------------------------------- */
static void output_json(
    const struct felentry *entries,
    int                   n)
{
    printf("[\n");
    for(int ent_idx = 0; ent_idx < n; ent_idx++)
    {
        /* Escape double quotes in description */
        printf("  {\"name\": \"%s\", \"description\": \"",
               entries[ent_idx].name);
        for(const char *p = entries[ent_idx].desc; *p; p++)
        {
            if(*p == '"')
            {
                putchar('\\');
            }
            putchar(*p);
        }
        printf("\"}%s\n", ent_idx + 1 < n ? "," : "");
    }
    printf("]\n");
}

/* ----------------------------------------------------------------
 * main
 * -------------------------------------------------------------- */
int main(
    int argc,
    char *argv[])
{
    int action = milk_help_init(
                     argc, argv, FEL_DESC, FEL_DESC_LONG);

    if(action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if(action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    /* Parse arguments */
    int   fuzzy    = 0;
    int   json_out = 0;
    int   n_terms  = 0;
    const char *terms[64];

    for(int arg_idx = 1; arg_idx < argc; arg_idx++)
    {
        if(strcmp(argv[arg_idx], "-f") == 0 ||
                strcmp(argv[arg_idx], "--fuzzy") == 0)
        {
            fuzzy = 1;
        }
        else if(strcmp(argv[arg_idx], "-j") == 0 ||
                strcmp(argv[arg_idx], "--json") == 0)
        {
            json_out = 1;
        }
        else if(argv[arg_idx][0] != '-')
        {
            if(n_terms < 64)
            {
                terms[n_terms++] = argv[arg_idx];
            }
        }
        else
        {
            fprintf(stderr,
                    "\n\033[1;31mERROR\033[0m:"
                    " unknown option '%s'.\n\n", argv[arg_idx]);
            print_help(argv[0], 1);
            return 1;
        }
    } // for arg_idx

    /* Discover commands */
    discover_commands();

    /* Sort alphabetically before filtering */
    qsort(g_entries, (size_t)g_n_entries,
          sizeof(g_entries[0]), cmp_name);

    /* Filter */
    static struct felentry filtered[FEL_MAX_CMDS];
    int n_filtered = 0;

    for(int ent_idx = 0; ent_idx < g_n_entries; ent_idx++)
    {
        struct felentry *e = &g_entries[ent_idx];

        /* Build the "line" as "name  desc" for matching */
        char line[COL_NAME + FEL_DESC_MAX + 4];
        snprintf(line, sizeof(line),
                 "%-*s %s", COL_NAME, e->name, e->desc);

        if(n_terms == 0)
        {
            filtered[n_filtered++] = *e;
            continue;
        }

        if(fuzzy)
        {
            /* All terms must match; accumulate score + positions */
            int total_score = 0;
            static int all_pos[FEL_MAX_CMDS];
            int        n_all_pos = 0;
            int        matched = 1;

            for(int term_idx = 0; term_idx < n_terms; term_idx++)
            {
                static int pos[FEL_DESC_MAX + 128];
                int n_pos = 0;
                int sc = fuzzy_score(
                             terms[term_idx], line, pos,
                             (int)(sizeof(pos) / sizeof(pos[0])),
                             &n_pos);
                if(sc == 0)
                {
                    matched = 0;
                    break;
                }
                total_score += sc;
                for(int pos_idx = 0;
                        pos_idx < n_pos &&
                        n_all_pos < (int)(sizeof(all_pos) /
                                          sizeof(all_pos[0]));
                        pos_idx++)
                {
                    all_pos[n_all_pos++] = pos[pos_idx];
                }
            } // for term_idx

            if(!matched)
            {
                continue;
            }

            struct felentry fe = *e;
            fe.score = total_score;
            /* Copy highlight positions */
            int copy_n = n_all_pos <
                         (int)(sizeof(fe.hl_pos) /
                               sizeof(fe.hl_pos[0]))
                         ? n_all_pos
                         : (int)(sizeof(fe.hl_pos) /
                                 sizeof(fe.hl_pos[0]));
            memcpy(fe.hl_pos, all_pos,
                   (size_t)copy_n * sizeof(int));
            fe.hl_n = copy_n;
            filtered[n_filtered++] = fe;

        }
        else
        {
            /* Regex mode: first term only */
            regex_t re;
            if(regcomp(&re, terms[0],
                       REG_EXTENDED | REG_ICASE |
                       REG_NOSUB) != 0)
            {
                fprintf(stderr,
                        "\033[1;31mERROR\033[0m:"
                        " invalid regex '%s'\n", terms[0]);
                return 1;
            }
            if(regexec(&re, line, 0, NULL, 0) == 0)
            {
                filtered[n_filtered++] = *e;
            }
            regfree(&re);
        }
    } // for ent_idx

    /* Sort fuzzy results by score */
    if(fuzzy && n_terms > 0)
    {
        qsort(filtered, (size_t)n_filtered,
              sizeof(filtered[0]), cmp_score_desc);
    }

    /* Output */
    if(json_out)
    {
        output_json(filtered, n_filtered);
        return 0;
    }

    if(n_filtered == 0)
    {
        if(n_terms > 0)
        {
            printf("No matches found");
            if(fuzzy)
            {
                printf(" (fuzzy)");
            }
            printf(" for '%s", terms[0]);
            for(int term_idx = 1; term_idx < n_terms; term_idx++)
            {
                printf(" %s", terms[term_idx]);
            }
            printf("'.\n");
        }
        return 0;
    }

    print_header();

    for(int ent_idx = 0; ent_idx < n_filtered; ent_idx++)
    {
        struct felentry *e = &filtered[ent_idx];

        if(fuzzy && n_terms > 0)
        {
            char line[COL_NAME + FEL_DESC_MAX + 4];
            snprintf(line, sizeof(line),
                     "%-*s %s", COL_NAME,
                     e->name, e->desc);
            printf("[%3d] ", e->score);
            print_highlighted(line, e->hl_pos, e->hl_n);
            putchar('\n');
        }
        else
        {
            printf("%-*s %s\n",
                   COL_NAME, e->name, e->desc);
        }
    } // for ent_idx

    return 0;
}
