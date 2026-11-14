// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-fps-set.c
 * @brief Set one or more FPS parameter values from CLI arguments or batch input
 */

#include <ctype.h>
#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_types.h"
#include "fps_unit_state.h"
#include "milk_help.h"

/* Helper to check if string starts with prefix */
static int starts_with(const char *pre, const char *str)
{
    size_t lenpre = strlen(pre);
    size_t lenstr = strlen(str);
    return lenstr < lenpre ? 0 : strncmp(pre, str, lenpre) == 0;
}

static const char *get_type_name(uint32_t type)
{
    if (type & FPTYPE_INT32)
    {
        return "INT32";
    }
    if (type & FPTYPE_UINT32)
    {
        return "UINT32";
    }
    if (type & FPTYPE_INT64)
    {
        return "INT64";
    }
    if (type & FPTYPE_UINT64)
    {
        return "UINT64";
    }
    if (type & FPTYPE_FLOAT32)
    {
        return "FLOAT32";
    }
    if (type & FPTYPE_FLOAT64)
    {
        return "FLOAT64";
    }
    if (type & FPTYPE_STRING)
    {
        return "STRING";
    }
    if (type & FPTYPE_FILENAME)
    {
        return "FILENAME";
    }
    if (type & FPTYPE_FITSFILENAME)
    {
        return "FITSFILENAME";
    }
    if (type & FPTYPE_EXECFILENAME)
    {
        return "EXECFILENAME";
    }
    if (type & FPTYPE_DIRNAME)
    {
        return "DIRNAME";
    }
    if (type & FPTYPE_STREAMNAME)
    {
        return "STREAMNAME";
    }
    if (type & FPTYPE_FPSNAME)
    {
        return "FPSNAME";
    }
    if (type & FPTYPE_ONOFF)
    {
        return "ONOFF";
    }
    if (type & FPTYPE_TIMESPEC)
    {
        return "TIMESPEC";
    }
    if (type & FPTYPE_PID)
    {
        return "PID";
    }
    return "UNKNOWN";
} // get_type_name

#define FSET_DESC "set parameter values in Function Parameter Structures (FPS)"
#define FSET_DESC_LONG                                                              \
    "Write new values to one or more FPS parameters in shared memory, with\n"       \
    "two-pass validation (no parameter is modified if any assignment in the\n"      \
    "invocation fails validation) and per-FPS connection caching.\n\n"              \
    "Accepts classic '<fps>.<param> <val>', multi-arg '<fps>.<param>=<val> ...',\n" \
    "or '-b <file|->' batch input ('key=val', 'key val', or 'setval key val').\n"   \
    "TIMESPEC parameters accept floating-point seconds (e.g. 0.0015)."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FSET_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<fps>.<param>%s %s<value>%s\n", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s%s%s [%soptions%s] %s<fps>.<param>=<value>%s [%s...%s]\n", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s%s%s [%soptions%s] %s-b <file|->%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FSET_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_ARG, "<fps>.<param>", 25),
           "Target FPS name and parameter path");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<value>", 25),
           "New parameter value (number, string, ON/OFF, seconds)");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-b, --batch <file|->", 25),
           "Read assignments from file or '-' (stdin)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--skip-running", 25),
           "Skip all writes to FPS units whose run loop is active");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--respect-writerun", 25),
           "When running, write WRITERUN params and skip locked ones");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-f, --force", 25),
           "Allow writing non-WRITERUN params even while running");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-q, --quiet", 25),
           "Suppress confirmation and skip messages");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 25), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 25),
           "Print one-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 25),
           "Print verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-hm, --help-mono", 25),
           "Full help without ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %smyfps00.procinfo.enabled%s %sON%s\n", mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "", mh_color ? MH_DFLT : "", mh_color ? MH_RST : "");
    printf("  %s$ %s%s%s %smfilt-7.loopgain=0.2 mfilt-7.loopmult=0.99%s\n", mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s$ %s%s%s %s--skip-running -b -%s <<EOF\n"
           "    dmcomb-10.DMmode = 0\n"
           "    dmcomb-10.xsize  = 50\n"
           "  EOF\n\n",
           mh_color ? MH_DIM : "", mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-fps-info:inspect FPS directory contents",
                               "milk-fps-list:list active FPS instances",
                               "milk-fps-track:monitor FPS parameters",
                               "milk-fps-runstart:start FPS run loop" };
    milk_help_see_also(see_also, 4, mh_color);
} // print_help

/**
 * @brief Scan FPS entries for bash tab completion.
 */
static void do_completion_scan(const char *word)
{
    const char *dot = strchr(word, '.');

    if (dot == NULL)
    {
        FPS               *fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
        KEYWORD_TREE_NODE *keywnode =
            (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
        int  NBkwn    = 0;
        int  NBfps    = 0;
        long NBpindex = 0;

        if (fpsarray != NULL && keywnode != NULL)
        {
            functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);
            for (int ii = 0; ii < NBfps; ii++)
            {
                if (starts_with(word, fpsarray[ii].md->name))
                {
                    printf("%s.\n", fpsarray[ii].md->name);
                }
                fps_disconnect(&fpsarray[ii]);
            }
        }
        free(fpsarray);
        free(keywnode);
    }
    else
    {
        char   fpsname[128];
        size_t len = (size_t) (dot - word);
        if (len >= sizeof(fpsname))
        {
            len = sizeof(fpsname) - 1;
        }
        memcpy(fpsname, word, len);
        fpsname[len] = '\0';

        const char *param_prefix = dot + 1;
        FPS         fps;
        long        NBparam = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
        if (NBparam != -1)
        {
            fps.NBparam = NBparam;
            for (int ii = 0; ii < fps.NBparam; ii++)
            {
                if (!(fps.parray[ii].fpflag & FPFLAG_ACTIVE))
                {
                    continue;
                }
                const char *kw = fps.parray[ii].keywordfull;
                if (strncmp(kw, fpsname, strlen(fpsname)) == 0 && kw[strlen(fpsname)] == '.')
                {
                    const char *suffix = kw + strlen(fpsname) + 1;
                    if (starts_with(param_prefix, suffix))
                    {
                        printf("%s\n", kw);
                    }
                }
                else if (kw[0] == '.' && starts_with(param_prefix, kw + 1))
                {
                    printf("%s%s\n", fpsname, kw);
                }
            }
            fps_disconnect(&fps);
        }
    } // if/else dot
} // do_completion_scan

/* ---------------------------------------------------------
 * Raw assignment list, connection cache, and 2-pass items
 * --------------------------------------------------------- */

typedef struct
{
    char *fullkey;
    char *valstr;
    char *source;
    int   lineno;
} raw_assign_t;

typedef struct
{
    raw_assign_t *items;
    size_t        len;
    size_t        cap;
} raw_list_t;

typedef struct
{
    char name[STRINGMAXLEN_FPS_NAME];
    FPS  fps;
    int  run_alive;
    int  skip_logged;
} fps_cache_entry_t;

typedef struct
{
    int           fps_idx;
    int           pindex;
    int           skip;
    CLICMDARGDATA ad;
    const char   *fullkey;
    const char   *valstr;
} resolved_set_t;

static char *dup_cstr(const char *s)
{
    size_t n = strlen(s) + 1;
    char  *p = (char *) malloc(n);
    if (p != NULL)
    {
        memcpy(p, s, n);
    }
    return p;
}

static int raw_list_push(raw_list_t *lst,
                         const char *key,
                         const char *val,
                         const char *src,
                         int         lineno)
{
    if (lst->len == lst->cap)
    {
        size_t        ncap = (lst->cap == 0) ? 32 : lst->cap * 2;
        raw_assign_t *nmem = (raw_assign_t *) realloc(lst->items, ncap * sizeof(*nmem));
        if (nmem == NULL)
        {
            return -1;
        }
        lst->items = nmem;
        lst->cap   = ncap;
    }
    raw_assign_t *it = &lst->items[lst->len];
    it->fullkey      = dup_cstr(key);
    it->valstr       = dup_cstr(val);
    it->source       = (src != NULL) ? dup_cstr(src) : NULL;
    it->lineno       = lineno;
    if (it->fullkey == NULL || it->valstr == NULL || (src != NULL && it->source == NULL))
    {
        free(it->fullkey);
        free(it->valstr);
        free(it->source);
        return -1;
    }
    lst->len++;
    return 0;
} // raw_list_push

static void raw_list_free(raw_list_t *lst)
{
    for (size_t ii = 0; ii < lst->len; ii++)
    {
        free(lst->items[ii].fullkey);
        free(lst->items[ii].valstr);
        free(lst->items[ii].source);
    }
    free(lst->items);
    lst->items = NULL;
    lst->len   = 0;
    lst->cap   = 0;
}

static char *trim_ws(char *s)
{
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')
    {
        s++;
    }
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r' || s[n - 1] == '\n'))
    {
        s[--n] = '\0';
    }
    return s;
}

static void clean_batch_value(char *val)
{
    size_t n = strlen(val);
    if (n >= 2 && ((val[0] == '"' && val[n - 1] == '"') || (val[0] == '\'' && val[n - 1] == '\'')))
    {
        val[n - 1] = '\0';
        memmove(val, val + 1, n - 1);
        return;
    }
    /* Strip trailing inline '# comment' on unquoted values when preceded by whitespace */
    for (size_t ii = 1; val[ii] != '\0'; ii++)
    {
        if (val[ii] == '#' && (val[ii - 1] == ' ' || val[ii - 1] == '\t'))
        {
            val[ii] = '\0';
            trim_ws(val);
            break;
        }
    }
} // clean_batch_value

static int load_batch_stream(FILE *fp, const char *srcname, raw_list_t *lst, const char *prog)
{
    char line[4096];
    int  lineno = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        lineno++;
        size_t rawlen = strlen(line);
        if (rawlen == sizeof(line) - 1 && line[rawlen - 1] != '\n')
        {
            fprintf(stderr, "%s: %s:%d: line exceeds %zu bytes\n", prog, srcname, lineno,
                    sizeof(line) - 1);
            return 2;
        }

        char *p = trim_ws(line);
        if (*p == '\0' || *p == '#')
        {
            continue;
        }

        if (strncmp(p, "setval", 6) == 0 && (p[6] == ' ' || p[6] == '\t'))
        {
            p = trim_ws(p + 6);
        }

        char *sep = p;
        while (*sep != '\0' && *sep != '=' && *sep != ' ' && *sep != '\t')
        {
            sep++;
        }
        if (*sep == '\0')
        {
            fprintf(stderr, "%s: %s:%d: missing value for '%s'\n", prog, srcname, lineno, p);
            return 2;
        }

        char  delim = *sep;
        char *rest  = sep + 1;
        *sep        = '\0';
        rest        = trim_ws(rest);
        if (delim != '=' && *rest == '=')
        {
            rest = trim_ws(rest + 1);
        }
        clean_batch_value(rest);
        if (*p == '\0' || *rest == '\0')
        {
            fprintf(stderr, "%s: %s:%d: malformed assignment\n", prog, srcname, lineno);
            return 2;
        }

        if (raw_list_push(lst, p, rest, srcname, lineno) != 0)
        {
            fprintf(stderr, "%s: out of memory\n", prog);
            return 1;
        }
    } // while fgets
    return 0;
} // load_batch_stream

static int load_batch_file(const char *path, raw_list_t *lst, const char *prog)
{
    if (strcmp(path, "-") == 0)
    {
        return load_batch_stream(stdin, "<stdin>", lst, prog);
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "%s: cannot open batch file '%s': %s\n", prog, path, strerror(errno));
        return 1;
    }
    int rc = load_batch_stream(fp, path, lst, prog);
    fclose(fp);
    return rc;
}

static int cache_get_fps(fps_cache_entry_t **cache_ptr,
                         int                *ncache,
                         int                *capcache,
                         const char         *fpsname)
{
    fps_cache_entry_t *cache = *cache_ptr;
    for (int ii = 0; ii < *ncache; ii++)
    {
        if (strcmp(cache[ii].name, fpsname) == 0)
        {
            return ii;
        }
    }

    if (*ncache == *capcache)
    {
        int                ncap = (*capcache == 0) ? 16 : (*capcache * 2);
        fps_cache_entry_t *nmem =
            (fps_cache_entry_t *) realloc(cache, (size_t) ncap * sizeof(*nmem));
        if (nmem == NULL)
        {
            return -2;
        }
        *cache_ptr = nmem;
        cache      = nmem;
        *capcache  = ncap;
    }

    fps_cache_entry_t *ce = &cache[*ncache];
    memset(ce, 0, sizeof(*ce));
    ce->fps.SMfd = -1;
    long nbparam = fps_connect(fpsname, &ce->fps, FPSCONNECT_SIMPLE);
    if (nbparam == -1)
    {
        return -1;
    }
    ce->fps.NBparam = nbparam;
    snprintf(ce->name, sizeof(ce->name), "%s", fpsname);

    FPS_UNIT_STATUS st;
    fps_unit_status_fps(&ce->fps, FPS_UNIT_RUN, &st, FPS_UNIT_REPAIR);
    ce->run_alive = fps_unit_state_alive(st.state);

    return (*ncache)++;
} // cache_get_fps

static void cache_disconnect_all(fps_cache_entry_t *cache, int ncache)
{
    for (int ii = 0; ii < ncache; ii++)
    {
        fps_disconnect(&cache[ii].fps);
    }
    free(cache);
}

int main(int argc, char *argv[])
{
    char       *progbuf  = strdup(argv[0]);
    const char *progname = (progbuf != NULL) ? basename(progbuf) : argv[0];

    int action = milk_help_init(argc, argv, FSET_DESC, FSET_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        free(progbuf);
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        free(progbuf);
        return 0;
    }

    /* Bash completion hook */
    if (getenv("COMP_LINE") != NULL)
    {
        if (argc >= 3)
        {
            do_completion_scan(argv[2]);
            free(progbuf);
            return 0;
        }
    }

    int         opt_skip_running     = 0;
    int         opt_respect_writerun = 0;
    int         opt_force            = 0;
    int         opt_quiet            = 0;
    int         opts_done            = 0;
    raw_list_t  raw                  = { NULL, 0, 0 };
    const char *pos_args[1024];
    int         npos = 0;
    int         rc   = 0;

    for (int ii = 1; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (!opts_done && strcmp(arg, "--") == 0)
        {
            opts_done = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-b") == 0 || strcmp(arg, "--batch") == 0))
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires a file argument\n", progname, arg);
                rc = 2;
                goto done_raw;
            }
            rc = load_batch_file(argv[++ii], &raw, progname);
            if (rc != 0)
            {
                goto done_raw;
            }
            continue;
        }
        if (!opts_done && strncmp(arg, "--batch=", 8) == 0)
        {
            rc = load_batch_file(arg + 8, &raw, progname);
            if (rc != 0)
            {
                goto done_raw;
            }
            continue;
        }
        if (!opts_done && strcmp(arg, "--skip-running") == 0)
        {
            opt_skip_running = 1;
            continue;
        }
        if (!opts_done && strcmp(arg, "--respect-writerun") == 0)
        {
            opt_respect_writerun = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-f") == 0 || strcmp(arg, "--force") == 0))
        {
            opt_force = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0))
        {
            opt_quiet = 1;
            continue;
        }
        if (!opts_done && arg[0] == '-')
        {
            fprintf(stderr, "%s: invalid option '%s'\n", progname, arg);
            rc = 2;
            goto done_raw;
        }

        /* First positional argument stops option scanning so negative values like -1 work */
        opts_done = 1;
        if (npos >= (int) (sizeof(pos_args) / sizeof(pos_args[0])))
        {
            fprintf(stderr, "%s: too many positional arguments (use -b)\n", progname);
            rc = 2;
            goto done_raw;
        }
        pos_args[npos++] = arg;
    } // for ii

    if (npos == 2 && strchr(pos_args[0], '=') == NULL)
    {
        if (raw_list_push(&raw, pos_args[0], pos_args[1], NULL, 0) != 0)
        {
            rc = 1;
            goto done_raw;
        }
    }
    else if (npos > 0)
    {
        for (int ii = 0; ii < npos; ii++)
        {
            const char *eq = strchr(pos_args[ii], '=');
            if (eq == NULL)
            {
                if (npos == 1)
                {
                    fprintf(stderr, "%s: missing value for '%s'\n", progname, pos_args[ii]);
                }
                else
                {
                    fprintf(stderr,
                            "%s: unexpected argument '%s' (use '<fps>.<param>=<val>' for multiple "
                            "assignments)\n",
                            progname, pos_args[ii]);
                }
                rc = 2;
                goto done_raw;
            }
            size_t klen = (size_t) (eq - pos_args[ii]);
            if (klen == 0 || klen >= 512 || eq[1] == '\0')
            {
                fprintf(stderr, "%s: malformed assignment '%s'\n", progname, pos_args[ii]);
                rc = 2;
                goto done_raw;
            }
            char kbuf[512];
            memcpy(kbuf, pos_args[ii], klen);
            kbuf[klen] = '\0';
            if (raw_list_push(&raw, kbuf, eq + 1, NULL, 0) != 0)
            {
                rc = 1;
                goto done_raw;
            }
        }
    }

    if (raw.len == 0)
    {
        fprintf(stderr, "%s: missing parameter assignment\n", progname);
        print_help(progname, 1);
        rc = 2;
        goto done_raw;
    }

    {
        fps_cache_entry_t *cache    = NULL;
        int                ncache   = 0;
        int                capcache = 0;
        resolved_set_t    *plan     = (resolved_set_t *) calloc(raw.len, sizeof(*plan));
        if (plan == NULL)
        {
            fprintf(stderr, "%s: out of memory\n", progname);
            rc = 1;
            goto done_raw;
        }

        /* Pass 1: Connect once per FPS, resolve indices, check running guards, parse values */
        for (size_t ii = 0; ii < raw.len; ii++)
        {
            const raw_assign_t *ra       = &raw.items[ii];
            const char         *dot      = strchr(ra->fullkey, '.');
            char                loc[256] = "";
            if (ra->source != NULL)
            {
                snprintf(loc, sizeof(loc), "%s:%d: ", ra->source, ra->lineno);
            }

            if (dot == NULL || dot == ra->fullkey || dot[1] == '\0')
            {
                fprintf(stderr, "%s: %sinvalid format '%s' (expected <fps>.<param>)\n", progname,
                        loc, ra->fullkey);
                rc = 2;
                free(plan);
                cache_disconnect_all(cache, ncache);
                goto done_raw;
            }

            char   fpsname[STRINGMAXLEN_FPS_NAME];
            size_t flen = (size_t) (dot - ra->fullkey);
            if (flen >= sizeof(fpsname))
            {
                fprintf(stderr, "%s: %sFPS name too long in '%s'\n", progname, loc, ra->fullkey);
                rc = 2;
                free(plan);
                cache_disconnect_all(cache, ncache);
                goto done_raw;
            }
            memcpy(fpsname, ra->fullkey, flen);
            fpsname[flen] = '\0';

            int cidx = cache_get_fps(&cache, &ncache, &capcache, fpsname);
            if (cidx < 0)
            {
                fprintf(stderr, "%s: %scannot connect to FPS '%s'\n", progname, loc, fpsname);
                rc = 1;
                free(plan);
                cache_disconnect_all(cache, ncache);
                goto done_raw;
            }

            fps_cache_entry_t *ce     = &cache[cidx];
            int                pindex = functionparameter_GetParamIndex(&ce->fps, dot);
            if (pindex < 0)
            {
                pindex = functionparameter_GetParamIndex(&ce->fps, dot + 1);
            }
            if (pindex < 0)
            {
                fprintf(stderr, "%s: %sparameter '%s' not found in FPS '%s'\n", progname, loc, dot,
                        fpsname);
                rc = 1;
                free(plan);
                cache_disconnect_all(cache, ncache);
                goto done_raw;
            }

            resolved_set_t *pl = &plan[ii];
            pl->fps_idx        = cidx;
            pl->pindex         = pindex;
            pl->fullkey        = ra->fullkey;
            pl->valstr         = ra->valstr;
            pl->ad.type        = ce->fps.parray[pindex].type;

            if (milk_argval_parse(pl->ad.type, ra->valstr, &pl->ad) != 0 ||
                (FPTYPE_IS_STRING(pl->ad.type) &&
                 strlen(pl->ad.val.s) >= FUNCTION_PARAMETER_STRMAXLEN))
            {
                fprintf(stderr, "%s: %sinvalid %s value '%s' for '%s'\n", progname, loc,
                        get_type_name(pl->ad.type), ra->valstr, ra->fullkey);
                rc = 2;
                free(plan);
                cache_disconnect_all(cache, ncache);
                goto done_raw;
            }

            if (ce->run_alive)
            {
                int writerun = (ce->fps.parray[pindex].fpflag & FPFLAG_WRITERUN) != 0;
                if (opt_skip_running)
                {
                    pl->skip = 1;
                    if (!opt_quiet && !ce->skip_logged)
                    {
                        fprintf(stderr, "%s: skipping running unit '%s'\n", progname, fpsname);
                        ce->skip_logged = 1;
                    }
                }
                else if (!writerun && opt_respect_writerun)
                {
                    pl->skip = 1;
                    if (!opt_quiet)
                    {
                        fprintf(stderr, "%s: skipping non-WRITERUN parameter '%s' (running)\n",
                                progname, ra->fullkey);
                    }
                }
                else if (!writerun && !opt_force)
                {
                    fprintf(stderr,
                            "%s: %sparameter '%s' lacks FPFLAG_WRITERUN while '%s' is running "
                            "(use --skip-running, --respect-writerun, or -f)\n",
                            progname, loc, ra->fullkey, fpsname);
                    rc = 1;
                    free(plan);
                    cache_disconnect_all(cache, ncache);
                    goto done_raw;
                }
            } // if run_alive
        } // for Pass 1

        /* Pass 2: Commit validated parameter updates */
        for (size_t ii = 0; ii < raw.len; ii++)
        {
            resolved_set_t *pl = &plan[ii];
            if (pl->skip)
            {
                continue;
            }
            FPS *fps = &cache[pl->fps_idx].fps;
            if (functionparameter_SetParamValue_arg(fps, pl->pindex, &pl->ad) != 0)
            {
                fprintf(stderr, "%s: failed to write '%s'\n", progname, pl->fullkey);
                rc = 1;
                break;
            }
            char tname[STRINGMAXLEN_FPSTYPE];
            char shown[STRINGMAXLEN_CLICMDARG];
            functionparameter_GetTypeString((uint32_t) pl->ad.type, tname);
            milk_argval_format(&pl->ad, shown, sizeof(shown));
            functionparameter_outlog("SETVAL", "%s%s%s", fps->parray[pl->pindex].keywordfull, tname,
                                     shown);
            if (!opt_quiet && raw.len == 1)
            {
                printf("Parameter '%s' set to '%s'\n", pl->fullkey, pl->valstr);
            }
        } // for Pass 2

        free(plan);
        cache_disconnect_all(cache, ncache);
    }

done_raw:
    raw_list_free(&raw);
    free(progbuf);
    return rc;
} // main
