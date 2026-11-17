// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-fps-init.c
 * @brief   Initialize one or more FPS compute units in parallel with idempotent skip.
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "computeunit.h"
#include "fps.h"
#include "milk_help.h"
#include "milk_spawn.h"

#define FI_DESC "initialize one or more FPS compute units in parallel"
#define FI_DESC_LONG                                                                     \
    "Create shared-memory FPS structures for one or more standalone fpsexec units.\n"    \
    "By default, units whose <name>.fps.shm already exists are skipped in place\n"       \
    "(idempotent warm fast path). Missing units (or all listed units under --restart)\n" \
    "are initialized concurrently via posix_spawn."

typedef struct
{
    char  *fpsname;
    char  *bin;
    char **args;
    int    nargs;
    int    existed_before;
    int    needs_init;
} INIT_SPEC;

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FI_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<fpsname>%s %s<fpsexec_bin>%s [%sinit_args...%s]\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s%s%s [%soptions%s] %s-b%s %s<file|->%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FI_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  %s%-24s%s %s\n", mh_color ? MH_ARG : "", "<fpsname>", mh_color ? MH_RST : "",
           "Target FPS instance name");
    printf("  %s%-24s%s %s\n", mh_color ? MH_ARG : "", "<fpsexec_bin>", mh_color ? MH_RST : "",
           "Standalone *-fpsexec-* executable name or path");
    printf("  %s%-24s%s %s\n\n", mh_color ? MH_ARG : "", "[init_args...]", mh_color ? MH_RST : "",
           "Optional unit flags (-loops, -loopd SEC) and .param=val bindings");

    milk_help_section("Options", mh_color);
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-b, --batch <file|->",
           mh_color ? MH_RST : "", "Read '<fpsname> <fpsexec_bin> [args...]' lines");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "--restart", mh_color ? MH_RST : "",
           "Stop active run/conf workers and re-initialize existing units");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "--print-created", mh_color ? MH_RST : "",
           "Print newly created FPS names to stdout (one per line)");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-q, --quiet", mh_color ? MH_RST : "",
           "Suppress unit fpsinit stdout messages");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-24s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$%s %smilk-fps-init%s %smfilt-7 cacao-fpsexec-mfilt -loops .AOloopindex=7%s\n",
           mh_color ? MH_DIM : "", mh_color ? MH_RST : "", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$%s %smilk-fps-init%s %s-q --print-created -b - <<EOF%s\n\n",
           mh_color ? MH_DIM : "", mh_color ? MH_RST : "", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-fps-set:set FPS parameter values",
                               "milk-fps-runstart:start FPS run loop",
                               "milk-fps-rm:remove FPS from shared memory" };
    milk_help_see_also(see_also, 3, mh_color);
} // print_help

static char *dup_str(const char *src, size_t len)
{
    char *out = malloc(len + 1);
    if (out != NULL)
    {
        memcpy(out, src, len);
        out[len] = '\0';
    }
    return out;
} // dup_str

static void free_specs(INIT_SPEC *specs, int nspecs)
{
    if (specs == NULL)
    {
        return;
    }
    for (int spec_idx = 0; spec_idx < nspecs; spec_idx++)
    {
        free(specs[spec_idx].fpsname);
        free(specs[spec_idx].bin);
        for (int arg_idx = 0; arg_idx < specs[spec_idx].nargs; arg_idx++)
        {
            free(specs[spec_idx].args[arg_idx]);
        }
        free(specs[spec_idx].args);
    } // for spec_idx
    free(specs);
} // free_specs

static int valid_fpsname(const char *name)
{
    if (name == NULL || name[0] == '\0' || strlen(name) >= STRINGMAXLEN_FPS_NAME)
    {
        return 0;
    }
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0 || strchr(name, '/') != NULL)
    {
        return 0;
    }
    return 1;
} // valid_fpsname

static int append_spec(INIT_SPEC **specs_ptr,
                       int        *nspecs_ptr,
                       int        *cap_ptr,
                       char      **tokens,
                       int         ntokens,
                       int         lineno)
{
    if (ntokens < 2)
    {
        fprintf(stderr, "Error: line %d requires '<fpsname> <fpsexec_bin> [args...]'\n", lineno);
        return -1;
    }
    if (!valid_fpsname(tokens[0]))
    {
        fprintf(stderr, "Error: invalid FPS name '%s'\n", tokens[0]);
        return -1;
    }
    for (int prev_idx = 0; prev_idx < *nspecs_ptr; prev_idx++)
    {
        if (strcmp((*specs_ptr)[prev_idx].fpsname, tokens[0]) == 0)
        {
            fprintf(stderr, "Error: duplicate FPS name '%s' in batch\n", tokens[0]);
            return -1;
        }
    } // for prev_idx

    if (*nspecs_ptr >= *cap_ptr)
    {
        int        new_cap   = (*cap_ptr == 0) ? 16 : (*cap_ptr * 2);
        INIT_SPEC *new_specs = realloc(*specs_ptr, (size_t) new_cap * sizeof(INIT_SPEC));
        if (new_specs == NULL)
        {
            return -1;
        }
        *specs_ptr = new_specs;
        *cap_ptr   = new_cap;
    }

    INIT_SPEC *sp = &(*specs_ptr)[*nspecs_ptr];
    memset(sp, 0, sizeof(*sp));
    sp->fpsname = tokens[0];
    sp->bin     = tokens[1];
    sp->nargs   = ntokens - 2;
    if (sp->nargs > 0)
    {
        sp->args = calloc((size_t) sp->nargs, sizeof(char *));
        if (sp->args == NULL)
        {
            free(sp->fpsname);
            free(sp->bin);
            return -1;
        }
        for (int arg_idx = 0; arg_idx < sp->nargs; arg_idx++)
        {
            sp->args[arg_idx] = tokens[arg_idx + 2];
        }
    }
    (*nspecs_ptr)++;
    return 0;
} // append_spec

static int tokenize_line(const char *line, char ***tokens_out, int *ntokens_out, int lineno)
{
    int    cap    = 8;
    int    ntok   = 0;
    char **tokens = calloc((size_t) cap, sizeof(char *));
    if (tokens == NULL)
    {
        return -1;
    }

    const char *cur = line;
    while (*cur != '\0')
    {
        while (*cur != '\0' && isspace((unsigned char) *cur))
        {
            cur++;
        }
        if (*cur == '\0' || *cur == '#')
        {
            break;
        }

        size_t buf_cap = strlen(cur) + 1;
        char  *tok     = malloc(buf_cap);
        if (tok == NULL)
        {
            goto fail;
        }
        size_t pos = 0;

        while (*cur != '\0' && !isspace((unsigned char) *cur))
        {
            if (*cur == '#')
            {
                break;
            }
            if (*cur == '\'' || *cur == '"')
            {
                char quote = *cur++;
                while (*cur != '\0' && *cur != quote)
                {
                    if (quote == '"' && *cur == '\\' && cur[1] != '\0')
                    {
                        cur++;
                    }
                    tok[pos++] = *cur++;
                }
                if (*cur != quote)
                {
                    fprintf(stderr, "Error: unterminated quote on line %d\n", lineno);
                    free(tok);
                    goto fail;
                }
                cur++;
            }
            else if (*cur == '\\' && cur[1] != '\0')
            {
                cur++;
                tok[pos++] = *cur++;
            }
            else
            {
                tok[pos++] = *cur++;
            }
        } // while token chars
        tok[pos] = '\0';

        if (ntok >= cap)
        {
            cap *= 2;
            char **grown = realloc(tokens, (size_t) cap * sizeof(char *));
            if (grown == NULL)
            {
                free(tok);
                goto fail;
            }
            tokens = grown;
        }
        tokens[ntok++] = tok;
    } // while (*cur != '\0')

    *tokens_out  = tokens;
    *ntokens_out = ntok;
    return 0;

fail:
    for (int tok_idx = 0; tok_idx < ntok; tok_idx++)
    {
        free(tokens[tok_idx]);
    }
    free(tokens);
    return -1;
} // tokenize_line

static int load_batch_file(const char *path, INIT_SPEC **specs, int *nspecs, int *cap)
{
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: cannot open '%s': %s\n", path, strerror(errno));
        return -1;
    }

    char  *line   = NULL;
    size_t lcap   = 0;
    int    lineno = 0;
    int    rc     = 0;

    while (getline(&line, &lcap, fp) >= 0)
    {
        lineno++;
        char **tokens  = NULL;
        int    ntokens = 0;
        if (tokenize_line(line, &tokens, &ntokens, lineno) != 0)
        {
            rc = -1;
            break;
        }
        if (ntokens == 0)
        {
            free(tokens);
            continue;
        }
        if (append_spec(specs, nspecs, cap, tokens, ntokens, lineno) != 0)
        {
            for (int tok_idx = 0; tok_idx < ntokens; tok_idx++)
            {
                free(tokens[tok_idx]);
            }
            free(tokens);
            rc = -1;
            break;
        }
        free(tokens);
    } // while getline

    free(line);
    if (fp != stdin)
    {
        fclose(fp);
    }
    return rc;
} // load_batch_file

static int is_unit_flag_1(const char *arg)
{
    return strcmp(arg, "-loops") == 0 || strcmp(arg, "--loops") == 0 ||
           strcmp(arg, "-procinfo") == 0 || strcmp(arg, "--procinfo") == 0 ||
           strcmp(arg, "--no-procinfo") == 0;
} // is_unit_flag_1

static int is_unit_flag_2(const char *arg)
{
    return strcmp(arg, "-loopd") == 0 || strcmp(arg, "--loopd") == 0 || strcmp(arg, "-k") == 0 ||
           strcmp(arg, "--keywords") == 0 || strcmp(arg, "-d") == 0 ||
           strcmp(arg, "--description") == 0 || strcmp(arg, "-t") == 0 || strcmp(arg, "-e") == 0;
} // is_unit_flag_2

static pid_t spawn_fpsinit(const INIT_SPEC *sp, int out_fd)
{
    char resolved_bin[PATH_MAX];
    if (strchr(sp->bin, '/') != NULL)
    {
        snprintf(resolved_bin, sizeof(resolved_bin), "%s", sp->bin);
    }
    else
    {
        milkspawn_tool_path(sp->bin, resolved_bin, sizeof(resolved_bin));
    }

    int has_explicit_cmd = 0;
    for (int arg_idx = 0; arg_idx < sp->nargs; arg_idx++)
    {
        if (strcmp(sp->args[arg_idx], "fpsinit") == 0)
        {
            has_explicit_cmd = 1;
            break;
        }
    } // for arg_idx

    const char **child_argv = calloc((size_t) sp->nargs + 6, sizeof(char *));
    if (child_argv == NULL)
    {
        return -1;
    }

    int pos           = 0;
    child_argv[pos++] = resolved_bin;
    child_argv[pos++] = "-n";
    child_argv[pos++] = sp->fpsname;

    if (has_explicit_cmd)
    {
        for (int arg_idx = 0; arg_idx < sp->nargs; arg_idx++)
        {
            child_argv[pos++] = sp->args[arg_idx];
        }
    }
    else
    {
        int *used = calloc((size_t) (sp->nargs > 0 ? sp->nargs : 1), sizeof(int));
        if (used == NULL)
        {
            free(child_argv);
            return -1;
        }
        for (int arg_idx = 0; arg_idx < sp->nargs; arg_idx++)
        {
            const char *cur = sp->args[arg_idx];
            if (is_unit_flag_1(cur))
            {
                child_argv[pos++] = cur;
                used[arg_idx]     = 1;
            }
            else if (is_unit_flag_2(cur))
            {
                child_argv[pos++] = cur;
                used[arg_idx]     = 1;
                if (arg_idx + 1 < sp->nargs)
                {
                    if ((strcmp(cur, "-loopd") == 0 || strcmp(cur, "--loopd") == 0) &&
                        !isdigit((unsigned char) sp->args[arg_idx + 1][0]) &&
                        sp->args[arg_idx + 1][0] != '.')
                    {
                        continue;
                    }
                    child_argv[pos++] = sp->args[arg_idx + 1];
                    used[arg_idx + 1] = 1;
                    arg_idx++;
                }
            }
        } // for unit pre-flags

        child_argv[pos++] = "fpsinit";
        for (int arg_idx = 0; arg_idx < sp->nargs; arg_idx++)
        {
            if (!used[arg_idx])
            {
                child_argv[pos++] = sp->args[arg_idx];
            }
        }
        free(used);
    } // if !has_explicit_cmd

    child_argv[pos] = NULL;
    pid_t pid       = milkspawn_start(child_argv, out_fd, STDERR_FILENO);
    free(child_argv);
    return pid;
} // spawn_fpsinit

static int stop_existing_units(const INIT_SPEC *specs, int nspecs, int out_fd)
{
    static const char *const stop_ops[2] = { "runstop", "confstop" };
    int                      rc          = 0;

    for (int op_idx = 0; op_idx < 2; op_idx++)
    {
        pid_t *pids = calloc((size_t) nspecs, sizeof(pid_t));
        if (pids == NULL)
        {
            return 1;
        }
        for (int spec_idx = 0; spec_idx < nspecs; spec_idx++)
        {
            pids[spec_idx] = -1;
            if (specs[spec_idx].existed_before && specs[spec_idx].needs_init)
            {
                pids[spec_idx] = fps_unit_spawn(specs[spec_idx].fpsname, stop_ops[op_idx], NULL,
                                                out_fd, STDERR_FILENO);
            }
        } // for spec_idx
        for (int spec_idx = 0; spec_idx < nspecs; spec_idx++)
        {
            if (pids[spec_idx] > 0 && milkspawn_wait(pids[spec_idx]) != 0)
            {
                rc = 1;
            }
        } // for spec_idx
        free(pids);
    } // for op_idx
    return rc;
} // stop_existing_units

int main(int argc, char *argv[])
{
    int         mh_color        = isatty(STDOUT_FILENO);
    char       *progbuf         = dup_str(argv[0], strlen(argv[0]));
    char       *prog            = (progbuf != NULL) ? basename(progbuf) : argv[0];
    int         do_restart      = 0;
    int         do_print_create = 0;
    int         do_quiet        = 0;
    const char *batch_arg       = NULL;

    int arg_pos = 1;
    while (arg_pos < argc && argv[arg_pos][0] == '-')
    {
        const char *opt = argv[arg_pos];
        if (strcmp(opt, "--") == 0)
        {
            arg_pos++;
            break;
        }
        if (strcmp(opt, "-h1") == 0 || strcmp(opt, "--help-oneline") == 0)
        {
            printf("%s\n", FI_DESC);
            free(progbuf);
            return 0;
        }
        if (strcmp(opt, "-h2") == 0 || strcmp(opt, "--help-description") == 0)
        {
            printf("%s\n", FI_DESC_LONG);
            free(progbuf);
            return 0;
        }
        if (strcmp(opt, "-hm") == 0 || strcmp(opt, "--help-mono") == 0)
        {
            print_help(prog, 0);
            free(progbuf);
            return 0;
        }
        if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0)
        {
            print_help(prog, mh_color);
            free(progbuf);
            return 0;
        }
        if (strcmp(opt, "--restart") == 0)
        {
            do_restart = 1;
            arg_pos++;
        }
        else if (strcmp(opt, "--print-created") == 0)
        {
            do_print_create = 1;
            arg_pos++;
        }
        else if (strcmp(opt, "-q") == 0 || strcmp(opt, "--quiet") == 0)
        {
            do_quiet = 1;
            arg_pos++;
        }
        else if (strcmp(opt, "-b") == 0 || strcmp(opt, "--batch") == 0)
        {
            if (arg_pos + 1 >= argc)
            {
                fprintf(stderr, "Error: %s requires a file argument\n", opt);
                free(progbuf);
                return 2;
            }
            batch_arg = argv[arg_pos + 1];
            arg_pos += 2;
        }
        else if (strncmp(opt, "--batch=", 8) == 0)
        {
            batch_arg = opt + 8;
            arg_pos++;
        }
        else
        {
            fprintf(stderr, "Error: unknown option '%s' (run %s -h for usage)\n", opt, prog);
            free(progbuf);
            return 2;
        }
    } // while options

    INIT_SPEC *specs  = NULL;
    int        nspecs = 0;
    int        cap    = 0;

    if (batch_arg != NULL)
    {
        if (arg_pos < argc)
        {
            fprintf(stderr, "Error: positional arguments cannot be combined with -b/--batch\n");
            free(progbuf);
            return 2;
        }
        if (load_batch_file(batch_arg, &specs, &nspecs, &cap) != 0)
        {
            free_specs(specs, nspecs);
            free(progbuf);
            return 2;
        }
    }
    else
    {
        int rem = argc - arg_pos;
        if (rem < 2)
        {
            fprintf(stderr, "Error: expected '<fpsname> <fpsexec_bin> [args...]' or '-b <file>'\n");
            free(progbuf);
            return 2;
        }
        char **tokens = calloc((size_t) rem, sizeof(char *));
        if (tokens == NULL)
        {
            free(progbuf);
            return 1;
        }
        for (int tok_idx = 0; tok_idx < rem; tok_idx++)
        {
            tokens[tok_idx] = dup_str(argv[arg_pos + tok_idx], strlen(argv[arg_pos + tok_idx]));
        }
        if (append_spec(&specs, &nspecs, &cap, tokens, rem, 1) != 0)
        {
            for (int tok_idx = 0; tok_idx < rem; tok_idx++)
            {
                free(tokens[tok_idx]);
            }
            free(tokens);
            free(progbuf);
            return 2;
        }
        free(tokens);
    } // if batch_arg == NULL

    if (nspecs == 0)
    {
        free_specs(specs, nspecs);
        free(progbuf);
        return 0;
    }

    char shmdir[STRINGMAXLEN_DIRNAME];
    function_parameter_struct_shmdirname(shmdir);

    int need_any = 0;
    for (int spec_idx = 0; spec_idx < nspecs; spec_idx++)
    {
        char shmpath[PATH_MAX];
        snprintf(shmpath, sizeof(shmpath), "%s/%s.fps.shm", shmdir, specs[spec_idx].fpsname);
        specs[spec_idx].existed_before = (access(shmpath, F_OK) == 0);
        specs[spec_idx].needs_init     = (!specs[spec_idx].existed_before || do_restart);
        if (specs[spec_idx].needs_init)
        {
            need_any = 1;
        }
    } // for spec_idx

    if (!need_any)
    {
        free_specs(specs, nspecs);
        free(progbuf);
        return 0;
    }

    int devnull_fd = -1;
    int out_fd     = (do_quiet || do_print_create) ? -1 : STDOUT_FILENO;
    if (out_fd < 0)
    {
        devnull_fd = open("/dev/null", O_WRONLY);
        out_fd     = (devnull_fd >= 0) ? devnull_fd : STDOUT_FILENO;
    }

    int exit_rc = 0;
    if (do_restart && stop_existing_units(specs, nspecs, out_fd) != 0)
    {
        exit_rc = 1;
    }

    pid_t *pids = calloc((size_t) nspecs, sizeof(pid_t));
    if (pids == NULL)
    {
        if (devnull_fd >= 0)
        {
            close(devnull_fd);
        }
        free_specs(specs, nspecs);
        free(progbuf);
        return 1;
    }

    for (int spec_idx = 0; spec_idx < nspecs; spec_idx++)
    {
        pids[spec_idx] = -1;
        if (specs[spec_idx].needs_init)
        {
            pids[spec_idx] = spawn_fpsinit(&specs[spec_idx], out_fd);
            if (pids[spec_idx] < 0)
            {
                fprintf(stderr, "Error: failed to spawn '%s' for '%s': %s\n", specs[spec_idx].bin,
                        specs[spec_idx].fpsname, strerror(errno));
                exit_rc = 1;
            }
        }
    } // for spec_idx

    for (int spec_idx = 0; spec_idx < nspecs; spec_idx++)
    {
        if (pids[spec_idx] > 0)
        {
            int wrc = milkspawn_wait(pids[spec_idx]);
            if (wrc != 0)
            {
                exit_rc = (exit_rc == 0) ? wrc : exit_rc;
            }
            else if (do_print_create && !specs[spec_idx].existed_before)
            {
                printf("%s\n", specs[spec_idx].fpsname);
            }
        }
    } // for spec_idx

    free(pids);
    if (devnull_fd >= 0)
    {
        close(devnull_fd);
    }
    free_specs(specs, nspecs);
    free(progbuf);
    return exit_rc;
} // main
