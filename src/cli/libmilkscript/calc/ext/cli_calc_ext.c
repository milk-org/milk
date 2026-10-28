// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_ext.c
 * @brief Dynamic loader, `.c` auto-compiler, registry, and call-site scanner for user C extensions
 *        (`milk_calc_ext.h`) in `milk.calc` and `@calc`.
 */

#define _GNU_SOURCE

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_tokenizer.h"
#include "milk_calc_ext.h"

#define CALC_EXT_MAX_LIBS 32

static char loaded_libs[CALC_EXT_MAX_LIBS][PATH_MAX];
static int  n_loaded_libs = 0;

/**
 * fnv1a_bytes - Update 64-bit FNV-1a hash @h with @len bytes from @buf
 */
static uint64_t fnv1a_bytes(uint64_t h, const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *) buf;
    for (size_t i = 0; i < len; i++)
    {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/**
 * find_milk_include_dirs - Populate `-I<dir>` compiler arguments to locate `milk_calc_ext.h`
 */
static void find_milk_include_dirs(char inc_dirs[6][PATH_MAX], int *n_inc)
{
    *n_inc          = 0;
    const char *env = getenv("MILK_INSTALLDIR");
    if (env != NULL && env[0] != '\0')
    {
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include", env);
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include/milkscript", env);
    }

    Dl_info dli;
    if (dladdr((void *) &cli_calc_ext_load, &dli) != 0 && dli.dli_fname != NULL)
    {
        char  rpath[PATH_MAX];
        char *rp = realpath(dli.dli_fname, rpath);
        char *sl = (rp != NULL) ? strrchr(rp, '/') : NULL;
        if (sl != NULL)
        {
            *sl      = '\0';
            char *p2 = strrchr(rp, '/');
            if (p2 != NULL && strcmp(p2 + 1, "lib") == 0)
            {
                *p2 = '\0';
                if (*n_inc < 6)
                {
                    snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include", rp);
                }
                if (*n_inc < 6)
                {
                    snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include/milkscript", rp);
                }
            }
        }
    }

    /* Fallback to build-tree source paths if uninstalled */
    char src_file[PATH_MAX];
    snprintf(src_file, sizeof(src_file), "%s", __FILE__);
    char *last_sl = strrchr(src_file, '/');
    if (last_sl != NULL && *n_inc < 6)
    {
        *last_sl      = '\0';
        char *prev_sl = strrchr(src_file, '/');
        if (prev_sl != NULL && strcmp(prev_sl, "/ext") == 0)
        {
            *prev_sl = '\0';
        }
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", src_file);
        /* Also walk up to `<repo>/src/engine/libfps` if present */
        char repo_inc[PATH_MAX];
        snprintf(repo_inc, sizeof(repo_inc), "%s/../../../engine/libfps", src_file);
        if (access(repo_inc, R_OK) == 0 && *n_inc < 6)
        {
            snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", repo_inc);
        }
    }
}

/**
 * hash_ext_source_file - Compute FNV-1a cache hash of @c_path and compiler @cc
 */
static int hash_ext_source_file(const char *c_path,
                                const char *cc,
                                uint64_t   *out_hash,
                                char       *err,
                                size_t      errsz)
{
    FILE *fp = fopen(c_path, "rb");
    if (fp == NULL)
    {
        snprintf(err, errsz, "cannot open '%s': %s", c_path, strerror(errno));
        return -1;
    }
    uint64_t h   = 14695981039346656037ULL;
    uint32_t abi = MILK_CALC_EXT_ABI_VERSION;
    h            = fnv1a_bytes(h, &abi, sizeof(abi));
    h            = fnv1a_bytes(h, c_path, strlen(c_path));
    h            = fnv1a_bytes(h, cc, strlen(cc));

    unsigned char buf[4096];
    size_t        nread;
    while ((nread = fread(buf, 1, sizeof(buf), fp)) > 0)
    {
        h = fnv1a_bytes(h, buf, nread);
    }
    fclose(fp);
    *out_hash = h;
    return 0;
}

/**
 * compile_c_to_so - Compile user `.c` file @c_path into cached shared object @so_out
 */
static int compile_c_to_so(const char *c_path,
                           char       *so_out,
                           size_t      so_out_sz,
                           char       *err,
                           size_t      errsz)
{
    const char *cc_env = getenv("CC");
    const char *cc     = (cc_env != NULL && cc_env[0] != '\0') ? cc_env : "gcc";
    uint64_t    h      = 0;
    if (hash_ext_source_file(c_path, cc, &h, err, errsz) != 0)
    {
        return -1;
    }

    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL || tmpdir[0] == '\0')
    {
        tmpdir = "/tmp";
    }
    char cache_dir[PATH_MAX];
    snprintf(cache_dir, sizeof(cache_dir), "%s/milk-calc-cache-%u", tmpdir, (unsigned) getuid());
    (void) mkdir(cache_dir, 0700);

    snprintf(so_out, so_out_sz, "%s/ext_%016llx.so", cache_dir, (unsigned long long) h);
    if (access(so_out, R_OK) == 0)
    {
        return 0;
    }

    char tmp_so[PATH_MAX];
    snprintf(tmp_so, sizeof(tmp_so), "%s/ext_%016llx.%d.tmp.so", cache_dir, (unsigned long long) h,
             (int) getpid());

    char inc_dirs[6][PATH_MAX];
    int  n_inc = 0;
    find_milk_include_dirs(inc_dirs, &n_inc);

    char        inc_flags[6][PATH_MAX + 4];
    const char *argv[32];
    int         ac = 0;
    argv[ac++]     = cc;
    argv[ac++]     = "-O3";
    argv[ac++]     = "-march=native";
    argv[ac++]     = "-ffast-math";
    argv[ac++]     = "-fopenmp-simd";
    argv[ac++]     = "-shared";
    argv[ac++]     = "-fPIC";
    argv[ac++]     = "-Wall";
    argv[ac++]     = "-Wextra";
    argv[ac++]     = "-Wno-unused-parameter";
    for (int i = 0; i < n_inc; i++)
    {
        snprintf(inc_flags[i], sizeof(inc_flags[i]), "-I%s", inc_dirs[i]);
        argv[ac++] = inc_flags[i];
    }
    argv[ac++] = c_path;
    argv[ac++] = "-o";
    argv[ac++] = tmp_so;
    argv[ac++] = "-lm";
    argv[ac]   = NULL;

    pid_t pid = fork();
    if (pid < 0)
    {
        snprintf(err, errsz, "fork failed while compiling '%s'", c_path);
        return -1;
    }
    if (pid == 0)
    {
        execvp(cc, (char *const *) argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        unlink(tmp_so);
        snprintf(err, errsz, "compilation of '%s' failed (exit %d)", c_path,
                 WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return -1;
    }
    if (rename(tmp_so, so_out) != 0)
    {
        unlink(tmp_so);
        snprintf(err, errsz, "failed to move compiled '%s' into cache", c_path);
        return -1;
    }
    return 0;
}

/**
 * cli_calc_ext_load - Load a `.c` or `.so` user extension file and register its symbols
 */
int cli_calc_ext_load(const char *path, char *err, size_t errsz)
{
    if (path == NULL || path[0] == '\0')
    {
        return 0;
    }
    char        rpath_buf[PATH_MAX];
    const char *canon = realpath(path, rpath_buf);
    if (canon == NULL)
    {
        canon = path;
    }
    for (int i = 0; i < n_loaded_libs; i++)
    {
        if (strcmp(loaded_libs[i], canon) == 0)
        {
            return 0;
        }
    }

    char        so_path[PATH_MAX];
    const char *dl_target = canon;
    size_t      plen      = strlen(canon);
    if (plen > 2 && strcmp(canon + plen - 2, ".c") == 0)
    {
        if (compile_c_to_so(canon, so_path, sizeof(so_path), err, errsz) != 0)
        {
            return -1;
        }
        dl_target = so_path;
    }

    void *dl = dlopen(dl_target, RTLD_NOW | RTLD_LOCAL);
    if (dl == NULL)
    {
        snprintf(err, errsz, "dlopen('%s'): %s", path, dlerror());
        return -1;
    }

    const milk_calc_ext_table_t *tbl =
        (const milk_calc_ext_table_t *) dlsym(dl, "__milk_calc_ext_table");
    if (tbl == NULL)
    {
        snprintf(err, errsz, "'%s' is missing MILK_CALC_EXT_EXPORT(...)", path);
        dlclose(dl);
        return -1;
    }
    if (milk_calc_ext_register_table(tbl, err, errsz) != 0)
    {
        dlclose(dl);
        return -1;
    }
    if (n_loaded_libs < CALC_EXT_MAX_LIBS)
    {
        snprintf(loaded_libs[n_loaded_libs++], PATH_MAX, "%s", canon);
    }
    return 0;
}

/**
 * cli_calc_ext_load_list - Load a space-, colon-, or comma-separated list of `.c` / `.so` files
 */
int cli_calc_ext_load_list(const char *libs_str, char *err, size_t errsz)
{
    if (libs_str == NULL || libs_str[0] == '\0')
    {
        return 0;
    }
    const char *p = libs_str;
    while (*p != '\0')
    {
        while (*p != '\0' && (isspace((unsigned char) *p) || *p == ',' || *p == ':'))
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }
        const char *start = p;
        while (*p != '\0' && !isspace((unsigned char) *p) && *p != ',' && *p != ':')
        {
            p++;
        }
        char   item[PATH_MAX];
        size_t len = (size_t) (p - start);
        if (len >= sizeof(item))
        {
            len = sizeof(item) - 1;
        }
        memcpy(item, start, len);
        item[len] = '\0';
        if (cli_calc_ext_load(item, err, errsz) != 0)
        {
            return -1;
        }
    }
    return 0;
}

/**
 * cli_calc_ext_find - Look up a registered user kernel or operator by name
 */
int cli_calc_ext_find(const char *name)
{
    return milk_calc_ext_find(name);
}

/**
 * cli_calc_ext_get - Return the registered descriptor at index @idx
 */
const milk_calc_ext_entry_t *cli_calc_ext_get(int idx)
{
    return milk_calc_ext_get(idx);
}

/**
 * cli_calc_ext_count - Return the number of registered user kernels and operators
 */
int cli_calc_ext_count(void)
{
    return milk_calc_ext_count();
}

/**
 * scan_one_call - Inspect a `TOK_FUNC_USER` call at `toks[call_idx]` and invoke callbacks
 */
static void scan_one_call(const cli_token             *toks,
                          int                          ntok,
                          int                          call_idx,
                          int                          ndst,
                          char                         lhs[][STRINGMAXLEN_IMAGE_NAME],
                          cli_calc_ext_param_cb_t      param_cb,
                          cli_calc_ext_scalar_out_cb_t scalar_out_cb,
                          void                        *ud)
{
    const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) toks[call_idx].val_l);
    if (e == NULL)
    {
        return;
    }
    char func_name[64];
    char label[64];
    cli_calc_ext_split_call(toks[call_idx].sval, func_name, sizeof(func_name), label,
                            sizeof(label));

    const milk_calc_param_set_t *pset = (e->get_params != NULL) ? e->get_params() : NULL;
    int                          bound[MILK_CALC_MAX_PARAMS] = { 0 };
    int                          nin_img                     = (e->kind == MILK_EXT_BLK2) ? 2 : 1;
    int                          depth                       = 1;
    int                          arg_idx                     = 0;
    int                          arg_beg                     = call_idx + 1;

    for (int i = call_idx + 1; i < ntok && depth > 0; i++)
    {
        cli_token_type tt = toks[i].type;
        /* a nested call: the range also holds names, operators and commas, but only calls carry '(' */
        if (tt == TOK_LPAREN ||
            (tt >= TOK_FUNC_D_D && tt <= TOK_FUNC_USER && strchr(toks[i].sval, '(') != NULL))
        {
            depth++;
            continue;
        }
        if (tt == TOK_RPAREN && --depth > 0)
        {
            continue;
        }
        if ((tt == TOK_COMMA && depth == 1) || (tt == TOK_RPAREN && depth == 0))
        {
            if (i > arg_beg)
            {
                int is_kw = (arg_beg + 1 < i && toks[arg_beg + 1].type == TOK_EQUAL &&
                             (toks[arg_beg].type == TOK_VAR || toks[arg_beg].type == TOK_NVAR ||
                              toks[arg_beg].type == TOK_IMAGE));
                if (is_kw && pset != NULL)
                {
                    for (int p = 0; p < pset->nparams && p < MILK_CALC_MAX_PARAMS; p++)
                    {
                        if (strcmp(toks[arg_beg].sval, pset->specs[p].name) == 0)
                        {
                            bound[p] = 1;
                        }
                    }
                }
                else if (!is_kw && e->kind != MILK_EXT_OP && pset != NULL && arg_idx >= nin_img)
                {
                    int p = arg_idx - nin_img;
                    if (p >= 0 && p < pset->nparams && p < MILK_CALC_MAX_PARAMS)
                    {
                        bound[p] = 1;
                    }
                }
                arg_idx++;
            }
            arg_beg = i + 1;
        }
    }

    if (pset != NULL && param_cb != NULL)
    {
        int is_op = (e->kind == MILK_EXT_OP);
        for (int p = 0; p < pset->nparams && p < MILK_CALC_MAX_PARAMS; p++)
        {
            if (!bound[p])
            {
                param_cb(label, &pset->specs[p], is_op, ud);
            }
        }
    }
    if (e->kind == MILK_EXT_OP && e->get_ports != NULL && scalar_out_cb != NULL && ndst > 0)
    {
        const milk_calc_op_ports_t *ports = e->get_ports();
        for (int j = 0; j < ndst && j < ports->nout; j++)
        {
            if (ports->out_ports[j].kind == MILK_PORT_SCALAR && lhs[j][0] != '\0')
            {
                scalar_out_cb(lhs[j], ports->out_ports[j].descr, ud);
            }
        }
    }
}

static void scan_statement_line(const char                  *st,
                                size_t                       len,
                                cli_calc_ext_param_cb_t      param_cb,
                                cli_calc_ext_scalar_out_cb_t scalar_out_cb,
                                void                        *ud)
{
    if (len == 0 || len >= 4096)
    {
        return;
    }
    char line[4096];
    memcpy(line, st, len);
    line[len] = '\0';

    char lhs[MILK_CALC_MAX_PORTS][STRINGMAXLEN_IMAGE_NAME];
    int  ndst = 0;
    memset(lhs, 0, sizeof(lhs));

    cli_token toks[CLI_CALC_MAX_TOKENS];
    int       ntok = cli_tokenize(line, toks, CLI_CALC_MAX_TOKENS);
    if (ntok <= 0)
    {
        return;
    }
    int pos = 0;
    while (ndst < MILK_CALC_MAX_PORTS && pos < ntok)
    {
        if (toks[pos].type != TOK_VAR && toks[pos].type != TOK_NVAR && toks[pos].type != TOK_IMAGE)
        {
            ndst = 0;
            break;
        }
        snprintf(lhs[ndst++], STRINGMAXLEN_IMAGE_NAME, "%s", toks[pos].sval);
        pos++;
        if (pos < ntok && toks[pos].type == TOK_COMMA)
        {
            pos++;
        }
        else
        {
            break;
        }
    }
    if (pos >= ntok || toks[pos].type != TOK_EQUAL)
    {
        ndst = 0;
    }
    for (int i = 0; i < ntok; i++)
    {
        if (toks[i].type == TOK_FUNC_USER)
        {
            scan_one_call(toks, ntok, i, ndst, lhs, param_cb, scalar_out_cb, ud);
        }
    }
}

/**
 * cli_calc_ext_scan_source - Scan `.calc` source text for user extension calls and unbound params
 */
void cli_calc_ext_scan_source(const char                  *src,
                              cli_calc_ext_param_cb_t      param_cb,
                              cli_calc_ext_scalar_out_cb_t scalar_out_cb,
                              void                        *ud)
{
    if (src == NULL || src[0] == '\0' || milk_calc_ext_count() == 0)
    {
        return;
    }
    const char *p = src;
    while (*p != '\0')
    {
        while (*p != '\0' && (isspace((unsigned char) *p) || *p == ';'))
        {
            p++;
        }
        if (*p == '#')
        {
            while (*p != '\0' && *p != '\n')
            {
                p++;
            }
            continue;
        }
        if (*p == '\0')
        {
            break;
        }
        const char *st    = p;
        int         depth = 0;
        char        q     = 0;
        for (; *p != '\0'; p++)
        {
            if (q != 0)
            {
                q = (*p == q) ? 0 : q;
            }
            else if (*p == '"' || *p == '\'')
            {
                q = *p;
            }
            else if (*p == '(' || *p == '[')
            {
                depth++;
            }
            else if ((*p == ')' || *p == ']') && depth > 0)
            {
                depth--;
            }
            else if (*p == '#' ||
                     (depth == 0 && (*p == ';' || *p == '\n' || *p == '{' || *p == '}')))
            {
                break;
            }
        }
        scan_statement_line(st, (size_t) (p - st), param_cb, scalar_out_cb, ud);
        if (*p == '{' || *p == '}')
        {
            p++;
        }
    }
}
