// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_loader.c
 * @brief Dynamic `.c` / `.so` extension loader and static call-site scanner for `libmilkcalc`.
 *
 * Purpose:     let a user bring C kernels into `@calc` and milk.calc with `-l file.c` and no build
 *              step, and give the run time and the FPS of a unit one rule for which argument
 *              binds which input or parameter.
 * Does:        mcalc_ext_load() loads a file once per canonical path and modification time. A
 *              `.c` file is compiled
 *              with mc_cc_build() (the compiler, flags and cache directory of the JIT) into
 *              `ext_<key>.so`, keyed on the ABI version, the path and the file contents
 *              (mc_cc_key() adds compiler, flags and CPU); a cached `.so` is reused. The
 *              `__milk_calc_ext_table` of the library (MILK_CALC_EXT_EXPORT) is registered after
 *              dlopen(). mcalc_ext_load_list() loads a list separated by spaces, `,` or `:`.
 *              mc_ext_bind_args() binds the arguments of a call with the module-command rules
 *              (milk_args_bind()): `name=value` by name, the others to the inputs, then to the
 *              parameters, in order. mcalc_ext_scan_source() lexes a unit program and reports
 *              each parameter that no argument binds (param_cb: the unit exposes
 *              `<label>.<param>` in its FPS) and each target of a scalar output port
 *              (scalar_out_cb).
 * Used by:     mc_eval.c (`loadlib`, `@calc -l`, MILK_CALC_LIBS), units/milk_unit_calc.c (`-l`,
 *              `.unit.libs`, the scan), ext/mc_ext_exec.c (mc_ext_bind_args()). Declared in
 *              mcalc.h and plan/mc_plan.h.
 * Uses:        jit/mc_cc.c (compile step, cache directory, key), ext/mc_ext_registry.c,
 *              front/mc_lex.c (mc_lex()), fps_args.h (libfps, milk_args_bind()), libdl.
 * See also:    scripts/milk-cc (builds the same kind of `.so` ahead of time), milk_calc_ext.h;
 *              docs/guide/calculator.md, "Auto-Compilation and Caching (`-l` / `--lib`)";
 *              docs/guide/calc_c_extensions.md, "FPS Parameter Binding Rules".
 */

#define _GNU_SOURCE

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "front/mc_ast.h"
#include "plan/mc_plan.h"
#include "mcalc.h"
#include "fps_args.h"
#include "milk_calc_ext.h"

/* Maximum number of canonical library paths tracked for idempotent loading. */
#define CALC_EXT_MAX_LIBS 32

static char            loaded_libs[CALC_EXT_MAX_LIBS][PATH_MAX];
static struct timespec loaded_mtime[CALC_EXT_MAX_LIBS];
static int             n_loaded_libs = 0;

/* ============================================================================================== */
/* On-Demand `.c` -> `.so` Compilation                                                            */
/* ============================================================================================== */

/**
 * add_prefix_dirs - Add `<prefix>/include` and its `milkcalc/` and `libfps/` (fps_types.h) to @inc
 * @prefix: Installation prefix
 * @inc:    Include directory list
 * @n_inc:  Count of populated entries in @inc (updated)
 *
 * Return: void
 */
static void add_prefix_dirs(const char *prefix, char inc[8][PATH_MAX], int *n_inc)
{
    static const char *const sub[] = { "", "/milkcalc", "/libfps" };
    for (int i = 0; i < 3 && *n_inc < 8; i++)
    {
        snprintf(inc[(*n_inc)++], PATH_MAX, "%s/include%s", prefix, sub[i]);
    }
}

/**
 * find_milk_include_dirs - Discover `-I` header search paths for compiling `.c` extensions
 * @inc_dirs: Output array of up to 8 directory path strings
 * @n_inc:    Output count of populated entries in @inc_dirs
 *
 * Collects include directories from three sources in order:
 *   1. The `$MILK_INSTALLDIR` prefix (add_prefix_dirs()).
 *   2. The prefix of `libmilkcalc.so`, discovered via `dladdr(&mcalc_ext_load)`.
 *   3. The calculator source directory, two levels up from this file (milk_calc_ext.h), and
 *      its `../engine/libfps` (for in-tree unit tests before install).
 *
 * Return: void
 */
static void find_milk_include_dirs(char inc_dirs[8][PATH_MAX], int *n_inc)
{
    *n_inc          = 0;
    const char *env = getenv("MILK_INSTALLDIR");
    if (env != NULL && env[0] != '\0')
    {
        add_prefix_dirs(env, inc_dirs, n_inc);
    }
    Dl_info dli;
    if (dladdr((void *) &mcalc_ext_load, &dli) != 0 && dli.dli_fname != NULL)
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
                add_prefix_dirs(rp, inc_dirs, n_inc);
            }
        }
    }
    char src_file[PATH_MAX];
    snprintf(src_file, sizeof(src_file), "%s", __FILE__);
    /* this file is <calc>/ext/mc_ext_loader.c: two levels up is <calc>, where milk_calc_ext.h is */
    char *last_sl = strrchr(src_file, '/');
    if (last_sl != NULL)
    {
        *last_sl = '\0';
        last_sl  = strrchr(src_file, '/');
    }
    if (last_sl != NULL && *n_inc < 8)
    {
        *last_sl = '\0';
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", src_file);
        char repo_inc[PATH_MAX];
        snprintf(repo_inc, sizeof(repo_inc), "%s/../engine/libfps", src_file);
        if (access(repo_inc, R_OK) == 0 && *n_inc < 8)
        {
            snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", repo_inc);
        }
    }
}

/**
 * compile_c_to_so - Compile a user `.c` extension source file into a cached `.so` shared library
 * @c_path:    Canonical path to the `.c` extension source file
 * @so_out:    Output buffer receiving the compiled `.so` path
 * @so_out_sz: Capacity of @so_out in bytes
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Keys `ext_<key>.so` in the kernel cache directory on the ABI version, the path (relative
 * `#include`s) and the file bytes; mc_cc_key() adds compiler, flags and CPU. A cached `.so` is
 * reused; otherwise mc_cc_build() compiles it as it compiles JIT kernels, with the milk include
 * directories.
 *
 * Return: 0 on success, -1 on file I/O or compilation error
 */
static int compile_c_to_so(const char *c_path,
                           char       *so_out,
                           size_t      so_out_sz,
                           char       *err,
                           size_t      errsz)
{
    FILE *fp = fopen(c_path, "rb");
    if (fp == NULL)
    {
        snprintf(err, errsz, "cannot open '%s': %s", c_path, strerror(errno));
        return -1;
    }
    char  *text = NULL;
    size_t tlen = 0;
    FILE  *ms   = open_memstream(&text, &tlen);
    if (ms != NULL)
    {
        char   buf[4096];
        size_t nread;
        fprintf(ms, "milk_calc_ext %u %s\n", (unsigned) MILK_CALC_EXT_ABI_VERSION, c_path);
        while ((nread = fread(buf, 1, sizeof(buf), fp)) > 0)
        {
            fwrite(buf, 1, nread, ms);
        }
        fclose(ms);
    }
    fclose(fp);

    char dir[384];
    char key[33];
    if (text == NULL)
    {
        snprintf(err, errsz, "cannot read '%s'", c_path);
        return -1;
    }
    if (mc_cc_cache_dir(dir, sizeof(dir), err, errsz) != 0)
    {
        free(text);
        return -1;
    }
    mc_cc_key(text, key);
    free(text);
    snprintf(so_out, so_out_sz, "%s/ext_%s.so", dir, key);
    if (access(so_out, R_OK) == 0)
    {
        return 0;
    }

    char        inc_dirs[8][PATH_MAX];
    char        inc_flags[8][PATH_MAX + 2];
    const char *args[9];
    int         n_inc = 0;
    find_milk_include_dirs(inc_dirs, &n_inc);
    for (int i = 0; i < n_inc; i++)
    {
        snprintf(inc_flags[i], sizeof(inc_flags[i]), "-I%s", inc_dirs[i]);
        args[i] = inc_flags[i];
    }
    args[n_inc] = NULL;
    return mc_cc_build(c_path, so_out, args, err, errsz);
}

/* ============================================================================================== */
/* Public Extension Library Loader API                                                            */
/* ============================================================================================== */

/**
 * mcalc_ext_load - Load a `.c` or `.so` extension file and register its `__milk_calc_ext_table`
 * @path:  Path to a `.c` source file or `.so` shared library
 * @err:   Error message buffer
 * @errsz: Capacity of @err in bytes
 *
 * Deduplicates canonicalized paths against `loaded_libs[]` so loading the same file repeatedly is
 * a fast no-op. A file modified since it was loaded is loaded again: its entries replace the old
 * ones, and the cached statements, which may hold state the old code made, are dropped first. A
 * `.so` rebuilt at the same path stays as loaded: dlopen() returns the library it already has.
 *
 * Return: 0 on success, -1 on compilation, `dlopen`, `dlsym`, or registration failure
 */
int mcalc_ext_load(const char *path, char *err, size_t errsz)
{
    if (path == NULL || path[0] == '\0')
    {
        return 0;
    }
    char        rpath_buf[PATH_MAX];
    char        so_path[PATH_MAX];
    const char *canon = realpath(path, rpath_buf);
    canon             = (canon != NULL) ? canon : path;
    struct stat st    = { 0 };
    int         slot  = -1;
    (void) stat(canon, &st); /* a missing file fails below, at fopen() or dlopen() */
    for (int i = 0; i < n_loaded_libs && slot < 0; i++)
    {
        if (strcmp(loaded_libs[i], canon) == 0)
        {
            if (st.st_mtim.tv_sec == loaded_mtime[i].tv_sec &&
                st.st_mtim.tv_nsec == loaded_mtime[i].tv_nsec)
            {
                return 0;
            }
            slot = i;
            mc_stmt_cache_clear();
        }
    }
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
    if (tbl == NULL || milk_calc_ext_register_table(tbl, err, errsz) != 0)
    {
        if (tbl == NULL)
        {
            snprintf(err, errsz, "'%s' is missing MILK_CALC_EXT_EXPORT(...)", path);
        }
        dlclose(dl);
        return -1;
    }
    if (slot < 0 && n_loaded_libs < CALC_EXT_MAX_LIBS)
    {
        slot = n_loaded_libs++;
        snprintf(loaded_libs[slot], PATH_MAX, "%s", canon);
    }
    if (slot >= 0)
    {
        loaded_mtime[slot] = st.st_mtim;
    }
    return 0;
}

/**
 * mcalc_ext_load_list - Load a comma-, colon-, or whitespace-separated list of `.c`/`.so` paths
 * @libs_str: Delimited list of extension library paths (may be NULL or empty)
 * @err:      Error message buffer
 * @errsz:    Capacity of @err in bytes
 *
 * Return: 0 if all libraries loaded successfully, -1 on the first failure
 */
int mcalc_ext_load_list(const char *libs_str, char *err, size_t errsz)
{
    for (const char *p = (libs_str != NULL) ? libs_str : ""; *p != '\0';)
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
        len        = (len >= sizeof(item)) ? (sizeof(item) - 1) : len;
        memcpy(item, start, len);
        item[len] = '\0';
        if (mcalc_ext_load(item, err, errsz) != 0)
        {
            return -1;
        }
    }
    return 0;
}

/**
 * mcalc_ext_count - Return the total number of registered C extension entries
 *
 * Return: Count of registered extensions
 */
int mcalc_ext_count(void)
{
    return milk_calc_ext_count();
}

/**
 * mcalc_ext_get - Retrieve a registered C extension entry by index
 * @idx: Index in `[0, mcalc_ext_count())`
 *
 * Return: Pointer to `milk_calc_ext_entry_t`, or NULL if @idx is out of range
 */
const milk_calc_ext_entry_t *mcalc_ext_get(int idx)
{
    return milk_calc_ext_get(idx);
}

/* ============================================================================================== */
/* Call Arguments and the Static Call-Site Scanner                                                */
/* ============================================================================================== */

/**
 * mc_ext_bind_args - Which argument of a call binds each input and each parameter of an extension
 * @ext_id:  Registered extension index
 * @nargs:   Number of call arguments
 * @kw:      Keyword of each argument (`name` of `name=value`), or "" for a positional one
 * @in_arg:  Out: the argument of each input (image inputs, or operator input ports), or -1
 * @par_arg: Out: the argument of each parameter, or -1
 * @err:     Message buffer
 * @errsz:   Size of @err
 *
 * The module-command rules (milk_args_bind()): keyword arguments bind by name (an operator port
 * also as `<op>_<port>`), the others fill the inputs, then the parameters, not named, in order.
 * The runtime and the unit's FPS parameters (mcalc_ext_scan_source()) both bind with it.
 *
 * Return: the number of inputs, or -1 with @err (unknown name, name given twice, too many)
 */
int mc_ext_bind_args(int ext_id,
                     int nargs,
                     const char (*kw)[64],
                     int   *in_arg,
                     int   *par_arg,
                     char  *err,
                     size_t errsz)
{
    const milk_calc_ext_entry_t *e = milk_calc_ext_get(ext_id);
    if (e == NULL || nargs > MC_MAX_ARGS)
    {
        snprintf(err, errsz, "%s: too many arguments", (e != NULL) ? e->name : "extension");
        return -1;
    }
    const milk_calc_param_set_t *ps = (e->get_params != NULL) ? e->get_params() : NULL;
    const milk_calc_op_ports_t  *pts =
        (e->kind == MILK_EXT_OP && e->get_ports != NULL) ? e->get_ports() : NULL;
    int            nin  = (pts != NULL) ? pts->nin : (e->kind == MILK_EXT_BLK2) ? 2 : 1;
    int            npar = (ps != NULL) ? ps->nparams : 0;
    int            npos = 0;
    milk_argslot_t slots[MILK_CALC_MAX_PORTS + MILK_CALC_MAX_PARAMS];
    const char    *val[MILK_CALC_MAX_PORTS + MILK_CALC_MAX_PARAMS];
    int            word[MILK_CALC_MAX_PORTS + MILK_CALC_MAX_PARAMS];
    char           wbuf[MC_MAX_ARGS][72];
    char          *w[MC_MAX_ARGS];
    char           msg[160];

    for (int p = 0; p < nin; p++)
    {
        slots[p] = (milk_argslot_t) { (pts != NULL) ? pts->in_ports[p].name : "", 1 };
    }
    for (int k = 0; k < npar; k++)
    {
        const char *n  = ps->specs[k].name;
        slots[nin + k] = (milk_argslot_t) { n + (n[0] == '.'), 1 };
    }
    for (int i = 0; i < nargs; i++)
    {
        const char *k = kw[i] + (kw[i][0] == '.');
        for (int p = 0; pts != NULL && p < nin; p++)
        {
            k = milk_calc_port_match_(e->name, pts->in_ports[p].name, k) ? pts->in_ports[p].name
                                                                         : k;
        }
        npos += (k[0] == '\0');
        snprintf(wbuf[i], sizeof(wbuf[i]), (k[0] != '\0') ? ".%s=" : "%s",
                 (k[0] != '\0') ? k : "_");
        w[i] = wbuf[i];
    }
    if (npos > nin + npar - (nargs - npos))
    {
        snprintf(err, errsz, "%s: too many arguments (at most %d)", e->name, nin + npar);
        return -1;
    }
    if (milk_args_bind(nargs, w, nargs, nin + npar, slots, val, word, msg, sizeof(msg)) != 0)
    {
        snprintf(err, errsz, "%s: %s", e->name, msg);
        return -1;
    }
    for (int p = 0; p < nin; p++)
    {
        in_arg[p] = word[p];
    }
    for (int k = 0; k < npar; k++)
    {
        par_arg[k] = word[nin + k];
    }
    return nin;
}

/**
 * scan_call - Report the parameters and scalar outputs of the extension call at @toks[@at]
 * @toks:          Tokens of the program
 * @ntok:          Number of tokens
 * @at:            Index of the call's name (`toks[at + 1]` is its `(`)
 * @ndst:          Number of assignment targets of the statement
 * @lhs:           The targets
 * @param_cb:      Called for each parameter no argument binds
 * @scalar_out_cb: Called for each target that receives a `MILK_PORT_SCALAR` output port
 * @ud:            Passed to the callbacks
 *
 * Splits the arguments at the call's top-level commas and binds them with mc_ext_bind_args(), as
 * the runtime does. A call that does not bind (too many arguments, ...) reports nothing: it fails
 * when the program runs.
 */
static void scan_call(const mc_tok_t           *toks,
                      int                       ntok,
                      int                       at,
                      int                       ndst,
                      const char               *lhs[],
                      mcalc_ext_param_cb_t      param_cb,
                      mcalc_ext_scalar_out_cb_t scalar_out_cb,
                      void                     *ud)
{
    char        func[64];
    char        label[64];
    const char *colon = strchr(toks[at].sval, ':');
    size_t      flen  = (colon != NULL) ? (size_t) (colon - toks[at].sval) : strlen(toks[at].sval);
    snprintf(func, sizeof(func), "%.*s", (int) flen, toks[at].sval);
    snprintf(label, sizeof(label), "%s", (colon != NULL && colon[1] != '\0') ? colon + 1 : func);
    int                          ext_id = milk_calc_ext_find(func);
    const milk_calc_ext_entry_t *e      = milk_calc_ext_get(ext_id);
    if (e == NULL)
    {
        return;
    }

    char kw[MC_MAX_ARGS][64];
    int  nargs = 0;
    int  depth = 1;
    int  beg   = at + 2;
    for (int i = at + 2; i < ntok && depth > 0; i++)
    {
        mc_tok_type_t tt = toks[i].type;
        depth += (tt == MC_TOK_LPAREN || tt == MC_TOK_LBRACK);
        depth -= (tt == MC_TOK_RPAREN || tt == MC_TOK_RBRACK);
        if ((tt == MC_TOK_COMMA && depth == 1) || depth == 0)
        {
            if (i > beg && nargs < MC_MAX_ARGS)
            {
                int named = (i > beg + 1 && toks[beg].type == MC_TOK_IDENT &&
                             toks[beg + 1].type == MC_TOK_ASSIGN);
                snprintf(kw[nargs++], sizeof(kw[0]), "%s", named ? toks[beg].sval : "");
            }
            beg = i + 1;
        }
    }

    const milk_calc_param_set_t *ps = (e->get_params != NULL) ? e->get_params() : NULL;
    int                          in_arg[MILK_CALC_MAX_PORTS];
    int                          par_arg[MILK_CALC_MAX_PARAMS];
    char                         err[256];
    if (mc_ext_bind_args(ext_id, nargs, (const char (*)[64]) kw, in_arg, par_arg, err,
                         sizeof(err)) < 0)
    {
        return;
    }
    for (int k = 0; ps != NULL && param_cb != NULL && k < ps->nparams; k++)
    {
        if (par_arg[k] < 0)
        {
            param_cb(label, &ps->specs[k], e->kind == MILK_EXT_OP, ud);
        }
    }
    if (e->kind == MILK_EXT_OP && e->get_ports != NULL && scalar_out_cb != NULL)
    {
        const milk_calc_op_ports_t *ports = e->get_ports();
        for (int j = 0; j < ndst && j < ports->nout; j++)
        {
            if (ports->out_ports[j].kind == MILK_PORT_SCALAR)
            {
                scalar_out_cb(lhs[j], ports->out_ports[j].descr, ud);
            }
        }
    }
}

/**
 * mcalc_ext_scan_source - Report the extension parameters and scalar outputs of program @src
 * @src:           Program source text
 * @param_cb:      Called for each parameter of an extension call that no argument binds
 * @scalar_out_cb: Called for each assignment target that receives a scalar operator output
 * @ud:            Passed to the callbacks
 *
 * Lexes @src with the calculator lexer and walks its statements (split at top-level newlines,
 * `;`, `{` and `}`): the targets of `[local] a[, b ...] =` and every `name(` call to a registered
 * extension. A program that does not lex reports nothing: it fails when it runs.
 */
void mcalc_ext_scan_source(const char               *src,
                           mcalc_ext_param_cb_t      param_cb,
                           mcalc_ext_scalar_out_cb_t scalar_out_cb,
                           void                     *ud)
{
    mc_tok_list_t tl = { 0 };
    if (src == NULL || milk_calc_ext_count() == 0 || mc_lex(src, &tl) != 0)
    {
        mc_tok_list_free(&tl);
        return;
    }
    for (int st = 0; st < tl.ntok;)
    {
        int end   = st;
        int depth = 0;
        for (; end < tl.ntok && tl.toks[end].type != MC_TOK_EOF; end++)
        {
            mc_tok_type_t tt = tl.toks[end].type;
            depth += (tt == MC_TOK_LPAREN || tt == MC_TOK_LBRACK);
            depth -= (depth > 0 && (tt == MC_TOK_RPAREN || tt == MC_TOK_RBRACK));
            if (depth == 0 && (tt == MC_TOK_NEWLINE || tt == MC_TOK_SEMI || tt == MC_TOK_LBRACE ||
                               tt == MC_TOK_RBRACE))
            {
                break;
            }
        }

        const char *lhs[MILK_CALC_MAX_PORTS];
        int         ndst = 0;
        int         pos  = st;
        if (pos < end && tl.toks[pos].type == MC_TOK_IDENT &&
            strcmp(tl.toks[pos].sval, "local") == 0)
        {
            pos++;
        }
        while (ndst < MILK_CALC_MAX_PORTS && pos < end && tl.toks[pos].type == MC_TOK_IDENT)
        {
            lhs[ndst++] = tl.toks[pos++].sval;
            if (pos >= end || tl.toks[pos].type != MC_TOK_COMMA)
            {
                break;
            }
            pos++;
        }
        if (pos >= end || tl.toks[pos].type != MC_TOK_ASSIGN)
        {
            ndst = 0;
        }

        for (int i = st; i + 1 < end; i++)
        {
            if (tl.toks[i].type == MC_TOK_IDENT && tl.toks[i].no_space_after &&
                tl.toks[i + 1].type == MC_TOK_LPAREN)
            {
                scan_call(tl.toks, end, i, ndst, lhs, param_cb, scalar_out_cb, ud);
            }
        }
        st = end + 1;
    }
    mc_tok_list_free(&tl);
}
