// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_loader.c
 * @brief Dynamic `.c` / `.so` extension loader and static call-site scanner for `libmilkcalc`.
 *
 * Implements two subsystems supporting user-defined C extensions (`milk_calc_ext.h`):
 *   1. **Dynamic compilation and loading (`mcalc_ext_load`, `mcalc_ext_load_list`)**:
 *      Accepts either pre-compiled shared objects (`.so`) or raw `.c` source files (via `-l`,
 *      `loadlib`, or `MILK_CALC_LIBS`). Raw `.c` files are hashed with 64-bit FNV-1a over the ABI
 *      version, compiler name, and file contents, compiled once via `fork`/`execvp` (`gcc -O3
 *      -march=native -ffast-math -fopenmp-simd -shared -fPIC`) into
 *      `${TMPDIR:-/tmp}/milk-calc-cache-<uid>/ext_<hash>.so`, and loaded via `dlopen`. The exported
 *      `__milk_calc_ext_table` symbol is then registered in the global extension registry.
 *   2. **Static call-site scanner (`mcalc_ext_scan_source`)**:
 *      Tokenizes a `milk.calc` unit program before execution to discover every call to a
 *      registered extension (`func(...)` or `func:label(...)`). For any extension parameter not
 *      bound inline at the call site, invokes `param_cb` so the `milk.calc` FPS setup stage can
 *      auto-expose `<label>.<param>` in the unit's FPS parameter table, and invokes `scalar_out_cb`
 *      for scalar output ports unpacked on the LHS.
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
#include <sys/wait.h>
#include <unistd.h>

#include "mc_ast.h"
#include "mcalc.h"
#include "milk_calc_ext.h"

/* Maximum number of canonical library paths tracked for idempotent loading. */
#define CALC_EXT_MAX_LIBS 32

static char loaded_libs[CALC_EXT_MAX_LIBS][PATH_MAX];
static int  n_loaded_libs = 0;

/* ============================================================================================== */
/* Content Hashing and On-Demand `.c` -> `.so` Compilation                                        */
/* ============================================================================================== */

/**
 * fnv1a_bytes - Update a 64-bit FNV-1a hash state over a byte buffer
 * @h:   Initial or running 64-bit FNV-1a hash state
 * @buf: Input byte buffer
 * @len: Number of bytes in @buf
 *
 * Return: Updated 64-bit FNV-1a hash value
 */
static uint64_t fnv1a_bytes(uint64_t h, const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *) buf;
    for (size_t i = 0; i < len; i++)
    {
        h = (h ^ (uint64_t) p[i]) * 1099511628211ULL;
    }
    return h;
}

/**
 * find_milk_include_dirs - Discover `-I` header search paths for compiling `.c` extensions
 * @inc_dirs: Output array of up to 6 directory path strings
 * @n_inc:    Output count of populated entries in @inc_dirs
 *
 * Collects include directories from three sources in order:
 *   1. `$MILK_INSTALLDIR/include` and `$MILK_INSTALLDIR/include/milkcalc`.
 *   2. The prefix adjacent to `libmilkcalc.so` discovered via `dladdr(&mcalc_ext_load)`.
 *   3. The source tree directory containing `mc_ext_loader.c` and `../engine/libfps` (for
 *      in-tree unit tests before install).
 *
 * Return: void
 */
static void find_milk_include_dirs(char inc_dirs[6][PATH_MAX], int *n_inc)
{
    *n_inc          = 0;
    const char *env = getenv("MILK_INSTALLDIR");
    if (env != NULL && env[0] != '\0')
    {
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include", env);
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include/milkcalc", env);
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
                if (*n_inc < 6)
                {
                    snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include", rp);
                }
                if (*n_inc < 6)
                {
                    snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include/milkcalc", rp);
                }
            }
        }
    }
    char src_file[PATH_MAX];
    snprintf(src_file, sizeof(src_file), "%s", __FILE__);
    char *last_sl = strrchr(src_file, '/');
    if (last_sl != NULL && *n_inc < 6)
    {
        *last_sl = '\0';
        snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", src_file);
        char repo_inc[PATH_MAX];
        snprintf(repo_inc, sizeof(repo_inc), "%s/../engine/libfps", src_file);
        if (access(repo_inc, R_OK) == 0 && *n_inc < 6)
        {
            snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", repo_inc);
        }
    }
}

/**
 * compile_c_to_so - Compile a user `.c` extension source file into a cached `.so` shared library
 * @c_path:    Path to the `.c` extension source file
 * @so_out:    Output buffer receiving the compiled `.so` path
 * @so_out_sz: Capacity of @so_out in bytes
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Hashes the ABI version, source path, compiler binary (`$CC` or `"gcc"`), and file bytes to
 * derive a content-addressed cache path `ext_<hash>.so`. If the cached `.so` already exists on
 * disk, returns immediately; otherwise compiles to a PID-qualified temporary file and atomically
 * renames it into place.
 *
 * Return: 0 on success, -1 on file I/O or compilation error
 */
static int compile_c_to_so(const char *c_path,
                           char       *so_out,
                           size_t      so_out_sz,
                           char       *err,
                           size_t      errsz)
{
    const char *cc_env = getenv("CC");
    const char *cc     = (cc_env != NULL && cc_env[0] != '\0') ? cc_env : "gcc";
    FILE       *fp     = fopen(c_path, "rb");
    if (fp == NULL)
    {
        snprintf(err, errsz, "cannot open '%s': %s", c_path, strerror(errno));
        return -1;
    }
    uint32_t      abi = MILK_CALC_EXT_ABI_VERSION;
    uint64_t      h   = fnv1a_bytes(14695981039346656037ULL, &abi, sizeof(abi));
    unsigned char buf[4096];
    size_t        nread;
    h = fnv1a_bytes(fnv1a_bytes(h, c_path, strlen(c_path)), cc, strlen(cc));
    while ((nread = fread(buf, 1, sizeof(buf), fp)) > 0)
    {
        h = fnv1a_bytes(h, buf, nread);
    }
    fclose(fp);

    const char *tmpdir = getenv("TMPDIR");
    tmpdir             = (tmpdir != NULL && tmpdir[0] != '\0') ? tmpdir : "/tmp";
    char cache_dir[PATH_MAX];
    char tmp_so[PATH_MAX];
    snprintf(cache_dir, sizeof(cache_dir), "%s/milk-calc-cache-%u", tmpdir, (unsigned) getuid());
    (void) mkdir(cache_dir, 0700);
    snprintf(so_out, so_out_sz, "%s/ext_%016llx.so", cache_dir, (unsigned long long) h);
    if (access(so_out, R_OK) == 0)
    {
        return 0;
    }
    snprintf(tmp_so, sizeof(tmp_so), "%s/ext_%016llx.%d.tmp.so", cache_dir, (unsigned long long) h,
             (int) getpid());

    char inc_dirs[6][PATH_MAX];
    char inc_flags[6][PATH_MAX + 4];
    int  n_inc = 0;
    int  ac    = 0;
    find_milk_include_dirs(inc_dirs, &n_inc);
    const char *argv[32] = {
        cc,        "-O3",   "-march=native", "-ffast-math", "-fopenmp-simd",
        "-shared", "-fPIC", "-Wall",         "-Wextra",     "-Wno-unused-parameter"
    };
    ac = 10;
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
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        rename(tmp_so, so_out) != 0)
    {
        unlink(tmp_so);
        snprintf(err, errsz, "compilation of '%s' failed", c_path);
        return -1;
    }
    return 0;
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
 * a fast no-op.
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
    for (int i = 0; i < n_loaded_libs; i++)
    {
        if (strcmp(loaded_libs[i], canon) == 0)
        {
            return 0;
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
    if (n_loaded_libs < CALC_EXT_MAX_LIBS)
    {
        snprintf(loaded_libs[n_loaded_libs++], PATH_MAX, "%s", canon);
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
/* Static Source Call-Site Scanner for FPS Parameter Discovery                                    */
/* ============================================================================================== */

/**
 * split_ext_call_name - Split an extension call token `"func:label"` into function name and label
 * @raw:       Raw identifier token from lexer (e.g. `"bblur1pix:wfs"` or `"bblur1pix"`)
 * @func_out:  Output buffer receiving the base extension function name
 * @fsz:       Capacity of @func_out in bytes
 * @label_out: Output buffer receiving the instance label (defaults to @func_out if no `:` present)
 * @lsz:       Capacity of @label_out in bytes
 *
 * Return: void
 */
static void split_ext_call_name(const char *raw,
                                char       *func_out,
                                size_t      fsz,
                                char       *label_out,
                                size_t      lsz)
{
    const char *colon = (raw != NULL) ? strchr(raw, ':') : NULL;
    if (colon == NULL)
    {
        snprintf(func_out, fsz, "%s", (raw != NULL) ? raw : "");
        snprintf(label_out, lsz, "%s", (raw != NULL) ? raw : "");
        return;
    }
    size_t flen = (size_t) (colon - raw);
    if (flen >= fsz)
    {
        flen = fsz - 1;
    }
    memcpy(func_out, raw, flen);
    func_out[flen] = '\0';
    snprintf(label_out, lsz, "%s", (colon[1] != '\0') ? (colon + 1) : func_out);
}

/**
 * scan_one_ext_call - Inspect one extension call token sequence and report unbound FPS parameters
 * @toks:          Token array for the current statement
 * @ntok:          Total number of tokens in @toks
 * @call_idx:      Index of the function identifier token in @toks (`toks[call_idx + 1]` is `(`)
 * @ndst:          Number of LHS assignment target identifiers on this statement
 * @lhs:           Array of LHS target identifier strings
 * @param_cb:      Callback invoked for each extension parameter not bound inline at the call site
 * @scalar_out_cb: Callback invoked for each LHS target bound to a `MILK_PORT_SCALAR` output port
 * @ud:            Opaque user data pointer forwarded to @param_cb and @scalar_out_cb
 *
 * Walks the argument tokens between `(` and its matching `)` to determine which parameters in
 * `e->get_params()` are bound either by keyword (`param = val`) or by positional scalar argument.
 * Any remaining unbound parameters are reported via `param_cb(label, &pset->specs[p], is_op, ud)`.
 *
 * Return: void
 */
static void scan_one_ext_call(const mc_tok_t           *toks,
                              int                       ntok,
                              int                       call_idx,
                              int                       ndst,
                              char                      lhs[][256],
                              mcalc_ext_param_cb_t      param_cb,
                              mcalc_ext_scalar_out_cb_t scalar_out_cb,
                              void                     *ud)
{
    char func_name[64];
    char label[64];
    split_ext_call_name(toks[call_idx].sval, func_name, sizeof(func_name), label, sizeof(label));
    int                          ext_id = milk_calc_ext_find(func_name);
    const milk_calc_ext_entry_t *e      = milk_calc_ext_get(ext_id);
    if (e == NULL)
    {
        return;
    }

    const milk_calc_param_set_t *pset = e->get_params ? e->get_params() : NULL;
    int                          bound[MILK_CALC_MAX_PARAMS] = { 0 };
    int                          nin_img                     = (e->kind == MILK_EXT_BLK2) ? 2 : 1;
    int                          depth                       = 1;
    int                          arg_idx                     = 0;
    int                          arg_beg                     = call_idx + 2;

    for (int i = call_idx + 2; i < ntok && depth > 0; i++)
    {
        mc_tok_type_t tt = toks[i].type;
        if (tt == MC_TOK_LPAREN || tt == MC_TOK_LBRACK)
        {
            depth++;
            continue;
        }
        if ((tt == MC_TOK_RPAREN || tt == MC_TOK_RBRACK) && --depth > 0)
        {
            continue;
        }
        if ((tt == MC_TOK_COMMA && depth == 1) || (tt == MC_TOK_RPAREN && depth == 0))
        {
            if (i > arg_beg)
            {
                int is_kw = (arg_beg + 1 < i && toks[arg_beg].type == MC_TOK_IDENT &&
                             toks[arg_beg + 1].type == MC_TOK_ASSIGN);
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

/**
 * scan_statement_tokens - Lex a single statement slice and scan all function calls within it
 * @st:            Pointer to statement start in source text
 * @len:           Length of the statement in bytes
 * @param_cb:      Callback for unbound extension parameters
 * @scalar_out_cb: Callback for scalar output ports unpacked into LHS variables
 * @ud:            Opaque user data pointer
 *
 * Extracts any comma-separated LHS target identifiers preceding `=` and invokes
 * `scan_one_ext_call()` on every `IDENT(` token pair in the statement.
 *
 * Return: void
 */
static void scan_statement_tokens(const char               *st,
                                  size_t                    len,
                                  mcalc_ext_param_cb_t      param_cb,
                                  mcalc_ext_scalar_out_cb_t scalar_out_cb,
                                  void                     *ud)
{
    if (len == 0 || len >= 4096)
    {
        return;
    }
    char line[4096];
    memcpy(line, st, len);
    line[len] = '\0';

    mc_tok_list_t *tl = calloc(1, sizeof(*tl));
    if (tl == NULL || mc_lex(line, tl) != 0)
    {
        free(tl);
        return;
    }

    char lhs[MILK_CALC_MAX_PORTS][256];
    int  ndst = 0;
    int  pos  = 0;
    memset(lhs, 0, sizeof(lhs));
    if (pos < tl->ntok && tl->toks[pos].type == MC_TOK_IDENT &&
        strcmp(tl->toks[pos].sval, "local") == 0)
    {
        pos++;
    }
    while (ndst < MILK_CALC_MAX_PORTS && pos < tl->ntok)
    {
        if (tl->toks[pos].type != MC_TOK_IDENT)
        {
            ndst = 0;
            break;
        }
        snprintf(lhs[ndst++], sizeof(lhs[0]), "%s", tl->toks[pos].sval);
        pos++;
        if (pos < tl->ntok && tl->toks[pos].type == MC_TOK_COMMA)
        {
            pos++;
        }
        else
        {
            break;
        }
    }
    if (pos >= tl->ntok || tl->toks[pos].type != MC_TOK_ASSIGN)
    {
        ndst = 0;
    }

    for (int i = 0; i + 1 < tl->ntok; i++)
    {
        if (tl->toks[i].type == MC_TOK_IDENT && tl->toks[i].no_space_after &&
            tl->toks[i + 1].type == MC_TOK_LPAREN)
        {
            scan_one_ext_call(tl->toks, tl->ntok, i, ndst, lhs, param_cb, scalar_out_cb, ud);
        }
    }
    free(tl);
}

/**
 * mcalc_ext_scan_source - Scan program source @src for extension calls and unbound FPS parameters
 * @src:           Program source text
 * @param_cb:      Callback invoked for each unbound extension parameter
 * @scalar_out_cb: Callback invoked for each LHS variable receiving a scalar operator output port
 * @ud:            Opaque user data pointer passed to callbacks
 *
 * Splits @src into individual statements (respecting quotes, comments, parentheses, and braces)
 * and delegates each statement to `scan_statement_tokens()`.
 *
 * Return: void
 */
void mcalc_ext_scan_source(const char               *src,
                           mcalc_ext_param_cb_t      param_cb,
                           mcalc_ext_scalar_out_cb_t scalar_out_cb,
                           void                     *ud)
{
    if (src == NULL || src[0] == '\0' || milk_calc_ext_count() == 0)
    {
        return;
    }
    for (const char *p = src; *p != '\0';)
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
        scan_statement_tokens(st, (size_t) (p - st), param_cb, scalar_out_cb, ud);
        if (*p == '{' || *p == '}')
        {
            p++;
        }
    }
}
