// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_registry.c
 * @brief In-memory registry for `milk_calc_ext` kernels and operators.
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

#include "mcalc.h"
#include "milk_calc_ext.h"

#define CALC_EXT_MAX_ENTRIES 64
#define CALC_EXT_MAX_LIBS 32

static milk_calc_ext_entry_t ext_entries_storage[CALC_EXT_MAX_ENTRIES];
static char                  ext_names_storage[CALC_EXT_MAX_ENTRIES][64];
static int                   ext_nentries = 0;
static char                  loaded_libs[CALC_EXT_MAX_LIBS][PATH_MAX];
static int                   n_loaded_libs = 0;

static size_t param_fptype_size(uint32_t fptype)
{
    switch (fptype)
    {
    case FPTYPE_FLOAT64:
    case FPTYPE_INT64:
    case FPTYPE_UINT64:
        return 8;
    case FPTYPE_FLOAT32:
    case FPTYPE_INT32:
    case FPTYPE_UINT32:
        return 4;
    case FPTYPE_ONOFF:
        return sizeof(int);
    default:
        return 0;
    }
}

static int check_param_set(const char                  *cname,
                           const milk_calc_ext_entry_t *e,
                           char                        *err,
                           size_t                       errsz)
{
    const milk_calc_param_set_t *ps = (e->get_params != NULL) ? e->get_params() : NULL;
    if (ps == NULL || ps->nparams == 0)
    {
        return 0;
    }
    if (ps->nparams < 0 || ps->nparams > MILK_CALC_MAX_PARAMS || ps->specs == NULL)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "'%s': %d parameters, the engine packs at most %d", cname,
                     ps->nparams, MILK_CALC_MAX_PARAMS);
        }
        return -1;
    }
    if (ps->struct_size > MILK_CALC_PARAM_BYTES)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "'%s': parameter struct of %zu bytes, the engine packs at most %d",
                     cname, ps->struct_size, MILK_CALC_PARAM_BYTES);
        }
        return -1;
    }
    for (int i = 0; i < ps->nparams; i++)
    {
        const milk_calc_param_spec_t *sp   = &ps->specs[i];
        const char                   *pn   = (sp->name != NULL) ? sp->name : "?";
        size_t                        need = param_fptype_size(sp->fptype);
        if (need == 0)
        {
            if (err != NULL && errsz > 0)
            {
                snprintf(err, errsz,
                         "'%s': parameter '%s' has a type the engine cannot pack (numeric only)",
                         cname, pn);
            }
            return -1;
        }
        if (sp->size != need || sp->offset > ps->struct_size || sp->offset + need > ps->struct_size)
        {
            if (err != NULL && errsz > 0)
            {
                snprintf(err, errsz,
                         "'%s': parameter '%s' (%zu bytes at offset %zu) does not fit its type",
                         cname, pn, sp->size, sp->offset);
            }
            return -1;
        }
    }
    return 0;
}

int milk_calc_ext_register(const milk_calc_ext_entry_t *e, char *err, size_t errsz)
{
    if (e == NULL || e->name == NULL || e->name[0] == '\0')
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "invalid extension entry (empty name)");
        }
        return -1;
    }
    char        cname[64];
    const char *np   = e->name;
    size_t      nlen = strlen(np);
    if (nlen >= 2 && np[0] == '"' && np[nlen - 1] == '"')
    {
        np++;
        nlen -= 2;
    }
    if (nlen == 0 || nlen >= sizeof(cname))
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "invalid extension name '%s'", e->name);
        }
        return -1;
    }
    memcpy(cname, np, nlen);
    cname[nlen] = '\0';

    if (e->kind == MILK_EXT_STENCIL && (e->radius < 1 || e->radius > 3))
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "stencil '%s': radius %d out of range 1..3", cname, e->radius);
        }
        return -1;
    }
    if (check_param_set(cname, e, err, errsz) != 0)
    {
        return -1;
    }
    for (int i = 0; i < ext_nentries; i++)
    {
        if (strcmp(ext_entries_storage[i].name, cname) == 0)
        {
            snprintf(ext_names_storage[i], sizeof(ext_names_storage[i]), "%s", cname);
            ext_entries_storage[i]      = *e;
            ext_entries_storage[i].name = ext_names_storage[i];
            return 0;
        }
    }
    if (ext_nentries >= CALC_EXT_MAX_ENTRIES)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "too many registered calc extensions (max %d)",
                     CALC_EXT_MAX_ENTRIES);
        }
        return -1;
    }
    snprintf(ext_names_storage[ext_nentries], sizeof(ext_names_storage[ext_nentries]), "%s", cname);
    ext_entries_storage[ext_nentries]      = *e;
    ext_entries_storage[ext_nentries].name = ext_names_storage[ext_nentries];
    ext_nentries++;
    return 0;
}

int milk_calc_ext_register_table(const milk_calc_ext_table_t *tbl, char *err, size_t errsz)
{
    if (tbl == NULL)
    {
        return -1;
    }
    if (tbl->abi_version != MILK_CALC_EXT_ABI_VERSION)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "extension ABI version %u != expected %u", tbl->abi_version,
                     (unsigned) MILK_CALC_EXT_ABI_VERSION);
        }
        return -1;
    }
    for (int i = 0; i < tbl->nentries; i++)
    {
        if (milk_calc_ext_register(tbl->entries[i], err, errsz) != 0)
        {
            return -1;
        }
    }
    return 0;
}

int milk_calc_ext_find(const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    for (int i = 0; i < ext_nentries; i++)
    {
        if (strcmp(ext_entries_storage[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

const milk_calc_ext_entry_t *milk_calc_ext_get(int idx)
{
    return (idx >= 0 && idx < ext_nentries) ? &ext_entries_storage[idx] : NULL;
}

int milk_calc_ext_count(void)
{
    return ext_nentries;
}

static uint64_t fnv1a_bytes(uint64_t h, const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *) buf;
    for (size_t i = 0; i < len; i++)
    {
        h = (h ^ (uint64_t) p[i]) * 1099511628211ULL;
    }
    return h;
}

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
                (void) (*n_inc < 6 &&
                        (snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include", rp), 1));
                (void) (*n_inc < 6 &&
                        (snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s/include/milkcalc", rp), 1));
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
        (void) (access(repo_inc, R_OK) == 0 && *n_inc < 6 &&
                (snprintf(inc_dirs[(*n_inc)++], PATH_MAX, "%s", repo_inc), 1));
    }
}

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
    char cache_dir[PATH_MAX], tmp_so[PATH_MAX];
    snprintf(cache_dir, sizeof(cache_dir), "%s/milk-calc-cache-%u", tmpdir, (unsigned) getuid());
    (void) mkdir(cache_dir, 0700);
    snprintf(so_out, so_out_sz, "%s/ext_%016llx.so", cache_dir, (unsigned long long) h);
    if (access(so_out, R_OK) == 0)
    {
        return 0;
    }
    snprintf(tmp_so, sizeof(tmp_so), "%s/ext_%016llx.%d.tmp.so", cache_dir, (unsigned long long) h,
             (int) getpid());

    char inc_dirs[6][PATH_MAX], inc_flags[6][PATH_MAX + 4];
    int  n_inc = 0, ac = 0;
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

int mcalc_ext_load(const char *path, char *err, size_t errsz)
{
    if (path == NULL || path[0] == '\0')
    {
        return 0;
    }
    char        rpath_buf[PATH_MAX], so_path[PATH_MAX];
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
        (void) (tbl == NULL &&
                snprintf(err, errsz, "'%s' is missing MILK_CALC_EXT_EXPORT(...)", path));
        dlclose(dl);
        return -1;
    }
    (void) (n_loaded_libs < CALC_EXT_MAX_LIBS &&
            snprintf(loaded_libs[n_loaded_libs++], PATH_MAX, "%s", canon));
    return 0;
}

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

int mcalc_ext_count(void)
{
    return milk_calc_ext_count();
}

const milk_calc_ext_entry_t *mcalc_ext_get(int idx)
{
    return milk_calc_ext_get(idx);
}

void mcalc_ext_scan_source(const char               *src,
                           mcalc_ext_param_cb_t      param_cb,
                           mcalc_ext_scalar_out_cb_t scalar_out_cb,
                           void                     *ud)
{
    (void) src;
    (void) param_cb;
    (void) scalar_out_cb;
    (void) ud;
}
