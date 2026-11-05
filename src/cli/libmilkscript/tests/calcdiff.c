// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file calcdiff.c
 * @brief Differential test harness and micro-benchmark runner for the calculator engines.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <ftw.h>
#include <limits.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/image_coord.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "cli_calc_prog.h"
#include "cli_calc_scalar.h"
#include "mc_func.h"
#include "mcalc.h"
#include "mcalc_host.h"
#include "milkscript.h"

#define CALCDIFF_MAX_SETUP 256
#define CALCDIFF_OUT_SZ 65536

typedef struct
{
    char           name[STRINGMAXLEN_IMAGE_NAME];
    uint8_t        naxis;
    uint32_t       size[3];
    uint8_t        datatype;
    uint8_t        shared;
    uint16_t       nbkw;
    uint64_t       cnt0;
    uint64_t       nelement;
    size_t         nbytes;
    void          *data;
    IMAGE_KEYWORD *kw;
} img_snap_t;

typedef struct
{
    char   name[80];
    int    type;
    double f;
    long   l;
    char   s[512];
    int    from_clivar;
} var_snap_t;

typedef struct
{
    img_snap_t     *imgs;
    int             nimgs;
    var_snap_t     *vars;
    int             nvars;
    CLI_VAR         clivars[CLI_MAX_VARS];
    cli_calc_udef_t udefs[CALC_UDEF_MAX];
    int             nudefs;
} state_snap_t;

typedef struct
{
    int          ok;
    int          rc;
    int          not_impl;
    char         out[CALCDIFF_OUT_SZ];
    state_snap_t post;
} eval_rec_t;

static char g_tmpdir[PATH_MAX] = "";

/* Remove a file or directory entry during nftw cleanup. */
static int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
    (void) sb;
    (void) typeflag;
    (void) ftwbuf;
    return remove(fpath);
}

/* Remove the temporary shared-memory directory at exit. */
static void cleanup_tmpdir(void)
{
    if (g_tmpdir[0] != '\0')
    {
        nftw(g_tmpdir, unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
        g_tmpdir[0] = '\0';
    }
}

/* Read monotonic clock in nanoseconds. */
static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

/* Return 1 if @name is one of the predefined session constants (_PI, _e, ...) or internal loop vars. */
static int is_predefined_var(const char *name)
{
    if (strncmp(name, "__fstop", 7) == 0 || strncmp(name, "__fstep", 7) == 0)
    {
        return 1;
    }
    static const char *const names[] = { "_PI", "_e",  "_gamma", "_c", "_h",
                                         "_k",  "_pc", "_ly",    "_AU" };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
    {
        if (strcmp(name, names[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/* Return 1 if @name is an internal calculator temporary image. */
static int is_internal_img(const char *name)
{
    return strncmp(name, "_tmpcalc", 8) == 0 || strncmp(name, "_calcprog_", 10) == 0;
}

/* Suppress stdout and stderr while running a noisy function; returns saved fds in @saved. */
static void silence_begin(int saved[2])
{
    fflush(stdout);
    fflush(stderr);
    saved[0]    = dup(STDOUT_FILENO);
    saved[1]    = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0)
    {
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
}

/* Restore stdout and stderr from @saved. */
static void silence_end(const int saved[2])
{
    fflush(stdout);
    fflush(stderr);
    if (saved[0] >= 0)
    {
        dup2(saved[0], STDOUT_FILENO);
        close(saved[0]);
    }
    if (saved[1] >= 0)
    {
        dup2(saved[1], STDERR_FILENO);
        close(saved[1]);
    }
}

/* Initialize milkscript quietly in a private SHM directory. */
static int init_engine(void)
{
    snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/milk_calcdiff_%d", (int) getpid());
    mkdir(g_tmpdir, 0700);
    atexit(cleanup_tmpdir);
    setenv("MILK_SHM_DIR", g_tmpdir, 1);
    setenv("MILK_PROC_DIR", g_tmpdir, 1);
    int   saved[2];
    char *argv[] = { (char *) "milk-calcdiff", NULL };
    silence_begin(saved);
    int rc = milkscript_init(1, argv);
    silence_end(saved);
    return rc;
}

/* Free all dynamically allocated memory inside @snap. */
static void snap_free(state_snap_t *snap)
{
    for (int i = 0; i < snap->nimgs; i++)
    {
        free(snap->imgs[i].data);
        free(snap->imgs[i].kw);
    }
    free(snap->imgs);
    free(snap->vars);
    memset(snap, 0, sizeof(*snap));
}

/* Delete all images (including temporary pool images) and non-predefined variables. */
static void workspace_clear(void)
{
    cli_calc_tmp_cleanup();
    mcalc_tmp_cleanup();
    for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
    {
        if (data.core.image[i].used)
        {
            char nm[STRINGMAXLEN_IMAGE_NAME];
            snprintf(nm, sizeof(nm), "%s", data.core.image[i].name);
            delete_image_ID(nm, DELETE_IMAGE_ERRMODE_IGNORE);
        }
    }
    for (long i = 0; data.core.variable != NULL && i < data.core.NB_MAX_VARIABLE; i++)
    {
        VARIABLE *v = &data.core.variable[i];
        if (v->used && !is_predefined_var(v->name))
        {
            v->used    = 0;
            v->type    = -1;
            v->name[0] = '\0';
        }
    }
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used && !is_predefined_var(cli_vars[i].name))
        {
            cli_vars[i].used    = 0;
            cli_vars[i].name[0] = '\0';
            cli_vars[i].val[0]  = '\0';
        }
    }
    if (g_tmpdir[0] != '\0')
    {
        DIR *d = opendir(g_tmpdir);
        if (d != NULL)
        {
            struct dirent *de;
            while ((de = readdir(d)) != NULL)
            {
                if (strstr(de->d_name, ".im.shm") != NULL)
                {
                    char p[PATH_MAX];
                    snprintf(p, sizeof(p), "%s/%s", g_tmpdir, de->d_name);
                    unlink(p);
                }
            }
            closedir(d);
        }
    }
    milk_table_bump();
    cli_calc_udef_clear();
    mcalc_udef_clear();
    cli_float_digits   = 6;
    mcalc_float_digits = 6;
    dcprecision        = 0;
    cli_calc_out_mode  = CLI_CALC_OUT_VALUES;
    mcalc_out_mode     = MCALC_OUT_VALUES;
}

/* Delete any lingering _tmpcalc* or _tmpcalcpool* images so pool state starts clean. */
static void clear_tmp_pool(void)
{
    cli_calc_tmp_cleanup();
    mcalc_tmp_cleanup();
    for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
    {
        if (data.core.image[i].used && is_internal_img(data.core.image[i].name))
        {
            char nm[STRINGMAXLEN_IMAGE_NAME];
            snprintf(nm, sizeof(nm), "%s", data.core.image[i].name);
            delete_image_ID(nm, DELETE_IMAGE_ERRMODE_IGNORE);
        }
    }
}

/* Find image @name in @snap, or NULL if absent. */
static const img_snap_t *find_img(const state_snap_t *snap, const char *name)
{
    for (int i = 0; i < snap->nimgs; i++)
    {
        if (strcmp(snap->imgs[i].name, name) == 0)
        {
            return &snap->imgs[i];
        }
    }
    return NULL;
}

/* Capture all user images, variables, and udefs into @snap. */
static void snap_capture(state_snap_t *snap)
{
    memset(snap, 0, sizeof(*snap));
    clear_tmp_pool();

    int img_cap = 32;
    snap->imgs  = calloc((size_t) img_cap, sizeof(img_snap_t));
    for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
    {
        const IMAGE *im = &data.core.image[i];
        if (!im->used || im->md == NULL || im->array.raw == NULL || is_internal_img(im->name))
        {
            continue;
        }
        if (snap->nimgs >= img_cap)
        {
            img_cap *= 2;
            snap->imgs = realloc(snap->imgs, (size_t) img_cap * sizeof(img_snap_t));
        }
        img_snap_t *s = &snap->imgs[snap->nimgs++];
        memset(s, 0, sizeof(*s));
        snprintf(s->name, sizeof(s->name), "%s", im->name);
        s->naxis    = im->md[0].naxis;
        s->size[0]  = im->md[0].size[0];
        s->size[1]  = (s->naxis > 1) ? im->md[0].size[1] : 1;
        s->size[2]  = (s->naxis > 2) ? im->md[0].size[2] : 1;
        s->datatype = im->md[0].datatype;
        s->shared   = im->md[0].shared;
        s->nbkw     = im->md[0].NBkw;
        s->cnt0     = im->md[0].cnt0;
        s->nelement = im->md[0].nelement;
        s->nbytes   = (size_t) s->nelement * (size_t) ImageStreamIO_typesize(s->datatype);
        s->data     = malloc(s->nbytes > 0 ? s->nbytes : 1);
        if (s->nbytes > 0)
        {
            memcpy(s->data, im->array.raw, s->nbytes);
        }
        if (s->nbkw > 0 && im->kw != NULL)
        {
            s->kw = malloc((size_t) s->nbkw * sizeof(IMAGE_KEYWORD));
            memcpy(s->kw, im->kw, (size_t) s->nbkw * sizeof(IMAGE_KEYWORD));
        }
    }

    int var_cap = 64;
    snap->vars  = calloc((size_t) var_cap, sizeof(var_snap_t));
    for (long i = 0; data.core.variable != NULL && i < data.core.NB_MAX_VARIABLE; i++)
    {
        const VARIABLE *v = &data.core.variable[i];
        if (!v->used || is_predefined_var(v->name))
        {
            continue;
        }
        if (snap->nvars >= var_cap)
        {
            var_cap *= 2;
            snap->vars = realloc(snap->vars, (size_t) var_cap * sizeof(var_snap_t));
        }
        var_snap_t *vs = &snap->vars[snap->nvars++];
        memset(vs, 0, sizeof(*vs));
        snprintf(vs->name, sizeof(vs->name), "%s", v->name);
        vs->type = v->type;
        if (v->type == 0)
        {
            vs->f = v->value.f;
        }
        else if (v->type == 1)
        {
            vs->l = v->value.l;
        }
        else if (v->type == 2)
        {
            snprintf(vs->s, sizeof(vs->s), "%s", v->value.s);
        }
    }
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        const CLI_VAR *cv = &cli_vars[i];
        if (!cv->used || cv->type != 2 || is_predefined_var(cv->name) || variable_ID(cv->name) >= 0)
        {
            continue;
        }
        if (snap->nvars >= var_cap)
        {
            var_cap *= 2;
            snap->vars = realloc(snap->vars, (size_t) var_cap * sizeof(var_snap_t));
        }
        var_snap_t *vs = &snap->vars[snap->nvars++];
        memset(vs, 0, sizeof(*vs));
        snprintf(vs->name, sizeof(vs->name), "%s", cv->name);
        vs->type        = 2;
        vs->from_clivar = 1;
        snprintf(vs->s, sizeof(vs->s), "%s", cv->val);
    }
    memcpy(snap->clivars, cli_vars, sizeof(cli_vars));

    snap->nudefs = 0;
    for (int i = 0; i < CALC_UDEF_MAX; i++)
    {
        const cli_calc_udef_t *u = cli_calc_udef_get(i);
        if (u == NULL)
        {
            break;
        }
        snap->udefs[snap->nudefs++] = *u;
    }
}

/* Restore workspace images, variables, and udefs from @snap. */
static void snap_restore(const state_snap_t *snap)
{
    int saved[2];
    silence_begin(saved);
    clear_tmp_pool();

    int had_shared = 0;
    for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
    {
        const IMAGE *im = &data.core.image[i];
        if (!im->used)
        {
            continue;
        }
        const img_snap_t *s = find_img(snap, im->name);
        uint32_t          y = (im->md != NULL && im->md[0].naxis > 1) ? im->md[0].size[1] : 1;
        uint32_t          z = (im->md != NULL && im->md[0].naxis > 2) ? im->md[0].size[2] : 1;
        if (s == NULL || s->shared != 0 || im->md == NULL || im->md[0].shared != 0 ||
            im->md[0].naxis != s->naxis || im->md[0].datatype != s->datatype ||
            im->md[0].size[0] != s->size[0] || y != s->size[1] || z != s->size[2] ||
            im->md[0].NBkw < s->nbkw)
        {
            if (im->md != NULL && im->md[0].shared != 0)
            {
                had_shared = 1;
            }
            char nm[STRINGMAXLEN_IMAGE_NAME];
            snprintf(nm, sizeof(nm), "%s", im->name);
            delete_image_ID(nm, DELETE_IMAGE_ERRMODE_IGNORE);
        }
    }
    if (had_shared && g_tmpdir[0] != '\0')
    {
        DIR *d = opendir(g_tmpdir);
        if (d != NULL)
        {
            struct dirent *de;
            while ((de = readdir(d)) != NULL)
            {
                if (strstr(de->d_name, ".im.shm") != NULL)
                {
                    char p[PATH_MAX];
                    snprintf(p, sizeof(p), "%s/%s", g_tmpdir, de->d_name);
                    unlink(p);
                }
            }
            closedir(d);
        }
    }

    for (long i = 0; data.core.variable != NULL && i < data.core.NB_MAX_VARIABLE; i++)
    {
        VARIABLE *v = &data.core.variable[i];
        if (v->used && !is_predefined_var(v->name))
        {
            v->used    = 0;
            v->type    = -1;
            v->name[0] = '\0';
        }
    }
    for (int i = 0; i < snap->nvars; i++)
    {
        const var_snap_t *vs = &snap->vars[i];
        if (vs->type == 0)
        {
            create_variable_ID(vs->name, vs->f);
        }
        else if (vs->type == 1)
        {
            create_variable_long_ID(vs->name, vs->l);
        }
        else if (vs->type == 2 && !vs->from_clivar)
        {
            create_variable_string_ID(vs->name, vs->s);
        }
    }
    memcpy(cli_vars, snap->clivars, sizeof(cli_vars));

    for (int i = 0; i < snap->nimgs; i++)
    {
        const img_snap_t *s  = &snap->imgs[i];
        imageID           id = image_ID(s->name, data.core.image, data.core.NB_MAX_IMAGE);
        if (id < 0)
        {
            uint32_t sz[3] = { s->size[0], s->size[1], s->size[2] };
            create_image_ID(s->name, s->naxis, sz, s->datatype, s->shared,
                            s->nbkw > 0 ? s->nbkw : 10, 0, &id);
        }
        if (id >= 0 && s->nbytes > 0)
        {
            memcpy(data.core.image[id].array.raw, s->data, s->nbytes);
        }
        if (id >= 0 && data.core.image[id].md != NULL)
        {
            data.core.image[id].md[0].cnt0 = s->cnt0;
            if (data.core.image[id].kw != NULL && data.core.image[id].md[0].NBkw > 0)
            {
                memset(data.core.image[id].kw, 0,
                       (size_t) data.core.image[id].md[0].NBkw * sizeof(IMAGE_KEYWORD));
                if (s->nbkw > 0 && s->kw != NULL)
                {
                    memcpy(data.core.image[id].kw, s->kw, (size_t) s->nbkw * sizeof(IMAGE_KEYWORD));
                }
            }
        }
    }

    cli_calc_udef_clear();
    mcalc_udef_clear();
    for (int i = 0; i < snap->nudefs; i++)
    {
        const cli_calc_udef_t *u = &snap->udefs[i];
        cli_calc_udef_add(u->name, u->nargs, (const char (*)[64]) u->arg_names, u->body);
        mcalc_udef_add(u->name, u->nargs, (const char (*)[64]) u->arg_names, u->body);
    }
    cli_float_digits   = 6;
    mcalc_float_digits = 6;
    dcprecision        = 0;
    cli_calc_out_mode  = CLI_CALC_OUT_VALUES;
    mcalc_out_mode     = MCALC_OUT_VALUES;
    milk_table_bump();
    silence_end(saved);
}

/* Create a 2-D or 3-D local image of the given datatype and return its IMAGE pointer. */
static IMAGE *make_img(const char *name,
                       uint8_t     naxis,
                       uint32_t    nx,
                       uint32_t    ny,
                       uint32_t    nz,
                       uint8_t     dt)
{
    imageID  id    = -1;
    uint32_t sz[3] = { nx, ny, nz };
    create_image_ID(name, naxis, sz, dt, 0, 10, 0, &id);
    return (id >= 0) ? &data.core.image[id] : NULL;
}

/* Set a scalar double variable in both data.core.variable and cli_vars. */
static void set_fvar(const char *name, double val)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", val);
    cli_var_set(name, buf);
    create_variable_ID(name, val);
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used && strcmp(cli_vars[i].name, name) == 0)
        {
            cli_vars[i].type  = 0;
            cli_vars[i].num.f = val;
            break;
        }
    }
}

/* Set a scalar long variable in both data.core.variable and cli_vars. */
static void set_lvar(const char *name, long val)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%ld", val);
    cli_var_set(name, buf);
    create_variable_long_ID(name, val);
}

/* Populate the standard benchmark/default float32 images A B C D G O U V M at size NxN. */
static void build_base_images(uint32_t N, uint32_t NY)
{
    const char *names[] = { "A", "B", "C", "D", "G", "O", "U", "V", "M" };
    IMAGE      *ims[9];
    for (int k = 0; k < 9; k++)
    {
        ims[k] = make_img(names[k], 2, N, NY, 1, _DATATYPE_FLOAT);
    }
    double rlim = (N >= 256) ? 100.0 : (double) N * 0.35;
    double cx   = (double) (N / 2);
    double cy   = (double) (NY / 2);
    for (uint32_t y = 0; y < NY; y++)
    {
        for (uint32_t x = 0; x < N; x++)
        {
            size_t idx           = (size_t) y * N + x;
            float  fx            = (float) x;
            float  fy            = (float) y;
            float  a             = 0.2f + 0.8f * (fx / (float) N) + 0.3f * (fy / (float) NY);
            ims[0]->array.F[idx] = a;
            ims[1]->array.F[idx] = 0.15f + 0.6f * (fy / 256.0f) + 0.25f * (fx / 256.0f);
            ims[2]->array.F[idx] = a * 0.5f;
            ims[3]->array.F[idx] = 0.1f + 0.4f * ((fx + fy) / 512.0f);
            ims[4]->array.F[idx] = 0.9f + 0.2f * (fx / 256.0f);
            ims[5]->array.F[idx] = 0.05f + 0.01f * (fy / 256.0f);
            ims[6]->array.F[idx] = 0.0f;
            ims[7]->array.F[idx] = 0.0f;
            double r             = hypot((double) x - cx, (double) y - cy);
            ims[8]->array.F[idx] = (r < rlim) ? 1.0f : 0.0f;
        }
    }
    set_lvar("H", (long) (N / 2) - 1);
    cli_calc_try_parse_udef("def soft_thresh(x, t) = where(abs(x) > t, x, 0.0)");
    mcalc_udef_try_parse("def soft_thresh(x, t) = where(abs(x) > t, x, 0.0)");
}

/* Populate scalar variables used by generated expressions and benchmarks. */
static void build_base_scalars(void)
{
    const char *vnames[] = { "v", "m", "s", "lo", "hi", "cx", "cy", "flux" };
    for (size_t i = 0; i < sizeof(vnames) / sizeof(vnames[0]); i++)
    {
        set_fvar(vnames[i], 0.0);
    }
    set_fvar("x", 1.5);
    set_fvar("y", -0.75);
    set_lvar("n", 8);
    set_lvar("k", 3);
    set_fvar("t", 0.25);
    set_fvar("alpha", 0.2);
}

/* Populate dataset @id per section 6.2. */
static void build_dataset(const char *id)
{
    if (id == NULL || id[0] == '\0' || strcmp(id, "empty") == 0)
    {
        return;
    }
    if (strcmp(id, "corpus") == 0)
    {
        build_base_images(64, 64);
        return;
    }
    build_base_scalars();
    if (strcmp(id, "bench256") == 0 || strcmp(id, "size_256") == 0)
    {
        build_base_images(256, 256);
        return;
    }
    if (strcmp(id, "bench2048") == 0 || strcmp(id, "size_2048") == 0)
    {
        build_base_images(2048, 2048);
        return;
    }
    if (strncmp(id, "size_", 5) == 0)
    {
        uint32_t sz = (uint32_t) strtoul(id + 5, NULL, 10);
        if (sz > 0)
        {
            build_base_images(sz, sz);
            return;
        }
    }
    if (strcmp(id, "nonsquare") == 0)
    {
        build_base_images(23, 11);
        return;
    }
    if (strcmp(id, "default") == 0)
    {
        build_base_images(64, 64);
        return;
    }
    if (strcmp(id, "random") == 0)
    {
        build_base_images(64, 64);
        const char *rnames[] = { "A", "B", "C", "D", "G", "O" };
        for (int k = 0; k < 6; k++)
        {
            IMAGE *im =
                &data.core.image[image_ID(rnames[k], data.core.image, data.core.NB_MAX_IMAGE)];
            for (uint64_t i = 0; i < im->md[0].nelement; i++)
            {
                uint32_t h     = arith_hash32((uint32_t) i ^ ((uint32_t) (k + 1) * 0x9e3779b9u));
                im->array.F[i] = (float) (h >> 8) * (1.0f / 16777216.0f);
            }
        }
        IMAGE *imM = &data.core.image[image_ID("M", data.core.image, data.core.NB_MAX_IMAGE)];
        for (uint64_t i = 0; i < imM->md[0].nelement; i++)
        {
            imM->array.F[i] = (arith_hash32((uint32_t) i ^ 0x12345678u) & 1u) ? 1.0f : 0.0f;
        }
        return;
    }
    if (strncmp(id, "offset_", 7) == 0)
    {
        double off = strtod(id + 7, NULL);
        build_base_images(256, 256);
        IMAGE *imA = &data.core.image[image_ID("A", data.core.image, data.core.NB_MAX_IMAGE)];
        IMAGE *imB = &data.core.image[image_ID("B", data.core.image, data.core.NB_MAX_IMAGE)];
        for (uint32_t y = 0; y < 256; y++)
        {
            for (uint32_t x = 0; x < 256; x++)
            {
                size_t idx        = (size_t) y * 256 + x;
                imA->array.F[idx] = (float) (off + sin(0.37 * (double) x) + cos(0.21 * (double) y));
                imB->array.F[idx] = (float) (off + cos(0.29 * (double) x) - sin(0.17 * (double) y));
            }
        }
        return;
    }
    if (strcmp(id, "nan_special") == 0)
    {
        build_dataset("random");
        IMAGE      *imA = &data.core.image[image_ID("A", data.core.image, data.core.NB_MAX_IMAGE)];
        IMAGE      *imB = &data.core.image[image_ID("B", data.core.image, data.core.NB_MAX_IMAGE)];
        const float specials[] = { NAN,      INFINITY,  -INFINITY, 0.0f,     -0.0f,
                                   1.4e-45f, -1.4e-45f, 5.8e-39f,  -5.8e-39f };
        int         nspec      = (int) (sizeof(specials) / sizeof(specials[0]));
        for (int i = 0; i < 64; i++)
        {
            uint32_t pa      = arith_hash32((uint32_t) i * 17u + 3u) % imA->md[0].nelement;
            uint32_t pb      = arith_hash32((uint32_t) i * 31u + 7u) % imB->md[0].nelement;
            imA->array.F[pa] = specials[i % nspec];
            imB->array.F[pb] = specials[(i + 3) % nspec];
        }
        return;
    }
    if (strcmp(id, "int_limits") == 0)
    {
        build_base_images(16, 16);
        IMAGE         *u8       = make_img("im_u8", 2, 16, 16, 1, _DATATYPE_UINT8);
        IMAGE         *i8       = make_img("im_i8", 2, 16, 16, 1, _DATATYPE_INT8);
        IMAGE         *u16      = make_img("im_u16", 2, 16, 16, 1, _DATATYPE_UINT16);
        IMAGE         *i16      = make_img("im_i16", 2, 16, 16, 1, _DATATYPE_INT16);
        IMAGE         *u32      = make_img("im_u32", 2, 16, 16, 1, _DATATYPE_UINT32);
        IMAGE         *i32      = make_img("im_i32", 2, 16, 16, 1, _DATATYPE_INT32);
        IMAGE         *u64      = make_img("im_u64", 2, 16, 16, 1, _DATATYPE_UINT64);
        IMAGE         *i64      = make_img("im_i64", 2, 16, 16, 1, _DATATYPE_INT64);
        const uint8_t  v_u8[8]  = { 0, 1, 2, 127, 128, 253, 254, UINT8_MAX };
        const int8_t   v_i8[8]  = { INT8_MIN, INT8_MIN + 1, -2, -1, 0, 1, INT8_MAX - 1, INT8_MAX };
        const uint16_t v_u16[8] = { 0, 1, 2, 32767, 32768, 65533, 65534, UINT16_MAX };
        const int16_t  v_i16[8] = {
            INT16_MIN, INT16_MIN + 1, -2, -1, 0, 1, INT16_MAX - 1, INT16_MAX
        };
        const uint32_t v_u32[8] = {
            0, 1, 2, 0x7fffffffU, 0x80000000U, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX
        };
        const int32_t v_i32[8] = {
            INT32_MIN, INT32_MIN + 1, -2, -1, 0, 1, INT32_MAX - 1, INT32_MAX
        };
        const uint64_t v_u64[8] = { 0,
                                    1,
                                    2,
                                    0x7fffffffffffffffULL,
                                    0x8000000000000000ULL,
                                    UINT64_MAX - 2,
                                    UINT64_MAX - 1,
                                    UINT64_MAX };
        const int64_t  v_i64[8] = {
            INT64_MIN, INT64_MIN + 1, -2, -1, 0, 1, INT64_MAX - 1, INT64_MAX
        };
        for (int idx = 0; idx < 256; idx++)
        {
            u8->array.UI8[idx]   = v_u8[idx & 7];
            i8->array.SI8[idx]   = v_i8[idx & 7];
            u16->array.UI16[idx] = v_u16[idx & 7];
            i16->array.SI16[idx] = v_i16[idx & 7];
            u32->array.UI32[idx] = v_u32[idx & 7];
            i32->array.SI32[idx] = v_i32[idx & 7];
            u64->array.UI64[idx] = v_u64[idx & 7];
            i64->array.SI64[idx] = v_i64[idx & 7];
        }
        set_lvar("imin64", LONG_MIN);
        set_lvar("imax64", LONG_MAX);
        return;
    }
    if (strcmp(id, "complex") == 0)
    {
        build_base_images(32, 32);
        IMAGE *zA  = make_img("zA", 2, 32, 32, 1, _DATATYPE_COMPLEX_FLOAT);
        IMAGE *zB  = make_img("zB", 2, 32, 32, 1, _DATATYPE_COMPLEX_FLOAT);
        IMAGE *zC  = make_img("zC", 2, 32, 32, 1, _DATATYPE_COMPLEX_FLOAT);
        IMAGE *zAd = make_img("zAd", 2, 32, 32, 1, _DATATYPE_COMPLEX_DOUBLE);
        IMAGE *zBd = make_img("zBd", 2, 32, 32, 1, _DATATYPE_COMPLEX_DOUBLE);
        IMAGE *zCd = make_img("zCd", 2, 32, 32, 1, _DATATYPE_COMPLEX_DOUBLE);
        IMAGE *Ad  = make_img("Ad", 2, 32, 32, 1, _DATATYPE_DOUBLE);
        IMAGE *Bd  = make_img("Bd", 2, 32, 32, 1, _DATATYPE_DOUBLE);
        IMAGE *Cd  = make_img("Cd", 2, 32, 32, 1, _DATATYPE_DOUBLE);
        for (int i = 0; i < 32 * 32; i++)
        {
            double re           = 0.2 + 0.7 * ((double) (i % 32) / 32.0);
            double im           = -0.3 + 0.6 * ((double) (i / 32) / 32.0);
            double re2          = 0.5 + 0.4 * ((double) (i / 32) / 32.0);
            double im2          = 0.1 - 0.5 * ((double) (i % 32) / 32.0);
            zA->array.CF[i].re  = (float) re;
            zA->array.CF[i].im  = (float) im;
            zB->array.CF[i].re  = (float) re2;
            zB->array.CF[i].im  = (float) im2;
            zC->array.CF[i].re  = 0.0f;
            zC->array.CF[i].im  = 0.0f;
            zAd->array.CD[i].re = re;
            zAd->array.CD[i].im = im;
            zBd->array.CD[i].re = re2;
            zBd->array.CD[i].im = im2;
            zCd->array.CD[i].re = 0.0;
            zCd->array.CD[i].im = 0.0;
            Ad->array.D[i]      = re;
            Bd->array.D[i]      = re2;
            Cd->array.D[i]      = 0.0;
        }
        return;
    }
    if (strcmp(id, "masks") == 0)
    {
        build_base_images(64, 64);
        IMAGE *m0  = make_img("M0", 2, 64, 64, 1, _DATATYPE_FLOAT);
        IMAGE *m1  = make_img("M1", 2, 64, 64, 1, _DATATYPE_FLOAT);
        IMAGE *mr  = make_img("M_rand", 2, 64, 64, 1, _DATATYPE_FLOAT);
        IMAGE *mu8 = make_img("Mu8", 2, 64, 64, 1, _DATATYPE_UINT8);
        for (int i = 0; i < 64 * 64; i++)
        {
            uint8_t bit       = (arith_hash32((uint32_t) i ^ 0xdeadbeefu) & 1u) ? 1 : 0;
            m0->array.F[i]    = 0.0f;
            m1->array.F[i]    = 1.0f;
            mr->array.F[i]    = (float) bit;
            mu8->array.UI8[i] = bit;
        }
        return;
    }
    if (strcmp(id, "cubes") == 0)
    {
        build_base_images(16, 12);
        IMAGE *cA = make_img("cube", 3, 16, 12, 4, _DATATYPE_FLOAT);
        IMAGE *cB = make_img("cubeB", 3, 16, 12, 4, _DATATYPE_FLOAT);
        IMAGE *cC = make_img("cubeC", 3, 16, 12, 4, _DATATYPE_FLOAT);
        for (uint64_t i = 0; i < cA->md[0].nelement; i++)
        {
            uint32_t h1    = arith_hash32((uint32_t) i ^ 0x11111111u);
            uint32_t h2    = arith_hash32((uint32_t) i ^ 0x22222222u);
            cA->array.F[i] = 0.1f + 0.8f * (float) (h1 >> 8) * (1.0f / 16777216.0f);
            cB->array.F[i] = 0.2f + 0.7f * (float) (h2 >> 8) * (1.0f / 16777216.0f);
            cC->array.F[i] = 0.0f;
        }
        return;
    }
    if (strcmp(id, "broadcast") == 0)
    {
        build_base_images(32, 16);
        IMAGE *row   = make_img("row", 2, 32, 1, 1, _DATATYPE_FLOAT);
        IMAGE *col   = make_img("col", 2, 1, 16, 1, _DATATYPE_FLOAT);
        IMAGE *sc1   = make_img("sc1", 2, 1, 1, 1, _DATATYPE_FLOAT);
        IMAGE *cube  = make_img("cube", 3, 32, 16, 4, _DATATYPE_FLOAT);
        IMAGE *cubeC = make_img("cubeC", 3, 32, 16, 4, _DATATYPE_FLOAT);
        for (uint32_t x = 0; x < 32; x++)
        {
            row->array.F[x] = 0.1f + 0.02f * (float) x;
        }
        for (uint32_t y = 0; y < 16; y++)
        {
            col->array.F[y] = 0.2f + 0.03f * (float) y;
        }
        sc1->array.F[0] = 1.25f;
        for (uint64_t i = 0; i < cube->md[0].nelement; i++)
        {
            uint32_t h        = arith_hash32((uint32_t) i ^ 0x33333333u);
            cube->array.F[i]  = 0.1f + 0.8f * (float) (h >> 8) * (1.0f / 16777216.0f);
            cubeC->array.F[i] = 0.0f;
        }
        return;
    }
    build_base_images(64, 64);
}

/* Skip ASCII whitespace in @p. */
static const char *skip_ws(const char *p)
{
    while (*p != '\0' && isspace((unsigned char) *p))
    {
        p++;
    }
    return p;
}

/* Evaluate @input on engine @which (0 = old, 1 = new), splitting top-level `;` unless `if`/`for`. */
static int eval_stmts_on(int         which,
                         const char *input,
                         int         is_test,
                         int        *out_rc,
                         char       *out,
                         size_t      outsz)
{
    const char *lp = skip_ws(input);
    if ((strncmp(lp, "if", 2) == 0 && (isspace((unsigned char) lp[2]) || lp[2] == '(')) ||
        (strncmp(lp, "for", 3) == 0 && isspace((unsigned char) lp[3])) || strchr(lp, ';') == NULL)
    {
        return (which == 0) ? cli_calc_eval_str(input, is_test, out_rc, out, outsz)
                            : mcalc_eval_str(input, is_test, out_rc, out, outsz);
    }

    char  *copy  = strdup(input);
    char  *start = copy;
    int    depth = 0;
    char   quote = '\0';
    size_t opos  = 0;
    int    ok    = 1;
    if (out != NULL && outsz > 0)
    {
        out[0] = '\0';
    }
    *out_rc = 0;

    for (char *p = copy;; p++)
    {
        char c = *p;
        if (quote != '\0')
        {
            if (c == '\\' && p[1] != '\0')
            {
                p++;
            }
            else if (c == quote)
            {
                quote = '\0';
            }
        }
        else if (c == '"' || c == '\'')
        {
            quote = c;
        }
        else if (c == '(' || c == '[' || c == '{')
        {
            depth++;
        }
        else if ((c == ')' || c == ']' || c == '}') && depth > 0)
        {
            depth--;
        }
        else if ((c == ';' || c == '\0') && depth == 0)
        {
            *p              = '\0';
            const char *seg = skip_ws(start);
            if (*seg != '\0')
            {
                char seg_out[4096] = "";
                int  seg_rc        = 0;
                int  seg_ok =
                    (which == 0)
                        ? cli_calc_eval_str(seg, is_test, &seg_rc, seg_out, sizeof(seg_out))
                        : mcalc_eval_str(seg, is_test, &seg_rc, seg_out, sizeof(seg_out));
                if (!seg_ok || seg_rc != 0)
                {
                    ok      = seg_ok;
                    *out_rc = seg_rc;
                    break;
                }
                if (seg_out[0] != '\0' && out != NULL && opos + 2 < outsz)
                {
                    opos += (size_t) snprintf(out + opos, outsz - opos, "%s%s",
                                              (opos > 0) ? "\n" : "", seg_out);
                }
            }
            if (c == '\0')
            {
                break;
            }
            start = p + 1;
        }
        if (c == '\0')
        {
            break;
        }
    }
    free(copy);
    return ok;
}

/* Execute one setup string (`@cmd:mk2Dim/mk3Dim/rm ...` or a calc statement). */
static void run_setup_stmt(const char *s)
{
    int saved[2];
    silence_begin(saved);
    if (strncmp(s, "@cmd:", 5) == 0)
    {
        const char *cmd = s + 5;
        char        nm[STRINGMAXLEN_IMAGE_NAME];
        unsigned    nx = 0;
        unsigned    ny = 0;
        unsigned    nz = 0;
        if (sscanf(cmd, "mk2Dim %63s %u %u", nm, &nx, &ny) == 3)
        {
            delete_image_ID(nm, DELETE_IMAGE_ERRMODE_IGNORE);
            make_img(nm, 2, (uint32_t) nx, (uint32_t) ny, 1, _DATATYPE_FLOAT);
            milk_table_bump();
        }
        else if (sscanf(cmd, "mk3Dim %63s %u %u %u", nm, &nx, &ny, &nz) == 4)
        {
            delete_image_ID(nm, DELETE_IMAGE_ERRMODE_IGNORE);
            make_img(nm, 3, (uint32_t) nx, (uint32_t) ny, (uint32_t) nz, _DATATYPE_FLOAT);
            milk_table_bump();
        }
        else if (sscanf(cmd, "rm %63s", nm) == 1)
        {
            delete_image_ID(nm, DELETE_IMAGE_ERRMODE_IGNORE);
            milk_table_bump();
        }
    }
    else
    {
        char out[1024];
        int  rc = 0;
        eval_stmts_on(0, s, 0, &rc, out, sizeof(out));
    }
    silence_end(saved);
}

/* Run @stmt on engine @which (0 = old, 1 = new) and capture its result in @rec. */
static void run_one_engine(int which, const char *stmt, int is_test, eval_rec_t *rec)
{
    memset(rec, 0, sizeof(*rec));
    clear_tmp_pool();
    int saved[2];
    silence_begin(saved);
    if (which == 0)
    {
        rec->ok = eval_stmts_on(0, stmt, is_test, &rec->rc, rec->out, sizeof(rec->out));
    }
    else
    {
        mcalc_lasterr[0] = '\0';
        rec->ok          = eval_stmts_on(1, stmt, is_test, &rec->rc, rec->out, sizeof(rec->out));
        if (!rec->ok && strcmp(mcalc_lasterr, "not implemented in the new engine") == 0)
        {
            rec->not_impl = 1;
        }
    }
    silence_end(saved);
    snap_capture(&rec->post);
}

/* Return 1 if the workspace contains any float32 or complex-float image. */
static int has_float32_images(const state_snap_t *s1, const state_snap_t *s2)
{
    for (int i = 0; i < s1->nimgs; i++)
    {
        uint8_t dt = s1->imgs[i].datatype;
        if (dt == _DATATYPE_FLOAT || dt == _DATATYPE_COMPLEX_FLOAT)
        {
            return 1;
        }
    }
    for (int i = 0; i < s2->nimgs; i++)
    {
        uint8_t dt = s2->imgs[i].datatype;
        if (dt == _DATATYPE_FLOAT || dt == _DATATYPE_COMPLEX_FLOAT)
        {
            return 1;
        }
    }
    return 0;
}

/* Compare two double scalar values under section 6.1 tolerances. */
static int doubles_match(double a, double b, double rel_tol)
{
    int na = isnan(a) != 0;
    int nb = isnan(b) != 0;
    if (na || nb)
    {
        return na && nb;
    }
    int ia = isinf(a) != 0;
    int ib = isinf(b) != 0;
    if (ia || ib)
    {
        return ia && ib && ((a > 0.0) == (b > 0.0));
    }
    double diff  = fabs(a - b);
    double scale = fmax(fabs(a), fabs(b));
    if (diff == 0.0)
    {
        return 1;
    }
    return diff <= rel_tol * fmax(scale, 1.0);
}

/* Compare printed output strings @o1 and @o2 (allowing numeric rounding at cli_float_digits). */
static int outputs_match(const char *o1, const char *o2, double rel_tol)
{
    if (strcmp(o1, o2) == 0)
    {
        return 1;
    }
    const char *p1 = o1;
    const char *p2 = o2;
    while (*p1 != '\0' && *p2 != '\0')
    {
        const char *nl1 = strchr(p1, '\n');
        const char *nl2 = strchr(p2, '\n');
        size_t      l1  = (nl1 != NULL) ? (size_t) (nl1 - p1) : strlen(p1);
        size_t      l2  = (nl2 != NULL) ? (size_t) (nl2 - p2) : strlen(p2);
        if (l1 != l2 || strncmp(p1, p2, l1) != 0)
        {
            char  b1[256];
            char  b2[256];
            char *e1 = NULL;
            char *e2 = NULL;
            if (l1 >= sizeof(b1) || l2 >= sizeof(b2))
            {
                return 0;
            }
            memcpy(b1, p1, l1);
            b1[l1] = '\0';
            memcpy(b2, p2, l2);
            b2[l2]     = '\0';
            double v1  = strtod(b1, &e1);
            double v2  = strtod(b2, &e2);
            double tol = fmax(rel_tol, 2e-6);
            if (e1 == b1 || *e1 != '\0' || e2 == b2 || *e2 != '\0' || !doubles_match(v1, v2, tol))
            {
                return 0;
            }
        }
        p1 = (nl1 != NULL) ? (nl1 + 1) : (p1 + l1);
        p2 = (nl2 != NULL) ? (nl2 + 1) : (p2 + l2);
    }
    return *p1 == '\0' && *p2 == '\0';
}

/* Find variable @name in @snap, or NULL if absent. */
static const var_snap_t *find_var(const state_snap_t *snap, const char *name)
{
    for (int i = 0; i < snap->nvars; i++)
    {
        if (strcmp(snap->vars[i].name, name) == 0)
        {
            return &snap->vars[i];
        }
    }
    return NULL;
}

/* Compare two image snapshots @a (reference) and @b (new) per section 6.1; returns 0 on match. */
static int compare_img_data(const img_snap_t *a, const img_snap_t *b, char *detail, size_t detailsz)
{
    if (a->naxis != b->naxis || a->datatype != b->datatype || a->size[0] != b->size[0] ||
        a->size[1] != b->size[1] || a->size[2] != b->size[2])
    {
        snprintf(detail, detailsz,
                 "image %s metadata mismatch: old=(%ux%ux%u dt=%u) new=(%ux%ux%u dt=%u)", a->name,
                 a->size[0], a->size[1], a->size[2], a->datatype, b->size[0], b->size[1],
                 b->size[2], b->datatype);
        return -1;
    }
    uint8_t dt = a->datatype;
    if (dt == _DATATYPE_FLOAT || dt == _DATATYPE_COMPLEX_FLOAT)
    {
        uint64_t     n   = a->nelement * ((dt == _DATATYPE_COMPLEX_FLOAT) ? 2ULL : 1ULL);
        const float *pa  = (const float *) a->data;
        const float *pb  = (const float *) b->data;
        double       ref = 0.0;
        for (uint64_t i = 0; i < n; i++)
        {
            if (isfinite(pa[i]) && fabs((double) pa[i]) > ref)
            {
                ref = fabs((double) pa[i]);
            }
        }
        double tol = 1e-6 * fmax(ref, 1.0);
        for (uint64_t i = 0; i < n; i++)
        {
            int na = isnan(pa[i]) != 0;
            int nb = isnan(pb[i]) != 0;
            if (na != nb)
            {
                snprintf(detail, detailsz, "image %s pixel %llu NaN mismatch: old=%g new=%g",
                         a->name, (unsigned long long) i, (double) pa[i], (double) pb[i]);
                return -1;
            }
            if (na)
            {
                continue;
            }
            int ia = isinf(pa[i]) != 0;
            int ib = isinf(pb[i]) != 0;
            if (ia != ib || (ia && ((pa[i] > 0.0f) != (pb[i] > 0.0f))))
            {
                snprintf(detail, detailsz, "image %s pixel %llu inf mismatch: old=%g new=%g",
                         a->name, (unsigned long long) i, (double) pa[i], (double) pb[i]);
                return -1;
            }
            if (!ia && fabs((double) pa[i] - (double) pb[i]) > tol)
            {
                snprintf(detail, detailsz,
                         "image %s pixel %llu float32 diff=%g > tol=%g (old=%.8g new=%.8g)",
                         a->name, (unsigned long long) i, fabs((double) pa[i] - (double) pb[i]),
                         tol, (double) pa[i], (double) pb[i]);
                return -1;
            }
        }
        return 0;
    }
    if (dt == _DATATYPE_DOUBLE || dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        uint64_t      n   = a->nelement * ((dt == _DATATYPE_COMPLEX_DOUBLE) ? 2ULL : 1ULL);
        const double *pa  = (const double *) a->data;
        const double *pb  = (const double *) b->data;
        double        ref = 0.0;
        for (uint64_t i = 0; i < n; i++)
        {
            if (isfinite(pa[i]) && fabs(pa[i]) > ref)
            {
                ref = fabs(pa[i]);
            }
        }
        double tol = 1e-12 * fmax(ref, 1.0);
        for (uint64_t i = 0; i < n; i++)
        {
            int na = isnan(pa[i]) != 0;
            int nb = isnan(pb[i]) != 0;
            if (na != nb)
            {
                snprintf(detail, detailsz, "image %s pixel %llu NaN mismatch: old=%g new=%g",
                         a->name, (unsigned long long) i, pa[i], pb[i]);
                return -1;
            }
            if (na)
            {
                continue;
            }
            int ia = isinf(pa[i]) != 0;
            int ib = isinf(pb[i]) != 0;
            if (ia != ib || (ia && ((pa[i] > 0.0) != (pb[i] > 0.0))))
            {
                snprintf(detail, detailsz, "image %s pixel %llu inf mismatch: old=%g new=%g",
                         a->name, (unsigned long long) i, pa[i], pb[i]);
                return -1;
            }
            if (!ia && fabs(pa[i] - pb[i]) > tol)
            {
                snprintf(detail, detailsz,
                         "image %s pixel %llu double diff=%g > tol=%g (old=%.17g new=%.17g)",
                         a->name, (unsigned long long) i, fabs(pa[i] - pb[i]), tol, pa[i], pb[i]);
                return -1;
            }
        }
        return 0;
    }
    if (a->nbytes > 0 && memcmp(a->data, b->data, a->nbytes) != 0)
    {
        snprintf(detail, detailsz, "integer image %s bytes differ", a->name);
        return -1;
    }
    return 0;
}

/* Compare two evaluation records @r1 and @r2 per section 6.1; returns 0 on match. */
static int compare_records(const eval_rec_t *r1,
                           const eval_rec_t *r2,
                           int               is_test,
                           int               strict_mode,
                           char             *reason,
                           size_t            reasonsz,
                           char             *detail,
                           size_t            detailsz)
{
    if (!strict_mode && r2->not_impl)
    {
        return 0;
    }
    if (is_test)
    {
        int st1 = (!r1->ok || r1->rc == 2) ? 2 : r1->rc;
        int st2 = (!r2->ok || r2->rc == 2) ? 2 : r2->rc;
        if (st1 != st2)
        {
            snprintf(reason, reasonsz, "status_mismatch");
            snprintf(detail, detailsz, "test status old=%d new=%d", st1, st2);
            return -1;
        }
        if (st1 == 2)
        {
            return 0;
        }
    }
    else
    {
        int fail1 = (!r1->ok || r1->rc != 0);
        int fail2 = (!r2->ok || r2->rc != 0);
        if (fail1 != fail2)
        {
            snprintf(reason, reasonsz, "status_mismatch");
            snprintf(detail, detailsz, "eval status old=(ok=%d rc=%d) new=(ok=%d rc=%d err='%s')",
                     r1->ok, r1->rc, r2->ok, r2->rc, mcalc_lasterr);
            return -1;
        }
        if (fail1 && fail2)
        {
            return 0;
        }
    }

    double stol = has_float32_images(&r1->post, &r2->post) ? 1e-6 : 1e-12;
    if (!outputs_match(r1->out, r2->out, stol))
    {
        snprintf(reason, reasonsz, "output_mismatch");
        snprintf(detail, detailsz, "old_out='%s' new_out='%s'", r1->out, r2->out);
        return -1;
    }

    for (int i = 0; i < r1->post.nvars; i++)
    {
        const var_snap_t *v1 = &r1->post.vars[i];
        if (find_img(&r1->post, v1->name) != NULL)
        {
            continue;
        }
        const var_snap_t *v2 = find_var(&r2->post, v1->name);
        if (v2 == NULL)
        {
            snprintf(reason, reasonsz, "var_missing_in_new");
            snprintf(detail, detailsz, "variable %s missing in new", v1->name);
            return -1;
        }
        if (v1->type != v2->type)
        {
            snprintf(reason, reasonsz, "var_type_mismatch");
            snprintf(detail, detailsz, "variable %s type old=%d new=%d", v1->name, v1->type,
                     v2->type);
            return -1;
        }
        if (v1->type == 1 && v1->l != v2->l)
        {
            snprintf(reason, reasonsz, "var_int_mismatch");
            snprintf(detail, detailsz, "variable %s long old=%ld new=%ld", v1->name, v1->l, v2->l);
            return -1;
        }
        if (v1->type == 2 && strcmp(v1->s, v2->s) != 0)
        {
            snprintf(reason, reasonsz, "var_str_mismatch");
            snprintf(detail, detailsz, "variable %s str old='%s' new='%s'", v1->name, v1->s, v2->s);
            return -1;
        }
        if (v1->type == 0 && !doubles_match(v1->f, v2->f, stol))
        {
            snprintf(reason, reasonsz, "var_double_mismatch");
            snprintf(detail, detailsz, "variable %s double old=%.17g new=%.17g (tol=%g)", v1->name,
                     v1->f, v2->f, stol);
            return -1;
        }
    }
    for (int i = 0; i < r2->post.nvars; i++)
    {
        const var_snap_t *v2 = &r2->post.vars[i];
        if (find_img(&r2->post, v2->name) != NULL)
        {
            continue;
        }
        if (find_var(&r1->post, v2->name) == NULL)
        {
            snprintf(reason, reasonsz, "var_extra_in_new");
            snprintf(detail, detailsz, "extra variable %s in new", v2->name);
            return -1;
        }
    }

    for (int i = 0; i < r1->post.nimgs; i++)
    {
        const img_snap_t *im1 = &r1->post.imgs[i];
        const img_snap_t *im2 = find_img(&r2->post, im1->name);
        if (im2 == NULL)
        {
            snprintf(reason, reasonsz, "image_missing_in_new");
            snprintf(detail, detailsz, "image %s missing in new", im1->name);
            return -1;
        }
        if (compare_img_data(im1, im2, detail, detailsz) != 0)
        {
            snprintf(reason, reasonsz, "image_mismatch");
            return -1;
        }
    }
    for (int i = 0; i < r2->post.nimgs; i++)
    {
        const img_snap_t *im2 = &r2->post.imgs[i];
        if (find_img(&r1->post, im2->name) == NULL)
        {
            snprintf(reason, reasonsz, "image_extra_in_new");
            snprintf(detail, detailsz, "extra image %s in new", im2->name);
            return -1;
        }
    }
    return 0;
}

/* Print a JSON-escaped string (with surrounding quotes) to @fp. */
static void print_json_str(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (; s != NULL && *s != '\0'; s++)
    {
        unsigned char c = (unsigned char) *s;
        if (c == '"' || c == '\\')
        {
            fputc('\\', fp);
            fputc((int) c, fp);
        }
        else if (c == '\n')
        {
            fputs("\\n", fp);
        }
        else if (c == '\r')
        {
            fputs("\\r", fp);
        }
        else if (c == '\t')
        {
            fputs("\\t", fp);
        }
        else if (c < 0x20)
        {
            fprintf(fp, "\\u%04x", (unsigned int) c);
        }
        else
        {
            fputc((int) c, fp);
        }
    }
    fputc('"', fp);
}

/* Parse a JSON string literal starting at `"` into a newly allocated C string. */
static const char *parse_json_str(const char *p, char **out)
{
    p = skip_ws(p);
    if (*p != '"')
    {
        return NULL;
    }
    p++;
    size_t cap = 64;
    size_t len = 0;
    char  *buf = malloc(cap);
    while (*p != '\0' && *p != '"')
    {
        char c = *p++;
        if (c == '\\' && *p != '\0')
        {
            char esc = *p++;
            switch (esc)
            {
            case '"':
            case '\\':
            case '/':
                c = esc;
                break;
            case 'n':
                c = '\n';
                break;
            case 'r':
                c = '\r';
                break;
            case 't':
                c = '\t';
                break;
            case 'b':
                c = '\b';
                break;
            case 'f':
                c = '\f';
                break;
            case 'u':
                for (int k = 0; k < 4 && *p != '\0'; k++)
                {
                    p++;
                }
                c = '?';
                break;
            default:
                c = esc;
                break;
            }
        }
        if (len + 2 >= cap)
        {
            cap *= 2;
            buf = realloc(buf, cap);
        }
        buf[len++] = c;
    }
    if (*p != '"')
    {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    *out     = buf;
    return p + 1;
}

/* Skip any JSON value starting at @p. */
static const char *skip_json_val(const char *p)
{
    p = skip_ws(p);
    if (*p == '"')
    {
        char *tmp = NULL;
        p         = parse_json_str(p, &tmp);
        free(tmp);
        return p;
    }
    if (*p == '[' || *p == '{')
    {
        char open  = *p;
        char close = (open == '[') ? ']' : '}';
        int  depth = 1;
        p++;
        while (*p != '\0' && depth > 0)
        {
            if (*p == '"')
            {
                char *tmp = NULL;
                p         = parse_json_str(p, &tmp);
                free(tmp);
                if (p == NULL)
                {
                    return NULL;
                }
                continue;
            }
            if (*p == open)
            {
                depth++;
            }
            else if (*p == close)
            {
                depth--;
            }
            p++;
        }
        return p;
    }
    while (*p != '\0' && *p != ',' && *p != '}' && *p != ']' && !isspace((unsigned char) *p))
    {
        p++;
    }
    return p;
}

/* Parse one JSON input line into dataset, setup array, stmt, and is_test. */
static int parse_case_json(const char *line,
                           char      **dataset_out,
                           char      **setup_out,
                           int        *nsetup_out,
                           char      **stmt_out,
                           int        *is_test_out)
{
    *dataset_out = NULL;
    *nsetup_out  = 0;
    *stmt_out    = NULL;
    *is_test_out = 0;

    const char *p = skip_ws(line);
    if (*p != '{')
    {
        return -1;
    }
    p++;
    while (*p != '\0')
    {
        p = skip_ws(p);
        if (*p == '}')
        {
            break;
        }
        char *key = NULL;
        p         = parse_json_str(p, &key);
        if (p == NULL)
        {
            return -1;
        }
        p = skip_ws(p);
        if (*p != ':')
        {
            free(key);
            return -1;
        }
        p = skip_ws(p + 1);
        if (strcmp(key, "dataset") == 0)
        {
            free(*dataset_out);
            p = parse_json_str(p, dataset_out);
        }
        else if (strcmp(key, "stmt") == 0)
        {
            free(*stmt_out);
            p = parse_json_str(p, stmt_out);
        }
        else if (strcmp(key, "is_test") == 0)
        {
            if (strncmp(p, "true", 4) == 0 || *p == '1')
            {
                *is_test_out = 1;
            }
            p = skip_json_val(p);
        }
        else if (strcmp(key, "setup") == 0 && *p == '[')
        {
            p++;
            while (*p != '\0')
            {
                p = skip_ws(p);
                if (*p == ']')
                {
                    p++;
                    break;
                }
                char *item = NULL;
                p          = parse_json_str(p, &item);
                if (p == NULL)
                {
                    free(key);
                    return -1;
                }
                if (*nsetup_out < CALCDIFF_MAX_SETUP)
                {
                    setup_out[(*nsetup_out)++] = item;
                }
                else
                {
                    free(item);
                }
                p = skip_ws(p);
                if (*p == ',')
                {
                    p++;
                }
            }
        }
        else
        {
            p = skip_json_val(p);
        }
        free(key);
        if (p == NULL)
        {
            return -1;
        }
        p = skip_ws(p);
        if (*p == ',')
        {
            p++;
        }
    }
    return (*stmt_out != NULL) ? 0 : -1;
}

/* Time one benchmark statement or program on the old engine and write best of 7 runs (us). */
static int bench_one_old(int iters, const char *stmts, double *out_us)
{
    int    warmup   = (iters / 10 >= 5) ? (iters / 10) : 5;
    double best_ns  = 1e30;
    int    has_semi = (strchr(stmts, ';') != NULL);

    if (!has_semi)
    {
        char out[1024];
        int  rc = 0;
        if (!cli_calc_eval_str(stmts, 0, &rc, out, sizeof(out)) || rc != 0)
        {
            fprintf(stderr, "milk-calcdiff: bench eval failed for '%s'\n", stmts);
            return 1;
        }
        void *kept = cli_calc_keep_statement(stmts);
        if (kept != NULL)
        {
            for (int i = 0; i < warmup; i++)
            {
                if (cli_calc_fused_run(kept) != 0)
                {
                    cli_calc_kept_free(kept);
                    return 1;
                }
            }
            for (int r = 0; r < 7; r++)
            {
                uint64_t t0 = now_ns();
                for (int i = 0; i < iters; i++)
                {
                    if (cli_calc_fused_run(kept) != 0)
                    {
                        cli_calc_kept_free(kept);
                        return 1;
                    }
                }
                double ns = (double) (now_ns() - t0) / (double) iters;
                if (ns < best_ns)
                {
                    best_ns = ns;
                }
            }
            cli_calc_kept_free(kept);
            *out_us = best_ns / 1000.0;
            return 0;
        }
    }

    char        prog_buf[1024];
    const char *prog_src = stmts;
    const char *semi_c   = strstr(stmts, "; C =");
    if (semi_c != NULL)
    {
        size_t pre = (size_t) (semi_c - stmts);
        snprintf(prog_buf, sizeof(prog_buf), "%.*s; local C =%s", (int) pre, stmts, semi_c + 5);
        prog_src = prog_buf;
    }
    char           err[512] = "";
    cli_calc_prog *p        = cli_calc_prog_new(prog_src, NULL, 0, err, sizeof(err));
    if (p == NULL)
    {
        fprintf(stderr, "milk-calcdiff: cli_calc_prog_new failed: %s\n", err);
        return 1;
    }
    for (int i = 0; i < warmup; i++)
    {
        if (cli_calc_prog_run(p, err, sizeof(err)) != 0)
        {
            cli_calc_prog_free(p);
            return 1;
        }
    }
    for (int r = 0; r < 7; r++)
    {
        uint64_t t0 = now_ns();
        for (int i = 0; i < iters; i++)
        {
            if (cli_calc_prog_run(p, err, sizeof(err)) != 0)
            {
                cli_calc_prog_free(p);
                return 1;
            }
        }
        double ns = (double) (now_ns() - t0) / (double) iters;
        if (ns < best_ns)
        {
            best_ns = ns;
        }
    }
    cli_calc_prog_free(p);
    *out_us = best_ns / 1000.0;
    return 0;
}

/* Run `--bench --size N --iters K --engine old|new ["<stmts>"]` and print best of 7 runs (us). */
static int run_bench_mode(uint32_t size, int iters, const char *engine, const char *stmts)
{
    if (strcmp(engine, "new") == 0)
    {
        fprintf(stdout, "not available\n");
        return 2;
    }
    if (strcmp(engine, "old") != 0)
    {
        fprintf(stderr, "milk-calcdiff: unknown engine '%s'\n", engine);
        return 1;
    }

    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("MILK_CALC_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MILK_CALC_SPIN_US", "0", 1);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(8, &cpuset);
    (void) sched_setaffinity(0, sizeof(cpuset), &cpuset);

    if (init_engine() != 0)
    {
        return 1;
    }
    workspace_clear();
    build_base_images(size, size);
    build_base_scalars();

    if (stmts != NULL)
    {
        double us = 0.0;
        if (bench_one_old(iters, stmts, &us) != 0)
        {
            return 1;
        }
        printf("%.3f\n", us);
        return 0;
    }

    char   *line = NULL;
    size_t  cap  = 0;
    ssize_t nread;
    while ((nread = getline(&line, &cap, stdin)) != -1)
    {
        while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r'))
        {
            line[--nread] = '\0';
        }
        if (nread == 0)
        {
            continue;
        }
        int         k_iters = iters;
        const char *expr    = line;
        char       *tab     = strchr(line, '\t');
        if (tab != NULL)
        {
            *tab    = '\0';
            k_iters = (int) strtol(line, NULL, 10);
            expr    = tab + 1;
        }
        double us = 0.0;
        if (bench_one_old(k_iters, expr, &us) != 0)
        {
            free(line);
            return 1;
        }
        printf("%.3f\n", us);
        fflush(stdout);
    }
    free(line);
    return 0;
}

/* Map an mc_func_t entry to its gen_expr.py kind string. */
static const char *fn_kind_str(const mc_func_t *fn)
{
    if (fn->cast_dt != 0)
    {
        return "cast";
    }
    switch (fn->kind)
    {
    case MC_FN_SCALAR:
        return "elem";
    case MC_FN_WHERE:
        return "where";
    case MC_FN_COORD:
        return "coord";
    case MC_FN_PRNG:
        return "rng";
    case MC_FN_SHIFT:
        return "shift";
    case MC_FN_REDUCE_D:
    case MC_FN_REDUCE_L:
    case MC_FN_DOT:
    case MC_FN_EAGER_RED:
        return "reduce";
    case MC_FN_ISTATS:
    case MC_FN_ICENTROID:
        return "multi_red";
    case MC_FN_CPLX:
        return "complex";
    case MC_FN_STR:
        return "str";
    default:
        return "other";
    }
}

/* Dump the function table as JSON lines for gen_expr.py (`--dump-functions`). */
static int dump_functions(void)
{
    int              nfn = 0;
    const mc_func_t *tbl = mc_func_table(&nfn);
    for (int i = 0; i < nfn; i++)
    {
        printf("{\"name\":\"%s\",\"min_args\":%d,\"max_args\":%d,\"kind\":\"%s\",\"in_old\":%d}\n",
               tbl[i].name, tbl[i].min_args, tbl[i].max_args, fn_kind_str(&tbl[i]),
               (int) tbl[i].in_old);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int         bench_mode  = 0;
    int         old_vs_old  = 0;
    int         strict_mode = 0;
    int         test_local  = 0;
    uint32_t    bench_size  = 256;
    int         bench_iters = 2000;
    const char *bench_eng   = "old";
    const char *bench_stmts = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--dump-functions") == 0)
        {
            return dump_functions();
        }
        else if (strcmp(argv[i], "--old-vs-old") == 0)
        {
            old_vs_old = 1;
        }
        else if (strcmp(argv[i], "--strict") == 0)
        {
            strict_mode = 1;
        }
        else if (strcmp(argv[i], "--test-local") == 0)
        {
            test_local = 1;
        }
        else if (strcmp(argv[i], "--bench") == 0)
        {
            bench_mode = 1;
        }
        else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc)
        {
            bench_size = (uint32_t) strtoul(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
        {
            bench_iters = (int) strtol(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--engine") == 0 && i + 1 < argc)
        {
            bench_eng = argv[++i];
        }
        else if (bench_mode && bench_stmts == NULL)
        {
            bench_stmts = argv[i];
        }
    }

    if (bench_mode)
    {
        if (bench_stmts == NULL && isatty(STDIN_FILENO))
        {
            fprintf(stderr,
                    "Usage: milk-calcdiff --bench --size N --iters K --engine old|new <stmts>\n");
            return 1;
        }
        return run_bench_mode(bench_size, bench_iters, bench_eng, bench_stmts);
    }

    if (init_engine() != 0)
    {
        return 1;
    }

    char         cached_ds_name[64] = "";
    state_snap_t cached_ds_snap;
    memset(&cached_ds_snap, 0, sizeof(cached_ds_snap));

    char   *line = NULL;
    size_t  cap  = 0;
    ssize_t nread;
    while ((nread = getline(&line, &cap, stdin)) != -1)
    {
        if (nread <= 1)
        {
            continue;
        }
        char *dataset                   = NULL;
        char *setup[CALCDIFF_MAX_SETUP] = { NULL };
        int   nsetup                    = 0;
        char *stmt                      = NULL;
        int   is_test                   = 0;

        if (parse_case_json(line, &dataset, setup, &nsetup, &stmt, &is_test) != 0)
        {
            free(dataset);
            for (int k = 0; k < nsetup; k++)
            {
                free(setup[k]);
            }
            free(stmt);
            continue;
        }

        const char *ds_name = (dataset != NULL && dataset[0] != '\0') ? dataset : "default";
        if (test_local)
        {
            if (nsetup == 0)
            {
                if (strcmp(ds_name, cached_ds_name) != 0)
                {
                    snap_free(&cached_ds_snap);
                    workspace_clear();
                    build_dataset(ds_name);
                    snap_capture(&cached_ds_snap);
                    snprintf(cached_ds_name, sizeof(cached_ds_name), "%s", ds_name);
                }
            }
            else
            {
                cached_ds_name[0] = '\0';
                workspace_clear();
                build_dataset(ds_name);
                for (int k = 0; k < nsetup; k++)
                {
                    run_setup_stmt(setup[k]);
                }
            }

            char out1[4096] = "";
            char out2[4096] = "";
            int  tr1        = -1;
            int  tr2        = -1;
            int  rc1        = cli_calc_local(stmt, 6, out1, sizeof(out1), &tr1);
            int  rc2        = mcalc_local(stmt, 6, out2, sizeof(out2), &tr2);

            char reason[64]  = "";
            char detail[512] = "";
            int  diff        = 0;
            if (rc1 != rc2)
            {
                diff = 1;
                snprintf(reason, sizeof(reason), "local_decision_mismatch");
                snprintf(detail, sizeof(detail), "old_rc=%d new_rc=%d", rc1, rc2);
            }
            else if (rc1 == 1)
            {
                if (tr1 != tr2)
                {
                    diff = 1;
                    snprintf(reason, sizeof(reason), "local_truth_mismatch");
                    snprintf(detail, sizeof(detail), "old_truth=%d new_truth=%d", tr1, tr2);
                }
                else if (!outputs_match(out1, out2, 1e-12))
                {
                    diff = 1;
                    snprintf(reason, sizeof(reason), "local_output_mismatch");
                    snprintf(detail, sizeof(detail), "old_out='%s' new_out='%s'", out1, out2);
                }
            }
            if (diff)
            {
                fputs("{\"dataset\":", stdout);
                print_json_str(stdout, ds_name);
                fputs(",\"stmt\":", stdout);
                print_json_str(stdout, stmt);
                fputs(",\"reason\":", stdout);
                print_json_str(stdout, reason);
                fputs(",\"detail\":", stdout);
                print_json_str(stdout, detail);
                fputs("}\n", stdout);
                fflush(stdout);
            }
            free(dataset);
            for (int k = 0; k < nsetup; k++)
            {
                free(setup[k]);
            }
            free(stmt);
            continue;
        }

        state_snap_t        snap;
        const state_snap_t *base_snap = NULL;
        memset(&snap, 0, sizeof(snap));

        if (nsetup == 0)
        {
            if (strcmp(ds_name, cached_ds_name) != 0)
            {
                snap_free(&cached_ds_snap);
                workspace_clear();
                build_dataset(ds_name);
                snap_capture(&cached_ds_snap);
                snprintf(cached_ds_name, sizeof(cached_ds_name), "%s", ds_name);
            }
            else
            {
                snap_restore(&cached_ds_snap);
            }
            base_snap = &cached_ds_snap;
        }
        else
        {
            workspace_clear();
            build_dataset(ds_name);
            for (int k = 0; k < nsetup; k++)
            {
                run_setup_stmt(setup[k]);
            }
            snap_capture(&snap);
            base_snap = &snap;
        }

        eval_rec_t r1;
        eval_rec_t r2;
        int        eng2 = old_vs_old ? 0 : 1;
        run_one_engine(0, stmt, is_test, &r1);
        snap_restore(base_snap);
        run_one_engine(eng2, stmt, is_test, &r2);

        char reason[64]  = "";
        char detail[512] = "";
        if (compare_records(&r1, &r2, is_test, strict_mode, reason, sizeof(reason), detail,
                            sizeof(detail)) != 0)
        {
            fputs("{\"dataset\":", stdout);
            print_json_str(stdout, ds_name);
            fputs(",\"stmt\":", stdout);
            print_json_str(stdout, stmt);
            fputs(",\"reason\":", stdout);
            print_json_str(stdout, reason);
            fputs(",\"detail\":", stdout);
            print_json_str(stdout, detail);
            fputs("}\n", stdout);
            fflush(stdout);
        }

        snap_free(&r1.post);
        snap_free(&r2.post);
        snap_free(&snap);
        free(dataset);
        for (int k = 0; k < nsetup; k++)
        {
            free(setup[k]);
        }
        free(stmt);
    }
    snap_free(&cached_ds_snap);
    free(line);
    return 0;
}
