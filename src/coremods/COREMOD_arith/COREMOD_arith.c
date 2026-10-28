// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    COREMOD_arith.c
 * @brief   Arithmeric operations on images
 *
 * Addition, multiplication and much more
 *
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "arith"

// Module short description
#define MODULE_DESCRIPTION "Image arithmetic operations"

#include <stdio.h>
#include <string.h>

#include "milk_runnable.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_arith/milk_calc_ext.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_COREMOD_arith_extract3Dim,       &runnable_COREMOD_arith_extractim,
    &runnable_COREMOD_arith_image_merge3D,     &runnable_COREMOD_arith_image_cropmask,
    &runnable_COREMOD_arith_image_multicrop2D,
};

MILK_MODULE_RUNNABLES(COREMOD_arith, NULL, runnables);

/* =========================================================================
 * Core-Tier `milk_calc_ext` Registry & Built-In Module Kernels
 * ========================================================================= */

#define CALC_EXT_MAX_ENTRIES 64

static milk_calc_ext_entry_t ext_entries_storage[CALC_EXT_MAX_ENTRIES];
static char                  ext_names_storage[CALC_EXT_MAX_ENTRIES][64];
static int                   ext_nentries = 0;

/** param_fptype_size - Bytes the packer writes for FPS type @fptype, 0 if unsupported. */
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

/**
 * check_param_set - Validate parameter table against `MILK_CALC_PARAM_BYTES`
 */
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

/**
 * milk_calc_ext_register - Register or update a single kernel/operator entry in the global table
 * @e:     Extension descriptor
 * @err:   Optional error message buffer
 * @errsz: Size of @err
 *
 * Return: 0 on success, -1 on validation or capacity error.
 */
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

/**
 * milk_calc_ext_register_table - Register all entries in an exported `milk_calc_ext_table_t`
 * @tbl:   Exported extension table
 * @err:   Optional error message buffer
 * @errsz: Size of @err
 *
 * Return: 0 on success, -1 on ABI mismatch or entry validation error.
 */
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

/**
 * milk_calc_ext_find - Look up a registered kernel or operator by name
 * @name: Function/operator name
 *
 * Return: Entry index `0 .. count-1`, or `-1` if not found.
 */
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

/**
 * milk_calc_ext_get - Return the registered descriptor at index @idx
 * @idx: Entry index
 *
 * Return: Pointer to `milk_calc_ext_entry_t`, or `NULL` if out of range.
 */
const milk_calc_ext_entry_t *milk_calc_ext_get(int idx)
{
    return (idx >= 0 && idx < ext_nentries) ? &ext_entries_storage[idx] : NULL;
}

/**
 * milk_calc_ext_count - Return the number of registered kernels and operators
 */
int milk_calc_ext_count(void)
{
    return ext_nentries;
}

/* Built-in Core/Plugin spatial kernels exposed to `milk.calc` / `@calc` */
#define BBLUR_EXT_PARAMS(X)                                                                    \
    X(".amp", amp, FPTYPE_FLOAT64, 0.1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Side weight") \
    X(".niter", niter, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Blur passes")
MILK_CALC_PARAMS(bblur1pix, BBLUR_EXT_PARAMS);

static int bblur1pix_ext_img(IMAGE *out, const IMAGE *in, const struct bblur1pix_par *P)
{
    int    niter = (P != NULL && P->niter > 0) ? (int) P->niter : 1;
    double amp   = (P != NULL) ? P->amp : 0.1;
    return arith_image_bblur1pix_im(in, out, amp, niter);
}

#define ROT2D_EXT_PARAMS(X) \
    X(".angle", angle, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Angle (rad)")
MILK_CALC_PARAMS(rotate2d, ROT2D_EXT_PARAMS);

static int rotate2d_ext_img(IMAGE *out, const IMAGE *in, const struct rotate2d_par *P)
{
    return arith_image_rotate2d_im(in, out, (P != NULL) ? P->angle : 0.0);
}

#define CIRCSYM_EXT_PARAMS(X)                                                              \
    X(".cx", cx, FPTYPE_FLOAT64, -1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "X center") \
    X(".cy", cy, FPTYPE_FLOAT64, -1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Y center")
MILK_CALC_PARAMS(circsym, CIRCSYM_EXT_PARAMS);

static int circsym_ext_img(IMAGE *out, const IMAGE *in, const struct circsym_par *P)
{
    double cx = (P != NULL && P->cx >= 0.0) ? P->cx : 0.5 * (double) in->md[0].size[0];
    double cy = (P != NULL && P->cy >= 0.0)
                    ? P->cy
                    : 0.5 * (double) ((in->md[0].naxis > 1) ? in->md[0].size[1] : 1);
    return arith_image_circsym_im(in, out, cx, cy, 0);
}

static int circasym_ext_img(IMAGE *out, const IMAGE *in, const struct circsym_par *P)
{
    double cx = (P != NULL && P->cx >= 0.0) ? P->cx : 0.5 * (double) in->md[0].size[0];
    double cy = (P != NULL && P->cy >= 0.0)
                    ? P->cy
                    : 0.5 * (double) ((in->md[0].naxis > 1) ? in->md[0].size[1] : 1);
    return arith_image_circsym_im(in, out, cx, cy, 1);
}

MILK_CALC_MODULE_EXPORT(COREMOD_arith,
                        MILK_CALC_REG_IMG(bblur1pix, bblur1pix_ext_img, bblur1pix),
                        MILK_CALC_REG_IMG(rotate2d, rotate2d_ext_img, rotate2d),
                        MILK_CALC_REG_IMG(circsym, circsym_ext_img, circsym),
                        MILK_CALC_REG_IMG(circasym, circasym_ext_img, circsym));
