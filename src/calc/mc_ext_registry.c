// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_registry.c
 * @brief In-memory registry for `milk_calc_ext` kernels and operators.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "milk_calc_ext.h"

#define CALC_EXT_MAX_ENTRIES 64

static milk_calc_ext_entry_t ext_entries_storage[CALC_EXT_MAX_ENTRIES];
static char                  ext_names_storage[CALC_EXT_MAX_ENTRIES][64];
static int                   ext_nentries = 0;

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
