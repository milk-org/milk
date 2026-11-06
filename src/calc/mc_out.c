// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "libmilkdata/milkdata.h"
#include "libprocessinfo/processinfo.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/delete_image.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/image_copy.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_plan.h"
#include "mcalc.h"

#define MC_POOL_N 4
#define MC_POOL_BYTES ((size_t) 256 << 20)
#define MC_MAX_PIXELS 1048576

static imageID g_pool_ids[MC_POOL_N] = { -1, -1, -1, -1 };
static long    g_tmp_seq             = 1;
static int     g_post_defer          = 0;
static imageID g_post_ids[64];
static int     g_post_n = 0;

static size_t im_bytes(const IMAGE *im)
{
    return (size_t) im->md[0].nelement * (size_t) ImageStreamIO_typesize(im->md[0].datatype);
}

static IMAGE *pool_im(int k)
{
    char    name[32];
    imageID id = g_pool_ids[k];

    snprintf(name, sizeof(name), "_tmpcalcpool%d", k);
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used || strcmp(milk_data.image[id].name, name) != 0)
    {
        g_pool_ids[k] = -1;
        return NULL;
    }
    return &milk_data.image[id];
}

static int pool_put(imageID id)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        return 0;
    }
    IMAGE *im    = &milk_data.image[id];
    size_t total = 0;
    int    slot  = -1;
    char   name[32];

    if (im->md == NULL || im->md[0].shared != 0 || im->array.raw == NULL)
    {
        return 0;
    }
    for (int k = 0; k < MC_POOL_N; k++)
    {
        IMAGE *pk = pool_im(k);
        if (pk != NULL)
        {
            total += im_bytes(pk);
        }
        else if (slot < 0)
        {
            slot = k;
        }
    }
    if (slot < 0 || total + im_bytes(im) > MC_POOL_BYTES)
    {
        return 0;
    }
    snprintf(name, sizeof(name), "_tmpcalcpool%d", slot);
    chname_image_ID(im->name, name);
    if (strcmp(im->name, name) != 0)
    {
        return 0;
    }
    g_pool_ids[slot] = id;
    return 1;
}

imageID mc_tmp_image(uint8_t naxis, const uint32_t size[3], uint8_t dt)
{
    char tmpname[32];
    snprintf(tmpname, sizeof(tmpname), "_tmpcalc%ld", g_tmp_seq++);

    for (int k = 0; k < MC_POOL_N; k++)
    {
        IMAGE *im   = pool_im(k);
        int    same = (im != NULL && im->md[0].datatype == dt && im->md[0].naxis == naxis);
        for (uint8_t a = 0; same && a < naxis; a++)
        {
            same = (im->md[0].size[a] == size[a]);
        }
        if (!same)
        {
            continue;
        }
        chname_image_ID(im->name, tmpname);
        if (strcmp(im->name, tmpname) != 0)
        {
            continue;
        }
        imageID id      = g_pool_ids[k];
        g_pool_ids[k]   = -1;
        im->md[0].cnt0  = 0;
        im->md[0].cnt1  = 0;
        im->md[0].write = 0;
        for (int i = 0; im->kw != NULL && i < im->md[0].NBkw; i++)
        {
            im->kw[i].type = 'N';
        }
        return id;
    }

    uint32_t sz[3] = { size[0], (naxis >= 2) ? size[1] : 1, (naxis >= 3) ? size[2] : 1 };
    imageID  id    = -1;
    create_image_ID(tmpname, naxis, sz, dt, 0, 10, 0, &id);
    return id;
}

void mc_tmp_release(imageID tid)
{
    if (milk_data.image == NULL || tid < 0 || tid >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[tid].used)
    {
        return;
    }
    if (!pool_put(tid))
    {
        delete_image_ID(milk_data.image[tid].name, DELETE_IMAGE_ERRMODE_WARNING);
    }
}

void mc_ensure_pool_warm(uint8_t naxis, const uint32_t size[3], uint8_t dt)
{
    for (int k = 0; k < MC_POOL_N; k++)
    {
        if (pool_im(k) != NULL)
        {
            return;
        }
    }
    imageID tid = mc_tmp_image(naxis, size, dt);
    if (tid != -1)
    {
        mc_tmp_release(tid);
    }
}

void mcalc_tmp_cleanup(void)
{
    if (milk_data.image == NULL || milk_data.NB_MAX_IMAGE <= 0)
    {
        return;
    }
    for (long i = 0; i < milk_data.NB_MAX_IMAGE; i++)
    {
        if (milk_data.image[i].used && strncmp(milk_data.image[i].name, "_tmpcalc", 8) == 0 &&
            strncmp(milk_data.image[i].name, "_tmpcalcpool", 12) != 0)
        {
            if (!pool_put((imageID) i))
            {
                delete_image_ID(milk_data.image[i].name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
    }
    g_tmp_seq = 1;
}

imageID mc_out_prepare(mc_slot_t     *slot,
                       uint8_t        naxis,
                       const uint32_t size[3],
                       uint8_t        dt,
                       int            in_unit,
                       char          *err,
                       size_t         errsz)
{
    const char *bare   = (slot->bare[0] != '\0') ? slot->bare : slot->name;
    uint64_t    tgen   = milk_table_generation();
    int         is_tmp = (strncmp(bare, "_tmpcalc", 8) == 0);
    int want_shm = (is_tmp || slot->loc == 2 || slot->is_local) ? 0 : (slot->loc == 1 || in_unit);
    imageID id   = -1;

    if (slot->im_id >= 0 && slot->im_id < milk_data.NB_MAX_IMAGE &&
        milk_data.image[slot->im_id].used && slot->checked_gen == tgen && tgen != 0)
    {
        id = slot->im_id;
    }
    else
    {
        id       = (slot->im_id >= 0 && slot->im_id < milk_data.NB_MAX_IMAGE &&
                    milk_data.image[slot->im_id].used &&
                    strcmp(milk_data.image[slot->im_id].name, bare) == 0)
                       ? slot->im_id
                       : ((is_tmp || slot->loc == 2 || slot->is_local)
                              ? image_ID(bare, milk_data.image, milk_data.NB_MAX_IMAGE)
                              : mc_resolve_or_autoconnect_image(slot->name));
        long vid = variable_ID(bare);
        if (vid >= 0 && milk_data.variable[vid].type != 3)
        {
            if (err != NULL && errsz > 0)
            {
                if (strncmp(bare, "__g", 3) == 0)
                {
                    snprintf(err, errsz,
                             "an if or for needs a number, not an image: test "
                             "imin(...) for all pixels, imax(...) for any");
                }
                else
                {
                    snprintf(err, errsz, "'%s' is a variable (calculator or bash), not an image",
                             bare);
                }
            }
            return -1;
        }
    }

    uint64_t new_nel = 1;
    for (uint8_t a = 0; a < naxis; a++)
    {
        new_nel *= size[a];
    }

    if (id != -1)
    {
        IMAGE *im         = &milk_data.image[id];
        int    same_shape = (im->md[0].naxis == naxis && im->md[0].nelement == new_nel);
        for (uint8_t a = 0; same_shape && a < naxis; a++)
        {
            same_shape = (im->md[0].size[a] == size[a]);
        }

        if (!slot->is_local && im->md[0].shared == 1)
        {
            if (im->md[0].nelement != new_nel)
            {
                if (err != NULL && errsz > 0)
                {
                    snprintf(err, errsz, "Dimension mismatch in image assignment");
                }
                return -1;
            }
            im->md[0].write   = 1;
            slot->im_id       = id;
            slot->checked_gen = tgen;
            return id;
        }
        if (im->md[0].shared == 0 && same_shape && im->md[0].datatype == dt && !want_shm)
        {
            im->md[0].write   = 1;
            slot->im_id       = id;
            slot->checked_gen = tgen;
            return id;
        }
        if (im->md[0].shared == 1)
        {
            if (err != NULL && errsz > 0)
            {
                snprintf(err, errsz, "Dimension mismatch in image assignment");
            }
            return -1;
        }
        if (strncmp(bare, "_tmpcalc", 8) == 0)
        {
            mc_tmp_release(id);
        }
        else
        {
            delete_image_ID(bare, DELETE_IMAGE_ERRMODE_WARNING);
        }
        id = -1;
    }

    uint32_t sz[3] = { size[0], (naxis >= 2) ? size[1] : 1, (naxis >= 3) ? size[2] : 1 };
    if (!want_shm && strncmp(bare, "_tmpcalc", 8) == 0)
    {
        id = mc_tmp_image(naxis, sz, dt);
        if (id != -1 && strcmp(milk_data.image[id].name, bare) != 0)
        {
            chname_image_ID(milk_data.image[id].name, bare);
        }
    }
    else
    {
        create_image_ID(bare, naxis, sz, dt, want_shm, 10, 0, &id);
    }
    if (id == -1)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "cannot create image '%s'", bare);
        }
        return -1;
    }
    milk_data.image[id].md[0].write = 1;
    slot->im_id                     = id;
    slot->checked_gen               = milk_table_generation();
    return id;
}

void mc_out_post(imageID id)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        return;
    }
    IMAGE *im = &milk_data.image[id];
    if (im->md == NULL || !im->md[0].shared)
    {
        if (im->md != NULL)
        {
            im->md[0].write = 0;
            im->md[0].cnt0++;
        }
        return;
    }
    if (!g_post_defer)
    {
        processinfo_update_output_stream(NULL, im, NULL);
        return;
    }
    for (int i = 0; i < g_post_n; i++)
    {
        if (g_post_ids[i] == id)
        {
            return;
        }
    }
    if (g_post_n < (int) (sizeof(g_post_ids) / sizeof(g_post_ids[0])))
    {
        g_post_ids[g_post_n++] = id;
    }
    else
    {
        processinfo_update_output_stream(NULL, im, NULL);
    }
}

void mc_post_defer(int on)
{
    g_post_defer = on;
    g_post_n     = 0;
}

void mc_post_flush(void *processinfo)
{
    g_post_defer = 0;
    for (int i = 0; i < g_post_n; i++)
    {
        imageID id = g_post_ids[i];
        if (milk_data.image != NULL && id >= 0 && id < milk_data.NB_MAX_IMAGE)
        {
            IMAGE *im = &milk_data.image[id];
            if (im->used == 1 && im->md != NULL && im->md[0].shared)
            {
                processinfo_update_output_stream((PROCESSINFO *) processinfo, im, NULL);
            }
        }
    }
    g_post_n = 0;
}

void mc_format_image_summary(imageID id, const char *assign_name, char *buf, size_t bufsz)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        snprintf(buf, bufsz, "%s", (assign_name != NULL) ? assign_name : "");
        return;
    }
    const IMAGE_METADATA *md   = &milk_data.image[id].md[0];
    size_t                dpos = 0;
    char                  dims[128];
    dims[0] = '\0';
    for (uint8_t a = 0; a < md->naxis && dpos + 24 < sizeof(dims); a++)
    {
        dpos += (size_t) snprintf(dims + dpos, sizeof(dims) - dpos, "%s%lu", (a > 0) ? " x " : "",
                                  (unsigned long) md->size[a]);
    }
    const char *tstr = "?";
    static const struct
    {
        uint8_t     dt;
        const char *name;
    } dt_names[] = {
        { _DATATYPE_FLOAT, "f32" },          { _DATATYPE_DOUBLE, "f64" },
        { _DATATYPE_UINT8, "u8" },           { _DATATYPE_INT8, "i8" },
        { _DATATYPE_UINT16, "u16" },         { _DATATYPE_INT16, "i16" },
        { _DATATYPE_UINT32, "u32" },         { _DATATYPE_INT32, "i32" },
        { _DATATYPE_UINT64, "u64" },         { _DATATYPE_INT64, "i64" },
        { _DATATYPE_COMPLEX_FLOAT, "cf32" }, { _DATATYPE_COMPLEX_DOUBLE, "cf64" },
        { _DATATYPE_HALF, "f16" },
    };
    for (size_t i = 0; i < sizeof(dt_names) / sizeof(dt_names[0]); i++)
    {
        if (dt_names[i].dt == md->datatype)
        {
            tstr = dt_names[i].name;
            break;
        }
    }
    const char *nm =
        (assign_name != NULL && assign_name[0] != '\0') ? assign_name : milk_data.image[id].name;
    if ((assign_name != NULL && assign_name[0] != '\0') || strncmp(nm, "_tmpcalc", 8) != 0)
    {
        snprintf(buf, bufsz, "%s [%s] %s", nm, dims, tstr);
    }
    else
    {
        snprintf(buf, bufsz, "[%s] %s", dims, tstr);
    }
}

char *mc_format_pixel_line(imageID id)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        return NULL;
    }
    const IMAGE *im  = &milk_data.image[id];
    uint64_t     n   = im->md[0].nelement;
    uint8_t      dt  = im->md[0].datatype;
    char        *out = NULL;
    size_t       pos = 0;

    if (n > MC_MAX_PIXELS || (out = (char *) malloc(n * 26 + 1)) == NULL)
    {
        fprintf(stderr, "milk: calc error: %s has %lu pixels (-a takes %d at most)\n", im->name,
                (unsigned long) n, MC_MAX_PIXELS);
        return NULL;
    }
    out[0] = '\0';
#define PIX_INT(MBR)                                                                              \
    for (uint64_t i = 0; i < n; i++)                                                              \
    {                                                                                             \
        pos += (size_t) sprintf(out + pos, "%s%lld", i ? " " : "", (long long) im->array.MBR[i]); \
    }
#define PIX_REAL(MBR)                                                                  \
    for (uint64_t i = 0; i < n; i++)                                                   \
    {                                                                                  \
        pos += (size_t) sprintf(out + pos, "%s%.*g", i ? " " : "", mcalc_float_digits, \
                                (double) im->array.MBR[i]);                            \
    }
    if (dt < _DATATYPE_UINT8 || dt > _DATATYPE_DOUBLE)
    {
        fprintf(stderr, "milk: calc error: %s: -a takes real images\n", im->name);
        free(out);
        return NULL;
    }
    if (dt == _DATATYPE_FLOAT)
    {
        PIX_REAL(F)
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        PIX_REAL(D)
    }
    else
    {
        MILK_FOR_EACH_DATATYPE(dt, PIX_INT, PIX_INT(D))
    }
#undef PIX_INT
#undef PIX_REAL
    return out;
}
