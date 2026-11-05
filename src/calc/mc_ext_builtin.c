// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_builtin.c
 * @brief Built-in `MILK_CALC_REG_IMG` spatial kernels registered by `milkcalc`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libfps/fps_streamname_parse.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_ast.h"
#include "milk_calc_ext.h"

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

MILK_CALC_MODULE_EXPORT(milkcalc_builtin,
                        MILK_CALC_REG_IMG(bblur1pix, bblur1pix_ext_img, bblur1pix),
                        MILK_CALC_REG_IMG(rotate2d, rotate2d_ext_img, rotate2d),
                        MILK_CALC_REG_IMG(circsym, circsym_ext_img, circsym),
                        MILK_CALC_REG_IMG(circasym, circasym_ext_img, circsym));

int mc_is_numeric_env_var(const char *s)
{
    const char *ev   = getenv(s);
    char       *endp = NULL;
    return (ev != NULL && ev[0] != '\0' && (strtod(ev, &endp), endp != ev && *endp == '\0'));
}

long mc_resolve_or_autoconnect_image(const char *name)
{
    if (milk_data.image == NULL || milk_data.NB_MAX_IMAGE <= 0)
    {
        return -1;
    }
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (sp.error)
    {
        return -1;
    }
    const char *bare = (sp.name != NULL && sp.name[0] != '\0') ? sp.name : name;
    imageID     id   = image_ID(bare, milk_data.image, milk_data.NB_MAX_IMAGE);
    char        shmpath[512];
    struct stat st;

    if (id != -1)
    {
        IMAGE *im = &milk_data.image[id];
        if (im->used && im->md != NULL && im->md[0].shared == 1 &&
            ImageStreamIO_filename(shmpath, sizeof(shmpath), im->name) == IMAGESTREAMIO_SUCCESS &&
            stat(shmpath, &st) == 0 && st.st_ino != im->md[0].inode)
        {
            IMAGE fresh = { 0 };
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(im->name, &fresh) ==
                IMAGESTREAMIO_SUCCESS)
            {
                ImageStreamIO_closeIm(im);
                memcpy(im, &fresh, sizeof(IMAGE));
                im->used = 1;
                milk_table_bump();
            }
        }
        return id;
    }
    if (sp.loc == 'L' ||
        ImageStreamIO_filename(shmpath, sizeof(shmpath), bare) != IMAGESTREAMIO_SUCCESS ||
        access(shmpath, R_OK) != 0)
    {
        return -1;
    }
    IMAGE tmp_im = { 0 };
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(bare, &tmp_im) == IMAGESTREAMIO_SUCCESS)
    {
        id = next_avail_image_ID(-1);
        if (id != -1)
        {
            memcpy(&milk_data.image[id], &tmp_im, sizeof(IMAGE));
            milk_data.image[id].used = 1;
            milk_table_bump();
        }
        else
        {
            ImageStreamIO_closeIm(&tmp_im);
        }
    }
    return id;
}

void mc_parse_ident_streamname(char *name, char *bare, char *loc)
{
    long svid = variable_ID(name);
    if (svid != -1 && milk_data.variable[svid].type == 3 &&
        milk_data.variable[svid].value.s[0] != '\0')
    {
        snprintf(name, MC_TOK_STR_MAX, "%s", milk_data.variable[svid].value.s);
    }
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (loc != NULL)
    {
        *loc = sp.loc;
    }
    snprintf(bare, MC_TOK_STR_MAX, "%s",
             (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : name);
}

void mc_set_expr_from_im(mc_expr_t *e, const void *im_ptr)
{
    const IMAGE *im = (const IMAGE *) im_ptr;
    e->res_kind     = MC_RES_IMAGE;
    e->res_dt       = im->md[0].datatype;
    e->res_naxis    = im->md[0].naxis;
    e->res_size[0]  = (im->md[0].naxis > 0) ? im->md[0].size[0] : 1;
    e->res_size[1]  = (im->md[0].naxis > 1) ? im->md[0].size[1] : 1;
    e->res_size[2]  = (im->md[0].naxis > 2) ? im->md[0].size[2] : 1;
    e->needs_image  = 1;
}
