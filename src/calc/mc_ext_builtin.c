// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_builtin.c
 * @brief Built-in `MILK_CALC_REG_IMG` spatial kernels registered by `milkcalc`.
 */

#include "COREMOD_arith/image_spatial.h"
#include "milkDebugTools.h"
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
