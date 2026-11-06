// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_FFTW3
#    include <fftw3.h>
#endif

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/image_ID.h"
#include "linalg/mc_linalg.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

__attribute__((format(printf, 3, 4))) static int fs_err(char       *err,
                                                        size_t      errsz,
                                                        const char *fmt,
                                                        ...)
{
    if (err != NULL && errsz > 0 && err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errsz, fmt, ap);
        va_end(ap);
    }
    return -1;
}

#ifdef HAVE_FFTW3
typedef struct
{
    uint8_t  naxis;
    uint32_t sz[3];
    int      is_dbl;
    int      sign;
    void    *plan;
} fft_cache_entry_t;

#    define FFT_CACHE_CAP 16
static fft_cache_entry_t g_fft_cache[FFT_CACHE_CAP];
static int               g_fft_cache_n    = 0;
static int               g_fft_cache_next = 0;
static pthread_mutex_t   g_fft_lock       = PTHREAD_MUTEX_INITIALIZER;

static int run_fftw_cached(uint8_t        naxis,
                           const uint32_t sz[3],
                           int            is_dbl,
                           int            sign,
                           void          *in_buf,
                           void          *out_buf)
{
    pthread_mutex_lock(&g_fft_lock);
    void *plan = NULL;
    for (int i = 0; i < g_fft_cache_n; i++)
    {
        const fft_cache_entry_t *e = &g_fft_cache[i];
        if (e->naxis == naxis && e->sz[0] == sz[0] && e->sz[1] == sz[1] && e->sz[2] == sz[2] &&
            e->is_dbl == is_dbl && e->sign == sign)
        {
            plan = e->plan;
            break;
        }
    }
    if (plan == NULL)
    {
        unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        int      nx = (int) sz[0], ny = (int) sz[1], nz = (int) sz[2];
        if (is_dbl)
        {
            plan = (naxis == 1) ? (void *) fftw_plan_dft_1d(nx, (fftw_complex *) in_buf,
                                                            (fftw_complex *) out_buf, sign, flags)
                   : (naxis == 2)
                       ? (void *) fftw_plan_dft_2d(ny, nx, (fftw_complex *) in_buf,
                                                   (fftw_complex *) out_buf, sign, flags)
                       : (void *) fftw_plan_dft_3d(nz, ny, nx, (fftw_complex *) in_buf,
                                                   (fftw_complex *) out_buf, sign, flags);
        }
        else
        {
            plan = (naxis == 1) ? (void *) fftwf_plan_dft_1d(nx, (fftwf_complex *) in_buf,
                                                             (fftwf_complex *) out_buf, sign, flags)
                   : (naxis == 2)
                       ? (void *) fftwf_plan_dft_2d(ny, nx, (fftwf_complex *) in_buf,
                                                    (fftwf_complex *) out_buf, sign, flags)
                       : (void *) fftwf_plan_dft_3d(nz, ny, nx, (fftwf_complex *) in_buf,
                                                    (fftwf_complex *) out_buf, sign, flags);
        }
        if (plan != NULL)
        {
            int slot         = (g_fft_cache_n < FFT_CACHE_CAP) ? g_fft_cache_n++ : g_fft_cache_next;
            g_fft_cache_next = (slot + 1) % FFT_CACHE_CAP;
            if (g_fft_cache[slot].plan != NULL)
            {
                (void) (g_fft_cache[slot].is_dbl
                            ? (fftw_destroy_plan((fftw_plan) g_fft_cache[slot].plan), 1)
                            : (fftwf_destroy_plan((fftwf_plan) g_fft_cache[slot].plan), 1));
            }
            g_fft_cache[slot] = (fft_cache_entry_t) {
                .naxis  = naxis,
                .sz     = { sz[0], sz[1], sz[2] },
                .is_dbl = is_dbl,
                .sign   = sign,
                .plan   = plan,
            };
        }
    }
    if (plan != NULL)
    {
        (void) (is_dbl ? (fftw_execute_dft((fftw_plan) plan, (fftw_complex *) in_buf,
                                           (fftw_complex *) out_buf),
                          1)
                       : (fftwf_execute_dft((fftwf_plan) plan, (fftwf_complex *) in_buf,
                                            (fftwf_complex *) out_buf),
                          1));
    }
    pthread_mutex_unlock(&g_fft_lock);
    return (plan != NULL) ? 0 : -1;
}
#endif

static void dft_1d_line(uint32_t      N,
                        int           sign,
                        const double *in_re,
                        const double *in_im,
                        double       *out_re,
                        double       *out_im)
{
    double ang_base = (double) sign * 2.0 * M_PI / (double) N;
    for (uint32_t k = 0; k < N; k++)
    {
        double sr = 0.0, si = 0.0;
        for (uint32_t n = 0; n < N; n++)
        {
            double th = ang_base * (double) ((uint64_t) k * (uint64_t) n % (uint64_t) N);
            double c = cos(th), s = sin(th);
            sr += in_re[n] * c - in_im[n] * s;
            si += in_re[n] * s + in_im[n] * c;
        }
        out_re[k] = sr, out_im[k] = si;
    }
}

static void read_pix_any(const IMAGE *im, uint8_t dt, uint64_t idx, double *re, double *im_part)
{
    *re      = (dt == _DATATYPE_COMPLEX_FLOAT)    ? (double) im->array.CF[idx].re
               : (dt == _DATATYPE_COMPLEX_DOUBLE) ? im->array.CD[idx].re
                                                  : arith_read_im_pix(im, dt, idx);
    *im_part = (dt == _DATATYPE_COMPLEX_FLOAT)    ? (double) im->array.CF[idx].im
               : (dt == _DATATYPE_COMPLEX_DOUBLE) ? im->array.CD[idx].im
                                                  : 0.0;
}

static void write_pix_any(IMAGE *out, uint8_t dt, uint64_t idx, double re, double im_part)
{
    if (dt == _DATATYPE_COMPLEX_FLOAT)
    {
        out->array.CF[idx].re = (float) re, out->array.CF[idx].im = (float) im_part;
    }
    else if (dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        out->array.CD[idx].re = re, out->array.CD[idx].im = im_part;
    }
    else
    {
        mc_write_im_pix(out, dt, idx, re);
    }
}

static int exec_fft_or_ifft(mc_slot_t       *slot,
                            const mc_expr_t *rhs,
                            imageID          avoid_id,
                            const mc_expr_t *call_site,
                            int              in_unit,
                            char            *err,
                            size_t           errsz)
{
    int     is_ifft = (strcmp(rhs->u.call.fn->name, "ifft") == 0), src_tmp = 0, rc = 0;
    imageID src_id = -1;
    if (mc_resolve_or_eval_im(rhs->u.call.args[0], call_site, avoid_id, in_unit, &src_id, &src_tmp,
                              err, errsz) != 0)
    {
        return -1;
    }
    const IMAGE *src   = &milk_data.image[src_id];
    uint8_t      naxis = src->md[0].naxis;
    if (naxis < 1 || naxis > 3)
    {
        (void) (src_tmp && (mc_tmp_release(src_id), 1));
        return fs_err(err, errsz, "%s: expected 1D, 2D, or 3D image", rhs->u.call.fn->name);
    }
    uint32_t sz[3] = {
        src->md[0].size[0],
        (naxis > 1) ? src->md[0].size[1] : 1,
        (naxis > 2) ? src->md[0].size[2] : 1,
    };
    uint64_t nel    = (uint64_t) sz[0] * (uint64_t) sz[1] * (uint64_t) sz[2];
    uint8_t  sdt    = src->md[0].datatype;
    int      is_dbl = (sdt == _DATATYPE_DOUBLE || sdt == _DATATYPE_COMPLEX_DOUBLE);
    uint8_t  out_dt =
        mc_select_ctor_out_dt(slot, is_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT);
    is_dbl = (out_dt == _DATATYPE_COMPLEX_DOUBLE);

    imageID oid = mc_out_prepare(slot, naxis, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        (void) (src_tmp && (mc_tmp_release(src_id), 1));
        return -1;
    }
    IMAGE *out  = &milk_data.image[oid];
    int    sign = is_ifft ? 1 : -1;

#ifdef HAVE_FFTW3
    size_t elem_sz = is_dbl ? (2 * sizeof(double)) : (2 * sizeof(float));
    void  *in_buf  = malloc(elem_sz * (size_t) (nel > 0 ? nel : 1));
    if (in_buf == NULL)
    {
        (void) (src_tmp && (mc_tmp_release(src_id), 1));
        return fs_err(err, errsz, "%s: out of memory", rhs->u.call.fn->name);
    }
    for (uint64_t i = 0; i < nel; i++)
    {
        double re = 0.0, im_p = 0.0;
        read_pix_any(src, sdt, i, &re, &im_p);
        if (is_dbl)
        {
            ((double *) in_buf)[2 * i] = re, ((double *) in_buf)[2 * i + 1] = im_p;
        }
        else
        {
            ((float *) in_buf)[2 * i] = (float) re, ((float *) in_buf)[2 * i + 1] = (float) im_p;
        }
    }
    rc = run_fftw_cached(naxis, sz, is_dbl, sign, in_buf,
                         is_dbl ? (void *) out->array.CD : (void *) out->array.CF);
    if (rc == 0 && is_ifft)
    {
        double inv_n = 1.0 / (double) nel;
        for (uint64_t i = 0; i < nel; i++)
        {
            if (is_dbl)
            {
                out->array.CD[i].re *= inv_n, out->array.CD[i].im *= inv_n;
            }
            else
            {
                out->array.CF[i].re *= (float) inv_n, out->array.CF[i].im *= (float) inv_n;
            }
        }
    }
    free(in_buf);
#else
    double  *re    = calloc((size_t) nel, sizeof(double));
    double  *im    = calloc((size_t) nel, sizeof(double));
    uint32_t max_d = (sz[0] > sz[1]) ? sz[0] : sz[1];
    max_d          = (sz[2] > max_d) ? sz[2] : max_d;
    double *tr     = calloc((size_t) max_d, sizeof(double)),
           *ti     = calloc((size_t) max_d, sizeof(double));
    double *or_b   = calloc((size_t) max_d, sizeof(double)),
           *oi_b   = calloc((size_t) max_d, sizeof(double));
    for (uint64_t i = 0; i < nel; i++)
    {
        read_pix_any(src, sdt, i, &re[i], &im[i]);
    }
    for (uint32_t z = 0; z < sz[2]; z++)
    {
        for (uint32_t y = 0; y < sz[1]; y++)
        {
            uint64_t base = ((uint64_t) z * sz[1] + y) * sz[0];
            dft_1d_line(sz[0], sign, &re[base], &im[base], or_b, oi_b);
            memcpy(&re[base], or_b, sizeof(double) * sz[0]);
            memcpy(&im[base], oi_b, sizeof(double) * sz[0]);
        }
    }
    for (uint32_t z = 0; naxis >= 2 && z < sz[2]; z++)
    {
        for (uint32_t x = 0; x < sz[0]; x++)
        {
            for (uint32_t y = 0; y < sz[1]; y++)
            {
                uint64_t idx = ((uint64_t) z * sz[1] + y) * sz[0] + x;
                tr[y] = re[idx], ti[y] = im[idx];
            }
            dft_1d_line(sz[1], sign, tr, ti, or_b, oi_b);
            for (uint32_t y = 0; y < sz[1]; y++)
            {
                uint64_t idx = ((uint64_t) z * sz[1] + y) * sz[0] + x;
                re[idx] = or_b[y], im[idx] = oi_b[y];
            }
        }
    }
    for (uint32_t y = 0; naxis == 3 && y < sz[1]; y++)
    {
        for (uint32_t x = 0; x < sz[0]; x++)
        {
            for (uint32_t z = 0; z < sz[2]; z++)
            {
                uint64_t idx = ((uint64_t) z * sz[1] + y) * sz[0] + x;
                tr[z] = re[idx], ti[z] = im[idx];
            }
            dft_1d_line(sz[2], sign, tr, ti, or_b, oi_b);
            for (uint32_t z = 0; z < sz[2]; z++)
            {
                uint64_t idx = ((uint64_t) z * sz[1] + y) * sz[0] + x;
                re[idx] = or_b[z], im[idx] = oi_b[z];
            }
        }
    }
    double scale = is_ifft ? (1.0 / (double) nel) : 1.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        write_pix_any(out, out_dt, i, re[i] * scale, im[i] * scale);
    }
    free(re), free(im), free(tr), free(ti), free(or_b), free(oi_b);
#endif
    (void) dft_1d_line;
    (void) (rc == 0 && (mc_out_post(oid), 1));
    (void) (src_tmp && (mc_tmp_release(src_id), 1));
    return rc;
}

static int exec_fftshift(mc_slot_t       *slot,
                         const mc_expr_t *rhs,
                         imageID          avoid_id,
                         const mc_expr_t *call_site,
                         int              in_unit,
                         char            *err,
                         size_t           errsz)
{
    imageID src_id  = -1;
    int     src_tmp = 0;
    if (mc_resolve_or_eval_im(rhs->u.call.args[0], call_site, avoid_id, in_unit, &src_id, &src_tmp,
                              err, errsz) != 0)
    {
        return -1;
    }
    const IMAGE *src   = &milk_data.image[src_id];
    uint8_t      naxis = src->md[0].naxis;
    uint32_t     sz[3] = {
        src->md[0].size[0],
        (naxis > 1) ? src->md[0].size[1] : 1,
        (naxis > 2) ? src->md[0].size[2] : 1,
    };
    uint8_t sdt    = src->md[0].datatype;
    uint8_t out_dt = mc_select_ctor_out_dt(slot, sdt);
    imageID oid    = mc_out_prepare(slot, naxis, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        (void) (src_tmp && (mc_tmp_release(src_id), 1));
        return -1;
    }
    IMAGE   *out = &milk_data.image[oid];
    uint32_t hx = (sz[0] + 1) / 2, hy = (naxis > 1) ? ((sz[1] + 1) / 2) : 0;
    uint32_t hz  = (naxis > 2) ? ((sz[2] + 1) / 2) : 0;
    size_t   esz = (size_t) ImageStreamIO_typesize(sdt);

    for (uint32_t dz = 0; dz < sz[2]; dz++)
    {
        uint32_t sz_i = (dz + hz) % sz[2];
        for (uint32_t dy = 0; dy < sz[1]; dy++)
        {
            uint32_t sy_i = (dy + hy) % sz[1];
            for (uint32_t dx = 0; dx < sz[0]; dx++)
            {
                uint32_t sx_i = (dx + hx) % sz[0];
                uint64_t sidx = ((uint64_t) sz_i * sz[1] + sy_i) * sz[0] + sx_i;
                uint64_t didx = ((uint64_t) dz * sz[1] + dy) * sz[0] + dx;
                if (out_dt == sdt)
                {
                    memcpy((uint8_t *) out->array.raw + didx * esz,
                           (const uint8_t *) src->array.raw + sidx * esz, esz);
                }
                else
                {
                    mc_write_im_pix(out, out_dt, didx, arith_read_im_pix(src, sdt, sidx));
                }
            }
        }
    }
    mc_out_post(oid);
    (void) (src_tmp && (mc_tmp_release(src_id), 1));
    return 0;
}

static int exec_stack_or_cat(mc_slot_t       *slot,
                             const mc_expr_t *rhs,
                             imageID          avoid_id,
                             const mc_expr_t *call_site,
                             int              in_unit,
                             char            *err,
                             size_t           errsz)
{
    const char *fn     = rhs->u.call.fn->name;
    int         is_cat = (strcmp(fn, "cat") == 0);
    int         axis   = (strcmp(fn, "hstack") == 0) ? 0 : (strcmp(fn, "vstack") == 0) ? 1 : 2;
    if (is_cat)
    {
        double axd = 0.0;
        if (mc_eval_scalar_d(rhs->u.call.args[0], call_site, &axd, err, errsz) != 0)
        {
            return -1;
        }
        long axl = llround(axd);
        if (axl < 0 || axl > 2)
        {
            return fs_err(err, errsz, "cat: axis must be 0, 1, or 2");
        }
        axis = (int) axl;
    }

    int     arg0                = is_cat ? 1 : 0;
    int     nin                 = rhs->u.call.nargs - arg0;
    imageID ids[MC_MAX_ARGS]    = { 0 };
    int     is_tmp[MC_MAX_ARGS] = { 0 };
    int     rc                  = 0;

    for (int i = 0; i < nin; i++)
    {
        ids[i] = -1;
        if (mc_resolve_or_eval_im(rhs->u.call.args[arg0 + i], call_site, avoid_id, in_unit, &ids[i],
                                  &is_tmp[i], err, errsz) != 0)
        {
            rc = -1;
            goto done;
        }
    }

    uint32_t out_sz[3] = { 0, 0, 0 };
    uint8_t  max_naxis = (uint8_t) (axis + 1);
    int      any_cplx = 0, any_dbl = 0;
    for (int i = 0; i < nin; i++)
    {
        const IMAGE *im = &milk_data.image[ids[i]];
        uint8_t      dt = im->md[0].datatype;
        if (im->md[0].naxis > max_naxis)
        {
            max_naxis = im->md[0].naxis;
        }
        any_cplx |= (dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE);
        any_dbl |= (dt == _DATATYPE_DOUBLE || dt == _DATATYPE_COMPLEX_DOUBLE ||
                    dt == _DATATYPE_INT64 || dt == _DATATYPE_UINT64);
        uint32_t s[3] = {
            im->md[0].size[0],
            (im->md[0].naxis > 1) ? im->md[0].size[1] : 1,
            (im->md[0].naxis > 2) ? im->md[0].size[2] : 1,
        };
        if (i == 0)
        {
            out_sz[0] = s[0];
            out_sz[1] = s[1];
            out_sz[2] = s[2];
        }
        else
        {
            for (int d = 0; d < 3; d++)
            {
                if (d != axis && s[d] != out_sz[d])
                {
                    rc = fs_err(err, errsz, "%s: non-concatenated axis %d size mismatch (%u vs %u)",
                                fn, d, out_sz[d], s[d]);
                    goto done;
                }
            }
            out_sz[axis] += s[axis];
        }
    }

    uint8_t pref_dt = any_cplx ? (any_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT)
                               : (any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    uint8_t out_dt  = mc_select_ctor_out_dt(slot, pref_dt);
    imageID oid     = mc_out_prepare(slot, max_naxis, out_sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        rc = -1;
        goto done;
    }
    IMAGE   *out = &milk_data.image[oid];
    uint32_t off = 0;
    for (int i = 0; i < nin; i++)
    {
        const IMAGE *im   = &milk_data.image[ids[i]];
        uint8_t      dt   = im->md[0].datatype;
        uint32_t     s[3] = {
            im->md[0].size[0],
            (im->md[0].naxis > 1) ? im->md[0].size[1] : 1,
            (im->md[0].naxis > 2) ? im->md[0].size[2] : 1,
        };
        for (uint32_t z = 0; z < s[2]; z++)
        {
            uint32_t dz = (axis == 2) ? (off + z) : z;
            for (uint32_t y = 0; y < s[1]; y++)
            {
                uint32_t dy = (axis == 1) ? (off + y) : y;
                for (uint32_t x = 0; x < s[0]; x++)
                {
                    uint32_t dx   = (axis == 0) ? (off + x) : x;
                    uint64_t sidx = ((uint64_t) z * s[1] + y) * s[0] + x;
                    uint64_t didx = ((uint64_t) dz * out_sz[1] + dy) * out_sz[0] + dx;
                    double   re = 0.0, im_p = 0.0;
                    read_pix_any(im, dt, sidx, &re, &im_p);
                    write_pix_any(out, out_dt, didx, re, im_p);
                }
            }
        }
        off += s[axis];
    }
    mc_out_post(oid);

done:
    for (int i = 0; i < nin; i++)
    {
        (void) (is_tmp[i] && ids[i] >= 0 && (mc_tmp_release(ids[i]), 1));
    }
    return rc;
}

static int exec_rot90(mc_slot_t       *slot,
                      const mc_expr_t *rhs,
                      imageID          avoid_id,
                      const mc_expr_t *call_site,
                      int              in_unit,
                      char            *err,
                      size_t           errsz)
{
    double kd = 1.0;
    if (rhs->u.call.nargs == 2 &&
        mc_eval_scalar_d(rhs->u.call.args[1], call_site, &kd, err, errsz) != 0)
    {
        return -1;
    }
    int     k       = (int) (((llround(kd) % 4) + 4) % 4);
    imageID src_id  = -1;
    int     src_tmp = 0;
    if (mc_resolve_or_eval_im(rhs->u.call.args[0], call_site, avoid_id, in_unit, &src_id, &src_tmp,
                              err, errsz) != 0)
    {
        return -1;
    }
    const IMAGE *src = &milk_data.image[src_id];
    if (src->md[0].naxis != 2)
    {
        (void) (src_tmp && (mc_tmp_release(src_id), 1));
        return fs_err(err, errsz, "rot90: expected 2D image");
    }
    uint32_t nx = src->md[0].size[0], ny = src->md[0].size[1];
    uint32_t sz[3] = { (k % 2 == 1) ? ny : nx, (k % 2 == 1) ? nx : ny, 1 };
    uint8_t  sdt   = src->md[0].datatype;
    uint8_t  out_dt =
        mc_select_ctor_out_dt(slot, (sdt == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    imageID oid = mc_out_prepare(slot, 2, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        (void) (src_tmp && (mc_tmp_release(src_id), 1));
        return -1;
    }
    IMAGE *out = &milk_data.image[oid];
    for (uint32_t dy = 0; dy < sz[1]; dy++)
    {
        for (uint32_t dx = 0; dx < sz[0]; dx++)
        {
            uint32_t sx = (k == 0) ? dx : (k == 1) ? (nx - 1 - dy) : (k == 2) ? (nx - 1 - dx) : dy;
            uint32_t sy = (k == 0) ? dy : (k == 1) ? dx : (k == 2) ? (ny - 1 - dy) : (ny - 1 - dx);
            mc_write_im_pix(out, out_dt, (uint64_t) dy * sz[0] + dx,
                            arith_read_im_pix(src, sdt, (uint64_t) sy * nx + sx));
        }
    }
    mc_out_post(oid);
    (void) (src_tmp && (mc_tmp_release(src_id), 1));
    return 0;
}

typedef struct
{
    const mc_red_src_t *src;
    IMAGE              *out;
    uint8_t             out_dt;
    uint32_t            nx, ny;
    int                 kx, ky;
    const double       *kern;
    int                 box_r;
} conv_box_job_t;

static inline uint32_t clamp_i(int v, uint32_t n)
{
    return (v < 0) ? 0 : ((uint32_t) v >= n) ? (n - 1) : (uint32_t) v;
}

static inline double read_red_src_2d(const mc_red_src_t *s, uint32_t x, uint32_t y)
{
    return arith_read_im_pix(s->im, s->dt, (uint64_t) y * s->ishape[0] + x);
}

static void conv2_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    const conv_box_job_t *j  = (const conv_box_job_t *) ctx;
    int                   rx = j->kx / 2, ry = j->ky / 2;
    for (uint64_t y = lo; y < hi; y++)
    {
        for (uint32_t x = 0; x < j->nx; x++)
        {
            double acc = 0.0;
            for (int dy = -ry; dy <= ry; dy++)
            {
                uint32_t sy   = clamp_i((int) y + dy, j->ny);
                int      ky_i = ry - dy;
                for (int dx = -rx; dx <= rx; dx++)
                {
                    uint32_t sx   = clamp_i((int) x + dx, j->nx);
                    int      kx_i = rx - dx;
                    acc += read_red_src_2d(j->src, sx, sy) * j->kern[ky_i * j->kx + kx_i];
                }
            }
            mc_write_im_pix(j->out, j->out_dt, y * (uint64_t) j->nx + x, acc);
        }
    }
}

static void boxmean_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    const conv_box_job_t *j   = (const conv_box_job_t *) ctx;
    int                   r   = j->box_r;
    double                inv = 1.0 / (double) ((2 * r + 1) * (2 * r + 1));
    for (uint64_t y = lo; y < hi; y++)
    {
        for (uint32_t x = 0; x < j->nx; x++)
        {
            double acc = 0.0;
            for (int dy = -r; dy <= r; dy++)
            {
                uint32_t sy = clamp_i((int) y + dy, j->ny);
                for (int dx = -r; dx <= r; dx++)
                {
                    uint32_t sx = clamp_i((int) x + dx, j->nx);
                    acc += read_red_src_2d(j->src, sx, sy);
                }
            }
            mc_write_im_pix(j->out, j->out_dt, y * (uint64_t) j->nx + x, acc * inv);
        }
    }
}

int mc_exec_conv2_or_boxmean(mc_slot_t          *slot,
                             const mc_expr_t    *rhs,
                             const mc_expr_t    *call_site,
                             const mc_red_src_t *s1,
                             uint8_t             out_dt,
                             imageID             avoid_id,
                             int                 in_unit,
                             char               *err,
                             size_t              errsz)
{
    const char *fn = rhs->u.call.fn->name;
    if (s1->naxis != 2)
    {
        return fs_err(err, errsz, "%s: expected 2D image", fn);
    }
    uint32_t sz[3] = { s1->ishape[0], s1->ishape[1], 1 };

    if (strcmp(fn, "boxmean") == 0)
    {
        double rd = 0.0;
        if (mc_eval_scalar_d(rhs->u.call.args[1], call_site, &rd, err, errsz) != 0)
        {
            return -1;
        }
        long r = llround(rd);
        if (r < 1 || r > 64)
        {
            return fs_err(err, errsz, "boxmean: radius r must be in [1, 64]");
        }
        imageID oid = mc_out_prepare(slot, 2, sz, out_dt, in_unit, err, errsz);
        if (oid < 0)
        {
            return -1;
        }
        conv_box_job_t job = {
            .src    = s1,
            .out    = &milk_data.image[oid],
            .out_dt = out_dt,
            .nx     = sz[0],
            .ny     = sz[1],
            .box_r  = (int) r,
        };
        int nth =
            milk_omp_team_size((uint64_t) sz[0] * (uint64_t) sz[1] * (uint64_t) (2 * r + 1), 65536);
        (void) ((nth <= 1) ? (boxmean_team_cb(&job, 0, 1, 0, sz[1]), 1)
                           : (arith_team_run(nth, sz[1], boxmean_team_cb, &job), 1));
        mc_out_post(oid);
        return 0;
    }

    imageID kid   = -1;
    int     k_tmp = 0;
    if (mc_resolve_or_eval_im(rhs->u.call.args[1], call_site, avoid_id, in_unit, &kid, &k_tmp, err,
                              errsz) != 0)
    {
        return -1;
    }
    const IMAGE *kim = &milk_data.image[kid];
    uint32_t     kx  = kim->md[0].size[0];
    uint32_t     ky  = (kim->md[0].naxis > 1) ? kim->md[0].size[1] : 1;
    if (kim->md[0].naxis != 2 || kx < 1 || kx > 31 || ky < 1 || ky > 31 || (kx & 1) == 0 ||
        (ky & 1) == 0)
    {
        (void) (k_tmp && (mc_tmp_release(kid), 1));
        return fs_err(err, errsz, "conv2: kernel K must be 2D with odd dimensions in [1, 31]");
    }
    double kern[31 * 31];
    for (uint32_t i = 0; i < kx * ky; i++)
    {
        kern[i] = arith_read_im_pix(kim, kim->md[0].datatype, i);
    }
    (void) (k_tmp && (mc_tmp_release(kid), 1));

    imageID oid = mc_out_prepare(slot, 2, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        return -1;
    }
    conv_box_job_t job = {
        .src    = s1,
        .out    = &milk_data.image[oid],
        .out_dt = out_dt,
        .nx     = sz[0],
        .ny     = sz[1],
        .kx     = (int) kx,
        .ky     = (int) ky,
        .kern   = kern,
    };
    int nth = milk_omp_team_size((uint64_t) sz[0] * (uint64_t) sz[1] * (uint64_t) (kx * ky), 65536);
    (void) ((nth <= 1) ? (conv2_team_cb(&job, 0, 1, 0, sz[1]), 1)
                       : (arith_team_run(nth, sz[1], conv2_team_cb, &job), 1));
    mc_out_post(oid);
    return 0;
}

int mc_exec_fft_or_stack(mcalc_prog_t    *prog,
                         mc_ctor_step_t  *cs,
                         imageID          avoid_id,
                         const mc_expr_t *call_site,
                         int              in_unit,
                         char            *err,
                         size_t           errsz)
{
    mc_slot_t       *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs  = call_site ? cs->rhs_expr->u.call.udef_body : cs->rhs_expr;
    const char      *fn   = rhs->u.call.fn->name;

    if (strcmp(fn, "fft") == 0 || strcmp(fn, "ifft") == 0)
    {
        return exec_fft_or_ifft(slot, rhs, avoid_id, call_site, in_unit, err, errsz);
    }
    if (strcmp(fn, "fftshift") == 0)
    {
        return exec_fftshift(slot, rhs, avoid_id, call_site, in_unit, err, errsz);
    }
    if (strcmp(fn, "rot90") == 0)
    {
        return exec_rot90(slot, rhs, avoid_id, call_site, in_unit, err, errsz);
    }
    return exec_stack_or_cat(slot, rhs, avoid_id, call_site, in_unit, err, errsz);
}

int mc_exec_linalg_or_ext(mcalc_prog_t    *prog,
                          mc_ctor_step_t  *cs,
                          imageID          avoid_id,
                          const mc_expr_t *call_site,
                          int              in_unit,
                          char            *err,
                          size_t           errsz)
{
    if (cs->multi_stmt != NULL)
    {
        return mc_exec_multi_linalg_or_ext(prog, cs, cs->multi_stmt, in_unit, NULL, err, errsz);
    }
    const mc_expr_t *rhs = call_site ? cs->rhs_expr->u.call.udef_body : cs->rhs_expr;
    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.ext_id >= 0)
    {
        return mc_exec_ext_ctor(prog, cs, avoid_id, call_site, in_unit, err, errsz);
    }
    if (rhs->kind != MC_EXPR_CALL || rhs->u.call.fn == NULL)
    {
        return mc_exec_blas_call(prog, cs, avoid_id, call_site, in_unit, err, errsz);
    }
    const char *fn = rhs->u.call.fn->name;
    if (strcmp(fn, "fft") == 0 || strcmp(fn, "ifft") == 0 || strcmp(fn, "fftshift") == 0 ||
        strcmp(fn, "hstack") == 0 || strcmp(fn, "vstack") == 0 || strcmp(fn, "dstack") == 0 ||
        strcmp(fn, "cat") == 0 || strcmp(fn, "rot90") == 0)
    {
        return mc_exec_fft_or_stack(prog, cs, avoid_id, call_site, in_unit, err, errsz);
    }
    if (strcmp(fn, "trsv") == 0 || strcmp(fn, "trsm") == 0 || strcmp(fn, "gesv") == 0 ||
        strcmp(fn, "posv") == 0 || strcmp(fn, "gels") == 0 || strcmp(fn, "solve") == 0 ||
        strcmp(fn, "inv") == 0 || strcmp(fn, "pinv") == 0 || strcmp(fn, "svd") == 0 ||
        strcmp(fn, "eigh") == 0)
    {
        return mc_exec_lapack_call(prog, cs, avoid_id, call_site, in_unit, err, errsz);
    }
    return mc_exec_blas_call(prog, cs, avoid_id, call_site, in_unit, err, errsz);
}
