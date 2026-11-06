// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_fft_stack.c
 * @brief FFT, array stacking/concatenation, 90-degree rotation, 2D convolution, and top-level
 *        constructor dispatch for `libmilkcalc`.
 *
 * This module implements the Fourier transform, spatial filtering, array reshaping/concatenation,
 * and top-level linear-algebra/extension constructor dispatcher for `libmilkcalc`:
 *
 * - **1D/2D/3D Discrete Fourier Transforms (`fft`, `ifft`, `fftshift`)**: executes forward and
 *   normalized (`1/N`) inverse complex DFTs using a mutex-protected, ring-buffer cache of 16
 *   FFTW3 guru/unaligned plans (`FFTW_ESTIMATE | FFTW_UNALIGNED`) for both single-precision
 *   (`fftwf`) and double-precision (`fftw`) transforms, with a separable 1D DFT fallback when
 *   compiled without `HAVE_FFTW3`. `fftshift` circularly shifts the zero-frequency component to
 *   the center of the spectrum across all active axes.
 * - **Array stacking, concatenation, and rotation (`hstack`, `vstack`, `dstack`, `cat`, `rot90`)**:
 *   concatenates arbitrary sequences of images along axis 0, 1, or 2 with automatic real/complex
 *   and float32/float64 promotion, and rotates 2D images by `k * 90` degrees.
 * - **2D spatial filtering (`conv2`, `boxmean`)**: executes clamped-boundary 2D kernel convolution
 *   (odd kernel dimensions up to 31x31) and `(2r + 1) x (2r + 1)` box-car averaging in parallel
 *   across image rows via `arith_team_run()`.
 * - **Top-level constructor dispatcher (`mc_exec_linalg_or_ext`)**: routes `MC_STEP_CTOR` plan
 *   steps to multi-LHS handlers, C/FPS extension operators, FFT/stack helpers, LAPACK solvers,
 *   or BLAS routines.
 */

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

/* ============================================================================
 * Error Reporting and FFTW3 Plan Cache
 * ============================================================================ */

/**
 * fs_err - Format a diagnostic message into @err (if still empty) and return -1
 * @err:   Destination error buffer (may be NULL)
 * @errsz: Capacity of @err in bytes
 * @fmt:   printf-style format string
 *
 * Preserves the first error recorded during execution by only writing into @err
 * when `err[0] == '\0'`.
 *
 * Return: Always -1
 */
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
/**
 * struct fft_cache_entry_t - Cached FFTW3/FFTW3F out-of-place DFT plan entry
 * @naxis:  Transform dimensionality (1, 2, or 3)
 * @sz:     Transform dimensions `[nx, ny, nz]` in ImageStreamIO column-major order
 * @is_dbl: Non-zero for double-precision (`fftw_plan`), 0 for single-precision (`fftwf_plan`)
 * @sign:   Transform direction (`FFTW_FORWARD` = -1 or `FFTW_BACKWARD` = +1)
 * @plan:   Opaque `fftw_plan` or `fftwf_plan` handle created with `FFTW_UNALIGNED`
 */
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

/**
 * run_fftw_cached - Look up or create a cached FFTW3 plan and execute it on (@in_buf, @out_buf)
 * @naxis:   Transform dimensionality (1, 2, or 3)
 * @sz:      Transform dimensions `[nx, ny, nz]` (fast axis `nx = sz[0]`, slow axis `nz = sz[2]`)
 * @is_dbl:  Non-zero for `complex_double`, 0 for `complex_float`
 * @sign:    `-1` for forward DFT, `+1` for inverse DFT
 * @in_buf:  Input interleaved complex buffer
 * @out_buf: Output interleaved complex buffer
 *
 * Because FFTW planner routines are not thread-safe and plan creation is costly
 * inside real-time loops, plans are created with `FFTW_ESTIMATE | FFTW_UNALIGNED`
 * under `g_fft_lock`, stored in a 16-slot FIFO ring cache, and executed via the
 * new-array interface `fftw_execute_dft()`. Note that ImageStreamIO dimensions
 * `(nx, ny, nz)` map to row-major FFTW dimensions `(ny, nx)` in 2D and `(nz, ny, nx)`
 * in 3D.
 *
 * Return: 0 on success, -1 if FFTW plan creation failed
 */
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
                if (g_fft_cache[slot].is_dbl)
                {
                    fftw_destroy_plan((fftw_plan) g_fft_cache[slot].plan);
                }
                else
                {
                    fftwf_destroy_plan((fftwf_plan) g_fft_cache[slot].plan);
                }
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
        if (is_dbl)
        {
            fftw_execute_dft((fftw_plan) plan, (fftw_complex *) in_buf, (fftw_complex *) out_buf);
        }
        else
        {
            fftwf_execute_dft((fftwf_plan) plan, (fftwf_complex *) in_buf,
                              (fftwf_complex *) out_buf);
        }
    }
    pthread_mutex_unlock(&g_fft_lock);
    return (plan != NULL) ? 0 : -1;
}
#endif

/* ============================================================================
 * Fallback 1D DFT and Complex Pixel Accessors
 * ============================================================================ */

/**
 * dft_1d_line - Compute a 1D complex Discrete Fourier Transform of length @N (non-FFTW fallback)
 * @N:      Number of samples along the 1D line
 * @sign:   Transform exponent sign (`-1` for forward DFT, `+1` for inverse DFT)
 * @in_re:  Input real-part array of length @N
 * @in_im:  Input imaginary-part array of length @N
 * @out_re: Output real-part array of length @N
 * @out_im: Output imaginary-part array of length @N
 *
 * Return: void
 */
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
            double c  = cos(th);
            double s  = sin(th);
            sr += in_re[n] * c - in_im[n] * s;
            si += in_re[n] * s + in_im[n] * c;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

/**
 * read_pix_any - Read element @idx of real or complex image @im as a `(re, im_part)` double pair
 * @im:      Source image
 * @dt:      ImageStreamIO datatype code of @im
 * @idx:     Linear 0-based element index
 * @re:      Output real component
 * @im_part: Output imaginary component (set to 0.0 for real datatypes)
 *
 * Return: void
 */
static void read_pix_any(const IMAGE *im, uint8_t dt, uint64_t idx, double *re, double *im_part)
{
    *re      = (dt == _DATATYPE_COMPLEX_FLOAT)    ? (double) im->array.CF[idx].re
               : (dt == _DATATYPE_COMPLEX_DOUBLE) ? im->array.CD[idx].re
                                                  : arith_read_im_pix(im, dt, idx);
    *im_part = (dt == _DATATYPE_COMPLEX_FLOAT)    ? (double) im->array.CF[idx].im
               : (dt == _DATATYPE_COMPLEX_DOUBLE) ? im->array.CD[idx].im
                                                  : 0.0;
}

/**
 * write_pix_any - Write a `(re, im_part)` double pair to element @idx of image @out
 * @out:     Destination image
 * @dt:      ImageStreamIO datatype code of @out
 * @idx:     Linear 0-based element index
 * @re:      Real component to write
 * @im_part: Imaginary component to write (ignored if @dt is a real datatype)
 *
 * Return: void
 */
static void write_pix_any(IMAGE *out, uint8_t dt, uint64_t idx, double re, double im_part)
{
    if (dt == _DATATYPE_COMPLEX_FLOAT)
    {
        out->array.CF[idx].re = (float) re;
        out->array.CF[idx].im = (float) im_part;
    }
    else if (dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        out->array.CD[idx].re = re;
        out->array.CD[idx].im = im_part;
    }
    else
    {
        mc_write_im_pix(out, dt, idx, re);
    }
}

/* ============================================================================
 * FFT, IFFT, FFTShift, Stacking, Concatenation, and 90-Degree Rotation
 * ============================================================================ */

/**
 * exec_fft_or_ifft - Execute `fft(im)` or `ifft(im)` on a 1D, 2D, or 3D image into @slot
 * @slot:      Destination slot to receive the complex frequency/spatial output image
 * @rhs:       Call expression node for `fft` or `ifft`
 * @avoid_id:  Image ID to avoid clobbering when evaluating the input sub-expression
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Promotes `float64` / `complex_double` inputs to `_DATATYPE_COMPLEX_DOUBLE` and
 * all other types to `_DATATYPE_COMPLEX_FLOAT`. For `ifft`, normalizes the output
 * by `1 / nelement`.
 *
 * Return: 0 on success, -1 on rank error, allocation failure, or FFT execution failure
 */
static int exec_fft_or_ifft(mc_slot_t       *slot,
                            const mc_expr_t *rhs,
                            imageID          avoid_id,
                            const mc_expr_t *call_site,
                            int              in_unit,
                            char            *err,
                            size_t           errsz)
{
    int     is_ifft = (strcmp(rhs->u.call.fn->name, "ifft") == 0);
    int     src_tmp = 0, rc = 0;
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
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
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
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
        return -1;
    }
    IMAGE *out  = &milk_data.image[oid];
    int    sign = is_ifft ? 1 : -1;

#ifdef HAVE_FFTW3
    size_t elem_sz = is_dbl ? (2 * sizeof(double)) : (2 * sizeof(float));
    void  *in_buf  = malloc(elem_sz * (size_t) (nel > 0 ? nel : 1));
    if (in_buf == NULL)
    {
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
        return fs_err(err, errsz, "%s: out of memory", rhs->u.call.fn->name);
    }
    for (uint64_t i = 0; i < nel; i++)
    {
        double re = 0.0, im_p = 0.0;
        read_pix_any(src, sdt, i, &re, &im_p);
        if (is_dbl)
        {
            ((double *) in_buf)[2 * i]     = re;
            ((double *) in_buf)[2 * i + 1] = im_p;
        }
        else
        {
            ((float *) in_buf)[2 * i]     = (float) re;
            ((float *) in_buf)[2 * i + 1] = (float) im_p;
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
                out->array.CD[i].re *= inv_n;
                out->array.CD[i].im *= inv_n;
            }
            else
            {
                out->array.CF[i].re *= (float) inv_n;
                out->array.CF[i].im *= (float) inv_n;
            }
        }
    }
    free(in_buf);
#else
    double  *re    = calloc((size_t) nel, sizeof(double));
    double  *im    = calloc((size_t) nel, sizeof(double));
    uint32_t max_d = (sz[0] > sz[1]) ? sz[0] : sz[1];
    max_d          = (sz[2] > max_d) ? sz[2] : max_d;
    double *tr     = calloc((size_t) max_d, sizeof(double));
    double *ti     = calloc((size_t) max_d, sizeof(double));
    double *or_b   = calloc((size_t) max_d, sizeof(double));
    double *oi_b   = calloc((size_t) max_d, sizeof(double));
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
                tr[y]        = re[idx];
                ti[y]        = im[idx];
            }
            dft_1d_line(sz[1], sign, tr, ti, or_b, oi_b);
            for (uint32_t y = 0; y < sz[1]; y++)
            {
                uint64_t idx = ((uint64_t) z * sz[1] + y) * sz[0] + x;
                re[idx]      = or_b[y];
                im[idx]      = oi_b[y];
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
                tr[z]        = re[idx];
                ti[z]        = im[idx];
            }
            dft_1d_line(sz[2], sign, tr, ti, or_b, oi_b);
            for (uint32_t z = 0; z < sz[2]; z++)
            {
                uint64_t idx = ((uint64_t) z * sz[1] + y) * sz[0] + x;
                re[idx]      = or_b[z];
                im[idx]      = oi_b[z];
            }
        }
    }
    double scale = is_ifft ? (1.0 / (double) nel) : 1.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        write_pix_any(out, out_dt, i, re[i] * scale, im[i] * scale);
    }
    free(re);
    free(im);
    free(tr);
    free(ti);
    free(or_b);
    free(oi_b);
#endif
    (void) dft_1d_line;
    if (rc == 0)
    {
        mc_out_post(oid);
    }
    if (src_tmp)
    {
        mc_tmp_release(src_id);
    }
    return rc;
}

/**
 * exec_fftshift - Execute `fftshift(im)` to center the DC component along all active axes
 * @slot:      Destination slot to receive the shifted image
 * @rhs:       Call expression node for `fftshift`
 * @avoid_id:  Image ID to avoid clobbering when evaluating the input sub-expression
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Maps destination coordinate `d` along each active axis of length `N` from source
 * coordinate `(d + ceil(N / 2)) % N` (matching NumPy `fftshift` convention).
 * Preserves the input datatype (including complex types) when the output slot allows.
 *
 * Return: 0 on success, -1 on failure
 */
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
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
        return -1;
    }
    IMAGE   *out = &milk_data.image[oid];
    uint32_t hx  = (sz[0] + 1) / 2;
    uint32_t hy  = (naxis > 1) ? ((sz[1] + 1) / 2) : 0;
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
    if (src_tmp)
    {
        mc_tmp_release(src_id);
    }
    return 0;
}

/**
 * exec_stack_or_cat - Execute `hstack(...)`, `vstack(...)`, `dstack(...)`, or `cat(axis, ...)`
 * @slot:      Destination slot to receive the concatenated tensor
 * @rhs:       Call expression node
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Concatenates `nin` input images along `axis` (`0` for `hstack`, `1` for `vstack`,
 * `2` for `dstack`, or user-specified `0..2` for `cat`). Validates that all
 * non-concatenated dimensions match across all inputs and promotes the output
 * datatype to accommodate complex or 64-bit inputs.
 *
 * Return: 0 on success, -1 on invalid axis or non-concatenated dimension mismatch
 */
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

    /* Validate compatibility of non-concatenated axes and accumulate size along @axis */
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
        if (is_tmp[i] && ids[i] >= 0)
        {
            mc_tmp_release(ids[i]);
        }
    }
    return rc;
}

/**
 * exec_rot90 - Execute `rot90(im[, k])` to rotate a 2D image by `k * 90` degrees
 * @slot:      Destination slot to receive the rotated 2D image
 * @rhs:       Call expression node for `rot90`
 * @avoid_id:  Image ID to avoid clobbering when evaluating the input sub-expression
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Reduces `k` modulo 4 (supporting negative rotation counts) and remaps `(dx, dy)`
 * to source coordinates `(sx, sy)`, swapping `nx` and `ny` when `k` is odd.
 *
 * Return: 0 on success, -1 if input is not a 2D image or evaluation fails
 */
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
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
        return fs_err(err, errsz, "rot90: expected 2D image");
    }
    uint32_t nx    = src->md[0].size[0];
    uint32_t ny    = src->md[0].size[1];
    uint32_t sz[3] = { (k % 2 == 1) ? ny : nx, (k % 2 == 1) ? nx : ny, 1 };
    uint8_t  sdt   = src->md[0].datatype;
    uint8_t  out_dt =
        mc_select_ctor_out_dt(slot, (sdt == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    imageID oid = mc_out_prepare(slot, 2, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
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
    if (src_tmp)
    {
        mc_tmp_release(src_id);
    }
    return 0;
}

/* ============================================================================
 * 2D Convolution (conv2) and Box-Car Averaging (boxmean)
 * ============================================================================ */

/**
 * struct conv_box_job_t - Thread-team work descriptor for `conv2` and `boxmean`
 * @src:    Resolved 2D input source descriptor
 * @out:    Destination 2D output image
 * @out_dt: Output ImageStreamIO datatype code
 * @nx:     Image width (axis 0)
 * @ny:     Image height (axis 1)
 * @kx:     Kernel width for `conv2` (odd, `1..31`)
 * @ky:     Kernel height for `conv2` (odd, `1..31`)
 * @kern:   Flattened `kx * ky` convolution kernel weights (`NULL` for `boxmean`)
 * @box_r:  Half-window radius `r` for `(2r + 1) x (2r + 1)` `boxmean`
 */
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

/**
 * clamp_i - Clamp signed coordinate @v to the valid index range `[0, n - 1]`
 * @v: Signed pixel coordinate (may be negative or `>= n` near borders)
 * @n: Axis length (`> 0`)
 *
 * Return: Clamped unsigned coordinate in `[0, n - 1]`
 */
static inline uint32_t clamp_i(int v, uint32_t n)
{
    return (v < 0) ? 0 : ((uint32_t) v >= n) ? (n - 1) : (uint32_t) v;
}

/**
 * read_red_src_2d - Read pixel `(x, y)` from a 2D reduction source @s as `double`
 * @s: 2D reduction source descriptor
 * @x: Column coordinate (`0 <= x < s->ishape[0]`)
 * @y: Row coordinate (`0 <= y < s->ishape[1]`)
 *
 * Return: Pixel value at `(x, y)` promoted to `double`
 */
static inline double read_red_src_2d(const mc_red_src_t *s, uint32_t x, uint32_t y)
{
    return arith_read_im_pix(s->im, s->dt, (uint64_t) y * s->ishape[0] + x);
}

/**
 * conv2_team_cb - Worker callback computing 2D convolution over row range `[lo, hi)`
 * @ctx: Pointer to the shared `conv_box_job_t` descriptor
 * @tid: Thread index within the team (unused)
 * @nth: Total number of threads in the team (unused)
 * @lo:  First output row `y` (inclusive) assigned to this worker
 * @hi:  Last output row `y` (exclusive) assigned to this worker
 *
 * Evaluates true 2D convolution (with flipped kernel indices `rx - dx, ry - dy`)
 * and edge-clamped boundary conditions.
 *
 * Return: void
 */
static void conv2_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    const conv_box_job_t *j  = (const conv_box_job_t *) ctx;
    int                   rx = j->kx / 2;
    int                   ry = j->ky / 2;
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

/**
 * boxmean_team_cb - Worker callback computing `(2r + 1) x (2r + 1)` box average over `[lo, hi)`
 * @ctx: Pointer to the shared `conv_box_job_t` descriptor
 * @tid: Thread index within the team (unused)
 * @nth: Total number of threads in the team (unused)
 * @lo:  First output row `y` (inclusive) assigned to this worker
 * @hi:  Last output row `y` (exclusive) assigned to this worker
 *
 * Uses edge-clamped boundary coordinates matching `conv2`.
 *
 * Return: void
 */
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

/**
 * mc_exec_conv2_or_boxmean - Execute `conv2(im, K)` or `boxmean(im, r)` into @slot
 * @slot:      Destination slot to receive the filtered 2D image
 * @rhs:       Call expression node for `conv2` or `boxmean`
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @s1:        Resolved first argument (2D input image descriptor)
 * @out_dt:    Selected output datatype (`_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`)
 * @avoid_id:  Image ID to avoid clobbering when evaluating kernel argument `K`
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Validates that `@s1` is 2D, checks radius `r` in `[1, 64]` for `boxmean` or odd
 * kernel dimensions in `[1, 31]` for `conv2`, and dispatches row-parallel worker
 * teams via `arith_team_run()`.
 *
 * Return: 0 on success, -1 on validation or output preparation failure
 */
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
        if (nth <= 1)
        {
            boxmean_team_cb(&job, 0, 1, 0, sz[1]);
        }
        else
        {
            arith_team_run(nth, sz[1], boxmean_team_cb, &job);
        }
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
        if (k_tmp)
        {
            mc_tmp_release(kid);
        }
        return fs_err(err, errsz, "conv2: kernel K must be 2D with odd dimensions in [1, 31]");
    }
    double kern[31 * 31];
    for (uint32_t i = 0; i < kx * ky; i++)
    {
        kern[i] = arith_read_im_pix(kim, kim->md[0].datatype, i);
    }
    if (k_tmp)
    {
        mc_tmp_release(kid);
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
        .kx     = (int) kx,
        .ky     = (int) ky,
        .kern   = kern,
    };
    int nth = milk_omp_team_size((uint64_t) sz[0] * (uint64_t) sz[1] * (uint64_t) (kx * ky), 65536);
    if (nth <= 1)
    {
        conv2_team_cb(&job, 0, 1, 0, sz[1]);
    }
    else
    {
        arith_team_run(nth, sz[1], conv2_team_cb, &job);
    }
    mc_out_post(oid);
    return 0;
}

/* ============================================================================
 * Public Constructor Dispatchers
 * ============================================================================ */

/**
 * mc_exec_fft_or_stack - Dispatch FFT (`fft`, `ifft`, `fftshift`), rotation, and stacking calls
 * @prog:      Compiled calculator program holding the slot table
 * @cs:        Constructor step descriptor to execute
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Return: 0 on success, -1 on error
 */
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

/**
 * mc_exec_linalg_or_ext - Top-level router for `MC_FN_LINALG`, algebraic BLAS, and extension steps
 * @prog:      Compiled calculator program holding the slot table
 * @cs:        Constructor step descriptor to execute
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @call_site: Enclosing user-defined function call site (or NULL)
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Routes:
 * 1. Multi-LHS statements (`cs->multi_stmt != NULL`) to `mc_exec_multi_linalg_or_ext()`
 * 2. Registered C/FPS extension calls (`ext_id >= 0`) to `mc_exec_ext_ctor()`
 * 3. Algebraic BLAS expressions (`rhs->kind != MC_EXPR_CALL`) to `mc_exec_blas_call()`
 * 4. Fourier and array-stacking built-ins to `mc_exec_fft_or_stack()`
 * 5. LAPACK solvers and decompositions to `mc_exec_lapack_call()`
 * 6. Explicit BLAS calls and `einsum` to `mc_exec_blas_call()`
 *
 * Return: 0 on success, -1 on execution error
 */
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
