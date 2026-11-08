// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_fft_stack.c
 * @brief FFT, array stacking/concatenation, 90-degree rotation, and top-level constructor
 *        dispatch for `libmilkcalc`.
 *
 * This module implements the Fourier transform, array reshaping/concatenation,
 * and top-level linear-algebra/extension constructor dispatcher for `libmilkcalc`:
 *
 * - **1D/2D/3D Discrete Fourier Transforms (`fft`, `ifft`, `fftshift`)**: executes forward and
 *   normalized (`1/N`) inverse DFTs with a 16-slot cache of FFTW3 plans (`FFTW_MEASURE` in compute
 *   units, `FFTW_ESTIMATE` at the prompt) in single (`fftwf`) or double (`fftw`) precision: real-to-complex for real input (half the work), in
 *   64-byte aligned scratch (SIMD), threaded on the calculator's thread team when the FFTW threads
 *   libraries are present; a separable 1D DFT fallback when compiled without `HAVE_FFTW3`. `fftshift` circularly shifts the zero-frequency component to
 *   the center of the spectrum across all active axes.
 * - **Array stacking, concatenation, and rotation (`hstack`, `vstack`, `dstack`, `cat`, `rot90`)**:
 *   concatenates arbitrary sequences of images along axis 0, 1, or 2 with automatic real/complex
 *   and float32/float64 promotion, and rotates 2D images by `k * 90` degrees.
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
#include "COREMOD_arith/image_elem_block.h"
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
 * struct fft_cache_entry_t - Cached FFTW3/FFTW3F plan
 * @naxis:  Transform rank (1, 2, or 3)
 * @sz:     Transform dimensions `[nx, ny, nz]` in ImageStreamIO order
 * @is_dbl: Non-zero for double precision (`fftw_plan`), 0 for single precision (`fftwf_plan`)
 * @kind:   `FFT_R2C` (forward, real input), `FFT_FWD` or `FFT_INV` (in-place complex)
 * @nth:    Number of threads the plan runs on
 * @measure: Planned with `FFTW_MEASURE` (a unit's), else `FFTW_ESTIMATE`
 * @plan:   Opaque `fftw_plan` or `fftwf_plan`, made for 64-byte aligned buffers
 */
typedef struct
{
    uint8_t  naxis;
    uint32_t sz[3];
    int      is_dbl;
    int      kind;
    int      nth;
    int      measure;
    void    *plan;
} fft_cache_entry_t;

enum
{
    FFT_R2C,
    FFT_FWD,
    FFT_INV
};

#    define FFT_CACHE_CAP 16
static fft_cache_entry_t g_fft_cache[FFT_CACHE_CAP];
static int               g_fft_cache_n    = 0;
static int               g_fft_cache_next = 0;
/* FFTW's planner is not thread-safe, and plans run on the calculator's thread team, which serves
 * one caller at a time: plans are made and executed under this lock. */
static pthread_mutex_t g_fft_lock = PTHREAD_MUTEX_INITIALIZER;

#    ifdef HAVE_FFTW3_THREADS
/**
 * struct fft_loop_t - One FFTW parallel loop: `work(jobdata + i * elsize)` for each job `i`
 * @work:    FFTW job function
 * @jobdata: First job's data
 * @elsize:  Size of one job's data
 */
typedef struct
{
    void *(*work)(char *);
    char  *jobdata;
    size_t elsize;
} fft_loop_t;

/* Set while this thread runs FFTW jobs: a loop FFTW nests in a job runs here, since the team
 * serves one caller at a time */
static __thread int t_in_fft_job;

/** fft_loop_job - Team job: run FFTW jobs [@lo, @hi). */
static void fft_loop_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const fft_loop_t *l = (const fft_loop_t *) ctx;
    (void) tid;
    (void) nth;
    t_in_fft_job = 1;
    for (uint64_t i = lo; i < hi; i++)
    {
        l->work(l->jobdata + i * l->elsize);
    }
    t_in_fft_job = 0;
}

/** fft_parallel_loop - FFTW threads callback: run @njobs FFTW jobs on the calculator's team. */
static void fft_parallel_loop(void *(*work)(char *),
                              char  *jobdata,
                              size_t elsize,
                              int    njobs,
                              void  *data)
{
    fft_loop_t l = { work, jobdata, elsize };
    (void) data;
    if (t_in_fft_job)
    {
        fft_loop_job(&l, 0, 1, 0, (uint64_t) njobs);
        t_in_fft_job = 1;
        return;
    }
    arith_team_run(njobs, (uint64_t) njobs, fft_loop_job, &l);
}

static pthread_once_t g_fft_threads_once = PTHREAD_ONCE_INIT;

/** fft_threads_init - Make FFTW's threaded plans use the calculator's thread team. */
static void fft_threads_init(void)
{
    fftw_init_threads();
    fftwf_init_threads();
    fftw_threads_set_callback(fft_parallel_loop, NULL);
    fftwf_threads_set_callback(fft_parallel_loop, NULL);
}
#    endif

/**
 * fft_plan - Look up or make the plan of a transform on (@in, @out)
 * @naxis:  Transform rank (1, 2, or 3)
 * @sz:     Transform dimensions `[nx, ny, nz]` (fast axis `nx = sz[0]`)
 * @is_dbl: Non-zero for double precision
 * @kind:   `FFT_R2C`, `FFT_FWD` or `FFT_INV`
 * @nth:    Number of threads
 * @measure: Non-zero in a compute unit: plan with `FFTW_MEASURE` (it times candidate algorithms on
 *          scratch buffers, up to seconds once per shape), else `FFTW_ESTIMATE` (instant)
 * @in:     Input buffer (64-byte aligned)
 * @out:    Output buffer (64-byte aligned; @in for the in-place complex kinds)
 *
 * Context: caller holds `g_fft_lock`. Plans are kept in a 16-slot FIFO cache; a measured plan also
 * serves the prompt, an estimated one never serves a unit. ImageStreamIO `(nx, ny, nz)` is FFTW's
 * row-major `(ny, nx)` or `(nz, ny, nx)`.
 *
 * Return: The plan, or NULL if FFTW could not make it
 */
static void *fft_plan(uint8_t        naxis,
                      const uint32_t sz[3],
                      int            is_dbl,
                      int            kind,
                      int            nth,
                      int            measure,
                      void          *in,
                      void          *out)
{
    for (int i = 0; i < g_fft_cache_n; i++)
    {
        const fft_cache_entry_t *e = &g_fft_cache[i];
        if (e->naxis == naxis && memcmp(e->sz, sz, sizeof(e->sz)) == 0 && e->is_dbl == is_dbl &&
            e->kind == kind && e->nth == nth && e->measure >= measure)
        {
            return e->plan;
        }
    }
    /* FFTW_MEASURE overwrites the arrays it plans on: plan on scratch of the same sizes and
     * 64-byte alignment, which the plan then runs on the caller's buffers with */
    void *scratch = NULL;
    if (measure)
    {
        size_t   el    = is_dbl ? sizeof(double) : sizeof(float);
        uint64_t nel   = (uint64_t) sz[0] * sz[1] * sz[2];
        size_t   a_in  = ((size_t) nel * el * (kind == FFT_R2C ? 1 : 2) + 63) & ~(size_t) 63;
        size_t   n_out = (kind == FFT_R2C) ? 2 * (size_t) (sz[0] / 2 + 1) * sz[1] * sz[2] * el : 0;
        if (posix_memalign(&scratch, 64, a_in + n_out) != 0)
        {
            return NULL;
        }
        in  = scratch;
        out = (kind == FFT_R2C) ? (char *) scratch + a_in : scratch;
    }
    unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;
    int      dims[3];
    for (int a = 0; a < naxis; a++)
    {
        dims[a] = (int) sz[naxis - 1 - a];
    }
#    ifdef HAVE_FFTW3_THREADS
    pthread_once(&g_fft_threads_once, fft_threads_init);
    fftw_plan_with_nthreads(nth);
    fftwf_plan_with_nthreads(nth);
#    endif
    int   sign = (kind == FFT_INV) ? FFTW_BACKWARD : FFTW_FORWARD;
    void *plan = NULL;
    if (is_dbl)
    {
        plan = (kind == FFT_R2C) ? (void *) fftw_plan_dft_r2c(naxis, dims, (double *) in,
                                                              (fftw_complex *) out, flags)
                                 : (void *) fftw_plan_dft(naxis, dims, (fftw_complex *) in,
                                                          (fftw_complex *) out, sign, flags);
    }
    else
    {
        plan = (kind == FFT_R2C) ? (void *) fftwf_plan_dft_r2c(naxis, dims, (float *) in,
                                                               (fftwf_complex *) out, flags)
                                 : (void *) fftwf_plan_dft(naxis, dims, (fftwf_complex *) in,
                                                           (fftwf_complex *) out, sign, flags);
    }
    free(scratch);
    if (plan == NULL)
    {
        return NULL;
    }
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
        .naxis   = naxis,
        .sz      = { sz[0], sz[1], sz[2] },
        .is_dbl  = is_dbl,
        .kind    = kind,
        .nth     = nth,
        .measure = measure,
        .plan    = plan,
    };
    return plan;
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

#ifdef HAVE_FFTW3
/**
 * fft_load - Copy @nel pixels of @src into transform buffer @buf: reals (@real) or complex pairs
 * @src:    Source image (any real or complex type)
 * @sdt:    Datatype of @src
 * @buf:    Destination buffer (float or double elements)
 * @nel:    Number of pixels
 * @is_dbl: Non-zero for double elements
 * @real:   Non-zero to copy reals (real-to-complex input), else complex pairs
 *
 * Return: void
 */
static void fft_load(const IMAGE *src, uint8_t sdt, void *buf, uint64_t nel, int is_dbl, int real)
{
    if (real)
    {
        for (uint64_t off = 0; off < nel; off += ARITH_ELEM_BLK)
        {
            int len = (int) ((nel - off < ARITH_ELEM_BLK) ? nel - off : ARITH_ELEM_BLK);
            if (is_dbl)
            {
                arith_load_block_d_inline(src, off, len, (double *) buf + off);
            }
            else
            {
                arith_load_block_f_inline(src, off, len, (float *) buf + off);
            }
        }
        return;
    }
    if (sdt == (is_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT))
    {
        memcpy(buf, src->array.raw, 2 * (size_t) nel * (is_dbl ? sizeof(double) : sizeof(float)));
        return;
    }
    for (uint64_t i = 0; i < nel; i++)
    {
        double re = 0.0, im_p = 0.0;
        read_pix_any(src, sdt, i, &re, &im_p);
        if (is_dbl)
        {
            ((double *) buf)[2 * i]     = re;
            ((double *) buf)[2 * i + 1] = im_p;
        }
        else
        {
            ((float *) buf)[2 * i]     = (float) re;
            ((float *) buf)[2 * i + 1] = (float) im_p;
        }
    }
}

/**
 * struct fft_store_t - Copy of a transform result into the complex output image
 * @out:    Output image pixels (complex float or complex double)
 * @buf:    Transform result: half spectrum `(nz, ny, nx / 2 + 1)` (@r2c), else full spectrum
 * @nx:     X size
 * @ny:     Y size
 * @nz:     Z size
 * @is_dbl: Non-zero for double precision
 * @r2c:    Non-zero when @buf holds the half spectrum of a real input
 * @scale:  Factor applied when copying a full spectrum (`1 / N` for `ifft`)
 */
typedef struct
{
    void       *out;
    const void *buf;
    int64_t     nx;
    int64_t     ny;
    int64_t     nz;
    int         is_dbl;
    int         r2c;
    double      scale;
} fft_store_t;

/** FFT_STORE_ROW - write output row @r: a scaled copy, or the half spectrum and its mirror. */
#    define FFT_STORE_ROW(FT)                                                               \
        {                                                                                   \
            FT *o = (FT *) s->out + 2 * (int64_t) r * nx;                                   \
            if (!s->r2c)                                                                    \
            {                                                                               \
                const FT *b = (const FT *) s->buf + 2 * (int64_t) r * nx;                   \
                const FT  k = (FT) s->scale;                                                \
                _Pragma("omp simd") for (int64_t i = 0; i < 2 * nx; i++)                    \
                {                                                                           \
                    o[i] = k * b[i];                                                        \
                }                                                                           \
            }                                                                               \
            else                                                                            \
            {                                                                               \
                const int64_t z = (int64_t) r / ny;                                         \
                const int64_t y = (int64_t) r % ny;                                         \
                const FT     *h = (const FT *) s->buf + 2 * (int64_t) r * nhx;              \
                const FT     *mr =                                                          \
                    (const FT *) s->buf + 2 * (((nz - z) % nz) * ny + (ny - y) % ny) * nhx; \
                memcpy(o, h, 2 * (size_t) nhx * sizeof(FT));                                \
                for (int64_t kx = nhx; kx < nx; kx++)                                       \
                {                                                                           \
                    o[2 * kx]     = mr[2 * (nx - kx)];                                      \
                    o[2 * kx + 1] = -mr[2 * (nx - kx) + 1];                                 \
                }                                                                           \
            }                                                                               \
        }

/**
 * fft_store_job - Team job writing output rows [@lo, @hi) (Hermitian symmetry for real input:
 * X[kz][ky][kx] = conj(X[-kz][-ky][-kx]))
 */
static void fft_store_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const fft_store_t *s   = (const fft_store_t *) ctx;
    const int64_t      nx  = s->nx;
    const int64_t      ny  = s->ny;
    const int64_t      nz  = s->nz;
    const int64_t      nhx = nx / 2 + 1;
    (void) tid;
    (void) nth;
    for (uint64_t r = lo; r < hi; r++)
    {
        if (s->is_dbl)
        {
            FFT_STORE_ROW(double)
        }
        else
        {
            FFT_STORE_ROW(float)
        }
    }
}
#    undef FFT_STORE_ROW
#endif

/* ============================================================================
 * FFT, IFFT, FFTShift, Stacking, Concatenation, and 90-Degree Rotation
 * ============================================================================ */

/**
 * exec_fft_or_ifft - Execute `fft(im)` or `ifft(im)` on a 1D, 2D, or 3D image into @slot
 * @slot:      Destination slot to receive the complex frequency/spatial output image
 * @cs:        Constructor step (`fft` or `ifft` call; its scratch buffer is reused across frames)
 * @avoid_id:  Image ID to avoid clobbering when evaluating the input sub-expression
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Promotes `float64` / `complex_double` inputs to `_DATATYPE_COMPLEX_DOUBLE` and
 * all other types to `_DATATYPE_COMPLEX_FLOAT`. For `ifft`, normalizes the output
 * by `1 / nelement`. An existing real shared stream cannot take the complex result.
 *
 * Return: 0 on success, -1 on rank error, allocation failure, or FFT execution failure
 */
static int exec_fft_or_ifft(mc_slot_t      *slot,
                            mc_ctor_step_t *cs,
                            imageID         avoid_id,
                            int             in_unit,
                            char           *err,
                            size_t          errsz)
{
    const mc_expr_t *rhs = cs->rhs_expr;

    int     is_ifft = (strcmp(rhs->u.call.fn->name, "ifft") == 0);
    int     src_tmp = 0, rc = 0;
    imageID src_id = -1;
    if (mc_resolve_or_eval_im(rhs->u.call.args[0], avoid_id, in_unit, &src_id, &src_tmp, err,
                              errsz) != 0)
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
    if (out_dt != _DATATYPE_COMPLEX_FLOAT && out_dt != _DATATYPE_COMPLEX_DOUBLE)
    {
        if (src_tmp)
        {
            mc_tmp_release(src_id);
        }
        return fs_err(err, errsz, "%s: '%s' is a real stream; the result is complex",
                      rhs->u.call.fn->name, slot->bare);
    }

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
    /* Real input: a real-to-complex transform of half the spectrum, mirrored into the other half;
     * complex input and ifft: an in-place complex transform. Both run in 64-byte aligned scratch,
     * kept by the step, so FFTW's SIMD code applies. */
    (void) sign;
    int      r2c  = !is_ifft && sdt != _DATATYPE_COMPLEX_FLOAT && sdt != _DATATYPE_COMPLEX_DOUBLE;
    size_t   el   = is_dbl ? sizeof(double) : sizeof(float);
    uint64_t nh   = (uint64_t) (sz[0] / 2 + 1) * sz[1] * sz[2];
    size_t   a_in = ((size_t) nel * el * (r2c ? 1 : 2) + 63) & ~(size_t) 63;
    size_t   need = a_in + (r2c ? 2 * (size_t) nh * el : 0) + 64;
    if (cs->scratch_cap < need)
    {
        double *nb = realloc(cs->scratch_buf, need);
        if (nb == NULL)
        {
            if (src_tmp)
            {
                mc_tmp_release(src_id);
            }
            return fs_err(err, errsz, "%s: out of memory", rhs->u.call.fn->name);
        }
        cs->scratch_buf = nb;
        cs->scratch_cap = need;
    }
    char *in   = (char *) (((uintptr_t) cs->scratch_buf + 63) & ~(uintptr_t) 63);
    char *res  = r2c ? in + a_in : in;
    int   nth  = milk_omp_team_size(nel, 65536);
    int   kind = r2c ? FFT_R2C : (is_ifft ? FFT_INV : FFT_FWD);
    fft_load(src, sdt, in, nel, is_dbl, r2c);

    pthread_mutex_lock(&g_fft_lock);
    void *plan = fft_plan(naxis, sz, is_dbl, kind, nth, in_unit != 0, in, res);
    if (plan != NULL && is_dbl)
    {
        if (r2c)
        {
            fftw_execute_dft_r2c((fftw_plan) plan, (double *) in, (fftw_complex *) res);
        }
        else
        {
            fftw_execute_dft((fftw_plan) plan, (fftw_complex *) in, (fftw_complex *) res);
        }
    }
    else if (plan != NULL)
    {
        if (r2c)
        {
            fftwf_execute_dft_r2c((fftwf_plan) plan, (float *) in, (fftwf_complex *) res);
        }
        else
        {
            fftwf_execute_dft((fftwf_plan) plan, (fftwf_complex *) in, (fftwf_complex *) res);
        }
    }
    pthread_mutex_unlock(&g_fft_lock);
    if (plan == NULL)
    {
        rc = fs_err(err, errsz, "%s: FFTW cannot plan this transform", rhs->u.call.fn->name);
    }
    else
    {
        fft_store_t st = { is_dbl ? (void *) out->array.CD : (void *) out->array.CF,
                           res,
                           sz[0],
                           sz[1],
                           sz[2],
                           is_dbl,
                           r2c,
                           is_ifft ? 1.0 / (double) nel : 1.0 };
        arith_team_run(nth, (uint64_t) sz[1] * sz[2], fft_store_job, &st);
    }
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
                         int              in_unit,
                         char            *err,
                         size_t           errsz)
{
    imageID src_id  = -1;
    int     src_tmp = 0;
    if (mc_resolve_or_eval_im(rhs->u.call.args[0], avoid_id, in_unit, &src_id, &src_tmp, err,
                              errsz) != 0)
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
        if (mc_eval_scalar_d(rhs->u.call.args[0], &axd, err, errsz) != 0)
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
        if (mc_resolve_or_eval_im(rhs->u.call.args[arg0 + i], avoid_id, in_unit, &ids[i],
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
                      int              in_unit,
                      char            *err,
                      size_t           errsz)
{
    double kd = 1.0;
    if (rhs->u.call.nargs == 2 && mc_eval_scalar_d(rhs->u.call.args[1], &kd, err, errsz) != 0)
    {
        return -1;
    }
    int     k       = (int) (((llround(kd) % 4) + 4) % 4);
    imageID src_id  = -1;
    int     src_tmp = 0;
    if (mc_resolve_or_eval_im(rhs->u.call.args[0], avoid_id, in_unit, &src_id, &src_tmp, err,
                              errsz) != 0)
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
 * Public Constructor Dispatchers
 * ============================================================================ */

/**
 * mc_exec_fft_or_stack - Dispatch FFT (`fft`, `ifft`, `fftshift`), rotation, and stacking calls
 * @prog:      Compiled calculator program holding the slot table
 * @cs:        Constructor step descriptor to execute
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Return: 0 on success, -1 on error
 */
int mc_exec_fft_or_stack(mcalc_prog_t   *prog,
                         mc_ctor_step_t *cs,
                         imageID         avoid_id,
                         int             in_unit,
                         char           *err,
                         size_t          errsz)
{
    mc_slot_t       *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs  = cs->rhs_expr;
    const char      *fn   = rhs->u.call.fn->name;

    if (strcmp(fn, "fft") == 0 || strcmp(fn, "ifft") == 0)
    {
        return exec_fft_or_ifft(slot, cs, avoid_id, in_unit, err, errsz);
    }
    if (strcmp(fn, "fftshift") == 0)
    {
        return exec_fftshift(slot, rhs, avoid_id, in_unit, err, errsz);
    }
    if (strcmp(fn, "rot90") == 0)
    {
        return exec_rot90(slot, rhs, avoid_id, in_unit, err, errsz);
    }
    return exec_stack_or_cat(slot, rhs, avoid_id, in_unit, err, errsz);
}

/**
 * mc_exec_linalg_or_ext - Top-level router for `MC_FN_LINALG`, algebraic BLAS, and extension steps
 * @prog:      Compiled calculator program holding the slot table
 * @cs:        Constructor step descriptor to execute
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
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
int mc_exec_linalg_or_ext(mcalc_prog_t   *prog,
                          mc_ctor_step_t *cs,
                          imageID         avoid_id,
                          int             in_unit,
                          char           *err,
                          size_t          errsz)
{
    if (cs->multi_stmt != NULL)
    {
        return mc_exec_multi_linalg_or_ext(prog, cs, cs->multi_stmt, in_unit, NULL, err, errsz);
    }
    const mc_expr_t *rhs = cs->rhs_expr;
    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.ext_id >= 0)
    {
        return mc_exec_ext_ctor(prog, cs, avoid_id, in_unit, err, errsz);
    }
    if (rhs->kind != MC_EXPR_CALL || rhs->u.call.fn == NULL)
    {
        return mc_exec_blas_call(prog, cs, avoid_id, in_unit, err, errsz);
    }
    const char *fn = rhs->u.call.fn->name;
    if (strcmp(fn, "fft") == 0 || strcmp(fn, "ifft") == 0 || strcmp(fn, "fftshift") == 0 ||
        strcmp(fn, "hstack") == 0 || strcmp(fn, "vstack") == 0 || strcmp(fn, "dstack") == 0 ||
        strcmp(fn, "cat") == 0 || strcmp(fn, "rot90") == 0)
    {
        return mc_exec_fft_or_stack(prog, cs, avoid_id, in_unit, err, errsz);
    }
    if (strcmp(fn, "trsv") == 0 || strcmp(fn, "trsm") == 0 || strcmp(fn, "gesv") == 0 ||
        strcmp(fn, "posv") == 0 || strcmp(fn, "gels") == 0 || strcmp(fn, "solve") == 0 ||
        strcmp(fn, "inv") == 0 || strcmp(fn, "pinv") == 0 || strcmp(fn, "svd") == 0 ||
        strcmp(fn, "eigh") == 0)
    {
        return mc_exec_lapack_call(prog, cs, avoid_id, in_unit, err, errsz);
    }
    return mc_exec_blas_call(prog, cs, avoid_id, in_unit, err, errsz);
}
