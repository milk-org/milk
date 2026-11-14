// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_delay.c
 * @brief   Stream delay, fractional resampling, and causal temporal filtering
 *
 * Implements the reusable `MILK_STREAM_DELAY_STATE` ring-buffer engine and the
 * `runnable_COREMOD_memory_streamdelay` (`milk-fpsexec-mem-streamdelay` / `mem.streamdelay`)
 * compute unit.
 */

#include "milk_runnable.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "libmilkcommon/milk_omp_team.h"
#include "libmilkcommon/pixel_dispatch.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_delay.h"

/* L1d cache tile width (in elements) for multi-tap linear combinations */
#define SDELAY_TILE 2048U

/* ============================================================================================== */
/* Internal Ring-Buffer, Timestamp, and Weight-Assembly Helpers                                   */
/* ============================================================================================== */

/**
 * sdelay_ts_diff_sec - Return `t_end - t_start` in double-precision seconds
 */
static inline double sdelay_ts_diff_sec(struct timespec t_end, struct timespec t_start)
{
    return (double) (t_end.tv_sec - t_start.tv_sec) +
           1.0e-9 * (double) (t_end.tv_nsec - t_start.tv_nsec);
}

/**
 * sdelay_ts_less - Return 1 if `a < b`, 0 otherwise
 */
static inline int sdelay_ts_less(struct timespec a, struct timespec b)
{
    return (a.tv_sec < b.tv_sec) || (a.tv_sec == b.tv_sec && a.tv_nsec < b.tv_nsec);
}

/**
 * sdelay_slot_of_age - Map age index `0 .. valid_cnt - 1` (`0` = newest) to ring-buffer slot
 */
static inline uint64_t sdelay_slot_of_age(const MILK_STREAM_DELAY_STATE *state, uint64_t age_idx)
{
    if (state->head_idx > age_idx)
    {
        return state->head_idx - 1ULL - age_idx;
    }
    return state->buff_size + state->head_idx - 1ULL - age_idx;
}

/**
 * sdelay_age_of_slot - Map ring-buffer slot index back to age in frames (`0` = newest)
 */
static inline uint64_t sdelay_age_of_slot(const MILK_STREAM_DELAY_STATE *state, uint64_t slot_idx)
{
    if (state->head_idx > slot_idx)
    {
        return state->head_idx - 1ULL - slot_idx;
    }
    return state->buff_size + state->head_idx - 1ULL - slot_idx;
}

/**
 * sdelay_age_sec - Elapsed time in seconds (`>= 0`) of frame at @age_idx relative to @t_ref
 */
static inline double sdelay_age_sec(const MILK_STREAM_DELAY_STATE *state,
                                    uint64_t                       age_idx,
                                    struct timespec                t_ref)
{
    uint64_t slot_idx = sdelay_slot_of_age(state, age_idx);
    double   dt       = sdelay_ts_diff_sec(t_ref, state->ts_arr[slot_idx]);
    return (dt > 0.0) ? dt : 0.0;
}

/**
 * sdelay_add_slot_weight - Accumulate weight @w onto ring-buffer @slot_idx in O(1) time
 */
static inline void sdelay_add_slot_weight(MILK_STREAM_DELAY_STATE *state,
                                          uint64_t                 slot_idx,
                                          double                   w,
                                          uint64_t                *n_active)
{
    if (w == 0.0)
    {
        return;
    }
    if (state->slot_stamp[slot_idx] != state->step_stamp)
    {
        uint64_t pos                = (*n_active)++;
        state->slot_stamp[slot_idx] = state->step_stamp;
        state->slot_pos[slot_idx]   = (uint32_t) pos;
        state->active_slots[pos]    = slot_idx;
        state->active_weights[pos]  = w;
    }
    else
    {
        state->active_weights[state->slot_pos[slot_idx]] += w;
    }
}

/**
 * sdelay_read_tap_weight - Read tap element @tap_idx from @weights_im as `double`
 */
static inline double sdelay_read_tap_weight(const IMAGE *weights_im, uint64_t tap_idx)
{
    uint8_t dt = weights_im->md[0].datatype;
    if (dt == _DATATYPE_FLOAT)
    {
        return (double) weights_im->array.F[tap_idx];
    }
    if (dt == _DATATYPE_DOUBLE)
    {
        return weights_im->array.D[tap_idx];
    }

#define SDELAY_READ_W_CASE(DTYPE, ACC, CTYPE)           \
    else if (dt == DTYPE)                               \
    {                                                   \
        return (double) weights_im->array.ACC[tap_idx]; \
    }

    if (0)
    {
    }
    FOREACH_REAL_DATATYPE(SDELAY_READ_W_CASE)
#undef SDELAY_READ_W_CASE

    return 0.0;
}

/**
 * sdelay_bsearch_time_age - Find `a0` in `0 .. valid_cnt - 2` with `age(a0) <= target <= age(a0+1)`
 */
static uint64_t sdelay_bsearch_time_age(const MILK_STREAM_DELAY_STATE *state,
                                        double                         target_sec,
                                        struct timespec                t_ref)
{
    uint64_t lo = 0;
    uint64_t hi = state->valid_cnt - 1ULL;
    while (lo + 1ULL < hi)
    {
        uint64_t mid = lo + ((hi - lo) >> 1U);
        if (sdelay_age_sec(state, mid, t_ref) <= target_sec)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

/**
 * sdelay_accum_frame_lag - Add tap weight @tap_w at frame lag @lag_fr (`>= 0`)
 *
 * When @clamp_oldest is non-zero and the ring buffer is completely full (`valid_cnt == buff_size`),
 * lags past `buff_size - 1` clamp to the oldest buffered slot; otherwise out-of-range lags are
 * treated as zero-initialized pre-history.
 */
static void sdelay_accum_frame_lag(MILK_STREAM_DELAY_STATE *state,
                                   double                   lag_fr,
                                   double                   tap_w,
                                   int                      interp,
                                   int                      clamp_oldest,
                                   uint64_t                *n_active)
{
    if (lag_fr < 0.0)
    {
        lag_fr = 0.0;
    }
    double max_cap = (double) (state->buff_size - 1ULL);
    if (lag_fr > max_cap)
    {
        if (!clamp_oldest)
        {
            return;
        }
        lag_fr = max_cap;
    }

    if (interp)
    {
        double   fl   = floor(lag_fr);
        uint64_t a0   = (uint64_t) fl;
        double   frac = lag_fr - fl;
        uint64_t a1   = a0 + 1ULL;
        if (a0 < state->valid_cnt && (1.0 - frac) > 0.0)
        {
            sdelay_add_slot_weight(state, sdelay_slot_of_age(state, a0), tap_w * (1.0 - frac),
                                   n_active);
        }
        if (frac > 0.0)
        {
            if (a1 < state->valid_cnt)
            {
                sdelay_add_slot_weight(state, sdelay_slot_of_age(state, a1), tap_w * frac,
                                       n_active);
            }
            else if (clamp_oldest && state->valid_cnt == state->buff_size && a0 < state->valid_cnt)
            {
                sdelay_add_slot_weight(state, sdelay_slot_of_age(state, a0), tap_w * frac,
                                       n_active);
            }
        }
    }
    else
    {
        uint64_t age_idx = (uint64_t) llround(lag_fr);
        if (age_idx < state->valid_cnt)
        {
            sdelay_add_slot_weight(state, sdelay_slot_of_age(state, age_idx), tap_w, n_active);
        }
    } // if (interp) ... else
}

/**
 * sdelay_accum_time_lag - Add tap weight @tap_w at elapsed time lag @lag_sec (`>= 0`)
 *
 * In discrete mode (`interp == 0`), selects the newest buffered frame whose elapsed age is
 * `>= lag_sec` (strictly causal latency semantics matching legacy `mem.streamdelay`).
 * In linear mode (`interp == 1`), interpolates between adjacent `writetime` brackets.
 */
static void sdelay_accum_time_lag(MILK_STREAM_DELAY_STATE *state,
                                  double                   lag_sec,
                                  double                   tap_w,
                                  int                      interp,
                                  int                      clamp_oldest,
                                  struct timespec          t_ref,
                                  uint64_t                *n_active)
{
    if (state->valid_cnt == 0)
    {
        return;
    }
    if (lag_sec < 0.0)
    {
        lag_sec = 0.0;
    }

    double age_min = sdelay_age_sec(state, 0, t_ref);
    if (lag_sec <= age_min)
    {
        sdelay_add_slot_weight(state, sdelay_slot_of_age(state, 0), tap_w, n_active);
        return;
    }

    uint64_t oldest_age = state->valid_cnt - 1ULL;
    double   age_max    = sdelay_age_sec(state, oldest_age, t_ref);
    if (lag_sec > age_max)
    {
        if (clamp_oldest && state->valid_cnt == state->buff_size)
        {
            sdelay_add_slot_weight(state, sdelay_slot_of_age(state, oldest_age), tap_w, n_active);
        }
        return;
    }

    if (state->valid_cnt == 1ULL)
    {
        sdelay_add_slot_weight(state, sdelay_slot_of_age(state, 0), tap_w, n_active);
        return;
    }

    uint64_t a0 = sdelay_bsearch_time_age(state, lag_sec, t_ref);
    double   t0 = sdelay_age_sec(state, a0, t_ref);
    double   t1 = sdelay_age_sec(state, a0 + 1ULL, t_ref);

    if (interp)
    {
        double dt_seg = t1 - t0;
        double frac   = (dt_seg > 1.0e-12) ? ((lag_sec - t0) / dt_seg) : 0.0;
        if (frac < 0.0)
        {
            frac = 0.0;
        }
        else if (frac > 1.0)
        {
            frac = 1.0;
        }
        sdelay_add_slot_weight(state, sdelay_slot_of_age(state, a0), tap_w * (1.0 - frac),
                               n_active);
        sdelay_add_slot_weight(state, sdelay_slot_of_age(state, a0 + 1ULL), tap_w * frac, n_active);
    }
    else
    {
        uint64_t chosen_age = (lag_sec <= t0 + 1.0e-12) ? a0 : (a0 + 1ULL);
        sdelay_add_slot_weight(state, sdelay_slot_of_age(state, chosen_age), tap_w, n_active);
    } // if (interp) ... else
}

/**
 * sdelay_window_kernel - Evaluate unnormalized causal window kernel at normalized offset `u >= 0`
 */
static inline double sdelay_window_kernel(int ave_mode, double rel_lag, double width)
{
    double u = rel_lag / width;
    if (u < -1.0e-12)
    {
        return 0.0;
    }
    if (u < 0.0)
    {
        u = 0.0;
    }
    switch (ave_mode)
    {
    case MILK_SDELAY_AVE_BOXCAR:
        return (u <= 1.0 + 1.0e-12) ? 1.0 : 0.0;
    case MILK_SDELAY_AVE_LINEAR:
        return (u <= 1.0) ? (1.0 - u) : 0.0;
    case MILK_SDELAY_AVE_EXP:
        return (u <= 5.0) ? exp(-u) : 0.0;
    case MILK_SDELAY_AVE_GAUSS:
        return (u <= 4.0) ? exp(-0.5 * u * u) : 0.0;
    default:
        return (u <= 1.0 + 1.0e-12) ? 1.0 : 0.0;
    }
}

/* ============================================================================================== */
/* Vectorized and Multi-Type Pixel Evaluation Kernels                                             */
/* ============================================================================================== */

/**
 * sdelay_eval_f32_to_f32 - Compute weighted sum of float32 ring slots into float32 @out_im
 */
static MILK_HOT void sdelay_eval_f32_to_f32(const MILK_STREAM_DELAY_STATE *state,
                                            uint64_t                       n_active,
                                            IMAGE                         *out_im)
{
    uint64_t             nel = state->nelement;
    float *MILK_RESTRICT dst = out_im->array.F;
    int                  nth = milk_omp_team_size(nel, 262144);

    if (n_active == 2ULL)
    {
        const float *MILK_RESTRICT s0 = MILK_ASSUME_ALIGNED(
            (const float *) (state->ring_buf + state->active_slots[0] * state->slot_stride));
        const float *MILK_RESTRICT s1 = MILK_ASSUME_ALIGNED(
            (const float *) (state->ring_buf + state->active_slots[1] * state->slot_stride));
        const float w0 = (float) state->active_weights[0];
        const float w1 = (float) state->active_weights[1];

#ifdef _OPENMP
#    pragma omp parallel for simd schedule(static) num_threads(nth) if (parallel : nth > 1)
#endif
        for (uint64_t ii = 0; ii < nel; ii++)
        {
            dst[ii] = w0 * s0[ii] + w1 * s1[ii];
        }
        return;
    } // if (n_active == 2ULL)

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t base_idx = 0; base_idx < nel; base_idx += SDELAY_TILE)
    {
        uint64_t tile_len = nel - base_idx;
        if (tile_len > SDELAY_TILE)
        {
            tile_len = SDELAY_TILE;
        }
        float *MILK_RESTRICT       d_tile = dst + base_idx;
        const float *MILK_RESTRICT s0 =
            (const float *) (state->ring_buf + state->active_slots[0] * state->slot_stride) +
            base_idx;
        const float w0 = (float) state->active_weights[0];

#ifdef _OPENMP
#    pragma omp simd
#endif
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            d_tile[ii] = w0 * s0[ii];
        }

        for (uint64_t tap_idx = 1; tap_idx < n_active; tap_idx++)
        {
            const float *MILK_RESTRICT s_tap =
                (const float *) (state->ring_buf +
                                 state->active_slots[tap_idx] * state->slot_stride) +
                base_idx;
            const float w_tap = (float) state->active_weights[tap_idx];

#ifdef _OPENMP
#    pragma omp simd
#endif
            for (uint64_t ii = 0; ii < tile_len; ii++)
            {
                d_tile[ii] += w_tap * s_tap[ii];
            }
        }
    } // for (uint64_t base_idx = 0; base_idx < nel; base_idx += SDELAY_TILE)
}

/**
 * sdelay_eval_f64_to_f64 - Compute weighted sum of float64 ring slots into float64 @out_im
 */
static MILK_HOT void sdelay_eval_f64_to_f64(const MILK_STREAM_DELAY_STATE *state,
                                            uint64_t                       n_active,
                                            IMAGE                         *out_im)
{
    uint64_t              nel = state->nelement;
    double *MILK_RESTRICT dst = out_im->array.D;
    int                   nth = milk_omp_team_size(nel, 131072);

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t base_idx = 0; base_idx < nel; base_idx += SDELAY_TILE)
    {
        uint64_t tile_len = nel - base_idx;
        if (tile_len > SDELAY_TILE)
        {
            tile_len = SDELAY_TILE;
        }
        double *MILK_RESTRICT       d_tile = dst + base_idx;
        const double *MILK_RESTRICT s0 =
            (const double *) (state->ring_buf + state->active_slots[0] * state->slot_stride) +
            base_idx;
        const double w0 = state->active_weights[0];

#ifdef _OPENMP
#    pragma omp simd
#endif
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            d_tile[ii] = w0 * s0[ii];
        }

        for (uint64_t tap_idx = 1; tap_idx < n_active; tap_idx++)
        {
            const double *MILK_RESTRICT s_tap =
                (const double *) (state->ring_buf +
                                  state->active_slots[tap_idx] * state->slot_stride) +
                base_idx;
            const double w_tap = state->active_weights[tap_idx];

#ifdef _OPENMP
#    pragma omp simd
#endif
            for (uint64_t ii = 0; ii < tile_len; ii++)
            {
                d_tile[ii] += w_tap * s_tap[ii];
            }
        }
    } // for (uint64_t base_idx = 0; base_idx < nel; base_idx += SDELAY_TILE)
}

/**
 * sdelay_store_clamped_tile - Store `double` tile accumulator @acc into arbitrary real @out_im
 */
static void sdelay_store_clamped_tile(IMAGE        *out_im,
                                      uint64_t      base_idx,
                                      uint64_t      tile_len,
                                      const double *acc)
{
    uint8_t odt = out_im->md[0].datatype;
    if (odt == _DATATYPE_FLOAT)
    {
        float *MILK_RESTRICT d_ptr = out_im->array.F + base_idx;
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            d_ptr[ii] = (float) acc[ii];
        }
    }
    else if (odt == _DATATYPE_DOUBLE)
    {
        double *MILK_RESTRICT d_ptr = out_im->array.D + base_idx;
        __builtin_memcpy(d_ptr, acc, sizeof(double) * (size_t) tile_len);
    }
    else if (odt == _DATATYPE_UINT8)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < 0.0) ? 0.0 : (acc[ii] > 255.0 ? 255.0 : acc[ii]);
            out_im->array.UI8[base_idx + ii] = (uint8_t) llround(v);
        }
    }
    else if (odt == _DATATYPE_INT8)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < -128.0) ? -128.0 : (acc[ii] > 127.0 ? 127.0 : acc[ii]);
            out_im->array.SI8[base_idx + ii] = (int8_t) llround(v);
        }
    }
    else if (odt == _DATATYPE_UINT16)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < 0.0) ? 0.0 : (acc[ii] > 65535.0 ? 65535.0 : acc[ii]);
            out_im->array.UI16[base_idx + ii] = (uint16_t) llround(v);
        }
    }
    else if (odt == _DATATYPE_INT16)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < -32768.0) ? -32768.0 : (acc[ii] > 32767.0 ? 32767.0 : acc[ii]);
            out_im->array.SI16[base_idx + ii] = (int16_t) llround(v);
        }
    }
    else if (odt == _DATATYPE_UINT32)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < 0.0) ? 0.0 : (acc[ii] > 4294967295.0 ? 4294967295.0 : acc[ii]);
            out_im->array.UI32[base_idx + ii] = (uint32_t) llround(v);
        }
    }
    else if (odt == _DATATYPE_INT32)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < -2147483648.0)
                           ? -2147483648.0
                           : (acc[ii] > 2147483647.0 ? 2147483647.0 : acc[ii]);
            out_im->array.SI32[base_idx + ii] = (int32_t) llround(v);
        }
    }
    else if (odt == _DATATYPE_UINT64)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = (acc[ii] < 0.0) ? 0.0 : acc[ii];
            out_im->array.UI64[base_idx + ii] =
                (v >= 18446744073709551616.0) ? UINT64_MAX : (uint64_t) (v + 0.5);
        }
    }
    else if (odt == _DATATYPE_INT64)
    {
        for (uint64_t ii = 0; ii < tile_len; ii++)
        {
            double v = acc[ii];
            if (v <= (double) INT64_MIN)
            {
                out_im->array.SI64[base_idx + ii] = INT64_MIN;
            }
            else if (v >= 9223372036854775808.0)
            {
                out_im->array.SI64[base_idx + ii] = INT64_MAX;
            }
            else
            {
                out_im->array.SI64[base_idx + ii] = (int64_t) llround(v);
            }
        }
    }
}

/**
 * sdelay_eval_generic - L1-tiled multi-datatype linear combination for integer or mixed streams
 */
static void sdelay_eval_generic(const MILK_STREAM_DELAY_STATE *state,
                                uint64_t                       n_active,
                                IMAGE                         *out_im)
{
    uint64_t nel = state->nelement;
    uint8_t  idt = state->datatype;

    for (uint64_t base_idx = 0; base_idx < nel; base_idx += SDELAY_TILE)
    {
        double   MILK_ALIGNED(64) acc[SDELAY_TILE];
        uint64_t tile_len = nel - base_idx;
        if (tile_len > SDELAY_TILE)
        {
            tile_len = SDELAY_TILE;
        }
        memset(acc, 0, sizeof(double) * (size_t) tile_len);

#define SDELAY_ACCUM_TYPE(DTYPE, ACC, CTYPE)                                          \
    else if (idt == DTYPE)                                                            \
    {                                                                                 \
        for (uint64_t tap_idx = 0; tap_idx < n_active; tap_idx++)                     \
        {                                                                             \
            const CTYPE *MILK_RESTRICT s_tap =                                        \
                (const CTYPE *) (state->ring_buf +                                    \
                                 state->active_slots[tap_idx] * state->slot_stride) + \
                base_idx;                                                             \
            const double w_tap = state->active_weights[tap_idx];                      \
            for (uint64_t ii = 0; ii < tile_len; ii++)                                \
            {                                                                         \
                acc[ii] += w_tap * (double) s_tap[ii];                                \
            }                                                                         \
        }                                                                             \
    }

        if (0)
        {
        }
        FOREACH_REAL_DATATYPE(SDELAY_ACCUM_TYPE)
#undef SDELAY_ACCUM_TYPE

        sdelay_store_clamped_tile(out_im, base_idx, tile_len, acc);
    } // for (uint64_t base_idx = 0; base_idx < nel; base_idx += SDELAY_TILE)
}

/* ============================================================================================== */
/* Public `milk_stream_delay_*` State Engine API                                                  */
/* ============================================================================================== */

errno_t milk_stream_delay_init(MILK_STREAM_DELAY_STATE *state,
                               const IMAGE             *in_im,
                               uint64_t                 buff_size)
{
    if (UNLIKELY(state == NULL || in_im == NULL || in_im->md == NULL))
    {
        PRINT_ERROR("NULL state or input image");
        return RETURN_FAILURE;
    }

    memset(state, 0, sizeof(*state));

    uint8_t naxis = in_im->md[0].naxis;
    if (UNLIKELY(naxis < 1 || naxis > 3))
    {
        PRINT_ERROR("Unsupported naxis=%u (expected 1..3)", (unsigned) naxis);
        return RETURN_FAILURE;
    }

    uint32_t sx  = in_im->md[0].size[0];
    uint32_t sy  = (naxis > 1) ? in_im->md[0].size[1] : 1U;
    uint32_t sz  = (naxis > 2) ? in_im->md[0].size[2] : 1U;
    uint64_t nel = (uint64_t) sx * (uint64_t) sy * (uint64_t) sz;
    uint8_t  dt  = in_im->md[0].datatype;
    int      tsz = ImageStreamIO_typesize(dt);
    if (UNLIKELY(nel == 0 || tsz <= 0))
    {
        PRINT_ERROR("Invalid image geometry or datatype=%u", (unsigned) dt);
        return RETURN_FAILURE;
    }

    uint64_t cap = (buff_size < 2ULL) ? 2ULL : buff_size;
    if (UNLIKELY(nel > SIZE_MAX / (size_t) tsz))
    {
        PRINT_ERROR("Frame byte size overflow");
        return RETURN_FAILURE;
    }
    size_t frame_bytes = (size_t) nel * (size_t) tsz;
    size_t slot_stride = (frame_bytes + 63U) & ~(size_t) 63U;
    if (slot_stride == 0)
    {
        slot_stride = 64U;
    }
    if (UNLIKELY((size_t) cap != cap || slot_stride > SIZE_MAX / (size_t) cap))
    {
        PRINT_ERROR("Ring-buffer byte size overflow (buff_size=%llu)", (unsigned long long) cap);
        return RETURN_FAILURE;
    }

    void *raw_buf = NULL;
    if (UNLIKELY(posix_memalign(&raw_buf, 64U, slot_stride * (size_t) cap) != 0 || raw_buf == NULL))
    {
        PRINT_ERROR("Failed to allocate %zu-byte aligned ring buffer", slot_stride * (size_t) cap);
        return RETURN_FAILURE;
    }
    memset(raw_buf, 0, slot_stride * (size_t) cap);

    state->ring_buf       = (uint8_t *) raw_buf;
    state->ts_arr         = calloc((size_t) cap, sizeof(*state->ts_arr));
    state->seq_arr        = calloc((size_t) cap, sizeof(*state->seq_arr));
    state->slot_stamp     = calloc((size_t) cap, sizeof(*state->slot_stamp));
    state->slot_pos       = calloc((size_t) cap, sizeof(*state->slot_pos));
    state->active_slots   = calloc((size_t) cap, sizeof(*state->active_slots));
    state->active_weights = calloc((size_t) cap, sizeof(*state->active_weights));

    if (UNLIKELY(state->ts_arr == NULL || state->seq_arr == NULL || state->slot_stamp == NULL ||
                 state->slot_pos == NULL || state->active_slots == NULL ||
                 state->active_weights == NULL))
    {
        PRINT_ERROR("Failed to allocate stream_delay metadata arrays");
        milk_stream_delay_free(state);
        return RETURN_FAILURE;
    }

    state->buff_size   = cap;
    state->nelement    = nel;
    state->frame_bytes = frame_bytes;
    state->slot_stride = slot_stride;
    state->naxis       = naxis;
    state->size[0]     = sx;
    state->size[1]     = sy;
    state->size[2]     = sz;
    state->datatype    = dt;
    state->last_delay  = -1.0;
    state->last_unit   = -1;

    return RETURN_SUCCESS;
}

void milk_stream_delay_free(MILK_STREAM_DELAY_STATE *state)
{
    if (state == NULL)
    {
        return;
    }
    free(state->ring_buf);
    free(state->ts_arr);
    free(state->seq_arr);
    free(state->slot_stamp);
    free(state->slot_pos);
    free(state->active_slots);
    free(state->active_weights);
    memset(state, 0, sizeof(*state));
}

errno_t MILK_HOT milk_stream_delay_step(MILK_STREAM_DELAY_STATE     *state,
                                        const IMAGE                 *in_im,
                                        const IMAGE                 *weights_im,
                                        const MILK_STREAM_DELAY_CFG *cfg,
                                        IMAGE                       *out_im,
                                        int                         *out_updated)
{
    if (UNLIKELY(state == NULL || in_im == NULL || in_im->md == NULL || cfg == NULL ||
                 out_im == NULL || out_im->md == NULL || out_updated == NULL))
    {
        return RETURN_FAILURE;
    }
    *out_updated = 0;

    /* 1. Naive sleep mode: busy-wait in CLOCK_ISIO without ring-buffer storage */
    if (cfg->naive_mode)
    {
        double delay_sec = (cfg->delay > 0.0) ? cfg->delay : 0.0;
        if (delay_sec > 0.0)
        {
            struct timespec t_start = in_im->md[0].writetime;
            if (t_start.tv_sec == 0 && t_start.tv_nsec == 0)
            {
                clock_gettime(CLOCK_ISIO, &t_start);
            }
            struct timespec t_now;
            for (;;)
            {
                clock_gettime(CLOCK_ISIO, &t_now);
                if (sdelay_ts_diff_sec(t_now, t_start) >= delay_sec)
                {
                    break;
                }
                usleep(1);
            }
        }

        out_im->md[0].write = 1;
        if (out_im->md[0].datatype == in_im->md[0].datatype)
        {
            size_t nbytes = (size_t) in_im->md[0].nelement *
                            (size_t) ImageStreamIO_typesize(in_im->md[0].datatype);
            __builtin_memcpy(out_im->array.raw, in_im->array.raw, nbytes);
        }
        else if (out_im->md[0].datatype == _DATATYPE_FLOAT)
        {
            uint64_t nel = in_im->md[0].nelement;
            uint8_t  idt = in_im->md[0].datatype;
#define SDELAY_NAIVE_F32(DTYPE, ACC, CTYPE)                     \
    else if (idt == DTYPE)                                      \
    {                                                           \
        for (uint64_t ii = 0; ii < nel; ii++)                   \
        {                                                       \
            out_im->array.F[ii] = (float) in_im->array.ACC[ii]; \
        }                                                       \
    }
            if (0)
            {
            }
            FOREACH_REAL_DATATYPE(SDELAY_NAIVE_F32)
#undef SDELAY_NAIVE_F32
        }
        *out_updated = 1;
        return RETURN_SUCCESS;
    } // if (cfg->naive_mode)

    /* Verify ring buffer matches current input geometry */
    uint8_t  in_naxis = in_im->md[0].naxis;
    uint32_t sx       = in_im->md[0].size[0];
    uint32_t sy       = (in_naxis > 1) ? in_im->md[0].size[1] : 1U;
    uint32_t sz       = (in_naxis > 2) ? in_im->md[0].size[2] : 1U;
    if (UNLIKELY(state->ring_buf == NULL || state->naxis != in_naxis || state->size[0] != sx ||
                 state->size[1] != sy || state->size[2] != sz ||
                 state->datatype != in_im->md[0].datatype))
    {
        return RETURN_FAILURE;
    }

    /* 2. Push incoming frame into ring buffer when forced or when cnt0 advances */
    int             pushed = 0;
    struct timespec t_in   = in_im->md[0].writetime;
    uint64_t        cnt0   = in_im->md[0].cnt0;
    if (cfg->force_push || !state->has_cnt0 || cnt0 != state->last_cnt0)
    {
        if (t_in.tv_sec == 0 && t_in.tv_nsec == 0)
        {
            clock_gettime(CLOCK_ISIO, &t_in);
        }
        if (state->valid_cnt > 0)
        {
            struct timespec prev_ts = state->ts_arr[sdelay_slot_of_age(state, 0)];
            if (sdelay_ts_less(t_in, prev_ts))
            {
                t_in = prev_ts;
            }
        }

        uint64_t w_slot = state->head_idx;
        __builtin_memcpy(state->ring_buf + w_slot * state->slot_stride, in_im->array.raw,
                         state->frame_bytes);
        state->ts_arr[w_slot]  = t_in;
        state->seq_arr[w_slot] = ++state->push_seq;

        state->head_idx = (w_slot + 1ULL >= state->buff_size) ? 0ULL : (w_slot + 1ULL);
        if (state->valid_cnt < state->buff_size)
        {
            state->valid_cnt++;
        }
        state->last_cnt0   = cnt0;
        state->has_cnt0    = 1;
        state->status_kkin = w_slot;
        pushed             = 1;
    } // if (cfg->force_push || !state->has_cnt0 || cnt0 != state->last_cnt0)

    if (UNLIKELY(state->valid_cnt == 0))
    {
        return RETURN_SUCCESS;
    }

    /* Reference timestamp: pushed frame's timestamp, or current CLOCK_ISIO when polling */
    struct timespec t_ref;
    if (pushed)
    {
        t_ref = t_in;
    }
    else
    {
        clock_gettime(CLOCK_ISIO, &t_ref);
        struct timespec newest_ts = state->ts_arr[sdelay_slot_of_age(state, 0)];
        if (sdelay_ts_less(t_ref, newest_ts))
        {
            t_ref = newest_ts;
        }
    }

    int unit_mode =
        (cfg->unit_mode == MILK_SDELAY_UNIT_FRAME) ? MILK_SDELAY_UNIT_FRAME : MILK_SDELAY_UNIT_TIME;
    int    interp    = (cfg->interp != 0) ? 1 : 0;
    double delay_val = (cfg->delay > 0.0) ? cfg->delay : 0.0;
    double win_val   = (cfg->window > 0.0) ? cfg->window : 0.0;
    double max_age   = (unit_mode == MILK_SDELAY_UNIT_FRAME)
                           ? (double) (state->valid_cnt - 1ULL)
                           : sdelay_age_sec(state, state->valid_cnt - 1ULL, t_ref);

    /* In standalone stream mode (allow_skip == 1), hold output until delay warmup is reached */
    if (cfg->allow_skip && state->valid_cnt < state->buff_size)
    {
        double req_age = delay_val;
        if (unit_mode == MILK_SDELAY_UNIT_FRAME)
        {
            req_age = interp ? ceil(delay_val) : (double) llround(delay_val);
            if (req_age > (double) (state->buff_size - 1ULL))
            {
                req_age = (double) (state->buff_size - 1ULL);
            }
        }
        if (max_age + 1.0e-12 < req_age)
        {
            return RETURN_SUCCESS;
        }
        /* For multi-tap modes in a polling loop without new input, wait for next frame */
        if (!pushed && (weights_im != NULL || (cfg->ave_mode > 0 && win_val > 0.0) || interp))
        {
            return RETURN_SUCCESS;
        }
    }

    /* 3. Assemble sparse slot weights for the requested delay / filter mode */
    state->step_stamp++;
    if (UNLIKELY(state->step_stamp == 0))
    {
        memset(state->slot_stamp, 0, sizeof(*state->slot_stamp) * (size_t) state->buff_size);
        state->step_stamp = 1ULL;
    }
    uint64_t n_active = 0;

    if (weights_im != NULL && weights_im->md != NULL && weights_im->md[0].nelement > 0)
    {
        /* Mode A: Arbitrary 1D FIR linear combination of past frames / resampled lags */
        uint64_t n_taps = weights_im->md[0].nelement;
        if (unit_mode == MILK_SDELAY_UNIT_FRAME)
        {
            double step_fr = (win_val > 0.0) ? win_val : 1.0;
            for (uint64_t tap_idx = 0; tap_idx < n_taps; tap_idx++)
            {
                double w_tap = sdelay_read_tap_weight(weights_im, tap_idx);
                if (w_tap != 0.0)
                {
                    double lag_fr = delay_val + (double) tap_idx * step_fr;
                    sdelay_accum_frame_lag(state, lag_fr, w_tap, interp, (tap_idx == 0), &n_active);
                }
            }
        }
        else if (win_val > 0.0)
        {
            for (uint64_t tap_idx = 0; tap_idx < n_taps; tap_idx++)
            {
                double w_tap = sdelay_read_tap_weight(weights_im, tap_idx);
                if (w_tap != 0.0)
                {
                    double lag_sec = delay_val + (double) tap_idx * win_val;
                    sdelay_accum_time_lag(state, lag_sec, w_tap, interp, (tap_idx == 0), t_ref,
                                          &n_active);
                }
            }
        }
        else
        {
            /* Time-unit anchor at delay_val, stepping 1 buffered frame older per tap */
            uint64_t anchor_age = 0;
            if (delay_val > sdelay_age_sec(state, 0, t_ref))
            {
                if (delay_val > max_age)
                {
                    anchor_age = state->valid_cnt;
                }
                else if (state->valid_cnt > 1ULL)
                {
                    uint64_t a0 = sdelay_bsearch_time_age(state, delay_val, t_ref);
                    double   t0 = sdelay_age_sec(state, a0, t_ref);
                    anchor_age  = (delay_val <= t0 + 1.0e-12) ? a0 : (a0 + 1ULL);
                }
            }
            for (uint64_t tap_idx = 0; tap_idx < n_taps; tap_idx++)
            {
                double   w_tap   = sdelay_read_tap_weight(weights_im, tap_idx);
                uint64_t age_idx = anchor_age + tap_idx;
                if (w_tap != 0.0 && age_idx < state->valid_cnt)
                {
                    sdelay_add_slot_weight(state, sdelay_slot_of_age(state, age_idx), w_tap,
                                           &n_active);
                }
            }
        } // if (unit_mode == MILK_SDELAY_UNIT_FRAME) ... else
    }
    else if (cfg->ave_mode > 0 && win_val > 0.0)
    {
        /* Mode B: Normalized causal window filter (boxcar, linear, exponential, Gaussian) */
        double span_mult = (cfg->ave_mode == MILK_SDELAY_AVE_EXP)     ? 5.0
                           : (cfg->ave_mode == MILK_SDELAY_AVE_GAUSS) ? 4.0
                                                                      : 1.0;
        double max_span  = delay_val + span_mult * win_val;
        double w_sum     = 0.0;

        for (uint64_t age_idx = 0; age_idx < state->valid_cnt; age_idx++)
        {
            double cur_lag = (unit_mode == MILK_SDELAY_UNIT_FRAME)
                                 ? (double) age_idx
                                 : sdelay_age_sec(state, age_idx, t_ref);
            if (cur_lag + 1.0e-12 < delay_val)
            {
                continue;
            }
            if (cur_lag > max_span + 1.0e-12)
            {
                break;
            }
            double w = sdelay_window_kernel(cfg->ave_mode, cur_lag - delay_val, win_val);
            if (w > 0.0)
            {
                sdelay_add_slot_weight(state, sdelay_slot_of_age(state, age_idx), w, &n_active);
                w_sum += w;
            }
        }

        if (n_active > 0 && w_sum > 0.0)
        {
            double inv_sum = 1.0 / w_sum;
            for (uint64_t idx = 0; idx < n_active; idx++)
            {
                state->active_weights[idx] *= inv_sum;
            }
        }
        else if (max_age + 1.0e-12 >= delay_val || state->valid_cnt == state->buff_size)
        {
            /* Window narrower than inter-frame spacing: fall back to single-point lookup */
            if (unit_mode == MILK_SDELAY_UNIT_FRAME)
            {
                sdelay_accum_frame_lag(state, delay_val, 1.0, interp, 1, &n_active);
            }
            else
            {
                sdelay_accum_time_lag(state, delay_val, 1.0, interp, 1, t_ref, &n_active);
            }
        }
    }
    else
    {
        /* Mode C: Single-point discrete delay or 2-frame fractional linear interpolation */
        if (unit_mode == MILK_SDELAY_UNIT_FRAME)
        {
            sdelay_accum_frame_lag(state, delay_val, 1.0, interp, 1, &n_active);
        }
        else
        {
            sdelay_accum_time_lag(state, delay_val, 1.0, interp, 1, t_ref, &n_active);
        }
    } // if (weights_im != NULL ...) ... else

    /* Compact out any slots whose summed weights cancelled to zero */
    uint64_t compacted = 0;
    for (uint64_t idx = 0; idx < n_active; idx++)
    {
        if (state->active_weights[idx] != 0.0)
        {
            state->active_slots[compacted]   = state->active_slots[idx];
            state->active_weights[compacted] = state->active_weights[idx];
            compacted++;
        }
    }
    n_active = compacted;

    /* 4. Update telemetry and write output image */
    if (n_active == 0)
    {
        if (cfg->allow_skip)
        {
            return RETURN_SUCCESS;
        }
        size_t out_bytes    = (size_t) out_im->md[0].nelement *
                              (size_t) ImageStreamIO_typesize(out_im->md[0].datatype);
        out_im->md[0].write = 1;
        memset(out_im->array.raw, 0, out_bytes);
        *out_updated = 1;
        return RETURN_SUCCESS;
    }

    uint64_t slot0         = state->active_slots[0];
    state->status_kkout    = slot0;
    state->status_framelag = sdelay_age_of_slot(state, slot0);

    /* Fast path 1: single slot, weight == 1.0, matching datatype -> zero-math memcpy */
    if (n_active == 1ULL && fabs(state->active_weights[0] - 1.0) <= 1.0e-12 &&
        out_im->md[0].datatype == state->datatype)
    {
        if (cfg->allow_skip && cfg->ave_mode == 0 && !interp && weights_im == NULL)
        {
            uint64_t seq = state->seq_arr[slot0];
            if (seq == state->last_emitted_seq && delay_val == state->last_delay &&
                unit_mode == state->last_unit)
            {
                return RETURN_SUCCESS;
            }
            state->last_emitted_seq = seq;
            state->last_delay       = delay_val;
            state->last_unit        = unit_mode;
        }

        if (UNLIKELY(dcdebug > 0))
        {
            printf("streamdelay: write kkin=%llu kkout=%llu lag=%llu (%zu bytes)\n",
                   (unsigned long long) state->status_kkin,
                   (unsigned long long) state->status_kkout,
                   (unsigned long long) state->status_framelag, state->frame_bytes);
        }

        out_im->md[0].write = 1;
        __builtin_memcpy(out_im->array.raw, state->ring_buf + slot0 * state->slot_stride,
                         state->frame_bytes);
        *out_updated = 1;
        return RETURN_SUCCESS;
    }

    /* Fast path 2 & 3: SIMD / L1-tiled linear combination */
    out_im->md[0].write = 1;
    if (state->datatype == _DATATYPE_FLOAT && out_im->md[0].datatype == _DATATYPE_FLOAT)
    {
        sdelay_eval_f32_to_f32(state, n_active, out_im);
    }
    else if (state->datatype == _DATATYPE_DOUBLE && out_im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        sdelay_eval_f64_to_f64(state, n_active, out_im);
    }
    else
    {
        sdelay_eval_generic(state, n_active, out_im);
    }
    *out_updated = 1;
    return RETURN_SUCCESS;
}

/* ============================================================================================== */
/* Tier 3 Compute Unit (`milk-fpsexec-mem-streamdelay` / `mem.streamdelay`)                       */
/* ============================================================================================== */

static char     sdelay_inimname[FUNCTION_PARAMETER_STRMAXLEN]    = "imin";
static char     sdelay_outimname[FUNCTION_PARAMETER_STRMAXLEN]   = "imout";
static float    sdelay_delaysec                                  = 0.001f;
static int32_t  sdelay_naive_mode                                = 0;
static uint64_t sdelay_timebuffsize                              = 1000;
static int32_t  sdelay_unit                                      = 0;
static double   sdelay_delayfr                                   = 0.0;
static int32_t  sdelay_interp                                    = 0;
static int32_t  sdelay_avemode                                   = 0;
static uint64_t sdelay_avedtns                                   = 0;
static double   sdelay_avewidth                                  = 0.0;
static char     sdelay_weightsname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static int32_t  sdelay_out_float                                 = 0;
static uint64_t sdelay_statusframelag                            = 0;
static uint64_t sdelay_statuskkin                                = 0;
static uint64_t sdelay_statuskkout                               = 0;

#define FPS_PARAMS(X)                                                                           \
    X(".in_name", sdelay_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")   \
    X(".out_name", sdelay_outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")    \
    X(".delaysec", &sdelay_delaysec, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, \
      "delay [s] (when unit=0)")                                                                \
    X(".naive_mode", &sdelay_naive_mode, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Naive wait mode (ON: no buffer, busywait sleep)")                                        \
    X(".timebuffsize", &sdelay_timebuffsize, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,            \
      "time ring buffer depth [frames]")                                                        \
    X(".unit", &sdelay_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,           \
      "delay/window unit (0=seconds, 1=frames)")                                                \
    X(".delayfr", &sdelay_delayfr, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,   \
      "delay [frames] (when unit=1)")                                                           \
    X(".interp", &sdelay_interp, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,       \
      "interpolation (0=discrete, 1=linear)")                                                   \
    X(".option.timeavemode", &sdelay_avemode, FPTYPE_INT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                                   \
      "window filter (0=off, 1=boxcar, 2=linear, 3=exp, 4=gauss)")                              \
    X(".option.timeavedtns", &sdelay_avedtns, FPTYPE_UINT64, 0,                                 \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "window width or FIR tap step [ns] (unit=0)")     \
    X(".option.avewidth", &sdelay_avewidth, FPTYPE_FLOAT64, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "window width or FIR tap step [frames] (unit=1)") \
    X(".weights_name", sdelay_weightsname, FPTYPE_STREAMNAME, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "optional 1D FIR tap weights stream")             \
    X(".out_float", &sdelay_out_float, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                   \
      "force FLOAT32 output stream")                                                            \
    X(".status.framelag", &sdelay_statusframelag, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,      \
      "current lag in frames")                                                                  \
    X(".status.kkin", &sdelay_statuskkin, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,              \
      "ring buffer write index")                                                                \
    X(".status.kkout", &sdelay_statuskkout, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,            \
      "ring buffer read index")

/**
 * sdelay_confcheck - Validate `mem.streamdelay` configuration parameters
 */
static MILK_COLD errno_t sdelay_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        long fpi_bs = functionparameter_GetParamIndex(dcfpsptr, ".timebuffsize");
        if (fpi_bs >= 0 && dcfpsptr->parray[fpi_bs].val.ui64[0] < 2ULL)
        {
            dcfpsptr->parray[fpi_bs].val.ui64[0] = 2ULL;
        }
    }
    return RETURN_SUCCESS;
}

/**
 * sdelay_connect_create_nd - Connect or (re-)create output stream matching 1D/2D/3D @inimg shape
 */
static IMGID sdelay_connect_create_nd(const char *out_raw_name, IMGID inimg, uint8_t out_dt)
{
    IMGID   img    = imgid_make_from_name(out_raw_name);
    uint8_t naxis  = inimg.md->naxis;
    int     shared = (img.stream_loc == 'L') ? 0 : 1;

    uint32_t dims[3] = {
        inimg.md->size[0],
        (naxis > 1) ? inimg.md->size[1] : 1U,
        (naxis > 2) ? inimg.md->size[2] : 1U,
    };

    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID == -1 && shared)
    {
        read_sharedmem_image(img.name, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    }

    if (img.ID != -1)
    {
        IMGID imgc         = imgid_make();
        imgc.mdt->datatype = out_dt;
        imgc.mdt->naxis    = naxis;
        imgc.mdt->size[0]  = dims[0];
        imgc.mdt->size[1]  = dims[1];
        imgc.mdt->size[2]  = dims[2];
        imgc.mdt->NBkw     = IMGID_NB_KEYWO_MAX;
        uint64_t imgerr    = imgid_compare(img, imgc);
        imgid_free(&imgc);

        if (imgerr != 0)
        {
            delete_image_ID(img.name, DELETE_IMAGE_ERRMODE_WARNING);
            img.ID = -1;
        }
    }

    if (img.ID == -1)
    {
        create_image_ID(img.name, (long) naxis, dims, out_dt, shared, IMGID_NB_KEYWO_MAX, 0,
                        &img.ID);
    }

    if (img.ID != -1 && dcimg[img.ID].md != NULL)
    {
        imageID id    = img.ID;
        img.im        = &dcimg[id];
        img.md        = dcimg[id].md;
        img.createcnt = dcimg[id].createcnt;
        imgid_update_creationparams(&img);
    }
    else
    {
        img.ID        = -1;
        img.im        = NULL;
        img.md        = NULL;
        img.createcnt = 0;
    }

    return img;
}

/**
 * sdelay_resolve_weights - Resolve or attach optional live `.weights_name` stream
 */
static const IMAGE *sdelay_resolve_weights(const char *raw_name, IMGID *wimg, char *cached_name)
{
    if (raw_name == NULL || raw_name[0] == '\0' || strcmp(raw_name, "null") == 0 ||
        strcmp(raw_name, "none") == 0)
    {
        if (cached_name[0] != '\0')
        {
            imgid_free(wimg);
            memset(wimg, 0, sizeof(*wimg));
            wimg->ID       = -1;
            cached_name[0] = '\0';
        }
        return NULL;
    }

    if (strcmp(raw_name, cached_name) != 0 || wimg->ID == -1)
    {
        if (cached_name[0] != '\0')
        {
            imgid_free(wimg);
        }
        *wimg = imgid_make_from_name(raw_name);
        snprintf(cached_name, FUNCTION_PARAMETER_STRMAXLEN, "%s", raw_name);
        resolveIMGID(wimg, ERRMODE_NULL, dcimg, dcnimg);
        if (wimg->ID == -1 && wimg->stream_loc != 'L')
        {
            read_sharedmem_image(wimg->name, dcimg, dcnimg);
            resolveIMGID(wimg, ERRMODE_NULL, dcimg, dcnimg);
        }
    }
    else
    {
        resolveIMGID(wimg, ERRMODE_NULL, dcimg, dcnimg);
    }

    return (wimg->ID != -1) ? wimg->im : NULL;
}

MILK_PARAMS_DEFINE(streamdelay_fps, FPS_PARAMS);

static MILK_HOT errno_t streamdelay_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(sdelay_inimname);
    resolveIMGID(&inimg, ERRMODE_NULL, dcimg, dcnimg);
    if (inimg.ID == -1 && inimg.stream_loc != 'L')
    {
        read_sharedmem_image(inimg.name, dcimg, dcnimg);
    }
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg, dcnimg);

    uint8_t out_dt = (sdelay_out_float != 0) ? _DATATYPE_FLOAT : inimg.md->datatype;
    IMGID   outimg = sdelay_connect_create_nd(sdelay_outimname, inimg, out_dt);
    if (UNLIKELY(outimg.ID == -1 || outimg.im == NULL))
    {
        PRINT_ERROR("Failed to create output stream \"%s\"", sdelay_outimname);
        imgid_free(&inimg);
        imgid_free(&outimg);
        return RETURN_FAILURE;
    }

    MILK_STREAM_DELAY_STATE state = { 0 };
    if (!sdelay_naive_mode)
    {
        if (UNLIKELY(milk_stream_delay_init(&state, inimg.im, sdelay_timebuffsize) !=
                     RETURN_SUCCESS))
        {
            imgid_free(&inimg);
            imgid_free(&outimg);
            return RETURN_FAILURE;
        }
    }

    IMGID wimg                                      = { 0 };
    wimg.ID                                         = -1;
    char cached_wname[FUNCTION_PARAMETER_STRMAXLEN] = "";

    if (UNLIKELY(dcdebug > 0))
    {
        list_image_ID();
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    if (UNLIKELY(!sdelay_naive_mode && state.ring_buf == NULL))
    {
        if (milk_stream_delay_init(&state, inimg.im, sdelay_timebuffsize) != RETURN_SUCCESS)
        {
            break;
        }
    }

    const IMAGE *w_im = sdelay_resolve_weights(sdelay_weightsname, &wimg, cached_wname);

    int unit_m                = (sdelay_unit != 0) ? MILK_SDELAY_UNIT_FRAME : MILK_SDELAY_UNIT_TIME;
    MILK_STREAM_DELAY_CFG cfg = {
        .delay     = (unit_m == MILK_SDELAY_UNIT_FRAME) ? sdelay_delayfr : (double) sdelay_delaysec,
        .window    = (unit_m == MILK_SDELAY_UNIT_FRAME) ? sdelay_avewidth
                                                        : (1.0e-9 * (double) sdelay_avedtns),
        .unit_mode = unit_m,
        .interp    = (sdelay_interp != 0) ? 1 : 0,
        .ave_mode  = (int) sdelay_avemode,
        .naive_mode = (sdelay_naive_mode != 0) ? 1 : 0,
        .force_push = 0,
        .allow_skip = 1,
    };

    int updated = 0;
    if (LIKELY(milk_stream_delay_step(&state, inimg.im, w_im, &cfg, outimg.im, &updated) ==
               RETURN_SUCCESS))
    {
        sdelay_statusframelag = state.status_framelag;
        sdelay_statuskkin     = state.status_kkin;
        sdelay_statuskkout    = state.status_kkout;
        if (updated != 0)
        {
            processinfo_update_output_stream(processinfo, outimg.im, inimg.im);
        }
    }

    MILK_PROCINFO_END(self)

    if (cached_wname[0] != '\0')
    {
        imgid_free(&wimg);
    }
    milk_stream_delay_free(&state);
    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_streamdelay = {
    .kind = MILK_RUNNABLE_FPS,
    .app =
        {
            .fps_name    = "streamdelay",
            .cmdkey      = "streamdelay",
            .description = "delay and temporally filter input stream to output stream",
            .description_long =
                "Delay, fractionally interpolate, or apply causal temporal window / 1D FIR "
                "filtering to a 1D/2D/3D image stream in time (seconds) or frame units.",
        },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(streamdelay_fps),
    .compute   = streamdelay_compute,
    .confcheck = sdelay_confcheck,
};
EXPORT_FPSEXEC(milk, "mem-streamdelay", runnable_COREMOD_memory_streamdelay);
