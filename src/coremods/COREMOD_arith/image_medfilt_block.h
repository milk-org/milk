// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_medfilt_block.h
 * @brief   Two-pass 3x3, 5x5, and 7x7 Young-tableau SIMD median filter kernels, instantiated per type
 *
 * Included once per element type (`FT` = `double` or `float`, `FK(name)` = `name##_d` or `name##_f`).
 * No include guard: designed to be included once per element type.
 */

#ifndef FUSE_BLK
#    define FUSE_BLK 1024
#    define ARITH_MEDFILT_UNDEF_FUSE_BLK 1
#endif

#define PIX_SORT(a, b)                   \
    do                                   \
    {                                    \
        FT t_ = ((a) < (b)) ? (a) : (b); \
        (b)   = ((a) < (b)) ? (b) : (a); \
        (a)   = t_;                      \
    } while (0)

#define SORT5(a, b, c, e, f) \
    do                       \
    {                        \
        PIX_SORT(a, b);      \
        PIX_SORT(e, f);      \
        PIX_SORT(c, f);      \
        PIX_SORT(c, e);      \
        PIX_SORT(a, e);      \
        PIX_SORT(a, c);      \
        PIX_SORT(b, f);      \
        PIX_SORT(b, e);      \
        PIX_SORT(b, c);      \
    } while (0)

#define SORT6(a, b, c, e, f, g) \
    do                          \
    {                           \
        PIX_SORT(b, c);         \
        PIX_SORT(f, g);         \
        PIX_SORT(a, c);         \
        PIX_SORT(e, g);         \
        PIX_SORT(a, b);         \
        PIX_SORT(e, f);         \
        PIX_SORT(a, e);         \
        PIX_SORT(c, g);         \
        PIX_SORT(b, f);         \
        PIX_SORT(c, e);         \
        PIX_SORT(b, c);         \
        PIX_SORT(e, f);         \
    } while (0)

#define SORT7(a, b, c, e, f, g, h) \
    do                             \
    {                              \
        PIX_SORT(b, c);            \
        PIX_SORT(e, f);            \
        PIX_SORT(g, h);            \
        PIX_SORT(a, c);            \
        PIX_SORT(e, g);            \
        PIX_SORT(f, h);            \
        PIX_SORT(a, b);            \
        PIX_SORT(f, g);            \
        PIX_SORT(c, h);            \
        PIX_SORT(a, f);            \
        PIX_SORT(b, g);            \
        PIX_SORT(a, e);            \
        PIX_SORT(c, g);            \
        PIX_SORT(b, e);            \
        PIX_SORT(c, f);            \
        PIX_SORT(c, e);            \
    } while (0)

/**
 * FK(medfilt3_cols) - Run two-pass 3x3 Young-tableau median filter on clamped row strip @col
 * @N:   Number of output pixels in the row segment (`<= FUSE_BLK`)
 * @col: 3 clamped row strips of length `N + 2`
 * @d:   Destination buffer of length @N
 */
static inline void FK(medfilt3_cols)(int N, FT col[7][FUSE_BLK + 6], FT *restrict d)
{
    const int W = N + 2;
    _Pragma("omp simd") for (int u = 0; u < W; u++)
    {
        FT c0 = col[0][u], c1 = col[1][u], c2 = col[2][u];
        PIX_SORT(c0, c1);
        PIX_SORT(c1, c2);
        PIX_SORT(c0, c1);
        col[0][u] = c0;
        col[1][u] = c1;
        col[2][u] = c2;
    }
    const FT *r0 = col[0], *r1 = col[1], *r2 = col[2];
    _Pragma("omp simd") for (int i = 0; i < N; i++)
    {
        FT a0 = r0[i], a1 = r0[i + 1], a2 = r0[i + 2];
        FT a  = (a0 > a1) ? a0 : a1;
        a     = (a > a2) ? a : a2;
        FT b0 = r1[i], b1 = r1[i + 1], b2 = r1[i + 2];
        PIX_SORT(b0, b1);
        PIX_SORT(b1, b2);
        PIX_SORT(b0, b1);
        FT b  = b1;
        FT c0 = r2[i], c1 = r2[i + 1], c2 = r2[i + 2];
        FT c = (c0 < c1) ? c0 : c1;
        c    = (c < c2) ? c : c2;
        PIX_SORT(a, b);
        PIX_SORT(b, c);
        PIX_SORT(a, b);
        d[i] = b;
    }
}

/**
 * FK(medfilt5_cols) - Run two-pass 5x5 Young-tableau median filter on clamped row strip @col
 * @N:   Number of output pixels in the row segment (`<= FUSE_BLK`)
 * @col: 5 clamped row strips of length `N + 4`
 * @d:   Destination buffer of length @N
 */
static inline void FK(medfilt5_cols)(int N, FT col[7][FUSE_BLK + 6], FT *restrict d)
{
    const int W = N + 4;
    _Pragma("omp simd") for (int u = 0; u < W; u++)
    {
        FT c0 = col[0][u], c1 = col[1][u], c2 = col[2][u], c3 = col[3][u], c4 = col[4][u];
        SORT5(c0, c1, c2, c3, c4);
        col[0][u] = c0;
        col[1][u] = c1;
        col[2][u] = c2;
        col[3][u] = c3;
        col[4][u] = c4;
    }
    const FT *r0 = col[0], *r1 = col[1], *r2 = col[2], *r3 = col[3], *r4 = col[4];
    _Pragma("omp simd") for (int i = 0; i < N; i++)
    {
        FT w0 = r0[i], w1 = r0[i + 1], w2 = r0[i + 2], w3 = r0[i + 3], w4 = r0[i + 4];
        SORT5(w0, w1, w2, w3, w4);
        FT d3_max = w3, m0 = w4;

        w0 = r1[i];
        w1 = r1[i + 1];
        w2 = r1[i + 2];
        w3 = r1[i + 3];
        w4 = r1[i + 4];
        SORT5(w0, w1, w2, w3, w4);
        d3_max = (w2 > d3_max) ? w2 : d3_max;
        FT m1 = w3, d5_min = w4;

        w0 = r2[i];
        w1 = r2[i + 1];
        w2 = r2[i + 2];
        w3 = r2[i + 3];
        w4 = r2[i + 4];
        SORT5(w0, w1, w2, w3, w4);
        d3_max = (w1 > d3_max) ? w1 : d3_max;
        FT m2  = w2;
        d5_min = (w3 < d5_min) ? w3 : d5_min;

        w0 = r3[i];
        w1 = r3[i + 1];
        w2 = r3[i + 2];
        w3 = r3[i + 3];
        w4 = r3[i + 4];
        SORT5(w0, w1, w2, w3, w4);
        d3_max = (w0 > d3_max) ? w0 : d3_max;
        FT m3  = w1;
        d5_min = (w2 < d5_min) ? w2 : d5_min;

        w0 = r4[i];
        w1 = r4[i + 1];
        w2 = r4[i + 2];
        w3 = r4[i + 3];
        w4 = r4[i + 4];
        SORT5(w0, w1, w2, w3, w4);
        FT m4  = w0;
        d5_min = (w1 < d5_min) ? w1 : d5_min;

        SORT5(m0, m1, m2, m3, m4);
        PIX_SORT(d3_max, m2);
        PIX_SORT(m2, d5_min);
        PIX_SORT(d3_max, m2);
        d[i] = m2;
    } /* for (int i = 0; i < N; i++) */
}

#define LOAD_ROW7(row, i)                  \
    do                                     \
    {                                      \
        w0 = (row)[(i) + 0];               \
        w1 = (row)[(i) + 1];               \
        w2 = (row)[(i) + 2];               \
        w3 = (row)[(i) + 3];               \
        w4 = (row)[(i) + 4];               \
        w5 = (row)[(i) + 5];               \
        w6 = (row)[(i) + 6];               \
        SORT7(w0, w1, w2, w3, w4, w5, w6); \
    } while (0)

/**
 * FK(medfilt7_cols) - Run two-pass 7x7 Young-tableau median filter on clamped row strip @col
 * @N:   Number of output pixels in the row segment (`<= FUSE_BLK`)
 * @col: 7 clamped row strips of length `N + 6`
 * @d:   Destination buffer of length @N
 */
static inline void FK(medfilt7_cols)(int N, FT col[7][FUSE_BLK + 6], FT *restrict d)
{
    const int W = N + 6;
    _Pragma("omp simd") for (int u = 0; u < W; u++)
    {
        FT c0 = col[0][u], c1 = col[1][u], c2 = col[2][u], c3 = col[3][u];
        FT c4 = col[4][u], c5 = col[5][u], c6 = col[6][u];
        SORT7(c0, c1, c2, c3, c4, c5, c6);
        col[0][u] = c0;
        col[1][u] = c1;
        col[2][u] = c2;
        col[3][u] = c3;
        col[4][u] = c4;
        col[5][u] = c5;
        col[6][u] = c6;
    }
    const FT *r0 = col[0], *r1 = col[1], *r2 = col[2], *r3 = col[3];
    const FT *r4 = col[4], *r5 = col[5], *r6 = col[6];
    _Pragma("omp simd") for (int i = 0; i < N; i++)
    {
        FT w0, w1, w2, w3, w4, w5, w6;
        LOAD_ROW7(r0, i);
        FT d4_max = w4, p0 = w5, q0 = w6;
        LOAD_ROW7(r1, i);
        d4_max = (w3 > d4_max) ? w3 : d4_max;
        FT p1 = w4, q1 = w5, s0 = w6;
        LOAD_ROW7(r2, i);
        d4_max = (w2 > d4_max) ? w2 : d4_max;
        FT p2 = w3, q2 = w4, s1 = w5, d8_min = w6;
        LOAD_ROW7(r3, i);
        d4_max = (w1 > d4_max) ? w1 : d4_max;
        FT p3 = w2, q3 = w3, s2 = w4;
        d8_min = (w5 < d8_min) ? w5 : d8_min;
        LOAD_ROW7(r4, i);
        d4_max = (w0 > d4_max) ? w0 : d4_max;
        FT p4 = w1, q4 = w2, s3 = w3;
        d8_min = (w4 < d8_min) ? w4 : d8_min;
        LOAD_ROW7(r5, i);
        FT p5 = w0, q5 = w1, s4 = w2;
        d8_min = (w3 < d8_min) ? w3 : d8_min;
        LOAD_ROW7(r6, i);
        FT q6 = w0, s5 = w1;
        d8_min = (w2 < d8_min) ? w2 : d8_min;

        SORT6(p0, p1, p2, p3, p4, p5);
        SORT7(q0, q1, q2, q3, q4, q5, q6);
        SORT6(s0, s1, s2, s3, s4, s5);

        FT e0 = d4_max, e1 = p3, e2 = p4, e3 = p5, e4 = q2, e5 = q3;
        FT e6 = q4, e7 = s0, e8 = s1, e9 = s2, e10 = d8_min;
        PIX_SORT(e0, e1);
        PIX_SORT(e3, e4);
        PIX_SORT(e2, e4);
        PIX_SORT(e2, e3);
        PIX_SORT(e0, e3);
        PIX_SORT(e0, e2);
        PIX_SORT(e1, e4);
        PIX_SORT(e1, e3);
        PIX_SORT(e1, e2);
        PIX_SORT(e6, e7);
        PIX_SORT(e5, e7);
        PIX_SORT(e5, e6);
        PIX_SORT(e9, e10);
        PIX_SORT(e8, e10);
        PIX_SORT(e8, e9);
        PIX_SORT(e5, e8);
        PIX_SORT(e6, e9);
        PIX_SORT(e7, e10);
        PIX_SORT(e7, e9);
        PIX_SORT(e6, e8);
        PIX_SORT(e7, e8);
        PIX_SORT(e0, e6);
        PIX_SORT(e0, e5);
        PIX_SORT(e1, e7);
        PIX_SORT(e1, e6);
        PIX_SORT(e1, e5);
        PIX_SORT(e2, e8);
        PIX_SORT(e3, e9);
        PIX_SORT(e4, e10);
        PIX_SORT(e4, e9);
        PIX_SORT(e3, e8);
        PIX_SORT(e4, e8);
        PIX_SORT(e2, e5);
        PIX_SORT(e3, e6);
        PIX_SORT(e4, e7);
        PIX_SORT(e4, e6);
        PIX_SORT(e3, e5);
        PIX_SORT(e4, e5);
        d[i] = e5;
    } /* for (int i = 0; i < N; i++) */
}
#undef LOAD_ROW7

/**
 * FK(arith_medfilt_cols) - Dispatch two-pass KxK Young-tableau median filter on @col
 * @K:   Window width (3, 5, or 7)
 * @N:   Number of output pixels in the row segment (`<= FUSE_BLK`)
 * @col: K clamped row strips of length `N + K - 1`
 * @d:   Destination buffer of length @N
 */
static inline void FK(arith_medfilt_cols)(int K, int N, FT col[7][FUSE_BLK + 6], FT *restrict d)
{
    if (K == 3)
    {
        FK(medfilt3_cols)(N, col, d);
    }
    else if (K == 5)
    {
        FK(medfilt5_cols)(N, col, d);
    }
    else
    {
        FK(medfilt7_cols)(N, col, d);
    }
}

#undef SORT7
#undef SORT6
#undef SORT5
#undef PIX_SORT
#ifdef ARITH_MEDFILT_UNDEF_FUSE_BLK
#    undef ARITH_MEDFILT_UNDEF_FUSE_BLK
#    undef FUSE_BLK
#endif
