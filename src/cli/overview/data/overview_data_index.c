// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_index.c
 * @brief   Open-addressing hash index (see overview_data_index.h) and the hashes it is keyed with.
 */

#include <string.h>

#include "overview_data_index.h"

/** ov_ix_clear_ - empty an index of @cap slots. */
void ov_ix_clear_(int32_t *val, uint32_t cap)
{
    memset(val, 0xff, cap * sizeof(*val));
}

/** ov_ix_put_ - add @v under hash @h (duplicates allowed). Return: 0, or -1 when the table is full. */
int ov_ix_put_(uint64_t *key, int32_t *val, uint32_t cap, uint64_t h, int32_t v)
{
    uint32_t mask = cap - 1;
    for (uint32_t n = 0, i = (uint32_t) h & mask; n < cap; n++, i = (i + 1) & mask)
    {
        if (val[i] < 0)
        {
            key[i] = h;
            val[i] = v;
            return 0;
        }
    }
    return -1;
}

/** ov_ix_slot_ - slot of the first value under hash @h that @m accepts, or -1. */
int ov_ix_slot_(const uint64_t *key,
                const int32_t  *val,
                uint32_t        cap,
                uint64_t        h,
                ov_ix_match_fn  m,
                const void     *ctx)
{
    uint32_t mask = cap - 1;
    for (uint32_t n = 0, i = (uint32_t) h & mask; n < cap && val[i] >= 0; n++, i = (i + 1) & mask)
    {
        if (key[i] == h && m(ctx, val[i]))
        {
            return (int) i;
        }
    }
    return -1;
}

/** ov_ix_find_ - first value under hash @h that @m accepts, or -1. */
int32_t ov_ix_find_(const uint64_t *key,
                    const int32_t  *val,
                    uint32_t        cap,
                    uint64_t        h,
                    ov_ix_match_fn  m,
                    const void     *ctx)
{
    int s = ov_ix_slot_(key, val, cap, h, m, ctx);
    return (s >= 0) ? val[s] : -1;
}

/** ov_hash_str - 64-bit FNV-1a of a string. */
uint64_t ov_hash_str(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    for (; *s != '\0'; s++)
    {
        h = (h ^ (unsigned char) *s) * 1099511628211ULL;
    }
    return h;
}

/** ov_hash_u64 - mix a 64-bit integer (splitmix64 finalizer), so nearby numbers spread over the table. */
uint64_t ov_hash_u64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
