// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_data_index.h
 * @brief Open-addressing hash index: 64-bit key hash -> array position, for O(1) lookups by name, inode, PID
 *
 * Linear probing over a power-of-two table; the value -1 marks an empty slot. Hashes may collide: lookups
 * confirm each candidate with a match function. Keep tables at most half full.
 */

#ifndef OVERVIEW_DATA_INDEX_H
#define OVERVIEW_DATA_INDEX_H

#include <stdint.h>

/* An index type with @cap slots (a power of two) */
#define OV_INDEX_DEF(cap)  \
    struct                 \
    {                      \
        uint64_t key[cap]; \
        int32_t  val[cap]; \
    }
#define OV_IX_CAP(ix) ((uint32_t) (sizeof((ix)->val) / sizeof((ix)->val[0])))

typedef int (*ov_ix_match_fn)(const void *ctx, int32_t val);

void    ov_ix_clear_(int32_t *val, uint32_t cap);
int     ov_ix_put_(uint64_t *key, int32_t *val, uint32_t cap, uint64_t h, int32_t v);
int     ov_ix_slot_(const uint64_t *key,
                    const int32_t  *val,
                    uint32_t        cap,
                    uint64_t        h,
                    ov_ix_match_fn  m,
                    const void     *ctx);
int32_t ov_ix_find_(const uint64_t *key,
                    const int32_t  *val,
                    uint32_t        cap,
                    uint64_t        h,
                    ov_ix_match_fn  m,
                    const void     *ctx);

#define ov_ix_clear(ix) ov_ix_clear_((ix)->val, OV_IX_CAP(ix))
#define ov_ix_put(ix, h, v) ov_ix_put_((ix)->key, (ix)->val, OV_IX_CAP(ix), (h), (v))
#define ov_ix_slot(ix, h, m, c) ov_ix_slot_((ix)->key, (ix)->val, OV_IX_CAP(ix), (h), (m), (c))
#define ov_ix_find(ix, h, m, c) ov_ix_find_((ix)->key, (ix)->val, OV_IX_CAP(ix), (h), (m), (c))

uint64_t ov_hash_str(const char *s);
uint64_t ov_hash_u64(uint64_t x);

#endif /* OVERVIEW_DATA_INDEX_H */
