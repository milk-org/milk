// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_filter.c
 * @brief   Name filtering by regex (substring fallback) with a small compiled-regex cache.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE /* strcasestr */
#endif

#include "overview_data_internal.h"
#include <regex.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define OV_FILTER_CACHE_SIZE 4

typedef struct
{
    char     pattern[64];
    regex_t  re;
    int      state; /* 0 empty, 1 compiled, -1 regcomp failed (negative entry, nothing to free) */
    uint64_t used;  /* last-use tick, for LRU eviction */
} ov_filter_cache_entry_t;

static ov_filter_cache_entry_t s_filter_cache[OV_FILTER_CACHE_SIZE];
static uint64_t                s_filter_tick;
static unsigned long           s_filter_compiles;

/** ov_filter_compile_count - number of regcomp() calls so far (cache behaviour checks). */
unsigned long ov_filter_compile_count(void)
{
    return s_filter_compiles;
}

/**
 * get_cached_regex - compiled @pattern from a small LRU cache. Failed patterns are cached too, so an
 * invalid filter is not recompiled every frame and does not evict the other panels' regexes.
 * Return: the regex with *@out_reg_ok = 1, or NULL with 0 when @pattern does not compile.
 */
static regex_t *get_cached_regex(const char *pattern, int *out_reg_ok)
{
    int slot = 0;
    for (int i = 0; i < OV_FILTER_CACHE_SIZE; i++)
    {
        ov_filter_cache_entry_t *e = &s_filter_cache[i];
        if (e->state != 0 && strcmp(e->pattern, pattern) == 0)
        {
            e->used     = ++s_filter_tick;
            *out_reg_ok = (e->state == 1);
            return (e->state == 1) ? &e->re : NULL;
        }
        if (e->used < s_filter_cache[slot].used)
        {
            slot = i; /* least recently used (empty slots have used == 0) */
        }
    }

    ov_filter_cache_entry_t *e = &s_filter_cache[slot];
    if (e->state == 1)
    {
        regfree(&e->re);
    }
    snprintf(e->pattern, sizeof(e->pattern), "%s", pattern);
    s_filter_compiles++;
    e->state    = (regcomp(&e->re, pattern, REG_EXTENDED | REG_NOSUB | REG_ICASE) == 0) ? 1 : -1;
    e->used     = ++s_filter_tick;
    *out_reg_ok = (e->state == 1);
    return (e->state == 1) ? &e->re : NULL;
}

/**
 * ov_filter_build - filter a string list using POSIX regex or case-insensitive substring matching.
 * @pattern: Regex or substring pattern.
 * @names:   Array of name strings to test.
 * @count:   Number of items in names array.
 * @out:     Output buffer to receive indices of matching items.
 * @max_out: Maximum capacity of out buffer.
 *
 * Return: Number of matching indices written to out buffer.
 */
int ov_filter_build(const char *pattern, const char **names, int count, int *out, int max_out)
{
    if (pattern == NULL || pattern[0] == '\0')
    {
        /* No filter — all items match */
        int n = count < max_out ? count : max_out;
        for (int i = 0; i < n; i++)
        {
            out[i] = i;
        }
        return n;
    }

    int      reg_ok = 0;
    regex_t *re     = get_cached_regex(pattern, &reg_ok);
    int      n      = 0;

    if (reg_ok && re != NULL)
    {
        for (int i = 0; i < count && n < max_out; i++)
        {
            if (names[i] != NULL && regexec(re, names[i], 0, NULL, 0) == 0)
            {
                out[n++] = i;
            }
        }
    }
    else
    {
        /* Fallback: case-insensitive literal substring search */
        for (int i = 0; i < count && n < max_out; i++)
        {
            if (names[i] != NULL && strcasestr(names[i], pattern) != NULL)
            {
                out[n++] = i;
            }
        }
    }

    return n;
}
