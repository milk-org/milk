// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help_search.c
 * @brief   Help topics search scoring and visible rows calculation for milk-CTRL.
 */

#include "overview_render_help_internal.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int index; /* Index in g_help_entries[] array */
    int score; /* Composite relevance score */
} help_search_match_t;

/**
 * compare_search_matches - qsort comparator for help search results by relevance score.
 * @a: Pointer to first help_search_match_t.
 * @b: Pointer to second help_search_match_t.
 *
 * Return: Negative if a > b (descending score), positive if b > a, or tie-break on index.
 */
static int compare_search_matches(const void *a, const void *b)
{
    const help_search_match_t *ma = (const help_search_match_t *) a;
    const help_search_match_t *mb = (const help_search_match_t *) b;
    if (mb->score != ma->score)
    {
        return mb->score - ma->score; /* descending score */
    }
    return ma->index - mb->index; /* stable tie-breaker */
}

/**
 * help_score_entry - score a help entry against a search query.
 * @entry: help entry to evaluate
 * @query: user search string
 *
 * Supports multi-token queries where all tokens must match (AND-logic).
 *
 * Return: score >= 0 (0 means no match).
 */
static int help_score_entry(const help_entry_t *entry, const char *query)
{
    if (entry == NULL || query == NULL || query[0] == '\0')
    {
        return 0;
    }

    char qbuf[64];
    strncpy(qbuf, query, sizeof(qbuf) - 1);
    qbuf[sizeof(qbuf) - 1] = '\0';

    char *tokens[8];
    int   ntok    = 0;
    char *saveptr = NULL;
    char *tok     = strtok_r(qbuf, " \t", &saveptr);
    while (tok != NULL && ntok < 8)
    {
        tokens[ntok++] = tok;
        tok            = strtok_r(NULL, " \t", &saveptr);
    }

    if (ntok == 0)
    {
        return 0;
    }

    int         total_score = 0;
    const char *sec_name    = ov_help_section_name(entry->section);
    const char *sec_tag     = ov_help_section_tag(entry->section);

    for (int t = 0; t < ntok; t++)
    {
        const char *w    = tokens[t];
        int         wlen = (int) strlen(w);
        if (wlen == 0)
        {
            continue;
        }

        int tok_score = 0;

        /* 1. Keystroke exact or prefix match */
        if (entry->key != NULL)
        {
            if (strcasecmp(entry->key, w) == 0)
            {
                tok_score += 350; /* Exact match on key, e.g. "k" or "F2" */
            }
            else if (strncasecmp(entry->key, w, (size_t) wlen) == 0)
            {
                tok_score += 180; /* Key prefix match */
            }
            else if (strcasestr(entry->key, w) != NULL)
            {
                tok_score += 90;
            }
        }

        /* 2. Label match (primary title summary) */
        if (entry->label != NULL)
        {
            const char *p = strcasestr(entry->label, w);
            if (p != NULL)
            {
                int at_boundary = (p == entry->label || *(p - 1) == ' ' || *(p - 1) == '/' ||
                                   *(p - 1) == '(' || *(p - 1) == '[' || *(p - 1) == '-');
                if (at_boundary)
                {
                    if (p[wlen] == '\0' || p[wlen] == ' ' || p[wlen] == '/' || p[wlen] == ')' ||
                        p[wlen] == ']')
                    {
                        tok_score += 160;
                    }
                    else
                    {
                        tok_score += 120;
                    }
                }
                else
                {
                    tok_score += 60;
                }
            }
        }

        /* 3. Section/topic match */
        if (sec_name != NULL && strcasestr(sec_name, w) != NULL)
        {
            tok_score += 70;
        }
        if (sec_tag != NULL && strcasecmp(sec_tag, w) == 0)
        {
            tok_score += 80;
        }

        /* 4. Detail documentation match */
        if (entry->detail != NULL)
        {
            const char *p = strcasestr(entry->detail, w);
            if (p != NULL)
            {
                tok_score += 35;
            }
        }

        /* Every token must match somewhere (AND logic) */
        if (tok_score == 0)
        {
            return 0;
        }

        total_score += tok_score;
    }

    /* Command entries get a priority boost over section headers */
    if (entry->flags & HF_ENTRY)
    {
        total_score += 15;
    }

    return total_score;
}

/**
 * help_visible_rows - count visible rows and populate mapping array.
 * @lay: layout state (for expand bitmask or active search query)
 * @map: output array mapping visible row index to g_help_entries[] index
 *
 * Return: number of visible rows.
 */
int help_visible_rows(const OV_LAYOUT *lay, int *map)
{
    /* If search query is non-empty, populate map with ranked search matches */
    if (lay->help_search[0] != '\0')
    {
        help_search_match_t matches[128];
        int                 n_matches = 0;

        for (int i = 0; i < g_help_total; i++)
        {
            int s = help_score_entry(&g_help_entries[i], lay->help_search);
            if (s > 0 && n_matches < 128)
            {
                matches[n_matches].index = i;
                matches[n_matches].score = s;
                n_matches++;
            }
        }

        if (n_matches > 1)
        {
            qsort(matches, (size_t) n_matches, sizeof(help_search_match_t), compare_search_matches);
        }

        for (int i = 0; i < n_matches; i++)
        {
            map[i] = matches[i].index;
        }
        return n_matches;
    }

    int vis = 0;
    for (int i = 0; i < g_help_total; i++)
    {
        if (g_help_entries[i].flags & HF_SECTION)
        {
            map[vis++] = i;
        }
        else if (g_help_entries[i].flags & (HF_ENTRY | HF_COLORS))
        {
            if (help_is_expanded(lay, g_help_entries[i].section))
            {
                map[vis++] = i;
            }
        }
    }
    return vis;
}
