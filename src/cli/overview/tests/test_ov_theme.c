// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_theme: every palette sets every colour slot (a missed slot would draw black). */

#include "overview_theme.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

int main(void)
{
    /* Slots that may be black on purpose: the terminal background of a dark palette */
    static const size_t black_ok[] = { offsetof(ov_theme_t, bg_terminal) };
    const size_t        first      = offsetof(ov_theme_t, bg_terminal);
    const size_t        last       = offsetof(ov_theme_t, fg_accent) + sizeof(ov_rgb_t);
    const size_t        nslots     = (last - first) / sizeof(ov_rgb_t);

    CHECK(sizeof(ov_theme_t) - last <
          sizeof(ov_rgb_t)); /* fg_accent is the last slot: only padding after */
    CHECK(ov_theme_count() >= 10);
    for (int i = 0; i < ov_theme_count(); i++)
    {
        const ov_theme_t *t = ov_theme_get(i);
        CHECK(t->id != NULL && t->name != NULL && t->desc != NULL);
        for (size_t k = 0; k < nslots; k++)
        {
            size_t          off = first + k * sizeof(ov_rgb_t);
            const ov_rgb_t *c   = (const ov_rgb_t *) ((const char *) t + off);
            int             ok  = (c->r | c->g | c->b) != 0;
            for (size_t b = 0; b < sizeof(black_ok) / sizeof(black_ok[0]); b++)
            {
                ok |= (off == black_ok[b]);
            }
            if (!ok)
            {
                fprintf(stderr, "theme %s: slot %zu is black\n", t->id, k);
            }
            CHECK(ok);
            CHECK(c->r >= 0 && c->r <= 255 && c->g >= 0 && c->g <= 255 && c->b >= 0 && c->b <= 255);
        }
    }
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
