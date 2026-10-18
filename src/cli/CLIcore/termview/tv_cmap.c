// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_cmap.c
 * @brief Colormap, scale and range name tables.
 */

#include <strings.h>

#include "termview_internal.h"

#define TV_COUNT(a) ((int) (sizeof(a) / sizeof((a)[0])))

const char *const tv_cmap_names[COLORMAP_NB] = {
    "greyscale", "heat",  "cold",  "jet",    "inferno", "viridis", "magma",  "plasma", "bone",
    "rainbow",   "turbo", "ocean", "copper", "spring",  "summer",  "autumn", "winter",
};

const char *const tv_scale_names[SCALE_NB] = {
    "linear", "sqrt", "log", "log_strong", "log_extreme", "asinh", "squared", "cubed",
};

const char *const tv_range_names[RANGE_NB] = {
    "minmax", "0.1-99.9%", "0.5-99.5%", "1-99%", "5-95%", "10-90%", "15-85%", "20-80%",
};

/** Short CLI spellings of the range modes (lower percentile, or "minmax"). */
const char *const tv_range_keys[RANGE_NB] = {
    "minmax", "0.1", "0.5", "1", "5", "10", "15", "20",
};

_Static_assert(TV_COUNT(tv_cmap_names) == COLORMAP_NB, "colormap name table out of sync");
_Static_assert(TV_COUNT(tv_scale_names) == SCALE_NB, "scale name table out of sync");
_Static_assert(TV_COUNT(tv_range_names) == RANGE_NB, "range name table out of sync");
_Static_assert(TV_COUNT(tv_range_keys) == RANGE_NB, "range key table out of sync");

int tv_lookup_name(const char *const *names, int n, const char *s)
{
    if (s == NULL || *s == '\0')
    {
        return -1;
    }
    size_t len   = strlen(s);
    int    match = -1;
    for (int i = 0; i < n; i++)
    {
        if (strcasecmp(names[i], s) == 0)
        {
            return i;
        }
        if (strncasecmp(names[i], s, len) == 0)
        {
            match = (match == -1) ? i : -2;
        }
    }
    return (match >= 0) ? match : -1;
}
