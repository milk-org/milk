// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_cmap.c
 * @brief Colormaps, intensity scalings, LUT construction and the option name tables.
 */

#include <math.h>
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

tv_rgb_t tv_cmap_color(termview_colormap_t cmap, double v)
{
    if (v < 0.0)
    {
        v = 0.0;
    }
    if (v > 1.0)
    {
        v = 1.0;
    }

    tv_rgb_t c = { 0, 0, 0 };
    if (cmap == COLORMAP_GREYSCALE)
    {
        c.r = c.g = c.b = (uint8_t) (v * 255.0f);
    }
    else
    {
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (cmap == COLORMAP_HEAT)
        {
            if (v < 0.25f)
            {
                b = v * 4.0f;
            }
            else if (v < 0.5f)
            {
                b = 1.0f;
                g = (v - 0.25f) * 4.0f;
            }
            else if (v < 0.75f)
            {
                b = 1.0f - (v - 0.5f) * 4.0f;
                g = 1.0f;
                r = (v - 0.5f) * 4.0f;
            }
            else
            {
                g = 1.0f - (v - 0.75f) * 4.0f;
                r = 1.0f;
            }
        }
        else if (cmap == COLORMAP_COLD)
        {
            if (v < 0.33f)
            {
                b = v * 3.0f;
            }
            else if (v < 0.66f)
            {
                b = 1.0f;
                g = (v - 0.33f) * 3.0f;
            }
            else
            {
                b = 1.0f;
                g = 1.0f;
                r = (v - 0.66f) * 3.0f;
            }
        }
        else if (cmap == COLORMAP_JET)
        {
            if (v < 0.25f)
            {
                b = 1.0f;
                g = v * 4.0f;
            }
            else if (v < 0.5f)
            {
                b = 1.0f - (v - 0.25f) * 4.0f;
                g = 1.0f;
            }
            else if (v < 0.75f)
            {
                g = 1.0f;
                r = (v - 0.5f) * 4.0f;
            }
            else
            {
                g = 1.0f - (v - 0.75f) * 4.0f;
                r = 1.0f;
            }
        }
        else if (cmap == COLORMAP_INFERNO)
        {
            if (v < 0.33f)
            {
                r = v * 1.5f;
                b = v * 2.0f;
            }
            else if (v < 0.66f)
            {
                r = 0.5f + (v - 0.33f) * 1.5f;
                g = (v - 0.33f) * 1.5f;
                b = 0.66f - (v - 0.33f) * 2.0f;
            }
            else
            {
                r = 1.0f;
                g = 0.5f + (v - 0.66f) * 1.5f;
            }
        }
        else if (cmap == COLORMAP_VIRIDIS)
        {
            if (v < 0.33f)
            {
                r = 0.2f;
                g = v * 1.5f;
                b = 0.4f + v * 1.2f;
            }
            else if (v < 0.66f)
            {
                r = 0.2f;
                g = 0.5f + (v - 0.33f) * 1.0f;
                b = 0.8f - (v - 0.33f) * 1.5f;
            }
            else
            {
                r = 0.2f + (v - 0.66f) * 2.4f;
                g = 0.83f + (v - 0.66f) * 0.5f;
                b = 0.3f - (v - 0.66f) * 0.9f;
            }
        }
        else if (cmap == COLORMAP_MAGMA)
        {
            if (v < 0.25f)
            {
                r = v * 2.0f;
                g = 0.0f;
                b = v * 2.0f;
            }
            else if (v < 0.5f)
            {
                r = 0.5f + (v - 0.25f) * 2.0f;
                g = 0.0f;
                b = 0.5f - (v - 0.25f) * 2.0f;
            }
            else if (v < 0.75f)
            {
                r = 1.0f;
                g = (v - 0.5f) * 2.0f;
                b = 0.0f;
            }
            else
            {
                r = 1.0f;
                g = 0.5f + (v - 0.75f) * 2.0f;
                b = (v - 0.75f) * 4.0f;
            }
        }
        else if (cmap == COLORMAP_PLASMA)
        {
            if (v < 0.33f)
            {
                r = v * 1.5f;
                g = 0.0f;
                b = 0.5f + v * 1.5f;
            }
            else if (v < 0.66f)
            {
                r = 0.5f + (v - 0.33f) * 1.5f;
                g = (v - 0.33f) * 1.5f;
                b = 1.0f - (v - 0.33f) * 1.5f;
            }
            else
            {
                r = 1.0f;
                g = 0.5f + (v - 0.66f) * 1.5f;
                b = 0.5f - (v - 0.66f) * 1.5f;
            }
        }
        else if (cmap == COLORMAP_BONE)
        {
            if (v < 0.33f)
            {
                r = v * 1.5f;
                g = v * 1.5f;
                b = v * 2.0f;
            }
            else if (v < 0.66f)
            {
                r = 0.5f + (v - 0.33f) * 1.0f;
                g = 0.5f + (v - 0.33f) * 1.5f;
                b = 0.66f + (v - 0.33f) * 1.0f;
            }
            else
            {
                r = 0.83f + (v - 0.66f) * 0.5f;
                g = 1.0f;
                b = 1.0f;
            }
        }
        else if (cmap == COLORMAP_RAINBOW)
        {
            float h = (1.0f - v) * 5.0f;
            int   i = (int) h;
            float f = h - i;
            float q = 1.0f - f;
            switch (i)
            {
            case 0:
                r = 1.0f;
                g = f;
                b = 0.0f;
                break;
            case 1:
                r = q;
                g = 1.0f;
                b = 0.0f;
                break;
            case 2:
                r = 0.0f;
                g = 1.0f;
                b = f;
                break;
            case 3:
                r = 0.0f;
                g = q;
                b = 1.0f;
                break;
            case 4:
                r = f;
                g = 0.0f;
                b = 1.0f;
                break;
            case 5:
                r = 1.0f;
                g = 0.0f;
                b = 1.0f;
                break;
            default:
                r = 1.0f;
                g = 0.0f;
                b = 1.0f;
                break;
            }
        }
        else if (cmap == COLORMAP_TURBO)
        {
            float k1 =
                0.1357f +
                v * (4.5974f - v * (42.3277f - v * (130.5887f - v * (150.5666f - v * 58.1375f))));
            float k2 = 0.0914f +
                       v * (2.1941f + v * (4.8429f - v * (14.1850f - v * (4.2773f + v * 2.8251f))));
            float k3 =
                0.1066f +
                v * (12.6419f - v * (60.5820f - v * (110.3627f - v * (89.9031f - v * 27.3482f))));
            r = k1;
            g = k2;
            b = k3;
        }
        else if (cmap == COLORMAP_OCEAN)
        {
            r = v * 0.5f;
            g = v * 0.8f;
            b = 0.2f + v * 0.8f;
        }
        else if (cmap == COLORMAP_COPPER)
        {
            r = v * 1.25f;
            g = v * 0.7812f;
            b = v * 0.4975f;
        }
        else if (cmap == COLORMAP_SPRING)
        {
            r = 1.0f;
            g = v;
            b = 1.0f - v;
        }
        else if (cmap == COLORMAP_SUMMER)
        {
            r = v;
            g = 0.5f + v * 0.5f;
            b = 0.4f;
        }
        else if (cmap == COLORMAP_AUTUMN)
        {
            r = 1.0f;
            g = v;
            b = 0.0f;
        }
        else if (cmap == COLORMAP_WINTER)
        {
            r = 0.0f;
            g = v;
            b = 1.0f - v * 0.5f;
        }
        if (r > 1.0f)
        {
            r = 1.0f;
        }
        if (r < 0.0f)
        {
            r = 0.0f;
        }
        if (g > 1.0f)
        {
            g = 1.0f;
        }
        if (g < 0.0f)
        {
            g = 0.0f;
        }
        if (b > 1.0f)
        {
            b = 1.0f;
        }
        if (b < 0.0f)
        {
            b = 0.0f;
        }
        c.r = (uint8_t) (r * 255.0f);
        c.g = (uint8_t) (g * 255.0f);
        c.b = (uint8_t) (b * 255.0f);
    }
    return c;
}

void tv_lut_build(tv_rgb_t *lut, termview_colormap_t cmap, termview_scale_t scale)
{
    for (int i = 0; i < TV_LUT_N; i++)
    {
        double v = i / (double) (TV_LUT_N - 1);
        if (scale == SCALE_SQRT)
        {
            v = sqrt(v);
        }
        else if (scale == SCALE_LOG)
        {
            v = log(v * 9.0 + 1.0) / log(10.0);
        }
        else if (scale == SCALE_LOG_STRONG)
        {
            v = log(v * 99.0 + 1.0) / log(100.0);
        }
        else if (scale == SCALE_LOG_EXTREME)
        {
            v = log(v * 999.0 + 1.0) / log(1000.0);
        }
        else if (scale == SCALE_ASINH)
        {
            v = asinh(v * 10.0) / asinh(10.0);
        }
        else if (scale == SCALE_SQUARED)
        {
            v = v * v;
        }
        else if (scale == SCALE_CUBED)
        {
            v = v * v * v;
        }

        lut[i] = tv_cmap_color(cmap, v);
    }
    lut[TV_LUT_N] = (tv_rgb_t) { 255, 0, 255 };
}
