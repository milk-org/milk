// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_units.c
 * @brief   Number formatting with units, so values keep a bounded width.
 */

#include "overview_render_internal.h"

/** ov_fmt_dt_ns - a duration as "850ns", "12.3us", "1.25ms", "2.40s" (at most 7 columns); "-" when < 0. */
const char *ov_fmt_dt_ns(char *buf, size_t n, int64_t ns)
{
    double v = (double) ns;
    if (ns < 0)
    {
        snprintf(buf, n, "-");
    }
    else if (ns < 1000)
    {
        snprintf(buf, n, "%dns", (int) ns);
    }
    else if (ns < 1000000)
    {
        snprintf(buf, n, (v < 1e4) ? "%.2fus" : (v < 1e5) ? "%.1fus" : "%.0fus", v / 1e3);
    }
    else if (ns < 1000000000)
    {
        snprintf(buf, n, (v < 1e7) ? "%.2fms" : (v < 1e8) ? "%.1fms" : "%.0fms", v / 1e6);
    }
    else
    {
        snprintf(buf, n, (v < 1e10) ? "%.2fs" : (v < 1e11) ? "%.1fs" : "%.0fs", v / 1e9);
    }
    return buf;
}

/** ov_fmt_si - @v with an SI suffix in at most 6 columns: "999.9", "12.3k", "1.20M", "45G"; "-" when < 0. */
const char *ov_fmt_si(char *buf, size_t n, double v)
{
    static const char suffix[] = " kMGTP";
    int               k        = 0;
    if (v < 0)
    {
        snprintf(buf, n, "-");
        return buf;
    }
    while (v >= 999.95 && k < 5)
    {
        v /= 1000.0;
        k++;
    }
    if (k == 0)
    {
        snprintf(buf, n, (v == (double) (int64_t) v) ? "%.0f" : "%.1f", v);
    }
    else
    {
        snprintf(buf, n, (v < 9.995) ? "%.2f%c" : (v < 99.95) ? "%.1f%c" : "%.0f%c", v, suffix[k]);
    }
    return buf;
}
