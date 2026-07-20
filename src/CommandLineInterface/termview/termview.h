// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _TERMVIEW_H
#define _TERMVIEW_H

#include <stdbool.h>
#include "CommandLineInterface/CLIcore.h"

typedef enum
{
    COLORMAP_GREYSCALE = 0,
    COLORMAP_HEAT,
    COLORMAP_COLD,
    COLORMAP_JET,
    COLORMAP_NB
} termview_colormap_t;

typedef enum
{
    SCALE_LINEAR = 0,
    SCALE_SQRT,
    SCALE_LOG,
    SCALE_NB
} termview_scale_t;

typedef enum
{
    RANGE_MINMAX = 0,
    RANGE_01_99,
    RANGE_05_95,
    RANGE_10_90,
    RANGE_MANUAL, // User defined
    RANGE_NB
} termview_range_t;

typedef struct
{
    termview_colormap_t colormap;
    termview_scale_t    scale;
    termview_range_t    range;
    bool                range_locked; // If true, min/max are frozen (or manual)
    double              manual_min;
    double              manual_max;
} termview_options_t;

errno_t termview_screen(const char *imagename, termview_options_t options);

#endif
