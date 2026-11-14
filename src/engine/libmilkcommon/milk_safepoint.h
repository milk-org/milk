// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_safepoint.h
 * @brief Points of a long computation where its data is whole
 *
 * A computation that runs for long (a calculator loop, a processing loop) calls
 * milk_safe_point() where the images and variables it works on are in a state that another
 * reader may see: between two statements, between two iterations. A session's server sets
 * milk_safe_point_fn and answers there the requests that wait for the command running. In any
 * other process the pointer is NULL and the call is one test.
 */

#ifndef MILK_SAFEPOINT_H
#define MILK_SAFEPOINT_H

#include <stddef.h>

/** What milk_safe_point() calls, or NULL. Set once, before any thread computes. */
extern void (*milk_safe_point_fn)(void);

/** milk_safe_point() - tell the host that the data of this computation is whole at this point. */
static inline void milk_safe_point(void)
{
    if (milk_safe_point_fn != NULL)
    {
        milk_safe_point_fn();
    }
}

#endif
