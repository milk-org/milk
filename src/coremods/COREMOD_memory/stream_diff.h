// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_diff.h
 * @brief Compute stream difference
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_streamdiff;

imageID COREMOD_MEMORY_streamDiff(const char *IDstream0_name,
                                  const char *IDstream1_name,
                                  const char *IDstreammask_name,
                                  const char *IDstreamout_name,
                                  long        semtrig);
