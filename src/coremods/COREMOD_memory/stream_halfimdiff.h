// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_halfimdiff.h
 * @brief Half-image difference
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_streamhalfdiff;

imageID COREMOD_MEMORY_stream_halfimDiff(const char *IDstream_name,
                                         const char *IDstreamout_name,
                                         long        semtrig);
