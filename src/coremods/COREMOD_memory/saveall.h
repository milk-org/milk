// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file saveall.h
 * @brief Save all images/streams to disk
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_imsaveallsnap;
extern const MILK_RUNNABLE runnable_COREMOD_memory_imsaveallseq;

errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname);

errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname,
                                    const char *IDtrig_name,
                                    long        semtrig,
                                    long        NBframes);
