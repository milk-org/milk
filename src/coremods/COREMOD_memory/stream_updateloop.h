// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_updateloop.h
 * @brief Stream updateloop module
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_streamburst;
extern const MILK_RUNNABLE runnable_COREMOD_memory_creaimstream;
extern const MILK_RUNNABLE runnable_COREMOD_memory_creaimstreamstrig;

imageID COREMOD_MEMORY_image_streamupdateloop(const char *IDinname,
                                              const char *IDoutname,
                                              long        usperiod,
                                              long        NBcubes,
                                              long        period,
                                              long        offsetus,
                                              const char *IDsync_name,
                                              int         semtrig,
                                              int         timingmode);

imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(const char *IDinname,
                                                      const char *IDoutname,
                                                      long        period,
                                                      long        offsetus,
                                                      const char *IDsync_name,
                                                      int         semtrig,
                                                      int         timingmode);
