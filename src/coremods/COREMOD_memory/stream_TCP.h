// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_TCP.h
 * @brief TCP stream transfer
 */

#ifndef _STREAM_TCP_H
#define _STREAM_TCP_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_imnetwtransmit;
extern const MILK_RUNNABLE runnable_COREMOD_memory_imnetwreceive;

errno_t COREMOD_MEMORY_testfunction_semaphore(const char *IDname, int semtrig, int testmode);

imageID COREMOD_MEMORY_image_NETWORKtransmit(const char *IDname,
                                             const char *IPaddr,
                                             int         port,
                                             int         mode,
                                             int         RT_priority);

imageID COREMOD_MEMORY_image_NETWORKreceive(int port, int mode, int RT_priority);

#endif
