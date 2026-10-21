// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_UDP.h
 * @brief UDP stream transfer
 */

#ifndef _STREAM_UDP_H
#define _STREAM_UDP_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_imudptransmit;
extern const MILK_RUNNABLE runnable_COREMOD_memory_imudpreceive;

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
                                            const char *IPaddr,
                                            int         port,
                                            int         do_counter_sync,
                                            int         RT_priority);

imageID COREMOD_MEMORY_image_NETUDPreceive(int port, int do_counter_sync, int RT_priority);

#endif
