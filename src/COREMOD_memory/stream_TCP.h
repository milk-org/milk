// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_TCP.h
 */

#ifndef _STREAM_TCP_H
#define _STREAM_TCP_H

errno_t stream__NETW_addCLIcmd();

errno_t COREMOD_MEMORY_testfunction_semaphore(const char *IDname,
        int semtrig,
        int testmode);

imageID COREMOD_MEMORY_image_NETWORKtransmit(const char *IDname,
        const char *IPaddr,
        int port,
        int mode,
        int RT_priority);

imageID COREMOD_MEMORY_image_NETWORKreceive(
    int port, int mode, int RT_priority);

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
        const char *IPaddr,
        int port,
        int mode,
        int RT_priority);

imageID COREMOD_MEMORY_image_NETUDPreceive(
    int port, int mode, int RT_priority);

#endif
