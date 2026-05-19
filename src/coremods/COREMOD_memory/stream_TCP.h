/**
 * @file stream_TCP.h
 * @brief TCP stream transfer
 */

#ifndef _STREAM_TCP_H
#define _STREAM_TCP_H

errno_t
CLIADDCMD_COREMOD_memory__stream_TCP();

errno_t
COREMOD_MEMORY_testfunction_semaphore(
    const char *IDname,
    int        semtrig,
    int        testmode);

imageID
COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int        port,
    int        mode,
    int        RT_priority);

imageID
COREMOD_MEMORY_image_NETWORKreceive(
    int port,
    int mode,
    int RT_priority);

#endif
