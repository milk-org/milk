/**
 * @file    stream_TCP.h
 */

#ifndef _STREAM_TCP_H
#define _STREAM_TCP_H

errno_t stream__UDP_addCLIcmd();

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
                                             const char *IPaddr,
                                             int         port,
                                             int         mode,
                                             int         RT_priority);

imageID
COREMOD_MEMORY_image_NETUDPreceive(int port, int mode, int RT_priority);

#endif
