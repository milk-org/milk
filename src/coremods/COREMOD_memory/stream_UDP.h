/**
 * @file stream_UDP.h
 * @brief UDP stream transfer
 */

#ifndef _STREAM_UDP_H
#define _STREAM_UDP_H

errno_t CLIADDCMD_COREMOD_memory__stream_UDP();

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
                                            const char *IPaddr,
                                            int         port,
                                            int         do_counter_sync,
                                            int         RT_priority);

imageID COREMOD_MEMORY_image_NETUDPreceive(int port, int do_counter_sync, int RT_priority);

#endif
