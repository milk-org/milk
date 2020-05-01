/**
 * @file    stream_TCP.h
 */


errno_t COREMOD_MEMORY_testfunction_semaphore(
    const char *IDname,
    int         semtrig,
    int         testmode
);


imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int         port,
    int         mode,
    int         RT_priority
);

imageID COREMOD_MEMORY_image_NETWORKreceive(
    int port,
    int mode,
    int RT_priority
);

