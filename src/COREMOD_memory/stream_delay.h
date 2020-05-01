/** @file stream_delay,h
 */

errno_t COREMOD_MEMORY_streamDelay_FPCONF(
    char    *fpsname,
    uint32_t CMDmode
);

imageID COREMOD_MEMORY_streamDelay_RUN(
    char *fpsname
);


errno_t COREMOD_MEMORY_streamDelay(
    const char *IDin_name,
    const char *IDout_name,
    long        delayus,
    long        dtus
);
