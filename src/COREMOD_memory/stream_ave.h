/** @file stream_ave.h
 */
 

errno_t stream_ave_addCLIcmd();


imageID COREMOD_MEMORY_streamAve(
    const char *IDstream_name,
    int         NBave,
    int         mode,
    const char *IDout_name
);

