/**
 * @file stream_paste.h
 * @brief Paste two 2D streams
 */

errno_t CLIADDCMD_COREMOD_memory__stream_paste();

imageID COREMOD_MEMORY_streamPaste(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreamout_name,
    long        semtrig0,
    long        semtrig1,
    int         master);
