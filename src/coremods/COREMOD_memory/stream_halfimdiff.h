/**
 * @file stream_halfimdiff.h
 * @brief Half-image difference
 */

errno_t CLIADDCMD_COREMOD_memory__stream_halfimdiff();

imageID COREMOD_MEMORY_stream_halfimDiff(
    const char *IDstream_name,
    const char *IDstreamout_name,
    long       semtrig);
