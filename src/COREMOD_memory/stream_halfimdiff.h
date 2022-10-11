/**
 * @file stream_hlfimdiff.h
 */

errno_t stream_halfimdiff_addCLIcmd();

imageID COREMOD_MEMORY_stream_halfimDiff(const char *IDstream_name,
        const char *IDstreamout_name,
        long        semtrig);
