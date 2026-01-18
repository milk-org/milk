/**
 * @file    fps_loadmemstream_lite.c
 * @brief   lite version of load memory stream for standalone libfps
 */

#include <string.h>
#include <stdio.h>
#include "fps.h"
#include "fps_internal.h"
#include "ImageStreamIO/ImageStreamIO.h"

imageID COREMOD_IOFITS_LoadMemStream(
    const char *sname,
    uint64_t   *streamflag,
    uint32_t   *imLOC
)
{
    *imLOC = STREAM_LOAD_SOURCE_NOTFOUND;
    
    if(strcmp(sname, "NULL") == 0) {
        *imLOC = STREAM_LOAD_SOURCE_NULL;
        return -1;
    }

    // Lite version only checks shared memory via ImageStreamIO
    IMAGE tmpimg;
    if (ImageStreamIO_openIm(&tmpimg, sname) == IMAGESTREAMIO_SUCCESS) {
        *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
        ImageStreamIO_closeIm(&tmpimg);
        return 0; // Return a dummy positive ID to indicate success
    }

    return -1;
}

// Stubs for other potentially missing functions if they are used by libfps
int file_exists(const char *filename) {
    return access(filename, F_OK) != -1;
}

int is_fits_file(const char *filename) {
    // Basic check for .fits extension
    const char *ext = strrchr(filename, '.');
    if (ext && strcmp(ext, ".fits") == 0) return 1;
    return 0;
}

int save_fits(const char *imname, const char *filename) {
    printf("save_fits stub called for %s -> %s\n", imname, filename);
    return -1;
}

int load_fits(const char *filename, const char *imname, int verbose, imageID *ID) {
    printf("load_fits stub called for %s -> %s\n", filename, imname);
    return -1;
}

int copy_image_ID(const char *name1, const char *name2, int shared) {
    printf("copy_image_ID stub called for %s -> %s\n", name1, name2);
    return -1;
}
