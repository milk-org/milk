/**
 * @file    fps_loadmemstream_lite.c
 * @brief   Lite version of load memory stream for libfps.
 *
 * This file is compiled into libmilkfps.so which does
 * NOT link against libCLIcore.so. We use weak
 * references to accessor globals defined in
 * fps_standalone_data.c (linked into standalone
 * executables only).
 */

#include <string.h>
#include <unistd.h>
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"

/*
 * Module-local image array storage.
 * Set via milkfps_set_image_array() from the V2
 * macro after CLI_data_init().
 */
static IMAGE *milkfps_imarray  = NULL;
static long   milkfps_nb_max   = 0;

void milkfps_set_image_array(
    IMAGE *imarray,
    long   nb_max
)
{
    milkfps_imarray = imarray;
    milkfps_nb_max  = nb_max;
}

/**
 * @brief Load a shared-memory stream, registering
 *        in the image array if available.
 */
imageID COREMOD_IOFITS_LoadMemStream(
    const char *sname,
    uint64_t   *streamflag,
    uint32_t   *imLOC
)
{
    (void) streamflag;

    *imLOC = STREAM_LOAD_SOURCE_NOTFOUND;

    if (sname == NULL || strlen(sname) == 0 ||
        strcmp(sname, " ") == 0 ||
        strcmp(sname, "NULL") == 0)
    {
        *imLOC = STREAM_LOAD_SOURCE_NULL;
        return -1;
    }

    IMAGE *imarray = milkfps_imarray;
    long   nb_max  = milkfps_nb_max;

    if (imarray == NULL || nb_max <= 0)
    {
        /*
         * No image array (pure library context).
         * Just check existence.
         */
        IMAGE tmpimg;
        if (ImageStreamIO_openIm(&tmpimg, sname)
            == IMAGESTREAMIO_SUCCESS)
        {
            *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
            ImageStreamIO_closeIm(&tmpimg);
            return 0;
        }
        return -1;
    }

    /* Check if already loaded */
    for (long i = 0; i < nb_max; i++)
    {
        if (imarray[i].used == 1 &&
            strncmp(imarray[i].name, sname,
                    STRINGMAXLEN_IMAGE_NAME)
                == 0)
        {
            *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
            return i;
        }
    }

    /* Find a free slot */
    long slot = -1;
    for (long i = 0; i < nb_max; i++)
    {
        if (imarray[i].used == 0)
        {
            slot = i;
            break;
        }
    }
    if (slot == -1)
    {
        return -1;
    }

    /* Open stream into the slot */
    if (ImageStreamIO_openIm(
            &imarray[slot], sname)
            == IMAGESTREAMIO_SUCCESS)
    {
        imarray[slot].used = 1;
        strncpy(imarray[slot].name, sname,
                STRINGMAXLEN_IMAGE_NAME - 1);
        *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
        return slot;
    }

    return -1;
}

/*
 * Stubs for functions referenced by other code in
 * libmilkfps.so (e.g. fps_checkparameter.c).
 */
int file_exists(const char *filename)
{
    return access(filename, F_OK) != -1;
}

int is_fits_file(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if (ext && strcmp(ext, ".fits") == 0)
    {
        return 1;
    }
    return 0;
}

int save_fits(
    const char *imname,
    const char *filename
)
{
    (void) imname;
    (void) filename;
    return -1;
}

int load_fits(
    const char *filename,
    const char *imname,
    int         verbose,
    imageID    *ID
)
{
    (void) filename;
    (void) imname;
    (void) verbose;
    (void) ID;
    return -1;
}

int copy_image_ID(
    const char *name1,
    const char *name2,
    int         shared
)
{
    (void) name1;
    (void) name2;
    (void) shared;
    return -1;
}
