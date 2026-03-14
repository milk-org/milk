/**
 * @file    fps_loadmemstream_lite.c
 * @brief   Lite version of load memory stream for libfps.
 *
 * This file is compiled into libmilkfps.so which does
 * NOT link against libCLIcore.so. We use weak
 * references to accessor globals defined in
 * fps_standalone_data.c (linked into standalone
 * executables only).
 *
 * Supports @X: prefix modifiers on stream names:
 *   L  Only search local imarray (no SHM)
 *   S  Force shared memory (default behavior)
 *   E  Must exist (return -1 → caller error)
 *   N  Must not exist (return -1 → caller error)
 */

#include <string.h>
#include <unistd.h>
#include <stdio.h>

#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "fps_streamname_parse.h"

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
 * @brief Search local imarray for a name.
 *
 * @param sname  Bare stream name
 * @return imageID or -1 if not found
 */
static imageID find_in_local(const char *sname)
{
    IMAGE *imarray = milkfps_imarray;
    long   nb_max  = milkfps_nb_max;

    if (imarray == NULL || nb_max <= 0)
    {
        return -1;
    }

    for (long i = 0; i < nb_max; i++)
    {
        if (imarray[i].used == 1 &&
            strncmp(imarray[i].name, sname,
                    STRINGMAXLEN_IMAGE_NAME)
                == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Load a shared-memory stream, with @X:
 *        prefix support.
 *
 * Parses the sname for an optional modifier prefix
 * and adjusts load behavior accordingly.
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

    /* Parse prefix */
    FPS_STREAMNAME_PARSED sp =
        fps_streamname_parse(sname);

    if (sp.error)
    {
        printf("ERROR: invalid stream modifier "
               "in \"%s\"\n", sname);
        return -1;
    }

    const char *name = sp.name;

    IMAGE *imarray = milkfps_imarray;
    long   nb_max  = milkfps_nb_max;

    /* @N: must-not-exist check */
    if (sp.must_new)
    {
        imageID existing = find_in_local(name);
        if (existing >= 0)
        {
            printf("@N modifier: \"%s\" already "
                   "exists locally (ID %ld)\n",
                   name, (long) existing);
            return -1;
        }

        if (imarray != NULL)
        {
            IMAGE tmpimg;
            if (ImageStreamIO_openIm(
                    &tmpimg, name)
                == IMAGESTREAMIO_SUCCESS)
            {
                ImageStreamIO_closeIm(&tmpimg);
                printf("@N modifier: \"%s\" "
                       "exists in SHM\n", name);
                return -1;
            }
        }
    }

    /* @L: local-only — skip SHM */
    if (sp.loc == 'L')
    {
        imageID id = find_in_local(name);
        if (id >= 0)
        {
            *imLOC =
                STREAM_LOAD_SOURCE_LOCALMEM;
        }
        else if (sp.must_exist)
        {
            printf("@LE modifier: \"%s\" not "
                   "found in local memory\n",
                   name);
        }
        return id;
    }

    /* Default / @S: shared memory path */

    if (imarray == NULL || nb_max <= 0)
    {
        /*
         * No image array (pure library context).
         * Just check SHM existence.
         */
        IMAGE tmpimg;
        if (ImageStreamIO_openIm(&tmpimg, name)
            == IMAGESTREAMIO_SUCCESS)
        {
            *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
            ImageStreamIO_closeIm(&tmpimg);

            if (sp.must_exist == 0)
            {
                return 0;
            }
            return 0;
        }
        if (sp.must_exist)
        {
            printf("@E modifier: \"%s\" not "
                   "found in SHM\n", name);
        }
        return -1;
    }

    /* Check if already loaded locally */
    if (sp.loc != 'S')
    {
        /* Default: check local first */
        for (long i = 0; i < nb_max; i++)
        {
            if (imarray[i].used == 1 &&
                strncmp(imarray[i].name, name,
                        STRINGMAXLEN_IMAGE_NAME)
                    == 0)
            {
                *imLOC =
                    STREAM_LOAD_SOURCE_SHAREMEM;
                return i;
            }
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

    /* Check SHM file exists before trying to open
     * (avoids spurious WARNING from ImageStreamIO
     *  for output streams not yet created) */
    {
        char shmpath[512];
        snprintf(shmpath, sizeof(shmpath),
                 "/milk/shm/%s.im.shm", name);
        if (access(shmpath, F_OK) != 0)
        {
            /* SHM file does not exist */
            if (sp.must_exist)
            {
                printf("@E modifier: \"%s\" "
                       "not found\n", name);
            }
            return -1;
        }
    }

    /* Open stream into the slot */
    if (ImageStreamIO_openIm(
            &imarray[slot], name)
            == IMAGESTREAMIO_SUCCESS)
    {
        imarray[slot].used = 1;
        strncpy(imarray[slot].name, name,
                STRINGMAXLEN_IMAGE_NAME - 1);
        *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
        return slot;
    }

    if (sp.must_exist)
    {
        printf("@E modifier: \"%s\" not found\n",
               name);
    }

    return -1;
}

/*
 * Weak stubs for functions referenced by other code
 * in libmilkfps.so (e.g. fps_checkparameter.c).
 *
 * When linking statically (USE_STATIC_LTO), the
 * real implementations in COREMOD_iofits_compute
 * override these weak versions.
 */
__attribute__((weak))
int file_exists(const char *filename)
{
    return access(filename, F_OK) != -1;
}

__attribute__((weak))
int is_fits_file(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if (ext && strcmp(ext, ".fits") == 0)
    {
        return 1;
    }
    return 0;
}

__attribute__((weak))
int save_fits(
    const char *imname,
    const char *filename
)
{
    (void) imname;
    (void) filename;
    return -1;
}

__attribute__((weak))
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

__attribute__((weak))
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
