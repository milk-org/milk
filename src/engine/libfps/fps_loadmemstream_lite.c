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
 *   E  Must exist (return -1 -> caller error)
 *   N  Must not exist (return -1 -> caller error)
 */


#include "fps.h"

/*
 * Module-local image array storage.
 * Set via milkfps_set_image_array() from the V2
 * macro after CLI_data_init().
 */
static IMAGE *milkfps_imarray  = NULL;
static long   milkfps_nb_max   = 0;

/**
 * @brief Set the module-local image array for standalone
 *        stream loading.
 *
 * Called by FPS_MAIN_STANDALONE_V2 after CLI_data_init()
 * to provide the image array and its capacity.  Must be
 * called before any COREMOD_IOFITS_LoadMemStream() call.
 *
 * @param imarray  Image array allocated by CLI_data_init()
 * @param nb_max   Maximum number of images in the array
 */
void milkfps_set_image_array(
    IMAGE *imarray,
    long  nb_max)
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

    if(imarray == NULL || nb_max <= 0)
    {
        return -1;
    }

    for(long ii = 0; ii < nb_max; ii++)
    {
        if(imarray[ii].used == 1 &&
                strncmp(imarray[ii].name, sname,
                        STRINGMAXLEN_IMAGE_NAME)
                == 0)
        {
            return ii;
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
    uint32_t   *imLOC)
{
    (void) streamflag;

    *imLOC = STREAM_LOAD_SOURCE_NOTFOUND;

    if(sname == NULL || strlen(sname) == 0 ||
            strcmp(sname, " ") == 0 ||
            strcmp(sname, "NULL") == 0)
    {
        *imLOC = STREAM_LOAD_SOURCE_NULL;
        return -1;
    }

    /* Parse prefix */
    FPS_STREAMNAME_PARSED sp =
        fps_streamname_parse(sname);

    if(sp.error)
    {
        printf("ERROR: invalid stream modifier "
               "in \"%s\"\n", sname);
        return -1;
    }

    const char *name = sp.name;

    IMAGE *imarray = milkfps_imarray;
    long   nb_max  = milkfps_nb_max;

    /* @N: must-not-exist check */
    if(sp.must_new)
    {
        imageID existing = find_in_local(name);
        if(existing >= 0)
        {
            printf("@N modifier: \"%s\" already "
                   "exists locally (ID %ld)\n",
                   name, (long) existing);
            return -1;
        }

        if(imarray != NULL)
        {
            char shmpath_n[512];
            if(ImageStreamIO_filename(
                        shmpath_n,
                        sizeof(shmpath_n),
                        name)
                    == IMAGESTREAMIO_SUCCESS &&
                    access(shmpath_n, F_OK) == 0)
            {
                IMAGE tmpimg;
                if(ImageStreamIO_openIm(
                            &tmpimg, name)
                        == IMAGESTREAMIO_SUCCESS)
                {
                    ImageStreamIO_closeIm(
                        &tmpimg);
                    printf(
                        "@N modifier: \"%s\" "
                        "exists in SHM\n",
                        name);
                    return -1;
                }
            }
        }
    }

    /* @L: local-only -- skip SHM */
    if(sp.loc == 'L')
    {
        imageID id = find_in_local(name);
        if(id >= 0)
        {
            *imLOC =
                STREAM_LOAD_SOURCE_LOCALMEM;
        }
        else if(sp.must_exist)
        {
            printf("@LE modifier: \"%s\" not "
                   "found in local memory\n",
                   name);
        }
        return id;
    }

    /* Default / @S: shared memory path */

    if(imarray == NULL || nb_max <= 0)
    {
        /*
         * No image array (pure library context).
         * Just check SHM existence.
         */
        char shmpath_lib[STRINGMAXLEN_FILE_NAME];
        if(ImageStreamIO_filename(shmpath_lib, sizeof(shmpath_lib), name) != IMAGESTREAMIO_SUCCESS
                || access(shmpath_lib, F_OK) != 0)
        {
            if(sp.must_exist)
            {
                printf("@E modifier: \"%s\" "
                       "not found in SHM\n",
                       name);
            }
            return -1;
        }

        IMAGE tmpimg;
        if(ImageStreamIO_openIm(&tmpimg, name)
                == IMAGESTREAMIO_SUCCESS)
        {
            *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
            ImageStreamIO_closeIm(&tmpimg);
            return 0;
        }
        if(sp.must_exist)
        {
            printf("@E modifier: \"%s\" not "
                   "found in SHM\n", name);
        }
        return -1;
    }

    /* Check if already loaded locally */
    if(sp.loc != 'S')
    {
        /* Default: check local first */
        for(long ii = 0; ii < nb_max; ii++)
        {
            if(imarray[ii].used == 1 &&
                    strncmp(imarray[ii].name, name,
                            STRINGMAXLEN_IMAGE_NAME)
                    == 0)
            {
                *imLOC =
                    STREAM_LOAD_SOURCE_SHAREMEM;
                return ii;
            }
        }
    }

    /* Find a free slot */
    long slot = -1;
    for(long ii = 0; ii < nb_max; ii++)
    {
        if(imarray[ii].used == 0)
        {
            slot = ii;
            break;
        }
    }
    if(slot == -1)
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
        if(access(shmpath, F_OK) != 0)
        {
            /* SHM file does not exist */
            if(sp.must_exist)
            {
                printf("@E modifier: \"%s\" "
                       "not found\n", name);
            }
            return -1;
        }
    }

    /* Open stream into the slot */
    if(ImageStreamIO_openIm(
                &imarray[slot], name)
            == IMAGESTREAMIO_SUCCESS)
    {
        imarray[slot].used = 1;
        strncpy(imarray[slot].name, name,
                STRINGMAXLEN_IMAGE_NAME - 1);
        *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
        return slot;
    }

    if(sp.must_exist)
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
 *
 * visibility("hidden") prevents these stubs from
 * being exported by the shared library, avoiding
 * shadowing of the real symbols from
 * libmilkCOREMODiofits.so at runtime.
 */
/**
 * @brief Weak stub: check file existence.
 *
 * Overridden at link time by the real implementation
 * in COREMOD_iofits when building the full CLI.
 */
__attribute__((weak, visibility("hidden")))
int file_exists(const char *filename)
{
    return access(filename, F_OK) != -1;
}

__attribute__((weak, visibility("hidden")))
/**
 * @brief Checks if a given file name corresponds to a FITS file.
 */
int is_fits_file(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if(ext && strcmp(ext, ".fits") == 0)
    {
        return 1;
    }
    return 0;
}

/**
 * @brief Weak stub: save image to FITS file.
 *
 * Returns -1 (no-op) unless overridden by
 * COREMOD_iofits at link time.
 */
__attribute__((weak, visibility("hidden")))
int save_fits(
    const char *imname,
    const char *filename)
{
    (void) imname;
    (void) filename;
    return -1;
}

/**
 * @brief Weak stub: load FITS file into image array.
 *
 * Returns -1 (no-op) unless overridden by
 * COREMOD_iofits at link time.
 */
__attribute__((weak, visibility("hidden")))
int load_fits(
    const char *filename,
    const char *imname,
    int        verbose,
    imageID    *ID)
{
    (void) filename;
    (void) imname;
    (void) verbose;
    (void) ID;
    return -1;
}

/**
 * @brief Weak stub: copy an image by name.
 *
 * Returns -1 (no-op) unless overridden by
 * COREMOD_memory at link time.
 */
__attribute__((weak, visibility("hidden")))
int copy_image_ID(
    const char *name1,
    const char *name2,
    int        shared)
{
    (void) name1;
    (void) name2;
    (void) shared;
    return -1;
}
