#ifndef IMAGEID_H
#define IMAGEID_H

#include "CLIcore.h"
#include "image_ID.h"
#include <string.h>
#include <stdlib.h>

static inline imageID RegisterIMGID(
    IMGID *img,
    IMAGE *imagearray,
    long NB_images
)
{
    imageID ID = -1;

    if (imagearray == NULL)
    {
        // If no array provided, we close the image and return 0 (success) or -1 (fail)
        // This corresponds to non-CLI mode check
        if (img->ID != -1)
        {
            ImageStreamIO_closeIm(img->im);
            free(img->im);
            img->ID = 0;
            return 0;
        }
        return -1;
    }

    // Check if already loaded
    ID = image_ID(img->name, imagearray, NB_images);
    if(ID != -1)
    {
        // Already loaded: close the one we just opened and point to the existing one
        if (img->im != NULL)
        {
            ImageStreamIO_closeIm(img->im);
            free(img->im);
        }
        img->ID = ID;
        img->im = &imagearray[ID];
        img->md = &imagearray[ID].md[0];
        img->createcnt = imagearray[ID].createcnt;
        imgid_update_creationparams(img);
    }
    else
    {
        // Not loaded: find slot and move it
        ID = next_avail_image_ID(-1);
        if (ID != -1)
        {
            // We assume imagearray has enough space and ID is valid index
            // Move content
            memcpy(&imagearray[ID], img->im, sizeof(IMAGE));
            // Free temporary structure
            free(img->im);

            img->ID = ID;
            img->im = &imagearray[ID];
            img->md = &imagearray[ID].md[0];
            // img.createcnt = dcimg[img.ID].createcnt; // Should be set? ImageStreamIO doesn't set createcnt?
            // Actually createcnt is in IMAGE struct, so it was copied.

            imagearray[ID].used = 1; // next_avail_image_ID sets this, but just to be sure if we used different logic

            imgid_update_creationparams(img);
        }
        else
        {
            // No space available
            if (img->im != NULL)
            {
                ImageStreamIO_closeIm(img->im);
                free(img->im);
            }
            img->ID = -1;
        }
    }

    return ID;
}




/** @brief Resolve image already in memory
 *
 *
 *
 * ERRMODE values
 * ERRMODE_WARN : print warning
 * ERRMODE_FAIL : error
 * ERRMODE_ABORT : abort
 */
#define resolveIMGID(...) _resolveIMGID_impl(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

static inline imageID _resolveIMGID_impl(
    const char *caller_file,
    int caller_line,
    const char *caller_func,
    IMGID *img,
    int ERRMODE,
    IMAGE *imagearray __attribute__((unused)),
    long NB_images __attribute__((unused))
)
{
    // IF:
    // Not resolved before OR create counter mismatch OR not used.
    // Note: we are comparing img->createcnt to dcimg[img->ID].createcnt to check if the
    // image has been re-created, indicating that our pointers are stale.
    if((img->ID == -1)
            || (img->createcnt != imagearray[img->ID].createcnt)
            || (imagearray[img->ID].used != 1))
    {
        img->ID = image_ID(img->name, dcimg, dcnimg);
        if(img->ID > -1)  // Resolve success !
        {
            img->im        = &imagearray[img->ID];
            img->md        = &imagearray[img->ID].md[0];
            img->createcnt = imagearray[img->ID].createcnt;

            // Populate the IMGID from the imageID metadata
            if(img->mdt != NULL)
            {
                imgid_update_creationparams(img);
            }
        }
    }

    // if still unresolved
    //
    if(img->ID == -1)
    {
        if((ERRMODE == ERRMODE_FAIL) || (ERRMODE == ERRMODE_ABORT))
        {
            if (img->name[0] == '\0')
            {
                const char *fpskey =
                    (img->fpskeyword[0] != '\0')
                    ? img->fpskeyword
                    : "<unknown>";

                if(img->fpskeyword[0] != '\0')
                {
                    fprintf(stderr,
                        "\n\033[1;31mABORT\033[0m "
                        "resolveIMGID: stream name "
                        "is empty.\n"
                        "  FPS parameter : %s\n"
                        "  Called from   : %s:%d"
                        " in %s()\n"
                        "  Fix: set the missing "
                        "parameter, e.g.:\n"
                        "    milk-fps-set %s"
                        " <stream_name>\n",
                        fpskey,
                        caller_file, caller_line,
                        caller_func,
                        fpskey);
                }
                else
                {
                    fprintf(stderr,
                        "\n\033[1;31mABORT\033[0m "
                        "resolveIMGID: stream name "
                        "is empty.\n"
                        "  FPS parameter : %s\n"
                        "  Called from   : %s:%d"
                        " in %s()\n"
                        "  Fix: set the missing "
                        "parameter and tag this "
                        "IMGID with imgid_setfpskeyword() "
                        "to enable a specific "
                        "milk-fps-set suggestion.\n",
                        fpskey,
                        caller_file, caller_line,
                        caller_func);
                }
                fflush(stderr);
                abort();
            }
            else
            {
                PRINT_ERROR(
                    "Cannot resolve image \"%s\"\n",
                    img->name);
                abort();
            }
        }
        else if(ERRMODE == ERRMODE_WARN)
        {
            const char *imgname =
                (img->name[0] != '\0')
                ? img->name : "<empty name>";
            PRINT_WARNING(
                "Cannot resolve image \"%s\"\n",
                imgname);
        }
    }

    return img->ID;
}

static inline int imgid_exists(const char *name)
{
    if(name == NULL || name[0] == '\0')
    {
        return 0;
    }

    IMGID img = {0}; // Zero-initialize so img.mdt is NULL
    img.ID        = -1;
    img.im        = NULL;
    img.md        = NULL;
    img.createcnt = 0;
    strncpy(img.name, name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.name[STRINGMAXLEN_IMAGE_NAME - 1] = '\0';

    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);

    return (img.ID != -1);
}


static inline IMGID makesetIMGID(CONST_WORD name, imageID ID)
{
    IMGID img;

    img.ID = ID;
    strncpy(img.name, name, STRINGMAXLEN_IMAGE_NAME - 1);

    img.im        = &dcimg[ID];
    img.md        = &dcimg[ID].md[0];
    img.createcnt = dcimg[ID].createcnt;

    return img;
}





/**
 * @brief Connnect to stream
 *
 * @param imname  stream name
 * @return IMGID
 */
static inline IMGID
stream_connect(
    const char *__restrict imname
)
{
    IMGID img = imgid_make_from_name(imname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    if(img.ID == -1)
    {
        // try to connect to shared memory if not in local memory already
        read_sharedmem_image(imname, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    }

    return img;
}



static inline imageID createimagefromIMGID(IMGID *img)
{
    create_image_ID(img->name,
                    img->mdt->naxis,
                    img->mdt->size,
                    img->mdt->datatype,
                    img->mdt->shared,
                    img->mdt->NBkw,
                    img->mdt->CBsize,
                    &img->ID);

    img->im        = &dcimg[img->ID];
    img->md        = &dcimg[img->ID].md[0];
    img->createcnt = dcimg[img->ID].createcnt;

    return img->ID;
}




/** Create image according to IMGID entries of existing image
 */
static inline imageID imcreatelikewiseIMGID(
    IMGID *target_img,
    IMGID *source_img
)
{
    if(target_img->ID == -1)
    {
        /* Save createcnt of existing image (if
         * any) so we can detect re-create vs
         * re-use after the create_image_ID call.
         */
        imageID old_id = image_ID(
            target_img->name,
            dcimg,
            dcnimg);
        uint64_t old_createcnt = 0;
        int existed = 0;

        if(old_id != -1)
        {
            existed = 1;
            old_createcnt =
                dcimg[old_id].createcnt;
        }

        DEBUG_TRACEPOINT("Creating 2D image");
        create_image_ID(target_img->name,
                        source_img->mdt->naxis,
                        source_img->mdt->size,
                        source_img->mdt->datatype,
                        source_img->mdt->shared,
                        source_img->mdt->NBkw,
                        source_img->mdt->CBsize,
                        &target_img->ID);
        DEBUG_TRACEPOINT(" ");
        target_img->im        =
            &dcimg[target_img->ID];
        target_img->md        =
            &dcimg[target_img->ID].md[0];
        target_img->createcnt =
            dcimg[target_img->ID].createcnt;

        /* Determine if image was re-used or
         * (re-)created by comparing createcnt.
         */
        int reused = (existed
            && target_img->createcnt
               == old_createcnt);

        if(reused)
        {
            /* Image already exists with matching
             * parameters — nothing to do.
             */
        }
        else if(target_img != source_img)
        {
            printf("  "
                   "\033[33mCreating\033[0m"
                   " from %s,"
                   " shared=%d, kw=%d\n",
                   source_img->name,
                   source_img->mdt->shared,
                   source_img->mdt->NBkw);
        }
        else
        {
            printf("  "
                   "\033[33mCreating\033[0m"
                   " shared=%d, kw=%d\n",
                   source_img->mdt->shared,
                   source_img->mdt->NBkw);
        }

        target_img->mdt->size[0] =
            source_img->mdt->size[0];
        if(source_img->mdt->naxis > 1)
        {
            target_img->mdt->size[1] =
                source_img->mdt->size[1];
        }
        if(source_img->mdt->naxis > 2)
        {
            target_img->mdt->size[2] =
                source_img->mdt->size[2];
        }
    }
    else
    {
        /* Image already resolved — nothing to do */
    }
    return target_img->ID;
}



/** Create image according to IMGID entries
 *  See cloning creation function imcreatelikewiseIMGID()
 */
static inline imageID imcreateIMGID(IMGID *img)
{
    return imcreatelikewiseIMGID(img, img);
}



/**
 * @brief Internal implementation — do not call directly.
 *
 * Use the stream_connect_create_2D() or stream_connect_create_2Df32()
 * macros below, which inject caller context for actionable diagnostics.
 */
static inline IMGID _stream_connect_create_2D_impl(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize,
    uint8_t  datatype,
    const char *caller_file,
    int        caller_line,
    const char *caller_func
)
{
    /* Guard: an empty or NULL stream name means a required FPS
     * parameter was never configured.  Detect here and abort with
     * a clear, actionable message rather than crashing deep inside
     * ImageStreamIO with a cryptic "Cannot allocate memory".
     *
     * The caller location printed below (file:line in func) shows
     * which module variable was empty — look it up in source to
     * find the associated FPS parameter key.
     */
    if (imname == NULL || imname[0] == '\0')
    {
        fprintf(stderr,
                "\n\033[1;31mABORT\033[0m stream_connect_create_2D: "
                "stream name is empty or NULL.\n"
                "  Called from: %s:%d in %s()\n"
                "  A required FPS stream parameter has not been configured.\n"
                "  Identify the FPS key from the '[empty]' probe entries above,\n"
                "  then set the missing parameter, e.g.:\n"
                "    milk-fps-set <fps_name> <key> <stream_name>\n",
                caller_file, caller_line, caller_func);
        fflush(stderr);
        abort();
    }

    IMGID img = imgid_make_from_name(imname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);


    if(img.ID == -1)
    {
        // try to connect to shared memory if not in local memory already
        read_sharedmem_image(imname, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    }

    if(img.ID != -1)
    {
        // if in local memory,
        // create blank img for comparison
        IMGID imgc      = imgid_make();
        imgc.mdt->datatype   = datatype;
        imgc.mdt->naxis      = 2;
        imgc.mdt->size[0]    = xsize;
        imgc.mdt->size[1]    = ysize;
        imgc.mdt->NBkw       = NB_KEYWNODE_MAX;
        uint64_t imgerr = imgid_compare(img, imgc);
        imgid_free(&imgc);
        printf("%lu errors\n", imgerr);

        // if doesn't pass test, erase from local memory
        if(imgerr != 0)
        {
            delete_image_ID(imname, DELETE_IMAGE_ERRMODE_WARNING);
            img.ID = -1;
        }
    }

    // if not in local memory, (re)-create
    if(img.ID == -1)
    {
        uint32_t arraytmp[2];

        arraytmp[0] = xsize;
        arraytmp[1] = ysize;

        create_image_ID(imname, 2, arraytmp, datatype, 1, NB_KEYWNODE_MAX, 0, &img.ID);
    }


    if(img.ID != -1)
    {
        imageID ID    = img.ID;
        img.im        = &dcimg[ID];
        img.md        = dcimg[ID].md;
        img.createcnt = dcimg[ID].createcnt;
        imgid_update_creationparams(&img);
    }

    return img;
}

/**
 * @brief Connect to a 2D stream or create it if missing.
 *
 * Macro wrapper around _stream_connect_create_2D_impl that captures
 * the caller's __FILE__, __LINE__, and __FUNCTION__ at compile time.
 * On an empty stream name the abort message will identify the exact
 * source location, making it easy to trace back to the FPS parameter.
 */
#define stream_connect_create_2D(imname, xsize, ysize, datatype) \
    _stream_connect_create_2D_impl(imname, xsize, ysize, datatype, \
                                   __FILE__, __LINE__, __FUNCTION__)

/**
 * @brief Connect to a float32 2D stream or create it if missing.
 *
 * Convenience macro — equivalent to stream_connect_create_2D with
 * datatype = _DATATYPE_FLOAT.  Caller context is captured automatically.
 */
#define stream_connect_create_2Df32(imname, xsize, ysize) \
    _stream_connect_create_2D_impl(imname, xsize, ysize, _DATATYPE_FLOAT, \
                                   __FILE__, __LINE__, __FUNCTION__)

static inline IMGID stream_connect_create_3D(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize,
    uint8_t  datatype
)
{
    IMGID img = imgid_make_from_name(imname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);


    if(img.ID == -1)
    {
        // try to connect to shared memory if not in local memory already
        read_sharedmem_image(imname, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    }

    if(img.ID != -1)
    {
        // if in local memory,
        // create blank img for comparison
        IMGID imgc      = imgid_make();
        imgc.mdt->datatype   = datatype;
        imgc.mdt->naxis      = 3;
        imgc.mdt->size[0]    = xsize;
        imgc.mdt->size[1]    = ysize;
        imgc.mdt->size[2]    = zsize;
        imgc.mdt->NBkw       = NB_KEYWNODE_MAX;
        uint64_t imgerr = imgid_compare(img, imgc);
        imgid_free(&imgc);
        printf("%lu errors\n", imgerr);

        // if doesn't pass test, erase from local memory
        if(imgerr != 0)
        {
            delete_image_ID(imname, DELETE_IMAGE_ERRMODE_WARNING);
            img.ID = -1;
        }
    }

    // if not in local memory, (re)-create
    if(img.ID == -1)
    {
        uint32_t arraytmp[3];

        arraytmp[0] = xsize;
        arraytmp[1] = ysize;
        arraytmp[2] = zsize;

        printf("CREATING image size %u %u %u\n", xsize, ysize, zsize);

        create_image_ID(imname, 3, arraytmp, datatype, 1, NB_KEYWNODE_MAX, 0, &img.ID);
    }


    if(img.ID != -1)
    {
        imageID ID    = img.ID;
        img.im        = &dcimg[ID];
        img.md        = dcimg[ID].md;
        img.createcnt = dcimg[ID].createcnt;
        imgid_update_creationparams(&img);
    }

    return img;
}

/**
 * @brief Connnect to stream or create if doesn't exist
 *
 * If stream exists but has wrong size type, recreate
 *
 * @param imname  stream name
 * @param xsize   x size
 * @param ysize   y size
 * @param zsize   z size
 * @return IMGID
 */
static inline IMGID stream_connect_create_3Df32(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize)
{
    return stream_connect_create_3D(imname, xsize, ysize, zsize, _DATATYPE_FLOAT);
}





#endif
