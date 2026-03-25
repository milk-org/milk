/**
 * @file    IMGID.h
 * @brief   Image identifying structure
 *
 */

#ifndef IMGID_H
#define IMGID_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "fps_streamname_parse.h"
#include "imgid_slice.h"




#ifdef __cplusplus
typedef const char *CONST_WORD;
#else
typedef const char *__restrict CONST_WORD;
#endif

#ifndef errno_t
typedef int errno_t;
#endif

#define IMGID_NB_KEYWO_MAX 10

#define IMGID_CONNECT_NOCHECK      0
#define IMGID_CONNECT_CHECK_FAIL   1
#define IMGID_CONNECT_CHECK_CREATE 2

#define IMGID_CONNECTED            0
#define IMGID_CREATED              1
#define IMGID_RECREATED            2

#define IMGID_SUCCESS              IMGID_CONNECTED

#define IMGID_ERR_BADNAME          10
#define IMGID_ERR_ALLOCATION       11
#define IMGID_ERR_NOTFOUND         12
#define IMGID_ERR_MISMATCH         13



// Image identifier for internal use by milk
//
// This is the preferred format for handling images in milk as function arguments,
// providing additional information and context for the current process/application.
//
// When used with milk-CLI, the type avoids name-resolving imageID multiple times
// provides quick and convenient access to data and metadata pointers
// Pass this as argument to functions to have both input-by-ID (ID>-1)
// and input-by-name (ID=-1).
typedef struct
{
    imageID ID; // -1 if not resolved
    // If using an array of images (as done by milk-CLI) this is the index
    // of the image in the array.
    // If not using an array of images, ID=0 if the image is loaded in memory.

    // increments when image created, used to check if re-resolving needed
    int64_t createcnt;

    // used to resolve if needed
    char            name[STRINGMAXLEN_IMAGE_NAME];

    // image content, data and metadata
    IMAGE          *im;
    // md points at im.md
    IMAGE_METADATA *md;

    // METADATA TEMPLATE
    // Requested image params.
    // Used to create image or test if existing image matches.
    // These fields do not always match the image content.
    //
    IMAGE_METADATA *mdt;

    // Stream name modifier (from @X: prefix)
    // Parsed by imgid_make_from_name()
    char stream_loc;        // 'L','S','F', or '\0'
    char stream_must_exist; // 1 if @E
    char stream_must_new;   // 1 if @N

    // Array slice parsed from name[...]
    IMGID_SLICE slice;

    // Materialized slice buffer
    IMAGE      *slice_im;        // NULL until first use
    uint64_t    slice_last_cnt0;  // source frame counter
    int         slice_shared;    // 1 if @S: requested

} IMGID;



/** make blank IMGID
 *
 * All fields are uninitialized
 * Can be used for comparison
*/
static inline IMGID imgid_make()
{
    IMGID img;

    img.mdt = (IMAGE_METADATA*) malloc(sizeof(IMAGE_METADATA));

    // default values for image creation
    img.mdt->datatype = _DATATYPE_FLOAT;
    img.mdt->naxis    = 0;
    img.mdt->size[0]  = 0;
    img.mdt->size[1]  = 0;
    img.mdt->size[2]  = 0;
    img.mdt->shared   = 0;
    img.mdt->NBkw     = IMGID_NB_KEYWO_MAX;
    img.mdt->CBsize   = 0;

    img.ID        = -1;
    img.createcnt = 0;
    strncpy(img.mdt->name, "",
            STRINGMAXLEN_IMAGE_NAME - 1);
    img.im = NULL;
    img.md = NULL;

    img.stream_loc        = '\0';
    img.stream_must_exist = 0;
    img.stream_must_new   = 0;

    memset(&img.slice, 0, sizeof(IMGID_SLICE));
    img.slice_im       = NULL;
    img.slice_last_cnt0 = 0;
    img.slice_shared   = 0;

    return img;
}

static inline void imgid_free(
    IMGID *img
)
{
    if(img->mdt != NULL)
    {
        free(img->mdt);
    }
    img->mdt = NULL;

    if (img->slice_im != NULL)
    {
        free(img->slice_im);
        img->slice_im = NULL;
    }
}


/** make IMGID from name
 *
 * Settings can be embedded in the image name
 * string for convenience.
 *
 * @X: prefix modifiers (parsed first):
 *   @S:im1  shared memory (default)
 *   @L:im1  local memory only
 *   @E:im1  must already exist
 *   @N:im1  must not exist
 *
 * Legacy > prefixes (still supported):
 *   "tf64>im1"  datatype double
 *   "k10>im1"   number of keywords = 10
 *   "c20>im1"   20-sized circular buffer
 *
 * The legacy "s>" prefix has been removed.
 * Use @S: instead (or omit — shared is default).
*/
static inline IMGID imgid_make_from_name(CONST_WORD name)
{
    IMGID img = imgid_make();

    // default values for image creation
    img.mdt->datatype = _DATATYPE_FLOAT;
    img.mdt->naxis    = 2;
    img.mdt->size[0]  = 1;
    img.mdt->size[1]  = 1;
    img.mdt->shared   = 1; // shared by default
    img.mdt->NBkw     = IMGID_NB_KEYWO_MAX;
    img.mdt->CBsize   = 0;

    /* Parse @X: modifier prefix first */
    const char *effective_name = name;
    {
        FPS_STREAMNAME_PARSED sp =
            fps_streamname_parse(name);
        if (!sp.error)
        {
            img.stream_loc = sp.loc;
            img.stream_must_exist =
                sp.must_exist;
            img.stream_must_new =
                sp.must_new;
            effective_name = sp.name;

            if (sp.loc == 'L')
            {
                img.mdt->shared = 0;
            }
        }
    }

    char *pch;
    char *pch1;

    char  namestring[200];
    strncpy(namestring, effective_name, 199);

    pch1 = namestring;
    if(strlen(namestring) != 0)
    {
        pch = strtok(namestring, ">");
        while(pch != NULL)
        {
            pch1 = pch;
            //printf("[%2d] %s\n", nbword, pch);

            /* "s>" prefix removed — use @S:
             * instead (shared is default).
             */

            if(strcmp(pch, "tui8") == 0)
            {
                printf("    data type unsigned 8-bit int\n");
                img.mdt->datatype = _DATATYPE_UINT8;
            }
            if(strcmp(pch, "tsi8") == 0)
            {
                printf("    data type signed 8-bit int\n");
                img.mdt->datatype = _DATATYPE_INT8;
            }
            if(strcmp(pch, "tui16") == 0)
            {
                printf("    data type unsigned 16-bit int\n");
                img.mdt->datatype = _DATATYPE_UINT16;
            }
            if(strcmp(pch, "tsi16") == 0)
            {
                printf("    data type signed 16-bit int\n");
                img.mdt->datatype = _DATATYPE_INT16;
            }
            if(strcmp(pch, "tui32") == 0)
            {
                printf("    data type unsigned 32-bit int\n");
                img.mdt->datatype = _DATATYPE_UINT32;
            }
            if(strcmp(pch, "tsi32") == 0)
            {
                printf("    data type signed 32-bit int\n");
                img.mdt->datatype = _DATATYPE_INT32;
            }
            if(strcmp(pch, "tui64") == 0)
            {
                printf("    data type unsigned 64-bit int\n");
                img.mdt->datatype = _DATATYPE_UINT64;
            }
            if(strcmp(pch, "tsi64") == 0)
            {
                printf("    data type signed 64-bit int\n");
                img.mdt->datatype = _DATATYPE_INT64;
            }

            if(strcmp(pch, "tf32") == 0)
            {
                printf("    data type double (32)\n");
                img.mdt->datatype = _DATATYPE_FLOAT;
            }
            if(strcmp(pch, "tf64") == 0)
            {
                printf("    data type float (64)\n");
                img.mdt->datatype = _DATATYPE_DOUBLE;
            }

            /*            if(pch[0] == 'k')
                        {
                            int nbkw;
                            sscanf(pch, "k%d", &nbkw);
                            printf("    %d keywords\n", nbkw);
                            img.NBkw = nbkw;
                        }

                        if(pch[0] == 'c')
                        {
                            int cbsize;
                            sscanf(pch, "c%d", &cbsize);
                            printf("    %d circular buffer size\n", cbsize);
                            img.CBsize = cbsize;
                        }
            */
            pch = strtok(NULL, ">");
        }
    }

    img.ID        = -1;
    img.createcnt = -1;
    img.im = NULL;
    img.md = NULL;

    /* Parse bracket slice from the final name.
     * e.g. "im[0:19,10:29]" → bare name "im"
     *   + slice {0:19, 10:29}.
     */
    {
        char bare[STRINGMAXLEN_IMAGE_NAME];
        char slicebuf[128];
        int has_bracket =
            imgid_slice_split_name(
                pch1,
                bare,
                STRINGMAXLEN_IMAGE_NAME,
                slicebuf,
                sizeof(slicebuf));

        strncpy(img.name, bare,
                STRINGMAXLEN_IMAGE_NAME - 1);

        if (has_bracket)
        {
            img.slice =
                imgid_slice_parse(slicebuf);
        }
    }

    return img;
}




static inline IMGID imgid_make_from_name_2D(
    CONST_WORD name,
    uint32_t xsize,
    uint32_t ysize
)
{
    IMGID img   = imgid_make_from_name(name);
    img.mdt->naxis   = 2;
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;

    return img;
}

static inline IMGID imgid_make_from_name_3D(
    CONST_WORD name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    IMGID img   = imgid_make_from_name(name);
    img.mdt->naxis   = 3;
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;

    return img;
}



static inline void imgid_copy(
    IMGID *imgin,
    IMGID *imgout
)
{
    imgout->mdt->datatype = imgin->mdt->datatype;
    imgout->mdt->shared   = imgin->mdt->shared;

    imgout->mdt->naxis = imgin->mdt->naxis;

    imgout->mdt->size[0] = imgin->mdt->size[0];
    imgout->mdt->size[1] = imgin->mdt->size[1];
    imgout->mdt->size[2] = imgin->mdt->size[2];

    imgout->mdt->NBkw   = imgin->mdt->NBkw;
    imgout->mdt->CBsize = imgin->mdt->CBsize;
}



static inline void imgid_update_creationparams(IMGID *img)
{
    img->mdt->datatype = img->md->datatype;
    img->mdt->naxis    = img->md->naxis;
    for(int ii = 0; ii < 3; ++ii)
    {
        img->mdt->size[ii] = img->md->size[ii];
    }
    img->mdt->shared = img->md->shared;
    img->mdt->NBkw   = img->md->NBkw;
    img->mdt->CBsize = img->md->CBsize;
}


static inline uint64_t imgid_mdcompare(
    IMAGE_METADATA *md1,
    IMAGE_METADATA *md2
)
{
    uint64_t diff = 0;

    if (md1->datatype != md2->datatype) { diff |= (1ULL << 0); }
    if (md1->naxis != md2->naxis) { diff |= (1ULL << 1); }
    if (md1->size[0] != md2->size[0]) { diff |= (1ULL << 2); }
    if (md1->size[1] != md2->size[1]) { diff |= (1ULL << 3); }
    if (md1->size[2] != md2->size[2]) { diff |= (1ULL << 4); }
    if (md1->shared != md2->shared) { diff |= (1ULL << 5); }
    if (md1->NBkw != md2->NBkw) { diff |= (1ULL << 6); }
    if (md1->CBsize != md2->CBsize) { diff |= (1ULL << 7); }

    return diff;
}


/**
 * @brief Check if img complies to imgtemplate
 *
 */
static inline uint64_t imgid_compare(
    IMGID img,
    IMGID imgtemplate
)
{
    int compErr = 0;

    if(imgtemplate.mdt->datatype != _DATATYPE_UNINITIALIZED)
    {
        printf("Checking datatype       ");
        if(imgtemplate.mdt->datatype != img.mdt->datatype)
        {
            printf("%d %d", imgtemplate.mdt->datatype, img.mdt->datatype);
            printf("  -> FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if((int8_t)imgtemplate.mdt->naxis != -1)
    {
        printf("Checking naxis  %d %d    ", imgtemplate.mdt->naxis, img.mdt->naxis);
        if(imgtemplate.mdt->naxis != img.mdt->naxis)
        {
            printf("  -> FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.mdt->size[0] != 0)
    {
        printf("Checking size[0]        ");
        if(imgtemplate.mdt->size[0] != img.mdt->size[0])
        {
            printf(" %d %d", imgtemplate.mdt->size[0], img.mdt->size[0]);
            printf("  -> FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.mdt->size[1] != 0)
    {
        printf("Checking size[1]        ");
        if(imgtemplate.mdt->size[1] != img.mdt->size[1])
        {
            printf(" %d %d", imgtemplate.mdt->size[1], img.mdt->size[1]);
            printf("  -> FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.mdt->size[2] != 0)
    {
        printf("Checking size[2]        ");
        if(imgtemplate.mdt->size[2] != img.mdt->size[2])
        {
            printf(" %d %d", imgtemplate.mdt->size[2], img.mdt->size[2]);
            printf("  -> FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    printf("Checking NBkw           ");
    if(imgtemplate.mdt->NBkw != img.mdt->NBkw)
    {
        printf("FAIL\n");
        printf("   %4u  %s\n", imgtemplate.mdt->NBkw, imgtemplate.name);
        printf("   %4u  %s\n", img.mdt->NBkw, img.name);
        compErr++;
    }
    else
    {
        printf("PASS\n");
    }


    return compErr;
}




/**
 * @brief Check if img complies to imgtemplate
 *
 */
static inline uint64_t imgid_compare_md(
    IMGID img,
    IMGID imgtemplate
)
{
    int compErr = 0;

    printf("COMPARING %s %s\n", img.name, imgtemplate.name);

    if(imgtemplate.md->datatype != _DATATYPE_UNINITIALIZED)
    {
        printf("Checking md->datatype       ");
        if(imgtemplate.md->datatype != img.md->datatype)
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->naxis != 0)
    {
        printf("Checking md->naxis  %d %d    ", imgtemplate.md->naxis, img.md->naxis);
        if(imgtemplate.md->naxis != img.md->naxis)
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->size[0] != 0)
    {
        printf("Checking md->size[0]        ");
        if(imgtemplate.md->size[0] != img.md->size[0])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->size[1] != 0)
    {
        printf("Checking md->size[1]        ");
        if(imgtemplate.md->size[1] != img.md->size[1])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->size[2] != 0)
    {
        printf("Checking md->size[2]        ");
        if(imgtemplate.md->size[2] != img.md->size[2])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    printf("Checking NBkw           ");
    if(imgtemplate.md->NBkw != img.md->NBkw)
    {
        printf("FAIL\n");
        printf("   %4u  %s\n", imgtemplate.md->NBkw, imgtemplate.md->name);
        printf("   %4u  %s\n", img.md->NBkw, img.md->name);
        printf("    template : %u\n", imgtemplate.md->NBkw);
        printf("    dest     : %u\n", img.md->NBkw);
        compErr++;
    }
    else
    {
        printf("PASS\n");
    }

    return compErr;
}


// Create image from IMGID
static inline void imgid_mkimage(IMGID * img)
{
    ImageStreamIO_createIm(img->im, img->name, img->mdt->naxis, img->mdt->size, img->mdt->datatype, img->mdt->shared, img->mdt->NBkw, img->mdt->CBsize);
    img->createcnt++;
}


// Read ImageStreamIO from shared memory
//

static inline const char* imgid_strerror(errno_t err) {
    switch (err) {
    case IMGID_CONNECTED:
        return "CONNECTED";
    case IMGID_CREATED:
        return "CREATED";
    case IMGID_RECREATED:
        return "RECREATED";
    case IMGID_ERR_BADNAME:
        return "ERR_BADNAME";
    case IMGID_ERR_ALLOCATION:
        return "ERR_ALLOCATION";
    case IMGID_ERR_NOTFOUND:
        return "ERR_NOTFOUND";
    case IMGID_ERR_MISMATCH:
        return "ERR_MISMATCH";
    default:
        return "UNKNOWN_ERROR";
    }
}




static inline errno_t imgid_connect(
    IMGID *img,
    int FLAG
)
{
    printf("img template size:\n");
    printf("  naxes    = %d\n", img->mdt->naxis);
    printf("  datatype = %d\n", img->mdt->datatype);
    printf("  size[0]  = %d\n", img->mdt->size[0]);
    printf("  size[1]  = %d\n", img->mdt->size[1]);
    printf("  size[2]  = %d\n", img->mdt->size[2]);
    printf("  NBkw     = %d\n", img->mdt->NBkw);
    printf("  CBsize   = %d\n", img->mdt->CBsize);
    printf("  shared   = %d\n", img->mdt->shared);
    printf("  name     = %s\n", img->mdt->name);



    IMGID img_connected = {0}; // Local variable to hold the connected image
    IMAGE *image = NULL;
    int success = 0;
    errno_t retcode = IMGID_SUCCESS;

    img_connected.ID = -1;

    if (strlen(img->name) == 0) {
        retcode = IMGID_ERR_BADNAME;
        goto imgid_connect_report;
    }

    image = (IMAGE*) malloc(sizeof(IMAGE));
    if (image == NULL) {
        retcode = IMGID_ERR_ALLOCATION;
        goto imgid_connect_report;
    }

    if (ImageStreamIO_read_sharedmem_image_toIMAGE(img->name, image) == IMAGESTREAMIO_SUCCESS) {
        success = 1;
    } else {
        if (FLAG == IMGID_CONNECT_CHECK_CREATE) {
            success = 0;
        } else {
            free(image);
            img->ID = -1; // Propagate failure
            retcode = IMGID_ERR_NOTFOUND;
            goto imgid_connect_report;
        }
    }

    if (success) {
        // We have an image connected.
        img_connected.im = image;
        img_connected.md = image->md;
        strcpy(img_connected.name, img->name);
        img_connected.ID = 0;

        // Now check if it matches the template 'img' if FLAG is set
        if (FLAG == IMGID_CONNECT_CHECK_FAIL || FLAG == IMGID_CONNECT_CHECK_CREATE) {

            // Compare img_connected with img (template)
            // img is the template here.

            uint64_t diff = imgid_compare(img_connected, *img);

            if (diff == 0) {
                // Match!
                // Copy connection info to *img
                img->im = img_connected.im;
                img->md = img_connected.md;
                strcpy(img->name, img_connected.name);
                img->ID = 0;
                // Update creation params from what we found
                imgid_update_creationparams(img);
                retcode = IMGID_CONNECTED;
                goto imgid_connect_report;
            } else {
                // Mismatch
                if (FLAG == IMGID_CONNECT_CHECK_FAIL) {
                    printf("Image format mismatch\n");
                    free(image);
                    img->ID = -1;
                    retcode = IMGID_ERR_MISMATCH;
                    goto imgid_connect_report;
                }
                if (FLAG == IMGID_CONNECT_CHECK_CREATE) {
                    // Re-create
                    // First free the memory of the image we connected to (but shouldn't close it, just free wrapper)
                    free(image);

                    // Create new one using `img` params.

                    // Allocate new IMAGE for it?
                    img->im = (IMAGE*) malloc(sizeof(IMAGE));
                    if(img->im == NULL) {
                        img->ID = -1;
                        retcode = IMGID_ERR_ALLOCATION;
                        goto imgid_connect_report;
                    }
                    img->mdt->shared = 1; // Enforce shared for connect
                    imgid_mkimage(img);

                    if (img->createcnt > 0) {
                        img->ID = 0;
                        retcode = IMGID_RECREATED;
                        goto imgid_connect_report;
                    } else {
                        free(img->im);
                        img->ID = -1;
                        retcode = IMGID_ERR_ALLOCATION;
                        goto imgid_connect_report;
                    }
                }
            }
        } else {
            // No check, just copy info
            img->im = img_connected.im;
            img->md = img_connected.md;
            strcpy(img->name, img_connected.name);
            img->ID = 0;
            // Update creation params from what we found
            imgid_update_creationparams(img);
            retcode = IMGID_CONNECTED;
            goto imgid_connect_report;
        }
    } else {
        // Read failed (does not exist?)
        if (FLAG == IMGID_CONNECT_CHECK_CREATE) {
            // Create it
            img->im = image; // use the allocated struct
            img->mdt->shared = 1;
            imgid_mkimage(img);
            img->ID = 0;
            retcode = IMGID_CREATED;
            goto imgid_connect_report;
        }
        free(image);
        img->ID = -1;
        retcode = IMGID_ERR_NOTFOUND;
        goto imgid_connect_report;
    }

imgid_connect_report:
    printf("imgid_connect: image %-32s status %s\n", img->name, imgid_strerror(retcode));
    return retcode;
}


static inline void imgid_connect_create_2Df32(
    IMGID *img,
    uint32_t xsize,
    uint32_t ysize
)
{
    img->mdt->datatype = _DATATYPE_FLOAT;
    img->mdt->naxis    = 2;
    img->mdt->size[0]  = xsize;
    img->mdt->size[1]  = ysize;
    img->mdt->shared   = 1;
    img->mdt->NBkw     = IMGID_NB_KEYWO_MAX;
    img->mdt->CBsize   = 0;
    imgid_connect(img, IMGID_CONNECT_CHECK_CREATE);
}

static inline void imgid_connect_create_3Df32(
    IMGID *img,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    img->mdt->datatype = _DATATYPE_FLOAT;
    img->mdt->naxis    = 3;
    img->mdt->size[0]  = xsize;
    img->mdt->size[1]  = ysize;
    img->mdt->size[2]  = zsize;
    img->mdt->shared   = 1;
    img->mdt->NBkw     = IMGID_NB_KEYWO_MAX;
    img->mdt->CBsize   = 0;
    imgid_connect(img, IMGID_CONNECT_CHECK_CREATE);
}


#endif