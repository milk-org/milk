/**
 * @file    delete_image.c
 * @brief   delete image(s)
 *
 * Uses FPS V2 framework.
 */

#include <malloc.h>
#include <sys/mman.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"

// Forward declaration(s)
/**
 * @brief Delete an image from the image array.
 *
 * Frees pixel data and marks the slot as unused.
 * Removes SHM file if applicable.
 */
errno_t delete_image_ID(
    const char *__restrict imname,
    int                    errmode);


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "rmimg",
    .cmdkey      = "rm",
    .description = "remove image",
    .description_long =
    "Remove a single image from the current process memory. Frees the local buffer. If the image is shared, only the local mapping is removed."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char imname[FUNCTION_PARAMETER_STRMAXLEN]
    = "im";
static int64_t errmode_ptr = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imname", imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name") \
    X(".errmode", &errmode_ptr, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "errors mode (0:ign 1:warn 2:err 3:exit)")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/* (see exported functions below) */


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    IMGID img = imgid_make_from_name(imname);
    FUNC_CHECK_RETURN(
        delete_image_IMGID(
            &img, (int) errmode_ptr));

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata,
               my_bindings, nb_bindings,
               compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__delete_image()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    int cmdi = RegisterCLIcmd(
                   CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings =
        &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif


/* ================================================================
 *  EXPORTED UTILITY FUNCTIONS
 *  (unchanged — called from other translation units)
 * ============================================================= */

/** @brief deletes an ID
 *
 * errmode values:
 * DELETE_IMAGE_ERRMODE_IGNORE
 * DELETE_IMAGE_ERRMODE_WARNING
 * DELETE_IMAGE_ERRMODE_ERROR
 * DELETE_IMAGE_ERRMODE_EXIT
 *
 */
errno_t delete_image_IMGID(
    IMGID *img,
    int   errmode)
{
    return delete_image(img, errmode);
}

/** @brief deletes an ID
 *
 * errmode values:
 * DELETE_IMAGE_ERRMODE_IGNORE
 * DELETE_IMAGE_ERRMODE_WARNING
 * DELETE_IMAGE_ERRMODE_ERROR
 * DELETE_IMAGE_ERRMODE_EXIT
 *
 */
errno_t delete_image(
    IMGID *img,
    int   errmode)
{
    DEBUG_TRACE_FSTART();

    long s __attribute__((unused));
    char fname[STRINGMAXLEN_FULLFILENAME];

    imageID ID = img->ID;

    if(ID == -1)
    {
        if(errmode == DELETE_IMAGE_ERRMODE_IGNORE)
        {
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }

        if(errmode == DELETE_IMAGE_ERRMODE_WARNING)
        {
            PRINT_WARNING(
                "Image \"%s\" does not exist",
                img->name);
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }

        if(errmode == DELETE_IMAGE_ERRMODE_ERROR)
        {
            PRINT_WARNING(
                "Image \"%s\" does not exist",
                img->name);
            FUNC_RETURN_FAILURE(
                "Image \"%s\" does not exist",
                img->name);
        }

        if(errmode == DELETE_IMAGE_ERRMODE_EXIT)
        {
            abort();
        }
        return -1;
    }
    else
    {
        img->ID = -1;

        if(dcimg[ID].md[0].shared == 1)
        {
            free(dcimg[ID].semptr);
            dcimg[ID].semptr = NULL;

            if(dcimg[ID].semlog != NULL)
            {
                dcimg[ID].semlog = NULL;
            }

            if(munmap(dcimg[ID].md,
                      dcimg[ID].memsize) == -1)
            {
                printf(
                    "unmapping ID %ld : %p  %ld\n",
                    ID,
                    dcimg[ID].md,
                    dcimg[ID].memsize);
                PRINT_ERROR("Error un-mmapping the file: %s", strerror(errno));
            }

            close(dcimg[ID].shmfd);
            dcimg[ID].shmfd = -1;

            dcimg[ID].md = NULL;
            dcimg[ID].kw = NULL;

            dcimg[ID].memsize = 0;

            if(dcrmshm == 1)
            {
                EXECUTE_SYSTEM_COMMAND_NOCHECK(
                    "rm /dev/shm/sem.%s.%s_sem*",
                    dcshmsemdir,
                    img->name);
                WRITE_FULLFILENAME(
                    fname,
                    "/dev/shm/sem.%s.%s_semlog",
                    dcshmsemdir,
                    img->name);
                remove(fname);

                EXECUTE_SYSTEM_COMMAND_NOCHECK(
                    "rm %s/%s.im.shm",
                    dcshmdir, img->name);
            }
        }
        else
        {
            if(dcimg[ID].md[0].datatype
                    == _DATATYPE_UINT8)
            {
                if(dcimg[ID].array.UI8 == NULL)
                {
                    FUNC_RETURN_FAILURE(
                        "data array pointer is null");
                }
                free(dcimg[ID].array.UI8);
                dcimg[ID].array.UI8 = NULL;
            }
            else if(dcimg[ID].md[0].datatype
                    == _DATATYPE_INT32)
            {
                if(dcimg[ID].array.SI32 == NULL)
                {
                    FUNC_RETURN_FAILURE(
                        "data array pointer is null");
                }
                free(dcimg[ID].array.SI32);
                dcimg[ID].array.SI32 = NULL;
            }
            else if(dcimg[ID].md[0].datatype
                    == _DATATYPE_FLOAT)
            {
                if(dcimg[ID].array.F == NULL)
                {
                    FUNC_RETURN_FAILURE(
                        "data array pointer is null");
                }
                free(dcimg[ID].array.F);
                dcimg[ID].array.F = NULL;
            }
            else if(dcimg[ID].md[0].datatype
                    == _DATATYPE_DOUBLE)
            {
                if(dcimg[ID].array.D == NULL)
                {
                    FUNC_RETURN_FAILURE(
                        "data array pointer is null");
                }
                free(dcimg[ID].array.D);
                dcimg[ID].array.D = NULL;
            }
            else if(dcimg[ID].md[0].datatype
                    == _DATATYPE_COMPLEX_FLOAT)
            {
                if(dcimg[ID].array.CF == NULL)
                {
                    FUNC_RETURN_FAILURE(
                        "data array pointer is null");
                }
                free(dcimg[ID].array.CF);
                dcimg[ID].array.CF = NULL;
            }
            else if(dcimg[ID].md[0].datatype
                    == _DATATYPE_COMPLEX_DOUBLE)
            {
                if(dcimg[ID].array.CD == NULL)
                {
                    FUNC_RETURN_FAILURE(
                        "data array pointer is null");
                }
                free(dcimg[ID].array.CD);
                dcimg[ID].array.CD = NULL;
            }

            if(dcimg[ID].md == NULL)
            {
                FUNC_RETURN_FAILURE(
                    "data array pointer is null");
            }
            free(dcimg[ID].md);
            dcimg[ID].md = NULL;

            if(dcimg[ID].kw != NULL)
            {
                free(dcimg[ID].kw);
                dcimg[ID].kw = NULL;
            }
        }

        // Mark slot free LAST, after all cleanup
        dcimg[ID].used = 0;
    }



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/** @brief deletes an ID
 *
 * errmode values:
 * DELETE_IMAGE_ERRMODE_IGNORE
 * DELETE_IMAGE_ERRMODE_WARNING
 * DELETE_IMAGE_ERRMODE_ERROR
 * DELETE_IMAGE_ERRMODE_EXIT
 *
 */
errno_t delete_image_ID(
    const char *__restrict imname,
    int                    errmode)
{
    DEBUG_TRACE_FSTART();

    IMGID   img = imgid_make_from_name(imname);
    imageID ID  = resolveIMGID(
                      &img,  errmode,
                      dcimg, dcnimg);

    if(ID != -1)
    {
        delete_image(&img, errmode);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// delete all images with a prefix
/**
 * @brief Delete all images matching a name prefix.
 */
errno_t delete_image_ID_prefix(
    const char *prefix
)
{
    imageID i;

    for(i = 0; i < dcnimg; i++)
    {
        if(dcimg[i].used == 1)
        {
            if((strncmp(prefix,
                        dcimg[i].name,
                        strlen(prefix))) == 0)
            {
                printf("deleting image %s\n",
                       dcimg[i].name);
                delete_image_ID(
                    dcimg[i].name,
                    DELETE_IMAGE_ERRMODE_IGNORE);
            }
        }
    }
    return RETURN_SUCCESS;
}
