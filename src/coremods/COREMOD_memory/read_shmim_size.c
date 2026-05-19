/**
 * @file    read_shmim_size.c
 * @brief   read shared memory image size
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "readshmimsize",
    .cmdkey      = "readshmimsize",
    .description = "read shared memory image size",
    .description_long =
        "Read only the size metadata of a shared memory stream without mapping the full pixel buffer. Lightweight probe for stream dimensions."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char insname[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream";
static char outfname[FUNCTION_PARAMETER_STRMAXLEN]
    = "imsize.txt";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_sname", insname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".outfname", outfname, \
      FPTYPE_STRING_NOT_STREAM, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output file name")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Read shared memory image size
 *
 * @param name   stream name
 * @param fname  file name to write image size
 */
imageID read_sharedmem_image_size(
    const char *name,
    const char *fname)
{
    int             SM_fd;
    struct stat     file_stat;
    char            SM_fname[
        STRINGMAXLEN_FULLFILENAME];
    IMAGE_METADATA *map;

    FILE           *fp;

    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);

    if(img.ID == -1)
    {
        WRITE_FULLFILENAME(
            SM_fname, "%s/%s.im.shm",
            dcshmdir, name);

        SM_fd = open(SM_fname, O_RDWR);
        if(SM_fd == -1)
        {
            printf(
                "Cannot import file"
                " - continuing\n");
        }
        else
        {
            fstat(SM_fd, &file_stat);

            map = (IMAGE_METADATA *) mmap(
                0,
                sizeof(IMAGE_METADATA),
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                SM_fd,
                0);
            if(map == MAP_FAILED)
            {
                close(SM_fd);
                PRINT_ERROR(
                    "mmap failed for %s",
                    SM_fname);
                return -1;
            }

            fp = fopen(fname, "w");
            for( int i = 0; i < map[0].naxis; i++)
            {
                fprintf(fp, "%ld ",
                        (long) map[0].size[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);

            if(munmap(map,
                      sizeof(IMAGE_METADATA))
                == -1)
            {
                printf("unmapping %s\n",
                       SM_fname);
                PRINT_ERROR(
                    "Error un-mmapping the file: %s",
                    strerror(errno));
            }
            close(SM_fd);
        }
    }
    else
    {
        fp = fopen(fname, "w");
        for(int i = 0;
             i < img.im->md[0].naxis;
             i++)
        {
            fprintf(
                fp, "%ld ",
                (long) img.im->md[0].size[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }

    return img.ID;
}


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

    read_sharedmem_image_size(insname, outfname);

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
CLIADDCMD_COREMOD_memory__read_sharedmem_image_size()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
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
