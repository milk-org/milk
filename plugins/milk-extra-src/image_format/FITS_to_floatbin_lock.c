/**
 * @file FITS_to_floatbin_lock.c
 * @brief Write float binary with file locking
 */

#include <sys/file.h>

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_FORMAT_FITS_to_floatbin_lock(const char *__restrict IDname,
                                           const char *__restrict fname);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]    = "im";
static char p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "im.bin";

static FPS_APP_INFO FPS_app_info = { .fps_name    = "writefloatlock",
                                     .cmdkey      = "writefloatlock",
                                     .description = "write float with file locking",
                                     .description_long =
                                         "Write a FITS image as a raw 32-bit float binary file "
                                         "with file locking for safe concurrent access." };

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_fname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output binary file")

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    IMAGE_FORMAT_FITS_to_floatbin_lock(p_in, p_fname);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_format__floatbin_lock()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_FITS_to_floatbin_lock(const char *__restrict IDname,
                                           const char *__restrict fname)
{
    imageID ID = -1;
    long    xsize, ysize;
    long    ii;
    int     fd;
    float  *valarray;

    ID    = image_ID(IDname, dcimg, dcnimg);
    xsize = dcimg[ID].md[0].size[0];
    ysize = dcimg[ID].md[0].size[1];

    valarray = (float *) malloc(sizeof(float) * xsize * ysize);
    if (valarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if (dcimg[ID].md[0].datatype == _DATATYPE_FLOAT)
    {
        printf("WRITING float array\n");
        for (ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = dcimg[ID].array.F[ii];
        }
    }
    if (dcimg[ID].md[0].datatype == _DATATYPE_DOUBLE)
    {
        printf("WRITING double array\n");
        for (ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = (float) dcimg[ID].array.D[ii];
        }
    }

    if ((fd = open(fname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR)) == -1)
    {
        PRINT_ERROR("Cannot open file");
    }
    flock(fd, LOCK_EX);
    if (fd < 0)
    {
        printf("Error opening file: %s\n", strerror(errno));
    }

    if (write(fd, valarray, sizeof(float) * xsize * ysize) < 1)
    {
        PRINT_ERROR("write() returns <1 value");
    }
    //  for(ii=0;ii<xsize*ysize;ii++)
    //  printf("[%ld %f] ", ii, valarray[ii]);

    flock(fd, LOCK_UN);
    close(fd);

    free(valarray);

    return ID;
}
