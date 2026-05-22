/**
 * @file read_binary32f.c
 * @brief Read 32-bit float RAW image
 */

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_FORMAT_read_binary32f(const char *__restrict fname,
                                    long xsize,
                                    long ysize,
                                    const char *__restrict IDname);

static char      p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "im.bin";
static long long p_xsize                               = 512;
static long long p_ysize                               = 512;
static char      p_out[FUNCTION_PARAMETER_STRMAXLEN]   = "im";

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "readb32fim",
    .cmdkey           = "readb32fim",
    .description      = "read 32-bit float RAW image",
    .description_long = "Read a raw 32-bit float binary file into a shared memory image stream."
};

#define FPS_PARAMS(X)                                                              \
    X(".in_fname", p_fname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "binary file") \
    X(".xsize", &p_xsize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x size")         \
    X(".ysize", &p_ysize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y size")         \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };
static const int       nb_bindings   = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg[]        = { FPS_PARAMS(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata    = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS     cms           = { 0 };

static __attribute__((constructor)) void init_cms(void)
{
    strncpy(CLIcmddata.key, FPS_app_info.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info.description, sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL)
    {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    IMAGE_FORMAT_read_binary32f(p_fname, (long) p_xsize, (long) p_ysize, p_out);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_format__read_binary32f()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_read_binary32f(const char *__restrict fname,
                                    long xsize,
                                    long ysize,
                                    const char *__restrict IDname)
{
    DEBUG_TRACE_FSTART();

    FILE         *fp;
    float        *buffer;
    unsigned long fileLen;

    //Open file
    if ((fp = fopen(fname, "rb")) == NULL)
    {
        PRINT_ERROR("Cannot open file");
        return (0);
    }

    //Get file length
    fseek(fp, 0, SEEK_END);
    fileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    //Allocate memory
    buffer = (float *) malloc(fileLen + 1);
    if (!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(fp);
        return (0);
    }

    //Read file contents into buffer
    if (fread(buffer, fileLen, 1, fp) < 1)
    {
        PRINT_ERROR("fread() returns <1 value");
    }
    fclose(fp);

    IMGID imgout       = imgid_make_from_name_2D(IDname, xsize, ysize);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    long i = 0;
    for (long jj = 0; jj < ysize; jj++)
    {
        for (long ii = 0; ii < xsize; ii++)
        {
            imgout.im->array.F[jj * xsize + ii] = buffer[i];
            i++;
        }
    }

    free(buffer);

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}
