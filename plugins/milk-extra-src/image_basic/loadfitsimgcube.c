/**
 * @file loadfitsimgcube.c
 * @brief Load images into a cube
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
long load_fitsimages_cube(
    const char *__restrict strfilter,
    const char *__restrict ID_out_name);

static char p_pat[FUNCTION_PARAMETER_STRMAXLEN]
    = "im";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "out";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "loadfitsimgcube",
    .cmdkey      = "loadfitsimgcube",
    .description =
        "load images into a single cube"
};

#define FPS_PARAMS(X) \
    X(".in_pattern", p_pat, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "string pattern") \
    X(".out_name", p_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output cube")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};
static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static errno_t compute_function()
{
    load_fitsimages_cube(p_pat, p_out);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_basic__loadfitsimgcube()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

// load all images matching strfilter + .fits into a data cube
// return number of images loaded
// image name in buffer is same as file name without extension
long load_fitsimages_cube(const char *__restrict strfilter,
                          const char *__restrict ID_out_name)
{
    long     cnt = 0;
    char     fname[STRINGMAXLEN_FILENAME];
    char     fname1[STRINGMAXLEN_FILENAME];
    FILE    *fp;
    uint32_t xsize, ysize;
    imageID  ID;
    imageID  IDout;

    printf("Filter = %s\n", strfilter);

    EXECUTE_SYSTEM_COMMAND("ls %s > flist.tmp\n", strfilter);

    xsize = 0;
    ysize = 0;

    if((fp = fopen("flist.tmp", "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    while(fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        if(cnt == 0)
        {
            load_fits(fname, "imtmplfc", 1, &ID);
            xsize = dcimg[ID].md[0].size[0];
            ysize = dcimg[ID].md[0].size[1];
            delete_image_ID("imtmplfc", DELETE_IMAGE_ERRMODE_WARNING);
        }

        load_fits(fname, "imtmplfc", 1, &ID);
        if((dcimg[ID].md[0].size[0] != xsize) ||
                (dcimg[ID].md[0].size[1] != ysize))
        {
            fprintf(stderr,
                    "ERROR in load_fitsimages_cube: not all images have the "
                    "same size\n");
            exit(0);
        }
        delete_image_ID("imtmplfc", DELETE_IMAGE_ERRMODE_WARNING);
        cnt++;
    }
    fclose(fp);

    printf("Creating 3D cube ... ");
    fflush(stdout);
    create_3Dimage_ID(ID_out_name, xsize, ysize, cnt, &IDout);
    printf("\n");
    fflush(stdout);

    cnt = 0;
    if((fp = fopen("flist.tmp", "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    while(fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        strncpy(fname1, fname, STRINGMAXLEN_FILENAME);
        fname1[strlen(fname) - 5] = '\0';
        load_fits(fname, fname1, 1, NULL);
        printf("Image %s loaded -> %s\n", fname, fname1);
        ID = image_ID(fname1, dcimg, dcnimg);
        for(uint64_t ii = 0; ii < xsize * ysize; ii++)
        {
            dcimg[IDout].array.F[xsize * ysize * cnt + ii] =
                dcimg[ID].array.F[ii];
        }
        delete_image_ID(fname1, DELETE_IMAGE_ERRMODE_WARNING);
        cnt++;
    }

    fclose(fp);
    /*  n = snprintf(command,SBUFFERSIZE,"rm flist.tmp");
      if(n >= SBUFFERSIZE)
          PRINT_ERROR("Attempted to write string buffer with too many characters");

      if(system(command)==-1)
      {
          printf("WARNING: system(\"%s\") failed [function: %s  file: %s  line: %d ]\n",command,__func__,__FILE__,__LINE__);
          //exit(0);
      }

    */
    printf("%ld images loaded into cube %s\n", cnt, ID_out_name);

    return (cnt);
}
