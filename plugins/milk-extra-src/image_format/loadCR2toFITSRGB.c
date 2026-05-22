/**
 * @file loadCR2toFITSRGB.c
 * @brief Load CR2 file into R G B images
 */

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "FITStorgbFITSsimple.h"
#include "readPGM.h"

static int CR2toFITS_NORM = 0;

static float FLUXFACTOR = 1.0;

// Forward declaration
errno_t loadCR2toFITSRGB(const char *__restrict fnameCR2,
                         const char *__restrict fnameFITSr,
                         const char *__restrict fnameFITSg,
                         const char *__restrict fnameFITSb);

static char p_cr2[FUNCTION_PARAMETER_STRMAXLEN] = "im";
static char p_r[FUNCTION_PARAMETER_STRMAXLEN]   = "imR";
static char p_g[FUNCTION_PARAMETER_STRMAXLEN]   = "imG";
static char p_b[FUNCTION_PARAMETER_STRMAXLEN]   = "imB";

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "loadcr2torgb",
    .cmdkey           = "loadcr2torgb",
    .description      = "load CR2 file into R G B images",
    .description_long = "Load a Canon CR2 file and demosaic it into separate R, G, B FITS images."
};

#define FPS_PARAMS(X)                                                                   \
    X(".in_name", p_cr2, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input CR2 image") \
    X(".out_r", p_r, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output red")              \
    X(".out_g", p_g, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output green")            \
    X(".out_b", p_b, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output blue")

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
    loadCR2toFITSRGB(p_cr2, p_r, p_g, p_b);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_format__loadCR2toFITSRGB()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

// assumes dcraw is installed
errno_t loadCR2toFITSRGB(const char *__restrict fnameCR2,
                         const char *__restrict fnameFITSr,
                         const char *__restrict fnameFITSg,
                         const char *__restrict fnameFITSb)
{
    EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -t 0 -D -4 -c %s > _tmppgm.pgm", fnameCR2);

    read_PGMimage("_tmppgm.pgm", "tmpfits1");
    //  r = system("rm _tmppgm.pgm");

    if (CR2toFITS_NORM == 1)
    {
        FILE *fp;
        float iso;
        float shutter;
        float aperture;
        //imageID ID;
        //long xsize,ysize;

        EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -i -v %s | grep \"ISO speed\"| awk '{print $3}' > "
                                       "iso_tmp.txt",
                                       fnameCR2);

        if ((fp = fopen("iso_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        if (fscanf(fp, "%f\n", &iso) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if (system("rm iso_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("iso = %f\n", iso);

        EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -i -v %s | grep \"Shutter\"| awk '{print $2}' > "
                                       "shutter_tmp.txt",
                                       fnameCR2);

        if ((fp = fopen("shutter_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }

        if (fscanf(fp, "%f\n", &shutter) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if (system("rm shutter_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("shutter = %f\n", shutter);

        EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -i -v %s | grep \"Aperture\"| awk '{print $2}' > "
                                       "aperture_tmp.txt",
                                       fnameCR2);

        if ((fp = fopen("aperture_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        if (fscanf(fp, "f/%f\n", &aperture) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if (system("rm aperture_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("aperture = %f\n", aperture);

        //ID = image_ID("tmpfits1", dcimg, dcnimg);
        //        xsize = dcimg[ID].md[0].size[0];
        //        ysize = dcimg[ID].md[0].size[1];

        FLUXFACTOR = aperture * aperture / (shutter * iso);
    }
    else
    {
        FLUXFACTOR = 1.0;
    }

    printf("FLUXFACTOR = %g\n", FLUXFACTOR);

    if (variable_ID("RGBfullres") == -1)
    {
        convert_rawbayerFITStorgbFITS_simple("tmpfits1", fnameFITSr, fnameFITSg, fnameFITSb, 1);
    }
    else
    {
        convert_rawbayerFITStorgbFITS_simple("tmpfits1", fnameFITSr, fnameFITSg, fnameFITSb, 0);
    }

    delete_image_ID("tmpfits1", DELETE_IMAGE_ERRMODE_WARNING);

    FLUXFACTOR = 1.0;

    return RETURN_SUCCESS;
}
