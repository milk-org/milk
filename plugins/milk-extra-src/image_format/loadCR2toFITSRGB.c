/** @file loadCR2toFITSRGB.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "FITStorgbFITSsimple.h"
#include "readPGM.h"

static int CR2toFITS_NORM = 0;
// 1 if FITS should be normalized to ISO = 1, exposure = 1 sec, and F/1.0

static float FLUXFACTOR = 1.0;

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t loadCR2toFITSRGB(const char *__restrict fnameCR2,
                         const char *__restrict fnameFITSr,
                         const char *__restrict fnameFITSg,
                         const char *__restrict fnameFITSb);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_FORMAT_loadCR2toFITSRGB_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 3) +
            CLI_checkarg(4, 3) ==
            0)
    {
        loadCR2toFITSRGB(data.cmdargtoken[1].val.string,
                         data.cmdargtoken[2].val.string,
                         data.cmdargtoken[3].val.string,
                         data.cmdargtoken[4].val.string);
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t loadCR2toFITSRGB_addCLIcmd()
{

    RegisterCLIcommand("loadcr2torgb",
                       __FILE__,
                       IMAGE_FORMAT_loadCR2toFITSRGB_cli,
                       "load CR2 file into R G B images",
                       "<input image> <imR> <imG> <imB>",
                       "loadcr2torgb im imR imG imB",
                       "loadCR2toFITSRGB(const char *fnameCR2, const char "
                       "*fnameFITSr, const char *fnameFITSg, const "
                       "char *fnameFITSb)");

    return RETURN_SUCCESS;
}

// assumes dcraw is installed
errno_t loadCR2toFITSRGB(const char *__restrict fnameCR2,
                         const char *__restrict fnameFITSr,
                         const char *__restrict fnameFITSg,
                         const char *__restrict fnameFITSb)
{
    EXECUTE_SYSTEM_COMMAND("dcraw -t 0 -D -4 -c %s > _tmppgm.pgm", fnameCR2);

    read_PGMimage("_tmppgm.pgm", "tmpfits1");
    //  r = system("rm _tmppgm.pgm");

    if(CR2toFITS_NORM == 1)
    {
        FILE *fp;
        float iso;
        float shutter;
        float aperture;
        //imageID ID;
        //long xsize,ysize;

        EXECUTE_SYSTEM_COMMAND(
            "dcraw -i -v %s | grep \"ISO speed\"| awk '{print $3}' > "
            "iso_tmp.txt",
            fnameCR2);

        if((fp = fopen("iso_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        if(fscanf(fp, "%f\n", &iso) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if(system("rm iso_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("iso = %f\n", iso);

        EXECUTE_SYSTEM_COMMAND(
            "dcraw -i -v %s | grep \"Shutter\"| awk '{print $2}' > "
            "shutter_tmp.txt",
            fnameCR2);

        if((fp = fopen("shutter_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }

        if(fscanf(fp, "%f\n", &shutter) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if(system("rm shutter_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("shutter = %f\n", shutter);

        EXECUTE_SYSTEM_COMMAND(
            "dcraw -i -v %s | grep \"Aperture\"| awk '{print $2}' > "
            "aperture_tmp.txt",
            fnameCR2);

        if((fp = fopen("aperture_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        if(fscanf(fp, "f/%f\n", &aperture) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if(system("rm aperture_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("aperture = %f\n", aperture);

        //ID = image_ID("tmpfits1");
        //        xsize = data.image[ID].md[0].size[0];
        //        ysize = data.image[ID].md[0].size[1];

        FLUXFACTOR = aperture * aperture / (shutter * iso);
    }
    else
    {
        FLUXFACTOR = 1.0;
    }

    printf("FLUXFACTOR = %g\n", FLUXFACTOR);

    if(variable_ID("RGBfullres") == -1)
    {
        convert_rawbayerFITStorgbFITS_simple("tmpfits1",
                                             fnameFITSr,
                                             fnameFITSg,
                                             fnameFITSb,
                                             1);
    }
    else
    {
        convert_rawbayerFITStorgbFITS_simple("tmpfits1",
                                             fnameFITSr,
                                             fnameFITSg,
                                             fnameFITSb,
                                             0);
    }

    delete_image_ID("tmpfits1", DELETE_IMAGE_ERRMODE_WARNING);

    FLUXFACTOR = 1.0;

    return RETURN_SUCCESS;
}
