/** @file CR2toFITS.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "readPGM.h"

static int CR2toFITS_NORM = 0;
// 1 if FITS should be normalized to ISO = 1, exposure = 1 sec, and F/1.0

// ==========================================
// Forward declaration(s)
// ==========================================

imageID CR2toFITS(const char *__restrict fnameCR2,
                  const char *__restrict fnameFITS);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t CR2toFITS_cli()
{
    //  if(CLI_checkarg(1, 3)+CLI_checkarg(2, 3))
    CR2toFITS(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string);
    // else
    // return(0);

    return RETURN_SUCCESS;
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t CR2toFITS_addCLIcmd()
{

    RegisterCLIcommand(
        "cr2tofits",
        __FILE__,
        CR2toFITS_cli,
        "convert cr2 file to fits",
        "<input CR2 file> <output FITS file>",
        "cr2tofits im01.CR2 im01.fits",
        "int CR2toFITS(const char *fnameCR2, const char *fnameFITS)");

    return RETURN_SUCCESS;
}

/**
 * ## Purpose
 *
 *  Convert CR2 to FITS
 *
 * @note assumes dcraw is installed
 */
imageID CR2toFITS(const char *__restrict fnameCR2,
                  const char *__restrict fnameFITS)
{
    FILE *fp;

    float   iso;
    float   shutter;
    float   aperture;
    imageID ID;
    long    xsize, ysize;
    long    ii;

    EXECUTE_SYSTEM_COMMAND("dcraw -t 0 -D -4 -c %s > _tmppgm.pgm", fnameCR2);

    ID = read_PGMimage("_tmppgm.pgm", "tmpfits1");
    if(system("rm _tmppgm.pgm") != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    if(CR2toFITS_NORM == 1)
    {
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

        ID    = image_ID("tmpfits1");
        xsize = data.image[ID].md[0].size[0];
        ysize = data.image[ID].md[0].size[1];

        for(ii = 0; ii < xsize * ysize; ii++)
        {
            data.image[ID].array.F[ii] /= (shutter * aperture * aperture * iso);
        }
    }

    save_fl_fits("tmpfits1", fnameFITS);
    delete_image_ID("tmpfits1", DELETE_IMAGE_ERRMODE_WARNING);

    return ID;
}
