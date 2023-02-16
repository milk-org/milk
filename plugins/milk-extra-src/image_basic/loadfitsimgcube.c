/** @file loadfitsimgcube.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

long load_fitsimages_cube(const char *__restrict strfilter,
                          const char *__restrict ID_out_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_load_fitsimages_cube_cli()
{
    if(CLI_checkarg(1, 3) + CLI_checkarg(2, 3) == 0)
    {
        load_fitsimages_cube(data.cmdargtoken[1].val.string,
                             data.cmdargtoken[2].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t __attribute__((cold)) loadfitsimgcube_addCLIcmd()
{

    RegisterCLIcommand("loadfitsimgcube",
                       __FILE__,
                       image_basic_load_fitsimages_cube_cli,
                       "load multiple images into a single cube",
                       "loadfitsimgcube <string pattern> <outputcube>",
                       "loadfitsimgcube im out",
                       "long load_fitsimages_cube(const char *strfilter, const "
                       "char *ID_out_name)");

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
            xsize = data.image[ID].md[0].size[0];
            ysize = data.image[ID].md[0].size[1];
            delete_image_ID("imtmplfc", DELETE_IMAGE_ERRMODE_WARNING);
        }

        load_fits(fname, "imtmplfc", 1, &ID);
        if((data.image[ID].md[0].size[0] != xsize) ||
                (data.image[ID].md[0].size[1] != ysize))
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
        ID = image_ID(fname1);
        for(uint64_t ii = 0; ii < xsize * ysize; ii++)
        {
            data.image[IDout].array.F[xsize * ysize * cnt + ii] =
                data.image[ID].array.F[ii];
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
