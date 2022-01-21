/** @file saveall.c
 */

#include "CommandLineInterface/CLIcore.h"
#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "image_copy.h"
#include "list_image.h"

#include "COREMOD_iofits/COREMOD_iofits.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname);

errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname,
                                    const char *IDtrig_name,
                                    long        semtrig,
                                    long        NBframes);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_SaveAll_snapshot__cli()
{
    if (0 + CLI_checkarg(1, 5) == 0)
        {
            COREMOD_MEMORY_SaveAll_snapshot(data.cmdargtoken[1].val.string);
            return CLICMD_SUCCESS;
        }
    else
        {
            return CLICMD_INVALID_ARG;
        }
}

static errno_t COREMOD_MEMORY_SaveAll_sequ__cli()
{
    if (0 + CLI_checkarg(1, 5) + CLI_checkarg(2, CLIARG_IMG) +
            CLI_checkarg(3, CLIARG_LONG) + CLI_checkarg(4, CLIARG_LONG) ==
        0)
        {
            COREMOD_MEMORY_SaveAll_sequ(data.cmdargtoken[1].val.string,
                                        data.cmdargtoken[2].val.string,
                                        data.cmdargtoken[3].val.numl,
                                        data.cmdargtoken[4].val.numl);
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

errno_t saveall_addCLIcmd()
{

    RegisterCLIcommand(
        "imsaveallsnap",
        __FILE__,
        COREMOD_MEMORY_SaveAll_snapshot__cli,
        "save all images in directory",
        "<directory>",
        "imsaveallsnap dir1",
        "long COREMOD_MEMORY_SaveAll_snapshot(const char *dirname)");

    RegisterCLIcommand(
        "imsaveallseq",
        __FILE__,
        COREMOD_MEMORY_SaveAll_sequ__cli,
        "save all images in directory - sequence",
        "<directory> <trigger image name> <trigger semaphore> <NB frames>",
        "imsaveallsequ dir1 im1 3 20",
        "long COREMOD_MEMORY_SaveAll_sequ(const char *dirname, const char "
        "*IDtrig_name, long semtrig, long NBframes)");

    return RETURN_SUCCESS;
}

//
// save all current images/stream onto file
//
errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname)
{
    long *IDarray;
    long *IDarraycp;
    long  i;
    long  imcnt = 0;
    char  imnamecp[STRINGMAXLEN_IMGNAME];
    char  fnamecp[STRINGMAXLEN_FULLFILENAME];
    long  ID;

    for (i = 0; i < data.NB_MAX_IMAGE; i++)
        if (data.image[i].used == 1)
            {
                imcnt++;
            }

    IDarray   = (long *) malloc(sizeof(long) * imcnt);
    IDarraycp = (long *) malloc(sizeof(long) * imcnt);

    imcnt = 0;
    for (i = 0; i < data.NB_MAX_IMAGE; i++)
        {
            if (data.image[i].used == 1)
                {
                    IDarray[imcnt] = i;
                    imcnt++;
                }
        }

    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", dirname);

    // create array for each image
    for (i = 0; i < imcnt; i++)
        {
            ID = IDarray[i];
            WRITE_IMAGENAME(imnamecp, "%s_cp", data.image[ID].name);
            //printf("image %s\n", data.image[ID].name);
            IDarraycp[i] = copy_image_ID(data.image[ID].name, imnamecp, 0);
        }

    list_image_ID();

    for (i = 0; i < imcnt; i++)
        {
            ID = IDarray[i];
            WRITE_IMAGENAME(imnamecp, "%s_cp", data.image[ID].name);
            WRITE_FULLFILENAME(fnamecp,
                               "./%s/%s.fits",
                               dirname,
                               data.image[ID].name);
            save_fits(imnamecp, fnamecp);
        }

    free(IDarray);
    free(IDarraycp);

    return RETURN_SUCCESS;
}

//
// save all current images/stream onto file
// only saves 2D float streams into 3D cubes
//
errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname,
                                    const char *IDtrig_name,
                                    long        semtrig,
                                    long        NBframes)
{
    long   *IDarray;
    long   *IDarrayout;
    long    i;
    long    imcnt = 0;
    char    imnameout[200];
    char    fnameout[500];
    imageID ID;
    imageID IDtrig;

    long      frame = 0;
    char     *ptr0;
    char     *ptr1;
    uint32_t *imsizearray;

    for (i = 0; i < data.NB_MAX_IMAGE; i++)
        if (data.image[i].used == 1)
            {
                imcnt++;
            }

    IDarray    = (imageID *) malloc(sizeof(imageID) * imcnt);
    IDarrayout = (imageID *) malloc(sizeof(imageID) * imcnt);

    imcnt = 0;
    for (i = 0; i < data.NB_MAX_IMAGE; i++)
        if (data.image[i].used == 1)
            {
                IDarray[imcnt] = i;
                imcnt++;
            }
    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * imcnt);

    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", dirname);

    IDtrig = image_ID(IDtrig_name);

    printf("Creating arrays\n");
    fflush(stdout);

    // create 3D arrays
    for (i = 0; i < imcnt; i++)
        {
            sprintf(imnameout, "%s_out", data.image[IDarray[i]].name);
            imsizearray[i] = sizeof(float) *
                             data.image[IDarray[i]].md[0].size[0] *
                             data.image[IDarray[i]].md[0].size[1];
            printf("Creating image %s  size %d x %d x %ld\n",
                   imnameout,
                   data.image[IDarray[i]].md[0].size[0],
                   data.image[IDarray[i]].md[0].size[1],
                   NBframes);
            fflush(stdout);
            create_3Dimage_ID(imnameout,
                              data.image[IDarray[i]].md[0].size[0],
                              data.image[IDarray[i]].md[0].size[1],
                              NBframes,
                              &(IDarrayout[i]));
        }
    list_image_ID();

    printf("filling arrays\n");
    fflush(stdout);

    // drive semaphore to zero
    while (sem_trywait(data.image[IDtrig].semptr[semtrig]) == 0)
        {
        }

    frame = 0;
    while (frame < NBframes)
        {
            sem_wait(data.image[IDtrig].semptr[semtrig]);
            for (i = 0; i < imcnt; i++)
                {
                    ID   = IDarray[i];
                    ptr0 = (char *) data.image[IDarrayout[i]].array.F;
                    ptr1 = ptr0 + imsizearray[i] * frame;
                    memcpy(ptr1, data.image[ID].array.F, imsizearray[i]);
                }
            frame++;
        }

    printf("Saving images\n");
    fflush(stdout);

    list_image_ID();

    for (i = 0; i < imcnt; i++)
        {
            ID = IDarray[i];
            sprintf(imnameout, "%s_out", data.image[ID].name);
            sprintf(fnameout, "./%s/%s_out.fits", dirname, data.image[ID].name);
            save_fits(imnameout, fnameout);
        }

    free(IDarray);
    free(IDarrayout);
    free(imsizearray);

    return RETURN_SUCCESS;
}
