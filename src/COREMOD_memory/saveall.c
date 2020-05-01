/** @file saveall.c
 */


#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"
#include "create_image.h"
#include "delete_image.h"
#include "image_copy.h"

#include "COREMOD_iofits/COREMOD_iofits.h"



//
// save all current images/stream onto file
//
errno_t COREMOD_MEMORY_SaveAll_snapshot(
    const char *dirname
)
{
    long *IDarray;
    long *IDarraycp;
    long i;
    long imcnt = 0;
    char imnamecp[200];
    char fnamecp[500];
    long ID;


    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            imcnt++;
        }

    IDarray = (long *) malloc(sizeof(long) * imcnt);
    IDarraycp = (long *) malloc(sizeof(long) * imcnt);

    imcnt = 0;
    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            IDarray[imcnt] = i;
            imcnt++;
        }

	EXECUTE_SYSTEM_COMMAND("mkdir -p %s", dirname);

    // create array for each image
    for(i = 0; i < imcnt; i++)
    {
        ID = IDarray[i];
        sprintf(imnamecp, "%s_cp", data.image[ID].name);
        //printf("image %s\n", data.image[ID].name);
        IDarraycp[i] = copy_image_ID(data.image[ID].name, imnamecp, 0);
    }

    list_image_ID();

    for(i = 0; i < imcnt; i++)
    {
        ID = IDarray[i];
        sprintf(imnamecp, "%s_cp", data.image[ID].name);
        sprintf(fnamecp, "!./%s/%s.fits", dirname, data.image[ID].name);
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
errno_t COREMOD_MEMORY_SaveAll_sequ(
    const char *dirname,
    const char *IDtrig_name,
    long semtrig,
    long NBframes
)
{
    long *IDarray;
    long *IDarrayout;
    long i;
    long imcnt = 0;
    char imnameout[200];
    char fnameout[500];
    imageID ID;
    imageID IDtrig;

    long frame = 0;
    char *ptr0;
    char *ptr1;
    uint32_t *imsizearray;




    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            imcnt++;
        }

    IDarray = (imageID *) malloc(sizeof(imageID) * imcnt);
    IDarrayout = (imageID *) malloc(sizeof(imageID) * imcnt);

    imcnt = 0;
    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
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
    for(i = 0; i < imcnt; i++)
    {
        sprintf(imnameout, "%s_out", data.image[IDarray[i]].name);
        imsizearray[i] = sizeof(float) * data.image[IDarray[i]].md[0].size[0] *
                         data.image[IDarray[i]].md[0].size[1];
        printf("Creating image %s  size %d x %d x %ld\n", imnameout,
               data.image[IDarray[i]].md[0].size[0], data.image[IDarray[i]].md[0].size[1],
               NBframes);
        fflush(stdout);
        IDarrayout[i] = create_3Dimage_ID(imnameout,
                                          data.image[IDarray[i]].md[0].size[0], data.image[IDarray[i]].md[0].size[1],
                                          NBframes);
    }
    list_image_ID();

    printf("filling arrays\n");
    fflush(stdout);

    // drive semaphore to zero
    while(sem_trywait(data.image[IDtrig].semptr[semtrig]) == 0) {}

    frame = 0;
    while(frame < NBframes)
    {
        sem_wait(data.image[IDtrig].semptr[semtrig]);
        for(i = 0; i < imcnt; i++)
        {
            ID = IDarray[i];
            ptr0 = (char *) data.image[IDarrayout[i]].array.F;
            ptr1 = ptr0 + imsizearray[i] * frame;
            memcpy(ptr1, data.image[ID].array.F, imsizearray[i]);
        }
        frame++;
    }


    printf("Saving images\n");
    fflush(stdout);

    list_image_ID();


    for(i = 0; i < imcnt; i++)
    {
        ID = IDarray[i];
        sprintf(imnameout, "%s_out", data.image[ID].name);
        sprintf(fnameout, "!./%s/%s_out.fits", dirname, data.image[ID].name);
        save_fits(imnameout, fnameout);
    }

    free(IDarray);
    free(IDarrayout);
    free(imsizearray);

    return RETURN_SUCCESS;
}





