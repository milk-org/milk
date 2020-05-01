/**
 * @file    stream_sem.c
 * @brief   stream semaphores
 */


#include <pthread.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"




static pthread_t *thrarray_semwait;
static long NB_thrarray_semwait;


/**
 * @see ImageStreamIO_createsem
 */

imageID COREMOD_MEMORY_image_set_createsem(
    const char *IDname,
    long        NBsem
)
{
    imageID ID;

    ID = image_ID(IDname);

    if(ID != -1)
    {
        ImageStreamIO_createsem(&data.image[ID], NBsem);
    }

    return ID;
}




imageID COREMOD_MEMORY_image_seminfo(
    const char *IDname
)
{
    imageID ID;


    ID = image_ID(IDname);

    printf("  NB SEMAPHORES = %3d \n", data.image[ID].md[0].sem);
    printf(" semWritePID at %p\n", (void *) data.image[ID].semWritePID);
    printf(" semReadPID  at %p\n", (void *) data.image[ID].semReadPID);
    printf("----------------------------------\n");
    printf(" sem    value   writePID   readPID\n");
    printf("----------------------------------\n");
    int s;
    for(s = 0; s < data.image[ID].md[0].sem; s++)
    {
        int semval;

        sem_getvalue(data.image[ID].semptr[s], &semval);

        printf("  %2d   %6d   %8d  %8d\n", s, semval,
               (int) data.image[ID].semWritePID[s], (int) data.image[ID].semReadPID[s]);

    }
    printf("----------------------------------\n");
    int semval;
    sem_getvalue(data.image[ID].semlog, &semval);
    printf(" semlog = %3d\n", semval);
    printf("----------------------------------\n");

    return ID;
}




/**
 * @see ImageStreamIO_sempost
 */

imageID COREMOD_MEMORY_image_set_sempost(
    const char *IDname,
    long        index
)
{
    imageID ID;

    ID = image_ID(IDname);
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname);
    }

    ImageStreamIO_sempost(&data.image[ID], index);

    return ID;
}






/**
 * @see ImageStreamIO_sempost
 */
imageID COREMOD_MEMORY_image_set_sempost_byID(
    imageID ID,
    long    index
)
{
    ImageStreamIO_sempost(&data.image[ID], index);

    return ID;
}



/**
 * @see ImageStreamIO_sempost_excl
 */
imageID COREMOD_MEMORY_image_set_sempost_excl_byID(
    imageID ID,
    long index
)
{
    ImageStreamIO_sempost_excl(&data.image[ID], index);

    return ID;
}




/**
 * @see ImageStreamIO_sempost_loop
 */

imageID COREMOD_MEMORY_image_set_sempost_loop(
    const char *IDname,
    long        index,
    long        dtus
)
{
    imageID ID;

    ID = image_ID(IDname);
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname);
    }

    ImageStreamIO_sempost_loop(&data.image[ID], index, dtus);

    return ID;
}



/**
 * @see ImageStreamIO_semwait
 */
imageID COREMOD_MEMORY_image_set_semwait(
    const char *IDname,
    long        index
)
{
    imageID ID;

    ID = image_ID(IDname);
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname);
    }

    ImageStreamIO_semwait(&data.image[ID], index);

    return ID;
}





// only works for sem0
void *waitforsemID(
    void *ID
)
{
    pthread_t tid;
    int t;
    //    int semval;


    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    tid = pthread_self();

    //    sem_getvalue(data.image[(long) ID].semptr, &semval);
    //    printf("tid %u waiting for sem ID %ld   sem = %d   (%s)\n", (unsigned int) tid, (long) ID, semval, data.image[(long) ID].name);
    //    fflush(stdout);
    sem_wait(data.image[(imageID) ID].semptr[0]);
    //    printf("tid %u sem ID %ld done\n", (unsigned int) tid, (long) ID);
    //    fflush(stdout);

    for(t = 0; t < NB_thrarray_semwait; t++)
    {
        if(tid != thrarray_semwait[t])
        {
            //            printf("tid %u cancel thread %d tid %u\n", (unsigned int) tid, t, (unsigned int) (thrarray_semwait[t]));
            //           fflush(stdout);
            pthread_cancel(thrarray_semwait[t]);
        }
    }

    pthread_exit(NULL);
}




/// \brief Wait for multiple images semaphores [OR], only works for sem0
errno_t COREMOD_MEMORY_image_set_semwait_OR_IDarray(
    imageID *IDarray,
    long     NB_ID
)
{
    int t;
//    int semval;

    //   printf("======== ENTER COREMOD_MEMORY_image_set_semwait_OR_IDarray [%ld] =======\n", NB_ID);
    //   fflush(stdout);

    thrarray_semwait = (pthread_t *) malloc(sizeof(pthread_t) * NB_ID);
    NB_thrarray_semwait = NB_ID;

    for(t = 0; t < NB_ID; t++)
    {
        //      printf("thread %d create, ID = %ld\n", t, IDarray[t]);
        //      fflush(stdout);
        pthread_create(&thrarray_semwait[t], NULL, waitforsemID, (void *)IDarray[t]);
    }

    for(t = 0; t < NB_ID; t++)
    {
        //         printf("thread %d tid %u join waiting\n", t, (unsigned int) thrarray_semwait[t]);
        //fflush(stdout);
        pthread_join(thrarray_semwait[t], NULL);
        //    printf("thread %d tid %u joined\n", t, (unsigned int) thrarray_semwait[t]);
    }

    free(thrarray_semwait);
    // printf("======== EXIT COREMOD_MEMORY_image_set_semwait_OR_IDarray =======\n");
    //fflush(stdout);

    return RETURN_SUCCESS;
}







/// \brief flush multiple semaphores
errno_t COREMOD_MEMORY_image_set_semflush_IDarray(
    imageID *IDarray,
    long     NB_ID
)
{
    long i, cnt;
    int semval;
    int s;

    list_image_ID();
    for(i = 0; i < NB_ID; i++)
    {
        for(s = 0; s < data.image[IDarray[i]].md[0].sem; s++)
        {
            sem_getvalue(data.image[IDarray[i]].semptr[s], &semval);
            printf("sem %d/%d of %s [%ld] = %d\n", s, data.image[IDarray[i]].md[0].sem,
                   data.image[IDarray[i]].name, IDarray[i], semval);
            fflush(stdout);
            for(cnt = 0; cnt < semval; cnt++)
            {
                sem_trywait(data.image[IDarray[i]].semptr[s]);
            }
        }
    }

    return RETURN_SUCCESS;
}



/// set semaphore value to 0
// if index <0, flush all image semaphores
imageID COREMOD_MEMORY_image_set_semflush(
    const char *IDname,
    long        index
)
{
    imageID ID;

    ID = image_ID(IDname);
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname);
    }

    ImageStreamIO_semflush(&data.image[ID], index);


    return ID;
}



