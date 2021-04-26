/**
 * @file    delete_image.c
 * @brief   delete image(s)
 */

#include <malloc.h>
#include <sys/mman.h>


#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"




// ==========================================
// Forward declaration(s)
// ==========================================

errno_t delete_image_ID(
    const char *__restrict__ imname
);

errno_t destroy_shared_image_ID(
    const char *__restrict__ imname
);


// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t delete_image_ID__cli()
{
    long i = 1;
    printf("%ld : %d\n", i, data.cmdargtoken[i].type);
    while(data.cmdargtoken[i].type != 0)
    {
        if(data.cmdargtoken[i].type == 4)
        {
            delete_image_ID(data.cmdargtoken[i].val.string);
        }
        else
        {
            printf("Image %s does not exist\n", data.cmdargtoken[i].val.string);
        }
        i++;
    }

    return CLICMD_SUCCESS;
}


static errno_t destroy_shared_image_ID__cli()
{
    long i = 1;
    printf("%ld : %d\n", i, data.cmdargtoken[i].type);
    while(data.cmdargtoken[i].type != 0)
    {
        if(data.cmdargtoken[i].type == 4)
        {
            destroy_shared_image_ID(data.cmdargtoken[i].val.string);
        }
        else
        {
            printf("Image %s does not exist\n", data.cmdargtoken[i].val.string);
        }
        i++;
    }

    return CLICMD_SUCCESS;
}






// ==========================================
// Register CLI command(s)
// ==========================================

errno_t delete_image_addCLIcmd()
{
    RegisterCLIcommand(
        "rm",
        __FILE__,
        delete_image_ID__cli,
        "remove image(s)",
        "list of images",
        "rm im1 im4",
        "int delete_image_ID(char* imname)"
    );

    RegisterCLIcommand(
        "rmshmim",
        __FILE__,
        destroy_shared_image_ID__cli,
        "remove image(s) and files",
        "image name",
        "rmshmim im1",
        "int destroy_image_ID(char* imname)"
    );


    return RETURN_SUCCESS;
}








/* deletes an ID */
errno_t delete_image_ID(
    const char *__restrict__ imname
)
{
    imageID ID;
    long    s;
    char    fname[STRINGMAXLEN_FULLFILENAME];

    ID = image_ID(imname);

    if(ID != -1)
    {
        data.image[ID].used = 0;

        if(data.image[ID].md[0].shared == 1)
        {
            for(s = 0; s < data.image[ID].md[0].sem; s++)
            {
                sem_close(data.image[ID].semptr[s]);
            }

            free(data.image[ID].semptr);
            data.image[ID].semptr = NULL;


            if(data.image[ID].semlog != NULL)
            {
                sem_close(data.image[ID].semlog);
                data.image[ID].semlog = NULL;
            }

            if(munmap(data.image[ID].md, data.image[ID].memsize) == -1)
            {
                printf("unmapping ID %ld : %p  %ld\n", ID, data.image[ID].md,
                       data.image[ID].memsize);
                perror("Error un-mmapping the file");
            }

            close(data.image[ID].shmfd);
            data.image[ID].shmfd = -1;

            data.image[ID].md = NULL;
            data.image[ID].kw = NULL;

            data.image[ID].memsize = 0;

            if(data.rmSHMfile == 1)    // remove files from disk
            {
                EXECUTE_SYSTEM_COMMAND("rm /dev/shm/sem.%s.%s_sem*",
                                       data.shmsemdirname, imname);
                WRITE_FULLFILENAME(fname, "/dev/shm/sem.%s.%s_semlog", data.shmsemdirname, imname);
                remove(fname);

                EXECUTE_SYSTEM_COMMAND("rm %s/%s.im.shm", data.shmdir, imname);
            }

        }
        else
        {
            if(data.image[ID].md[0].datatype == _DATATYPE_UINT8)
            {
                if(data.image[ID].array.UI8 == NULL)
                {
                    PRINT_ERROR("data array pointer is null\n");
                    exit(EXIT_FAILURE);
                }
                free(data.image[ID].array.UI8);
                data.image[ID].array.UI8 = NULL;
            }
            if(data.image[ID].md[0].datatype == _DATATYPE_INT32)
            {
                if(data.image[ID].array.SI32 == NULL)
                {
                    PRINT_ERROR("data array pointer is null\n");
                    exit(EXIT_FAILURE);
                }
                free(data.image[ID].array.SI32);
                data.image[ID].array.SI32 = NULL;
            }
            if(data.image[ID].md[0].datatype == _DATATYPE_FLOAT)
            {
                if(data.image[ID].array.F == NULL)
                {
                    PRINT_ERROR("data array pointer is null\n");
                    exit(EXIT_FAILURE);
                }
                free(data.image[ID].array.F);
                data.image[ID].array.F = NULL;
            }
            if(data.image[ID].md[0].datatype == _DATATYPE_DOUBLE)
            {
                if(data.image[ID].array.D == NULL)
                {
                    PRINT_ERROR("data array pointer is null\n");
                    exit(EXIT_FAILURE);
                }
                free(data.image[ID].array.D);
                data.image[ID].array.D = NULL;
            }
            if(data.image[ID].md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                if(data.image[ID].array.CF == NULL)
                {
                    PRINT_ERROR("data array pointer is null\n");
                    exit(EXIT_FAILURE);
                }
                free(data.image[ID].array.CF);
                data.image[ID].array.CF = NULL;
            }
            if(data.image[ID].md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
            {
                if(data.image[ID].array.CD == NULL)
                {
                    PRINT_ERROR("data array pointer is null\n");
                    exit(EXIT_FAILURE);
                }
                free(data.image[ID].array.CD);
                data.image[ID].array.CD = NULL;
            }

            if(data.image[ID].md == NULL)
            {
                PRINT_ERROR("data array pointer is null\n");
                exit(0);
            }
            free(data.image[ID].md);
            data.image[ID].md = NULL;


            if(data.image[ID].kw != NULL)
            {
                free(data.image[ID].kw);
                data.image[ID].kw = NULL;
            }

        }
        //free(data.image[ID].logstatus);
        /*      free(data.image[ID].size);*/
        //      data.image[ID].md[0].last_access = 0;
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm WARNING: image %s does not exist [ %s  %s  %d ] %c[%d;m\n",
                (char) 27, 1, 31, imname, __FILE__, __func__, __LINE__, (char) 27, 0);
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    return RETURN_SUCCESS;
}



// delete all images with a prefix
errno_t delete_image_ID_prefix(
    const char *prefix
)
{
    imageID i;

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        if(data.image[i].used == 1)
            if((strncmp(prefix, data.image[i].name, strlen(prefix))) == 0)
            {
                printf("deleting image %s\n", data.image[i].name);
                delete_image_ID(data.image[i].name);
            }
    }
    return RETURN_SUCCESS;
}




errno_t destroy_shared_image_ID(
    const char *__restrict__ imname
)
{
    imageID ID;

    ID = image_ID(imname);
    if((ID != -1) && (data.image[ID].md[0].shared == 1))
    {
        ImageStreamIO_destroyIm(&data.image[ID]);
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm WARNING: shared image %s does not exist [ %s  %s  %d ] %c[%d;m\n",
                (char) 27, 1, 31, imname, __FILE__, __func__, __LINE__, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}
