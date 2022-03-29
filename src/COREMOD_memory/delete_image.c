/**
 * @file    delete_image.c
 * @brief   delete image(s)
 */

#include <malloc.h>
#include <sys/mman.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"

// Forward declaration(s)
errno_t delete_image_ID(const char *__restrict imname, int errmode);

// CLI function arguments and parameters
static char *imname;
static long *errmode;




// CLI function arguments and parameters
static CLICMDARGDEF farg[] = {
    {CLIARG_IMG,
     ".imname",
     "image name",
     "im",
     CLIARG_VISIBLE_DEFAULT,
     (void **) &imname,
     NULL},
    {CLIARG_LONG,
     ".errmode",
     "errors mode \n(0:ignore) (1:warning) (2:error) (3:exit)",
     "1",
     CLIARG_HIDDEN_DEFAULT,
     (void **) &errmode,
     NULL}};

// CLI function initialization data
static CLICMDDATA CLIcmddata = {"rm", "remove image", CLICMD_FIELDS_DEFAULTS};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    FUNC_CHECK_RETURN(delete_image_ID(imname, (int) *errmode));

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

    // Register function in CLI
    errno_t
    CLIADDCMD_COREMOD_memory__delete_image()
{
    //INSERT_STD_FPSCLIREGISTERFUNC

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}

/** @brief deletes an ID
 *
 * errmode values:
 * DELETE_IMAGE_ERRMODE_IGNORE
 * DELETE_IMAGE_ERRMODE_WARNING
 * DELETE_IMAGE_ERRMODE_ERROR
 * DELETE_IMAGE_ERRMODE_EXIT
 *
 */
errno_t delete_image_ID(const char *__restrict imname, int errmode)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    long    s;
    char    fname[STRINGMAXLEN_FULLFILENAME];

    ID = image_ID(imname);

    if (ID == -1)
    {
        if (errmode == DELETE_IMAGE_ERRMODE_IGNORE)
        {
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }

        if (errmode == DELETE_IMAGE_ERRMODE_WARNING)
        {
            PRINT_WARNING("Image \"%s\" does not exist", imname);
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }

        if (errmode == DELETE_IMAGE_ERRMODE_ERROR)
        {
            PRINT_WARNING("Image \"%s\" does not exist", imname);
            FUNC_RETURN_FAILURE("Image \"%s\" does not exist", imname);
        }

        if (errmode == DELETE_IMAGE_ERRMODE_EXIT)
        {
            abort();
        }
        return -1;
    }
    else
    {
        data.image[ID].used = 0;

        if (data.image[ID].md[0].shared == 1)
        {
            for (s = 0; s < data.image[ID].md[0].sem; s++)
            {
                sem_close(data.image[ID].semptr[s]);
            }

            free(data.image[ID].semptr);
            data.image[ID].semptr = NULL;

            if (data.image[ID].semlog != NULL)
            {
                sem_close(data.image[ID].semlog);
                data.image[ID].semlog = NULL;
            }

            if (munmap(data.image[ID].md, data.image[ID].memsize) == -1)
            {
                printf("unmapping ID %ld : %p  %ld\n",
                       ID,
                       data.image[ID].md,
                       data.image[ID].memsize);
                perror("Error un-mmapping the file");
            }

            close(data.image[ID].shmfd);
            data.image[ID].shmfd = -1;

            data.image[ID].md = NULL;
            data.image[ID].kw = NULL;

            data.image[ID].memsize = 0;

            if (data.rmSHMfile == 1) // remove files from disk
            {
                EXECUTE_SYSTEM_COMMAND("rm /dev/shm/sem.%s.%s_sem*",
                                       data.shmsemdirname,
                                       imname);
                WRITE_FULLFILENAME(fname,
                                   "/dev/shm/sem.%s.%s_semlog",
                                   data.shmsemdirname,
                                   imname);
                remove(fname);

                EXECUTE_SYSTEM_COMMAND("rm %s/%s.im.shm", data.shmdir, imname);
            }
        }
        else
        {
            if (data.image[ID].md[0].datatype == _DATATYPE_UINT8)
            {
                if (data.image[ID].array.UI8 == NULL)
                {
                    FUNC_RETURN_FAILURE("data array pointer is null");
                }
                free(data.image[ID].array.UI8);
                data.image[ID].array.UI8 = NULL;
            }
            if (data.image[ID].md[0].datatype == _DATATYPE_INT32)
            {
                if (data.image[ID].array.SI32 == NULL)
                {
                    FUNC_RETURN_FAILURE("data array pointer is null");
                }
                free(data.image[ID].array.SI32);
                data.image[ID].array.SI32 = NULL;
            }
            if (data.image[ID].md[0].datatype == _DATATYPE_FLOAT)
            {
                if (data.image[ID].array.F == NULL)
                {
                    FUNC_RETURN_FAILURE("data array pointer is null");
                }
                free(data.image[ID].array.F);
                data.image[ID].array.F = NULL;
            }
            if (data.image[ID].md[0].datatype == _DATATYPE_DOUBLE)
            {
                if (data.image[ID].array.D == NULL)
                {
                    FUNC_RETURN_FAILURE("data array pointer is null");
                }
                free(data.image[ID].array.D);
                data.image[ID].array.D = NULL;
            }
            if (data.image[ID].md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                if (data.image[ID].array.CF == NULL)
                {
                    FUNC_RETURN_FAILURE("data array pointer is null");
                }
                free(data.image[ID].array.CF);
                data.image[ID].array.CF = NULL;
            }
            if (data.image[ID].md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
            {
                if (data.image[ID].array.CD == NULL)
                {
                    FUNC_RETURN_FAILURE("data array pointer is null");
                }
                free(data.image[ID].array.CD);
                data.image[ID].array.CD = NULL;
            }

            if (data.image[ID].md == NULL)
            {
                FUNC_RETURN_FAILURE("data array pointer is null");
            }
            free(data.image[ID].md);
            data.image[ID].md = NULL;

            if (data.image[ID].kw != NULL)
            {
                free(data.image[ID].kw);
                data.image[ID].kw = NULL;
            }
        }
        //free(data.image[ID].logstatus);
        /*      free(data.image[ID].size);*/
        //      data.image[ID].md[0].last_access = 0;
    }

    if (data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// delete all images with a prefix
errno_t delete_image_ID_prefix(const char *prefix)
{
    imageID i;

    for (i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        if (data.image[i].used == 1)
            if ((strncmp(prefix, data.image[i].name, strlen(prefix))) == 0)
            {
                printf("deleting image %s\n", data.image[i].name);
                delete_image_ID(data.image[i].name,
                                DELETE_IMAGE_ERRMODE_IGNORE);
            }
    }
    return RETURN_SUCCESS;
}
