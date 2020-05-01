/**
 * @file    COREMOD_memory.c
 * @brief   milk memory functions
 *
 * Functions to handle images and streams
 *
 */



#define _GNU_SOURCE



/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT ""

// Module short description
#define MODULE_DESCRIPTION       "Memory management for images and variables"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */


#include <stdint.h>
#include <unistd.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <signal.h>
#include <ncurses.h>

#include <errno.h>
#include <signal.h>

#include <semaphore.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h> // for open
#include <unistd.h> // for close

#include <time.h>
#include <sys/time.h>



#include <fitsio.h>


#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_tools/COREMOD_tools.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"



#include "image_ID.h"
#include "image_keyword.h"
#include "variable_ID.h"
#include "list_image.h"

#include "create_image.h"
#include "delete_image.h"

#include "image_copy.h"
#include "image_complex.h"

#include "read_shmim.h"
#include "stream_TCP.h"
#include "logshmim.h"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */




# ifdef _OPENMP
# include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
# endif


#define SBUFFERSIZE 1000








/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_memory)




/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 1. MANAGE MEMORY AND IDENTIFIERS                                                                */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */


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





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 2. KEYWORDS                                                                                     */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */


static errno_t image_write_keyword_L__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_STR_NOT_IMG)
            == 0)
    {
        image_write_keyword_L(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




static errno_t image_list_keywords__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            == 0)
    {
        image_list_keywords(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}






/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 3. READ SHARED MEM IMAGE AND SIZE                                                               */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



static errno_t read_sharedmem_image_size__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            == 0)
    {

        read_sharedmem_image_size(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




static errno_t read_sharedmem_image__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            == 0)
    {

        read_sharedmem_image(
            data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 4. CREATE IMAGE                                                                                 */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



static errno_t create_image__cli()
{
    uint32_t *imsize;
    long naxis = 0;
    long i;
    uint8_t datatype;



    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg_noerrmsg(2, CLIARG_LONG)
            == 0)
    {
        naxis = 0;
        imsize = (uint32_t *) malloc(sizeof(uint32_t) * 5);
        i = 2;
        while(data.cmdargtoken[i].type == 2)
        {
            imsize[naxis] = data.cmdargtoken[i].val.numl;
            naxis++;
            i++;
        }
        switch(data.precision)
        {
            case 0:
                create_image_ID(data.cmdargtoken[1].val.string, naxis, imsize, _DATATYPE_FLOAT,
                                data.SHARED_DFT, data.NBKEWORD_DFT);
                break;
            case 1:
                create_image_ID(data.cmdargtoken[1].val.string, naxis, imsize, _DATATYPE_DOUBLE,
                                data.SHARED_DFT, data.NBKEWORD_DFT);
                break;
        }
        free(imsize);
        return CLICMD_SUCCESS;
    }
    else if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)  // type option exists
    {
        datatype = 0;

        if(strcmp(data.cmdargtoken[2].val.string, "c") == 0)
        {
            printf("type = CHAR\n");
            datatype = _DATATYPE_UINT8;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "i") == 0)
        {
            printf("type = INT\n");
            datatype = _DATATYPE_INT32;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "f") == 0)
        {
            printf("type = FLOAT\n");
            datatype = _DATATYPE_FLOAT;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "d") == 0)
        {
            printf("type = DOUBLE\n");
            datatype = _DATATYPE_DOUBLE;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "cf") == 0)
        {
            printf("type = COMPLEX_FLOAT\n");
            datatype = _DATATYPE_COMPLEX_FLOAT;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "cd") == 0)
        {
            printf("type = COMPLEX_DOUBLE\n");
            datatype = _DATATYPE_COMPLEX_DOUBLE;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "u") == 0)
        {
            printf("type = USHORT\n");
            datatype = _DATATYPE_UINT16;
        }

        if(strcmp(data.cmdargtoken[2].val.string, "l") == 0)
        {
            printf("type = LONG\n");
            datatype = _DATATYPE_INT64;
        }

        if(datatype == 0)
        {
            printf("Data type \"%s\" not recognized\n", data.cmdargtoken[2].val.string);
            printf("must be : \n");
            printf("  c : CHAR\n");
            printf("  i : INT32\n");
            printf("  f : FLOAT\n");
            printf("  d : DOUBLE\n");
            printf("  cf: COMPLEX FLOAT\n");
            printf("  cd: COMPLEX DOUBLE\n");
            printf("  u : USHORT16\n");
            printf("  l : LONG64\n");
            return CLICMD_INVALID_ARG;
        }
        naxis = 0;
        imsize = (uint32_t *) malloc(sizeof(uint32_t) * 5);
        i = 3;
        while(data.cmdargtoken[i].type == 2)
        {
            imsize[naxis] = data.cmdargtoken[i].val.numl;
            naxis++;
            i++;
        }

        create_image_ID(data.cmdargtoken[1].val.string, naxis, imsize, datatype,
                        data.SHARED_DFT, data.NBKEWORD_DFT);

        free(imsize);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}





static errno_t create_image_shared__cli() // default precision
{
    uint32_t *imsize;
    long naxis = 0;
    long i;


    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        naxis = 0;
        imsize = (uint32_t *) malloc(sizeof(uint32_t) * 5);
        i = 2;
        while(data.cmdargtoken[i].type == 2)
        {
            imsize[naxis] = data.cmdargtoken[i].val.numl;
            naxis++;
            i++;
        }
        switch(data.precision)
        {
            case 0:
                create_image_ID(data.cmdargtoken[1].val.string, naxis, imsize, _DATATYPE_FLOAT,
                                1, data.NBKEWORD_DFT);
                break;
            case 1:
                create_image_ID(data.cmdargtoken[1].val.string, naxis, imsize, _DATATYPE_DOUBLE,
                                1, data.NBKEWORD_DFT);
                break;
        }
        free(imsize);
        printf("Creating 10 semaphores\n");
        COREMOD_MEMORY_image_set_createsem(data.cmdargtoken[1].val.string,
                                           IMAGE_NB_SEMAPHORE);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




static errno_t create_ushort_image_shared__cli() // default precision
{
    uint32_t *imsize;
    long naxis = 0;
    long i;


    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        naxis = 0;
        imsize = (uint32_t *) malloc(sizeof(uint32_t) * 5);
        i = 2;
        while(data.cmdargtoken[i].type == 2)
        {
            imsize[naxis] = data.cmdargtoken[i].val.numl;
            naxis++;
            i++;
        }
        create_image_ID(data.cmdargtoken[1].val.string, naxis, imsize, _DATATYPE_UINT16,
                        1, data.NBKEWORD_DFT);

        free(imsize);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




static errno_t create_2Dimage_float()
{
    uint32_t *imsize;

    // CHECK ARGS
    //  printf("CREATING IMAGE\n");
    imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);

    imsize[0] = data.cmdargtoken[2].val.numl;
    imsize[1] = data.cmdargtoken[3].val.numl;

    create_image_ID(data.cmdargtoken[1].val.string, 2, imsize, _DATATYPE_FLOAT,
                    data.SHARED_DFT, data.NBKEWORD_DFT);

    free(imsize);

    return RETURN_SUCCESS;
}



static errno_t create_3Dimage_float()
{
    uint32_t *imsize;

    // CHECK ARGS
    //  printf("CREATING 3D IMAGE\n");
    imsize = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    imsize[0] = data.cmdargtoken[2].val.numl;
    imsize[1] = data.cmdargtoken[3].val.numl;
    imsize[2] = data.cmdargtoken[4].val.numl;

    create_image_ID(data.cmdargtoken[1].val.string, 3, imsize, _DATATYPE_FLOAT,
                    data.SHARED_DFT, data.NBKEWORD_DFT);

    free(imsize);

    return RETURN_SUCCESS;
}












/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 5. CREATE VARIABLE                                                                              */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 6. COPY IMAGE                                                                                   */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



static errno_t copy_image_ID__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    copy_image_ID(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string,
                  0);

    return CLICMD_SUCCESS;
}




static errno_t copy_image_ID_sharedmem__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    copy_image_ID(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string,
                  1);

    return CLICMD_SUCCESS;
}


static errno_t chname_image_ID__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    chname_image_ID(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string);

    return CLICMD_SUCCESS;
}



static errno_t COREMOD_MEMORY_cp2shm__cli()
{
    if(CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            == 0)
    {
        COREMOD_MEMORY_cp2shm(data.cmdargtoken[1].val.string,
                              data.cmdargtoken[2].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 7. DISPLAY / LISTS                                                                              */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



static errno_t memory_monitor__cli()
{
    memory_monitor(data.cmdargtoken[1].val.string);
    return CLICMD_SUCCESS;
}


static errno_t list_variable_ID_file__cli()
{
    if(CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
    {
        list_variable_ID_file(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 8. TYPE CONVERSIONS TO AND FROM COMPLEX                                                         */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */



static errno_t mk_complex_from_reim__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }
    if(data.cmdargtoken[2].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[2].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_complex_from_reim(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}



static errno_t mk_complex_from_amph__cli()
{
    if(data.cmdargtoken[1].type != 4)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }
    if(data.cmdargtoken[2].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[2].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_complex_from_amph(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}



static errno_t mk_reim_from_complex__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_reim_from_complex(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}



static errno_t mk_amph_from_complex__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_amph_from_complex(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}




/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 9. VERIFY SIZE                                                                                  */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 10. COORDINATE CHANGE                                                                           */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 11. SET IMAGE FLAGS / COUNTERS                                                                  */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */


static errno_t COREMOD_MEMORY_image_set_status__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_status(
            data.cmdargtoken[1].val.string,
            (int) data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_set_cnt0__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_cnt0(
            data.cmdargtoken[1].val.string,
            (int) data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_set_cnt1__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_cnt1(
            data.cmdargtoken[1].val.string,
            (int) data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 12. MANAGE SEMAPHORES                                                                           */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */




static errno_t COREMOD_MEMORY_image_set_createsem__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_createsem(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_seminfo__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG) == 0)
    {
        COREMOD_MEMORY_image_seminfo(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_set_sempost__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_sempost(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_set_sempost_loop__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_sempost_loop(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_image_set_semwait__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_semwait(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_set_semflush__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_set_semflush(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 13. SIMPLE OPERATIONS ON STREAMS                                                                */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */




static errno_t COREMOD_MEMORY_streamPoke__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_streamPoke(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_streamDiff__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, 5)
            + CLI_checkarg(4, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(5, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_streamDiff(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.string,
            data.cmdargtoken[4].val.string,
            data.cmdargtoken[5].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_streamPaste__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, 5)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            + CLI_checkarg(6, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_streamPaste(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.string,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl,
            data.cmdargtoken[6].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_stream_halfimDiff__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_stream_halfimDiff(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_streamAve__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, 5)
            == 0)
    {
        COREMOD_MEMORY_streamAve(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




static errno_t COREMOD_MEMORY_image_streamupdateloop__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, 5)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            + CLI_checkarg(6, CLIARG_LONG)
            + CLI_checkarg(7, 5)
            + CLI_checkarg(8, CLIARG_LONG)
            + CLI_checkarg(9, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_streamupdateloop(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl,
            data.cmdargtoken[6].val.numl,
            data.cmdargtoken[7].val.string,
            data.cmdargtoken[8].val.numl,
            data.cmdargtoken[9].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_image_streamupdateloop_semtrig__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, 5)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, 5)
            + CLI_checkarg(6, CLIARG_LONG)
            + CLI_checkarg(7, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_streamupdateloop_semtrig(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.string,
            data.cmdargtoken[6].val.numl,
            data.cmdargtoken[7].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


/*
int_fast8_t COREMOD_MEMORY_streamDelay__cli() {
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 5) + CLI_checkarg(3, 2) + CLI_checkarg(4, 2) == 0) {
        COREMOD_MEMORY_streamDelay(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string, data.cmdargtoken[3].val.numl, data.cmdargtoken[4].val.numl);
        return 0;
    } else {
        return 1;
    }
}
*/

static errno_t COREMOD_MEMORY_streamDelay__cli()
{
    char fpsname[200];

    // First, we try to execute function through FPS interface
    if(0
            + CLI_checkarg(1, 5)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)   // check that first arg is string, second arg is int
    {
        unsigned int OptionalArg00 = data.cmdargtoken[2].val.numl;

        // Set FPS interface name
        // By convention, if there are optional arguments, they should be appended to the fps name
        //
        if(data.processnameflag ==
                0)   // name fps to something different than the process name
        {
            sprintf(fpsname, "streamDelay-%06u", OptionalArg00);
        }
        else     // Automatically set fps name to be process name up to first instance of character '.'
        {
            strcpy(fpsname, data.processname0);
        }

        if(strcmp(data.cmdargtoken[1].val.string,
                  "_FPSINIT_") == 0)    // Initialize FPS and conf process
        {
            printf("Function parameters configure\n");
            COREMOD_MEMORY_streamDelay_FPCONF(fpsname, FPSCMDCODE_FPSINIT);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string,
                  "_CONFSTART_") == 0)    // Start conf process
        {
            printf("Function parameters configure\n");
            COREMOD_MEMORY_streamDelay_FPCONF(fpsname, FPSCMDCODE_CONFSTART);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string,
                  "_CONFSTOP_") == 0)   // Stop conf process
        {
            printf("Function parameters configure\n");
            COREMOD_MEMORY_streamDelay_FPCONF(fpsname, FPSCMDCODE_CONFSTOP);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTART_") == 0)   // Run process
        {
            printf("Run function\n");
            COREMOD_MEMORY_streamDelay_RUN(fpsname);
            return RETURN_SUCCESS;
        }
        /*
                if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTOP_") == 0) { // Cleanly stop process
                    printf("Run function\n");
                    COREMOD_MEMORY_streamDelay_STOP(OptionalArg00);
                    return RETURN_SUCCESS;
                }*/
    }

    // non FPS implementation - all parameters specified at function launch
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, 5)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_streamDelay(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }

}















static errno_t COREMOD_MEMORY_SaveAll_snapshot__cli()
{
    if(0
            + CLI_checkarg(1, 5) == 0)
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
    if(0
            + CLI_checkarg(1, 5)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_SaveAll_sequ(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_testfunction_semaphore__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_testfunction_semaphore(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_NETWORKtransmit__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_NETWORKtransmit(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_NETWORKreceive__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_LONG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_NETWORKreceive(
            data.cmdargtoken[1].val.numl,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_PixMapDecode_U__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(5, CLIARG_IMG)
            + CLI_checkarg(6, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(7, CLIARG_STR_NOT_IMG)
            == 0)
    {
        COREMOD_MEMORY_PixMapDecode_U(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.string,
            data.cmdargtoken[5].val.string,
            data.cmdargtoken[6].val.string,
            data.cmdargtoken[7].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 14. DATA LOGGING                                                                                */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */




static errno_t COREMOD_MEMORY_logshim_printstatus__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            == 0)
    {
        COREMOD_MEMORY_logshim_printstatus(
            data.cmdargtoken[1].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_logshim_set_on__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        printf("logshim_set_on ----------------------\n");
        COREMOD_MEMORY_logshim_set_on(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_logshim_set_logexit__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_logshim_set_logexit(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_sharedMem_2Dim_log__cli()
{

    if(CLI_checkarg_noerrmsg(4, CLIARG_STR_NOT_IMG) != 0)
    {
        sprintf(data.cmdargtoken[4].val.string, "null");
    }

    if(0
            + CLI_checkarg(1, 3)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, 3)
            == 0)
    {
        COREMOD_MEMORY_sharedMem_2Dim_log(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.string,
            data.cmdargtoken[4].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}













static errno_t init_module_CLI()
{
	
	data.MEM_MONITOR = 0; // 1 if memory monitor is on
	

    RegisterCLIcommand(
        "cmemtestf",
        __FILE__,
        COREMOD_MEMORY_testfunc,
        "testfunc",
        "no arg",
        "cmemtestf",
        "COREMOD_MEMORY_testfunc()");

    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 1. MANAGE MEMORY AND IDENTIFIERS                                                                */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "mmon",
        __FILE__,
        memory_monitor__cli,
        "Monitor memory content",
        "terminal tty name",
        "mmon /dev/pts/4",
        "int memory_monitor(const char *ttyname)");

    RegisterCLIcommand(
        "rm",
        __FILE__,
        delete_image_ID__cli,
        "remove image(s)",
        "list of images",
        "rm im1 im4",
        "int delete_image_ID(char* imname)");

    RegisterCLIcommand(
        "rmall",
        __FILE__,
        clearall,
        "remove all images",
        "no argument",
        "rmall",
        "int clearall()");



    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 2. KEYWORDS                                                                                     */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "imwritekwL",
        __FILE__,
        image_write_keyword_L__cli,
        "write long type keyword",
        "<imname> <kname> <value [long]> <comment>",
        "imwritekwL im1 kw2 34 my_keyword_comment",
        "long image_write_keyword_L(const char *IDname, const char *kname, long value, const char *comment)");

    RegisterCLIcommand(
        "imlistkw",
        __FILE__,
        image_list_keywords__cli,
        "list image keywords",
        "<imname>",
        "imlistkw im1",
        "long image_list_keywords(const char *IDname)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 3. READ SHARED MEM IMAGE AND SIZE                                                               */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "readshmimsize",
        __FILE__,
        read_sharedmem_image_size__cli,
        "read shared memory image size",
        "<name> <output file>",
        "readshmimsize im1 imsize.txt",
        "read_sharedmem_image_size(const char *name, const char *fname)");

    RegisterCLIcommand(
        "readshmim",
        __FILE__, read_sharedmem_image__cli,
        "read shared memory image",
        "<name>",
        "readshmim im1",
        "read_sharedmem_image(const char *name)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 4. CREATE IMAGE                                                                                 */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */




    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 5. CREATE VARIABLE                                                                              */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "creaim",
        __FILE__,
        create_image__cli,
        "create image, default precision",
        "<name> <xsize> <ysize> <opt: zsize>",
        "creaim imname 512 512",
        "long create_image_ID(const char *name, long naxis, uint32_t *size, uint8_t datatype, 0, 10)");

    RegisterCLIcommand(
        "creaimshm",
        __FILE__, create_image_shared__cli,
        "create image in shared mem, default precision",
        "<name> <xsize> <ysize> <opt: zsize>",
        "creaimshm imname 512 512",
        "long create_image_ID(const char *name, long naxis, uint32_t *size, uint8_t datatype, 0, 10)");

    RegisterCLIcommand(
        "creaushortimshm",
        __FILE__,
        create_ushort_image_shared__cli,
        "create unsigned short image in shared mem",
        "<name> <xsize> <ysize> <opt: zsize>",
        "creaushortimshm imname 512 512",
        "long create_image_ID(const char *name, long naxis, long *size, _DATATYPE_UINT16, 0, 10)");

    RegisterCLIcommand(
        "crea3dim",
        __FILE__,
        create_3Dimage_float,
        "creates 3D image, single precision",
        "<name> <xsize> <ysize> <zsize>",
        "crea3dim imname 512 512 100",
        "long create_image_ID(const char *name, long naxis, long *size, _DATATYPE_FLOAT, 0, 10)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 6. COPY IMAGE                                                                                   */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "cp",
        __FILE__, copy_image_ID__cli,
        "copy image",
        "source, dest",
        "cp im1 im4",
        "long copy_image_ID(const char *name, const char *newname, 0)");

    RegisterCLIcommand(
        "cpsh",
        __FILE__, copy_image_ID_sharedmem__cli,
        "copy image - create in shared mem if does not exist",
        "source, dest",
        "cp im1 im4",
        "long copy_image_ID(const char *name, const char *newname, 1)");

    RegisterCLIcommand(
        "mv",
        __FILE__, chname_image_ID__cli,
        "change image name",
        "source, dest",
        "mv im1 im4",
        "long chname_image_ID(const char *name, const char *newname)");

    RegisterCLIcommand(
        "imcp2shm",
        __FILE__,
        COREMOD_MEMORY_cp2shm__cli,
        "copy image ot shared memory",
        "<image> <shared mem image>",
        "imcp2shm im1 ims1",
        "long COREMOD_MEMORY_cp2shm(const char *IDname, const char *IDshmname)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 7. DISPLAY / LISTS                                                                              */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "listim",
        __FILE__,
        list_image_ID,
        "list images in memory",
        "no argument",
        "listim", "int_fast8_t list_image_ID()");

    RegisterCLIcommand(
        "listvar",
        __FILE__,
        list_variable_ID,
        "list variables in memory",
        "no argument",
        "listvar",
        "int list_variable_ID()");

    RegisterCLIcommand(
        "listvarf",
        __FILE__,
        list_variable_ID_file__cli,
        "list variables in memory, write to file",
        "<file name>",
        "listvarf var.txt",
        "int list_variable_ID_file()");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 8. TYPE CONVERSIONS TO AND FROM COMPLEX                                                         */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "ri2c",
        __FILE__,
        mk_complex_from_reim__cli,
        "real, imaginary -> complex",
        "real imaginary complex",
        "ri2c imr imi imc",
        "int mk_complex_from_reim(const char *re_name, const char *im_name, const char *out_name)");

    RegisterCLIcommand(
        "ap2c",
        __FILE__,
        mk_complex_from_amph__cli,
        "ampl, pha -> complex",
        "ampl pha complex",
        "ap2c ima imp imc",
        "int mk_complex_from_amph(const char *re_name, const char *im_name, const char *out_name, int sharedmem)");

    RegisterCLIcommand(
        "c2ri",
        __FILE__,
        mk_reim_from_complex__cli,
        "complex -> real, imaginary",
        "complex real imaginary",
        "c2ri imc imr imi",
        "int mk_reim_from_complex(const char *re_name, const char *im_name, const char *out_name)");

    RegisterCLIcommand(
        "c2ap",
        __FILE__,
        mk_amph_from_complex__cli,
        "complex -> ampl, pha",
        "complex ampl pha",
        "c2ap imc ima imp",
        "int mk_amph_from_complex(const char *re_name, const char *im_name, const char *out_name, int sharedmem)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 9. VERIFY SIZE                                                                                  */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */



    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 10. COORDINATE CHANGE                                                                           */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */



    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 11. SET IMAGE FLAGS / COUNTERS                                                                  */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "imsetstatus",
        __FILE__,
        COREMOD_MEMORY_image_set_status__cli,
        "set image status variable", "<image> <value [long]>",
        "imsetstatus im1 2",
        "long COREMOD_MEMORY_image_set_status(const char *IDname, int status)");

    RegisterCLIcommand(
        "imsetcnt0",
        __FILE__,
        COREMOD_MEMORY_image_set_cnt0__cli,
        "set image cnt0 variable", "<image> <value [long]>",
        "imsetcnt0 im1 2",
        "long COREMOD_MEMORY_image_set_cnt0(const char *IDname, int status)");

    RegisterCLIcommand(
        "imsetcnt1",
        __FILE__,
        COREMOD_MEMORY_image_set_cnt1__cli,
        "set image cnt1 variable", "<image> <value [long]>",
        "imsetcnt1 im1 2",
        "long COREMOD_MEMORY_image_set_cnt1(const char *IDname, int status)");



    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 12. MANAGE SEMAPHORES                                                                           */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "imsetcreatesem",
        __FILE__,
        COREMOD_MEMORY_image_set_createsem__cli,
        "create image semaphore",
        "<image> <NBsem>",
        "imsetcreatesem im1 5",
        "long COREMOD_MEMORY_image_set_createsem(const char *IDname, long NBsem)");

    RegisterCLIcommand(
        "imseminfo",
        __FILE__,
        COREMOD_MEMORY_image_seminfo__cli,
        "display semaphore info",
        "<image>",
        "imseminfo im1",
        "long COREMOD_MEMORY_image_seminfo(const char *IDname)");

    RegisterCLIcommand(
        "imsetsempost",
        __FILE__,
        COREMOD_MEMORY_image_set_sempost__cli,
        "post image semaphore. If sem index = -1, post all semaphores",
        "<image> <sem index>",
        "imsetsempost im1 2",
        "long COREMOD_MEMORY_image_set_sempost(const char *IDname, long index)");

    RegisterCLIcommand(
        "imsetsempostl",
        __FILE__,
        COREMOD_MEMORY_image_set_sempost_loop__cli,
        "post image semaphore loop. If sem index = -1, post all semaphores",
        "<image> <sem index> <time interval [us]>",
        "imsetsempostl im1 -1 1000",
        "long COREMOD_MEMORY_image_set_sempost_loop(const char *IDname, long index, long dtus)");

    RegisterCLIcommand(
        "imsetsemwait",
        __FILE__,
        COREMOD_MEMORY_image_set_semwait__cli,
        "wait image semaphore",
        "<image>",
        "imsetsemwait im1",
        "long COREMOD_MEMORY_image_set_semwait(const char *IDname)");

    RegisterCLIcommand(
        "imsetsemflush",
        __FILE__,
        COREMOD_MEMORY_image_set_semflush__cli,
        "flush image semaphore",
        "<image> <sem index>",
        "imsetsemflush im1 0",
        "long COREMOD_MEMORY_image_set_semflush(const char *IDname, long index)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 13. SIMPLE OPERATIONS ON STREAMS                                                                */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */

    RegisterCLIcommand(
        "creaimstream",
        __FILE__,
        COREMOD_MEMORY_image_streamupdateloop__cli,
        "create 2D image stream from 3D cube",
        "<image3d in> <image2d out> <interval [us]> <NBcubes> <period> <offsetus> <sync stream name> <semtrig> <timing mode>",
        "creaimstream imcube imstream 1000 3 3 154 ircam1 3 0",
        "long COREMOD_MEMORY_image_streamupdateloop(const char *IDinname, const char *IDoutname, long usperiod, long NBcubes, long period, long offsetus, const char *IDsync_name, int semtrig, int timingmode)");

    RegisterCLIcommand(
        "creaimstreamstrig",
        __FILE__,
        COREMOD_MEMORY_image_streamupdateloop_semtrig__cli,
        "create 2D image stream from 3D cube, use other stream to synchronize",
        "<image3d in> <image2d out> <period [int]> <delay [us]> <sync stream> <sync sem index> <timing mode>",
        "creaimstreamstrig imcube outstream 3 152 streamsync 3 0",
        "long COREMOD_MEMORY_image_streamupdateloop_semtrig(const char *IDinname, const char *IDoutname, long period, long offsetus, const char *IDsync_name, int semtrig, int timingmode)");

    RegisterCLIcommand(
        "streamdelay",
        __FILE__,
        COREMOD_MEMORY_streamDelay__cli,
        "delay 2D image stream",
        "<image2d in> <image2d out> <delay [us]> <resolution [us]>",
        "streamdelay instream outstream 1000 10",
        "long COREMOD_MEMORY_streamDelay(const char *IDin_name, const char *IDout_name, long delayus, long dtus)");

    RegisterCLIcommand(
        "imsaveallsnap",
        __FILE__,
        COREMOD_MEMORY_SaveAll_snapshot__cli, "save all images in directory",
        "<directory>", "imsaveallsnap dir1",
        "long COREMOD_MEMORY_SaveAll_snapshot(const char *dirname)");

    RegisterCLIcommand(
        "imsaveallseq",
        __FILE__,
        COREMOD_MEMORY_SaveAll_sequ__cli,
        "save all images in directory - sequence",
        "<directory> <trigger image name> <trigger semaphore> <NB frames>",
        "imsaveallsequ dir1 im1 3 20",
        "long COREMOD_MEMORY_SaveAll_sequ(const char *dirname, const char *IDtrig_name, long semtrig, long NBframes)");

    RegisterCLIcommand(
        "testfuncsem",
        __FILE__,
        COREMOD_MEMORY_testfunction_semaphore__cli,
        "test semaphore loop",
        "<image> <semindex> <testmode>",
        "testfuncsem im1 1 0",
        "int COREMOD_MEMORY_testfunction_semaphore(const char *IDname, int semtrig, int testmode)");

    RegisterCLIcommand(
        "imnetwtransmit",
        __FILE__,
        COREMOD_MEMORY_image_NETWORKtransmit__cli,
        "transmit image over network",
        "<image> <IP addr> <port [long]> <sync mode [int]>",
        "imnetwtransmit im1 127.0.0.1 0 8888 0",
        "long COREMOD_MEMORY_image_NETWORKtransmit(const char *IDname, const char *IPaddr, int port, int mode)");

    RegisterCLIcommand(
        "imnetwreceive",
        __FILE__,
        COREMOD_MEMORY_image_NETWORKreceive__cli,
        "receive image(s) over network. mode=1 uses counter instead of semaphore",
        "<port [long]> <mode [int]> <RT priority>",
        "imnetwreceive 8887 0 80",
        "long COREMOD_MEMORY_image_NETWORKreceive(int port, int mode, int RT_priority)");

    RegisterCLIcommand(
        "impixdecodeU",
        __FILE__,
        COREMOD_MEMORY_PixMapDecode_U__cli,
        "decode image stream",
        "<in stream> <xsize [long]> <ysize [long]> <nbpix per slice [ASCII file]> <decode map> <out stream> <out image slice index [FITS]>",
        "impixdecodeU streamin 120 120 pixsclienb.txt decmap outim outsliceindex.fits",
        "COREMOD_MEMORY_PixMapDecode_U(const char *inputstream_name, uint32_t xsizeim, uint32_t ysizeim, const char* NBpix_fname, const char* IDmap_name, const char *IDout_name, const char *IDout_pixslice_fname)");

    RegisterCLIcommand(
        "streampoke",
        __FILE__,
        COREMOD_MEMORY_streamPoke__cli,
        "Poke image stream at regular interval",
        "<in stream> <poke period [us]>",
        "streampoke stream 100",
        "long COREMOD_MEMORY_streamPoke(const char *IDstream_name, long usperiod)");

    RegisterCLIcommand(
        "streamdiff",
        __FILE__,
        COREMOD_MEMORY_streamDiff__cli,
        "compute difference between two image streams",
        "<in stream 0> <in stream 1> <out stream> <optional mask> <sem trigger index>",
        "streamdiff stream0 stream1 null outstream 3",
        "long COREMOD_MEMORY_streamDiff(const char *IDstream0_name, const char *IDstream1_name, const char *IDstreamout_name, long semtrig)");

    RegisterCLIcommand(
        "streampaste",
        __FILE__,
        COREMOD_MEMORY_streamPaste__cli,
        "paste two 2D image streams of same size",
        "<in stream 0> <in stream 1> <out stream> <sem trigger0> <sem trigger1> <master>",
        "streampaste stream0 stream1 outstream 3 3 0",
        "long COREMOD_MEMORY_streamPaste(const char *IDstream0_name, const char *IDstream1_name, const char *IDstreamout_name, long semtrig0, long semtrig1, int master)");

    RegisterCLIcommand(
        "streamhalfdiff",
        __FILE__,
        COREMOD_MEMORY_stream_halfimDiff__cli,
        "compute difference between two halves of an image stream",
        "<in stream> <out stream> <sem trigger index>",
        "streamhalfdiff stream outstream 3",
        "long COREMOD_MEMORY_stream_halfimDiff(const char *IDstream_name, const char *IDstreamout_name, long semtrig)");

    RegisterCLIcommand(
        "streamave",
        __FILE__,
        COREMOD_MEMORY_streamAve__cli,
        "averages stream",
        "<instream> <NBave> <mode, 1 for single local instance, 0 for loop> <outstream>",
        "streamave instream 100 0 outstream",
        "long COREMODE_MEMORY_streamAve(const char *IDstream_name, int NBave, int mode, const char *IDout_name)");


    /* =============================================================================================== */
    /* =============================================================================================== */
    /*                                                                                                 */
    /* 14. DATA LOGGING                                                                                */
    /*                                                                                                 */
    /* =============================================================================================== */
    /* =============================================================================================== */


    RegisterCLIcommand(
        "shmimstreamlog",
        __FILE__,
        COREMOD_MEMORY_sharedMem_2Dim_log__cli,
        "logs shared memory stream (run in current directory)",
        "<shm image> <cubesize [long]> <logdir>",
        "shmimstreamlog wfscamim 10000 /media/data",
        "long COREMOD_MEMORY_sharedMem_2Dim_log(const char *IDname, uint32_t zsize, const char *logdir, const char *IDlogdata_name)");

    RegisterCLIcommand(
        "shmimslogstat",
        __FILE__,
        COREMOD_MEMORY_logshim_printstatus__cli,
        "print log shared memory stream status",
        "<shm image>", "shmimslogstat wfscamim",
        "int COREMOD_MEMORY_logshim_printstatus(const char *IDname)");

    RegisterCLIcommand(
        "shmimslogonset", __FILE__,
        COREMOD_MEMORY_logshim_set_on__cli,
        "set on variable in log shared memory stream",
        "<shm image> <setv [long]>",
        "shmimslogonset imwfs 1",
        "int COREMOD_MEMORY_logshim_set_on(const char *IDname, int setv)");

    RegisterCLIcommand(
        "shmimslogexitset",
        __FILE__,
        COREMOD_MEMORY_logshim_set_logexit__cli,
        "set exit variable in log shared memory stream",
        "<shm image> <setv [long]>",
        "shmimslogexitset imwfs 1",
        "int COREMOD_MEMORY_logshim_set_logexit(const char *IDname, int setv)");



    // add atexit functions here

    return RETURN_SUCCESS;
}







/**
 *
 * Test function aimed at creating unsolved seg fault bug
 * Will crash under gcc-7 if -O3 or -Ofast gcc compilation flag
 *
 * UPDATE: has been resolved (2019) - kept it for reference
 */
errno_t COREMOD_MEMORY_testfunc()
{
//	imageID   ID;
//	imageID   IDimc;
    uint32_t  xsize;
    uint32_t  ysize;
    uint32_t  xysize;
    uint32_t  ii;
    uint32_t *imsize;
    IMAGE     testimage_in;
    IMAGE     testimage_out;



    printf("Entering test function\n");
    fflush(stdout);

    xsize = 50;
    ysize = 50;
    imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    imsize[0] = xsize;
    imsize[1] = ysize;
    xysize = xsize * ysize;

    // create image (shared memory)
    ImageStreamIO_createIm(&testimage_in, "testimshm", 2, imsize, _DATATYPE_UINT32,
                           1, 0);

    // create image (local memory only)
    ImageStreamIO_createIm(&testimage_out, "testimout", 2, imsize, _DATATYPE_UINT32,
                           0, 0);


    // crashes with seg fault with gcc-7.3.0 -O3
    // tested with float (.F) -> crashes
    // tested with uint32 (.UI32) -> crashes
    // no crash with gcc-6.3.0 -O3
    // crashes with gcc-7.2.0 -O3
    for(ii = 0; ii < xysize; ii++)
    {
        testimage_out.array.UI32[ii] = testimage_in.array.UI32[ii];
    }

    // no seg fault with gcc-7.3.0 -O3
    //memcpy(testimage_out.array.UI32, testimage_in.array.UI32, SIZEOF_DATATYPE_UINT32*xysize);


    free(imsize);

    printf("No bug... clean exit\n");
    fflush(stdout);

    return RETURN_SUCCESS;
}












/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 1. MANAGE MEMORY AND IDENTIFIERS                                                                */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */


errno_t memory_monitor(
    const char *termttyname
)
{
    if(data.Debug > 0)
    {
        printf("starting memory_monitor on \"%s\"\n", termttyname);
    }

    data.MEM_MONITOR = 1;
    init_list_image_ID_ncurses(termttyname);
    list_image_ID_ncurses();
    atexit(close_list_image_ID_ncurses);

    return RETURN_SUCCESS;
}






long compute_nb_variable()
{
    long NBvar = 0;

    for(variableID i = 0; i < data.NB_MAX_VARIABLE; i++)
    {
        if(data.variable[i].used == 1)
        {
            NBvar += 1;
        }
    }

    return NBvar;
}





long compute_variable_memory()
{
    long totalvmem = 0;

    for(variableID i = 0; i < data.NB_MAX_VARIABLE; i++)
    {
        totalvmem += sizeof(VARIABLE);
        if(data.variable[i].used == 1)
        {
            totalvmem += 0;
        }
    }
    return totalvmem;
}





/* deletes a variable ID */
errno_t delete_variable_ID(
    const char *varname
)
{
    imageID ID;

    ID = variable_ID(varname);
    if(ID != -1)
    {
        data.variable[ID].used = 0;
        /*      free(data.variable[ID].name);*/
    }
    else
        fprintf(stderr,
                "%c[%d;%dm WARNING: variable %s does not exist [ %s  %s  %d ] %c[%d;m\n",
                (char) 27, 1, 31, varname, __FILE__, __func__, __LINE__, (char) 27, 0);

    return RETURN_SUCCESS;
}



errno_t clearall()
{
    imageID ID;

    for(ID = 0; ID < data.NB_MAX_IMAGE; ID++)
    {
        if(data.image[ID].used == 1)
        {
            delete_image_ID(data.image[ID].name);
        }
    }
    for(ID = 0; ID < data.NB_MAX_VARIABLE; ID++)
    {
        if(data.variable[ID].used == 1)
        {
            delete_variable_ID(data.variable[ID].name);
        }
    }

    return RETURN_SUCCESS;
}































/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 5. CREATE VARIABLE                                                                              */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */





/* creates floating point variable */
variableID create_variable_ID(
    const char *name,
    double value
)
{
    variableID ID;
    long i1, i2;

    //printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);

    ID = -1;
    //printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);

    i1 = image_ID(name);
    //printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);


    i2 = variable_ID(name);
    //    printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);

    if(i1 != -1)
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an image\n",
               name);
    }
    else
    {
        if(i2 != -1)
        {
            //	  printf("Warning : variable name \"%s\" is already in use\n",name);
            ID = i2;
        }
        else
        {
            ID = next_avail_variable_ID();
        }

        data.variable[ID].used = 1;
        data.variable[ID].type = 0; /** floating point double */
        strcpy(data.variable[ID].name, name);
        data.variable[ID].value.f = value;

    }
    //    printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);
    return ID;
}



/* creates long variable */
variableID create_variable_long_ID(
    const char *name,
    long value
)
{
    variableID ID;
    long i1, i2;

    ID = -1;
    i1 = image_ID(name);
    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an image\n",
               name);
    }
    else
    {
        if(i2 != -1)
        {
            //	  printf("Warning : variable name \"%s\" is already in use\n",name);
            ID = i2;
        }
        else
        {
            ID = next_avail_variable_ID();
        }

        data.variable[ID].used = 1;
        data.variable[ID].type = 1; /** long */
        strcpy(data.variable[ID].name, name);
        data.variable[ID].value.l = value;

    }

    return ID;
}



/* creates long variable */
variableID create_variable_string_ID(
    const char *name,
    const char *value
)
{
    variableID ID;
    long i1, i2;

    ID = -1;
    i1 = image_ID(name);
    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an image\n",
               name);
    }
    else
    {
        if(i2 != -1)
        {
            //	  printf("Warning : variable name \"%s\" is already in use\n",name);
            ID = i2;
        }
        else
        {
            ID = next_avail_variable_ID();
        }

        data.variable[ID].used = 1;
        data.variable[ID].type = 2; /** string */
        strcpy(data.variable[ID].name, name);
        strcpy(data.variable[ID].value.s, value);
    }

    return ID;
}

























/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 9. VERIFY SIZE                                                                                  */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */






//  check only is size > 0
int check_2Dsize(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize
)
{
    int      retval;
    imageID  ID;

    retval = 1;
    ID = image_ID(ID_name);
    if(data.image[ID].md[0].naxis != 2)
    {
        retval = 0;
    }
    if(retval == 1)
    {
        if((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
        {
            retval = 0;
        }
        if((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
        {
            retval = 0;
        }
    }

    return retval;
}



int check_3Dsize(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize
)
{
    int     retval;
    imageID ID;

    retval = 1;
    ID = image_ID(ID_name);
    if(data.image[ID].md[0].naxis != 3)
    {
        /*      printf("Wrong naxis : %ld - should be 3\n",data.image[ID].md[0].naxis);*/
        retval = 0;
    }
    if(retval == 1)
    {
        if((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
        {
            /*	  printf("Wrong xsize : %ld - should be %ld\n",data.image[ID].md[0].size[0],xsize);*/
            retval = 0;
        }
        if((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
        {
            /*	  printf("Wrong ysize : %ld - should be %ld\n",data.image[ID].md[0].size[1],ysize);*/
            retval = 0;
        }
        if((zsize > 0) && (data.image[ID].md[0].size[2] != zsize))
        {
            /*	  printf("Wrong zsize : %ld - should be %ld\n",data.image[ID].md[0].size[2],zsize);*/
            retval = 0;
        }
    }
    /*  printf("CHECK = %d\n",value);*/

    return retval;
}







int COREMOD_MEMORY_check_2Dsize(
    const char *IDname,
    uint32_t    xsize,
    uint32_t    ysize
)
{
    int     sizeOK = 1; // 1 if size matches
    imageID ID;


    ID = image_ID(IDname);
    if(data.image[ID].md[0].naxis != 2)
    {
        printf("WARNING : image %s naxis = %d does not match expected value 2\n",
               IDname, (int) data.image[ID].md[0].naxis);
        sizeOK = 0;
    }
    if((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
    {
        printf("WARNING : image %s xsize = %d does not match expected value %d\n",
               IDname, (int) data.image[ID].md[0].size[0], (int) xsize);
        sizeOK = 0;
    }
    if((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
    {
        printf("WARNING : image %s ysize = %d does not match expected value %d\n",
               IDname, (int) data.image[ID].md[0].size[1], (int) ysize);
        sizeOK = 0;
    }

    return sizeOK;
}



int COREMOD_MEMORY_check_3Dsize(
    const char *IDname,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize
)
{
    int     sizeOK = 1; // 1 if size matches
    imageID ID;

    ID = image_ID(IDname);
    if(data.image[ID].md[0].naxis != 3)
    {
        printf("WARNING : image %s naxis = %d does not match expected value 3\n",
               IDname, (int) data.image[ID].md[0].naxis);
        sizeOK = 0;
    }
    if((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
    {
        printf("WARNING : image %s xsize = %d does not match expected value %d\n",
               IDname, (int) data.image[ID].md[0].size[0], (int) xsize);
        sizeOK = 0;
    }
    if((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
    {
        printf("WARNING : image %s ysize = %d does not match expected value %d\n",
               IDname, (int) data.image[ID].md[0].size[1], (int) ysize);
        sizeOK = 0;
    }
    if((zsize > 0) && (data.image[ID].md[0].size[2] != zsize))
    {
        printf("WARNING : image %s zsize = %d does not match expected value %d\n",
               IDname, (int) data.image[ID].md[0].size[2], (int) zsize);
        sizeOK = 0;
    }

    return sizeOK;
}





/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 10. COORDINATE CHANGE                                                                           */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */




errno_t rotate_cube(
    const char *ID_name,
    const char *ID_out_name,
    int         orientation
)
{
    /* 0 is from x axis */
    /* 1 is from y axis */
    imageID     ID;
    imageID     IDout;
    uint32_t    xsize, ysize, zsize;
    uint32_t    xsize1, ysize1, zsize1;
    uint32_t    ii, jj, kk;
    uint8_t     datatype;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    if(data.image[ID].md[0].naxis != 3)
    {
        PRINT_ERROR("Wrong naxis : %d - should be 3\n", (int) data.image[ID].md[0].naxis);
        exit(0);
    }
    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];
    zsize = data.image[ID].md[0].size[2];

    if(datatype == _DATATYPE_FLOAT) // single precision
    {
        if(orientation == 0)
        {
            xsize1 = zsize;
            ysize1 = ysize;
            zsize1 = xsize;
            IDout = create_3Dimage_ID_float(ID_out_name, xsize1, ysize1, zsize1);
            for(ii = 0; ii < xsize1; ii++)
                for(jj = 0; jj < ysize1; jj++)
                    for(kk = 0; kk < zsize1; kk++)
                    {
                        data.image[IDout].array.F[kk * ysize1 * xsize1 + jj * xsize1 + ii] =
                            data.image[ID].array.F[ii * xsize * ysize + jj * xsize + kk];
                    }
        }
        else
        {
            xsize1 = xsize;
            ysize1 = zsize;
            zsize1 = ysize;
            IDout = create_3Dimage_ID_float(ID_out_name, xsize1, ysize1, zsize1);
            for(ii = 0; ii < xsize1; ii++)
                for(jj = 0; jj < ysize1; jj++)
                    for(kk = 0; kk < zsize1; kk++)
                    {
                        data.image[IDout].array.F[kk * ysize1 * xsize1 + jj * xsize1 + ii] =
                            data.image[ID].array.F[jj * xsize * ysize + kk * xsize + ii];
                    }
        }
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        if(orientation == 0)
        {
            xsize1 = zsize;
            ysize1 = ysize;
            zsize1 = xsize;
            IDout = create_3Dimage_ID_double(ID_out_name, xsize1, ysize1, zsize1);
            for(ii = 0; ii < xsize1; ii++)
                for(jj = 0; jj < ysize1; jj++)
                    for(kk = 0; kk < zsize1; kk++)
                    {
                        data.image[IDout].array.D[kk * ysize1 * xsize1 + jj * xsize1 + ii] =
                            data.image[ID].array.D[ii * xsize * ysize + jj * xsize + kk];
                    }
        }
        else
        {
            xsize1 = xsize;
            ysize1 = zsize;
            zsize1 = ysize;
            IDout = create_3Dimage_ID_double(ID_out_name, xsize1, ysize1, zsize1);
            for(ii = 0; ii < xsize1; ii++)
                for(jj = 0; jj < ysize1; jj++)
                    for(kk = 0; kk < zsize1; kk++)
                    {
                        data.image[IDout].array.D[kk * ysize1 * xsize1 + jj * xsize1 + ii] =
                            data.image[ID].array.D[jj * xsize * ysize + kk * xsize + ii];
                    }
        }
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(0);
    }

    return RETURN_SUCCESS;
}










/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 11. SET IMAGE FLAGS / COUNTERS                                                                  */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */





errno_t COREMOD_MEMORY_image_set_status(
    const char *IDname,
    int         status
)
{
    imageID ID;

    ID = image_ID(IDname);
    data.image[ID].md[0].status = status;

    return RETURN_SUCCESS;
}


errno_t COREMOD_MEMORY_image_set_cnt0(
    const char *IDname,
    int         cnt0
)
{
    imageID ID;

    ID = image_ID(IDname);
    data.image[ID].md[0].cnt0 = cnt0;

    return RETURN_SUCCESS;
}


errno_t COREMOD_MEMORY_image_set_cnt1(
    const char *IDname,
    int         cnt1
)
{
    imageID ID;

    ID = image_ID(IDname);
    data.image[ID].md[0].cnt1 = cnt1;

    return RETURN_SUCCESS;
}


























/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 13. SIMPLE OPERATIONS ON STREAMS                                                                */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */


/**
 * ## Purpose
 *
 * Poke a stream at regular time interval\n
 * Does not change shared memory content\n
 *
 */
imageID COREMOD_MEMORY_streamPoke(
    const char *IDstream_name,
    long        usperiod
)
{
    imageID ID;
    long    twait1;
    struct  timespec t0;
    struct  timespec t1;
    double  tdiffv;
    struct  timespec tdiff;

    ID = image_ID(IDstream_name);



    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        sprintf(pinfoname, "streampoke-%s", IDstream_name);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        sprintf(msgstring, "%s", IDstream_name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    if(data.processinfo == 1)
    {
        processinfo->loopstat = 1;    // loop running
    }
    int loopOK = 1;
    int loopCTRLexit = 0; // toggles to 1 when loop is set to exit cleanly
    long loopcnt = 0;


    while(loopOK == 1)
    {
        // processinfo control
        if(data.processinfo == 1)
        {
            while(processinfo->CTRLval == 1)   // pause
            {
                struct timespec treq, trem;
                treq.tv_sec = 0;
                treq.tv_nsec = 50000;
                nanosleep(&treq, &trem);
            }

            if(processinfo->CTRLval == 2) // single iteration
            {
                processinfo->CTRLval = 1;
            }

            if(processinfo->CTRLval == 3) // exit loop
            {
                loopCTRLexit = 1;
            }
        }


        clock_gettime(CLOCK_REALTIME, &t0);

        data.image[ID].md[0].write = 1;
        data.image[ID].md[0].cnt0++;
        data.image[ID].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(ID, -1);



        usleep(twait1);

        clock_gettime(CLOCK_REALTIME, &t1);
        tdiff = timespec_diff(t0, t1);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        if(tdiffv < 1.0e-6 * usperiod)
        {
            twait1 ++;
        }
        else
        {
            twait1 --;
        }

        if(twait1 < 0)
        {
            twait1 = 0;
        }
        if(twait1 > usperiod)
        {
            twait1 = usperiod;
        }


        if(loopCTRLexit == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                struct timespec tstop;
                struct tm *tstoptm;
                char msgstring[200];

                clock_gettime(CLOCK_REALTIME, &tstop);
                tstoptm = gmtime(&tstop.tv_sec);

                sprintf(msgstring, "CTRLexit at %02d:%02d:%02d.%03d", tstoptm->tm_hour,
                        tstoptm->tm_min, tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
                strncpy(processinfo->statusmsg, msgstring, 200);

                processinfo->loopstat = 3; // clean exit
            }
        }

        loopcnt++;
        if(data.processinfo == 1)
        {
            processinfo->loopcnt = loopcnt;
        }
    }


    return ID;
}







/**
 * ## Purpose
 *
 * Compute difference between two 2D streams\n
 * Triggers on stream0\n
 *
 */
imageID COREMOD_MEMORY_streamDiff(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreammask_name,
    const char *IDstreamout_name,
    long        semtrig
)
{
    imageID    ID0;
    imageID    ID1;
    imageID    IDout;
    uint32_t   xsize;
    uint32_t   ysize;
    uint32_t   xysize;
    long       ii;
    uint32_t  *arraysize;
    unsigned long long  cnt;
    imageID    IDmask; // optional

    ID0 = image_ID(IDstream0_name);
    ID1 = image_ID(IDstream1_name);
    IDmask = image_ID(IDstreammask_name);

    xsize = data.image[ID0].md[0].size[0];
    ysize = data.image[ID0].md[0].size[1];
    xysize = xsize * ysize;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDstreamout_name, 2, arraysize, _DATATYPE_FLOAT, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDstreamout_name, IMAGE_NB_SEMAPHORE);
    }

    free(arraysize);


    while(1)
    {
        // has new frame arrived ?
        if(data.image[ID0].md[0].sem == 0)
        {
            while(cnt == data.image[ID0].md[0].cnt0)   // test if new frame exists
            {
                usleep(5);
            }
            cnt = data.image[ID0].md[0].cnt0;
        }
        else
        {
            sem_wait(data.image[ID0].semptr[semtrig]);
        }




        data.image[IDout].md[0].write = 1;
        if(IDmask == -1)
        {
            for(ii = 0; ii < xysize; ii++)
            {
                data.image[IDout].array.F[ii] = data.image[ID0].array.F[ii] -
                                                data.image[ID1].array.F[ii];
            }
        }
        else
        {
            for(ii = 0; ii < xysize; ii++)
            {
                data.image[IDout].array.F[ii] = (data.image[ID0].array.F[ii] -
                                                 data.image[ID1].array.F[ii]) * data.image[IDmask].array.F[ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);;
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }


    return IDout;
}







//
// compute difference between two 2D streams
// triggers alternatively on stream0 and stream1
//
imageID COREMOD_MEMORY_streamPaste(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreamout_name,
    long        semtrig0,
    long        semtrig1,
    int         master
)
{
    imageID     ID0;
    imageID     ID1;
    imageID     IDout;
    imageID     IDin;
    long        Xoffset;
    uint32_t    xsize;
    uint32_t    ysize;
//    uint32_t    xysize;
    long        ii;
    long        jj;
    uint32_t   *arraysize;
    unsigned long long   cnt;
    uint8_t     datatype;
    int         FrameIndex;

    ID0 = image_ID(IDstream0_name);
    ID1 = image_ID(IDstream1_name);

    xsize = data.image[ID0].md[0].size[0];
    ysize = data.image[ID0].md[0].size[1];
//    xysize = xsize*ysize;
    datatype = data.image[ID0].md[0].datatype;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    arraysize[0] = 2 * xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDstreamout_name, 2, arraysize, datatype, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDstreamout_name, IMAGE_NB_SEMAPHORE);
    }
    free(arraysize);


    FrameIndex = 0;

    while(1)
    {
        if(FrameIndex == 0)
        {
            // has new frame 0 arrived ?
            if(data.image[ID0].md[0].sem == 0)
            {
                while(cnt == data.image[ID0].md[0].cnt0) // test if new frame exists
                {
                    usleep(5);
                }
                cnt = data.image[ID0].md[0].cnt0;
            }
            else
            {
                sem_wait(data.image[ID0].semptr[semtrig0]);
            }
            Xoffset = 0;
            IDin = 0;
        }
        else
        {
            // has new frame 1 arrived ?
            if(data.image[ID1].md[0].sem == 0)
            {
                while(cnt == data.image[ID1].md[0].cnt0) // test if new frame exists
                {
                    usleep(5);
                }
                cnt = data.image[ID1].md[0].cnt0;
            }
            else
            {
                sem_wait(data.image[ID1].semptr[semtrig1]);
            }
            Xoffset = xsize;
            IDin = 1;
        }


        data.image[IDout].md[0].write = 1;

        switch(datatype)
        {
            case _DATATYPE_UINT8 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.UI8[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI8[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT16 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.UI16[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI16[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT32 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.UI32[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI32[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT64 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.UI64[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI64[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT8 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.SI8[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI8[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT16 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.SI16[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI16[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT32 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.SI32[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI32[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT64 :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.SI64[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI64[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_FLOAT :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.F[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.F[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_DOUBLE :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.D[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.D[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_COMPLEX_FLOAT :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.CF[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.CF[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_COMPLEX_DOUBLE :
                for(ii = 0; ii < xsize; ii++)
                    for(jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.CD[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.CD[jj * xsize + ii];
                    }
                break;

            default :
                printf("Unknown data type\n");
                exit(0);
                break;
        }
        if(FrameIndex == master)
        {
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);;
            data.image[IDout].md[0].cnt0++;
        }
        data.image[IDout].md[0].cnt1 = FrameIndex;
        data.image[IDout].md[0].write = 0;

        if(FrameIndex == 0)
        {
            FrameIndex = 1;
        }
        else
        {
            FrameIndex = 0;
        }
    }

    return IDout;
}












//
// compute difference between two halves of an image stream
// triggers on instream
//
imageID COREMOD_MEMORY_stream_halfimDiff(
    const char *IDstream_name,
    const char *IDstreamout_name,
    long        semtrig
)
{
    imageID    ID0;
    imageID    IDout;
    uint32_t   xsizein;
    uint32_t   ysizein;
//    uint32_t   xysizein;
    uint32_t   xsize;
    uint32_t   ysize;
    uint32_t   xysize;
    long       ii;
    uint32_t  *arraysize;
    unsigned long long  cnt;
    uint8_t    datatype;
    uint8_t    datatypeout;


    ID0 = image_ID(IDstream_name);

    xsizein = data.image[ID0].md[0].size[0];
    ysizein = data.image[ID0].md[0].size[1];
//    xysizein = xsizein*ysizein;

    xsize = xsizein;
    ysize = ysizein / 2;
    xysize = xsize * ysize;


    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    datatype = data.image[ID0].md[0].datatype;
    datatypeout = _DATATYPE_FLOAT;
    switch(datatype)
    {

        case _DATATYPE_UINT8:
            datatypeout = _DATATYPE_INT16;
            break;

        case _DATATYPE_UINT16:
            datatypeout = _DATATYPE_INT32;
            break;

        case _DATATYPE_UINT32:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_UINT64:
            datatypeout = _DATATYPE_INT64;
            break;


        case _DATATYPE_INT8:
            datatypeout = _DATATYPE_INT16;
            break;

        case _DATATYPE_INT16:
            datatypeout = _DATATYPE_INT32;
            break;

        case _DATATYPE_INT32:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_INT64:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_DOUBLE:
            datatypeout = _DATATYPE_DOUBLE;
            break;
    }

    IDout = image_ID(IDstreamout_name);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDstreamout_name, 2, arraysize, datatypeout, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDstreamout_name, IMAGE_NB_SEMAPHORE);
    }

    free(arraysize);



    while(1)
    {
        // has new frame arrived ?
        if(data.image[ID0].md[0].sem == 0)
        {
            while(cnt == data.image[ID0].md[0].cnt0) // test if new frame exists
            {
                usleep(5);
            }
            cnt = data.image[ID0].md[0].cnt0;
        }
        else
        {
            sem_wait(data.image[ID0].semptr[semtrig]);
        }

        data.image[IDout].md[0].write = 1;

        switch(datatype)
        {

            case _DATATYPE_UINT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI16[ii] = data.image[ID0].array.UI8[ii] -
                                                       data.image[ID0].array.UI8[xysize + ii];
                }
                break;

            case _DATATYPE_UINT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI32[ii] = data.image[ID0].array.UI16[ii] -
                                                       data.image[ID0].array.UI16[xysize + ii];
                }
                break;

            case _DATATYPE_UINT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.UI32[ii] -
                                                       data.image[ID0].array.UI32[xysize + ii];
                }
                break;

            case _DATATYPE_UINT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.UI64[ii] -
                                                       data.image[ID0].array.UI64[xysize + ii];
                }
                break;



            case _DATATYPE_INT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI16[ii] = data.image[ID0].array.SI8[ii] -
                                                       data.image[ID0].array.SI8[xysize + ii];
                }
                break;

            case _DATATYPE_INT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI32[ii] = data.image[ID0].array.SI16[ii] -
                                                       data.image[ID0].array.SI16[xysize + ii];
                }
                break;

            case _DATATYPE_INT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.SI32[ii] -
                                                       data.image[ID0].array.SI32[xysize + ii];
                }
                break;

            case _DATATYPE_INT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.SI64[ii] -
                                                       data.image[ID0].array.SI64[xysize + ii];
                }
                break;



            case _DATATYPE_FLOAT:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.F[ii] = data.image[ID0].array.F[ii] -
                                                    data.image[ID0].array.F[xysize + ii];
                }
                break;

            case _DATATYPE_DOUBLE:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.D[ii] = data.image[ID0].array.D[ii] -
                                                    data.image[ID0].array.D[xysize + ii];
                }
                break;

        }

        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }


    return IDout;
}





imageID COREMOD_MEMORY_streamAve(
    const char *IDstream_name,
    int         NBave,
    int         mode,
    const char *IDout_name
)
{
    imageID      IDout;
    imageID      IDout0;
    imageID      IDin;
    uint8_t      datatype;
    uint32_t     xsize;
    uint32_t     ysize;
    uint32_t     xysize;
    uint32_t    *imsize;
    int_fast8_t  OKloop;
    int          cntin = 0;
    long         dtus = 20;
    long         ii;
    long         cnt0;
    long         cnt0old;

    IDin = image_ID(IDstream_name);
    datatype = data.image[IDin].md[0].datatype;
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    xysize = xsize * ysize;


    IDout0 = create_2Dimage_ID("_streamAve_tmp", xsize, ysize);

    if(mode == 1) // local image
    {
        IDout = create_2Dimage_ID(IDout_name, xsize, ysize);
    }
    else // shared memory
    {
        IDout = image_ID(IDout_name);
        if(IDout == -1) // CREATE IT
        {
            imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
            imsize[0] = xsize;
            imsize[1] = ysize;
            IDout = create_image_ID(IDout_name, 2, imsize, _DATATYPE_FLOAT, 1, 0);
            COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);
            free(imsize);
        }
    }


    cntin = 0;
    cnt0old = data.image[IDin].md[0].cnt0;

    for(ii = 0; ii < xysize; ii++)
    {
        data.image[IDout].array.F[ii] = 0.0;
    }

    OKloop = 1;
    while(OKloop == 1)
    {
        // has new frame arrived ?
        cnt0 = data.image[IDin].md[0].cnt0;
        if(cnt0 != cnt0old)
        {
            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.D[ii];
                    }
                    break;
            }

            cntin++;
            if(cntin == NBave)
            {
                cntin = 0;
                data.image[IDout].md[0].write = 1;
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.F[ii] = data.image[IDout0].array.F[ii] / NBave;
                }
                data.image[IDout].md[0].cnt0++;
                data.image[IDout].md[0].write = 0;
                COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

                if(mode != 1)
                {
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[ii] = 0.0;
                    }
                }
                else
                {
                    OKloop = 0;
                }
            }
            cnt0old = cnt0;
        }
        usleep(dtus);
    }

    delete_image_ID("_streamAve_tmp");

    return IDout;
}






/** @brief takes a 3Dimage(s) (circular buffer(s)) and writes slices to a 2D image with time interval specified in us
 *
 *
 * If NBcubes=1, then the circular buffer named IDinname is sent to IDoutname at a frequency of 1/usperiod MHz
 * If NBcubes>1, several circular buffers are used, named ("%S_%03ld", IDinname, cubeindex). Semaphore semtrig of image IDsync_name triggers switch between circular buffers, with a delay of offsetus. The number of consecutive sem posts required to advance to the next circular buffer is period
 *
 * @param IDinname      Name of DM circular buffer (appended by _000, _001 etc... if NBcubes>1)
 * @param IDoutname     Output DM channel stream
 * @param usperiod      Interval between consecutive frames [us]
 * @param NBcubes       Number of input DM circular buffers
 * @param period        If NBcubes>1: number of input triggers required to advance to next input buffer
 * @param offsetus      If NBcubes>1: time offset [us] between input trigger and input buffer switch
 * @param IDsync_name   If NBcubes>1: Stream used for synchronization
 * @param semtrig       If NBcubes>1: semaphore used for synchronization
 * @param timingmode    Not used
 *
 *
 */
imageID COREMOD_MEMORY_image_streamupdateloop(
    const char *IDinname,
    const char *IDoutname,
    long        usperiod,
    long        NBcubes,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    __attribute__((unused)) int         timingmode
)
{
    imageID   *IDin;
    long       cubeindex;
    char       imname[200];
    long       IDsync;
    unsigned long long  cntsync;
    long       pcnt = 0;
    long       offsetfr = 0;
    long       offsetfrcnt = 0;
    int        cntDelayMode = 0;

    imageID    IDout;
    long       kk;
    uint32_t  *arraysize;
    long       naxis;
    uint8_t    datatype;
    char      *ptr0s; // source start 3D array ptr
    char      *ptr0; // source
    char      *ptr1; // dest
    long       framesize;
//    int        semval;

    int        RT_priority = 80; //any number from 0-99
    struct     sched_param schedpar;

    long       twait1;
    struct     timespec t0;
    struct     timespec t1;
    double     tdiffv;
    struct     timespec tdiff;

    int        SyncSlice = 0;



    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
#endif


    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        sprintf(pinfoname, "streamloop-%s", IDoutname);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        sprintf(msgstring, "%s->%s", IDinname, IDoutname);
        processinfo_WriteMessage(processinfo, msgstring);
    }




    if(NBcubes < 1)
    {
        printf("ERROR: invalid number of input cubes, needs to be >0");
        return RETURN_FAILURE;
    }


    int sync_semwaitindex;
    IDin = (long *) malloc(sizeof(long) * NBcubes);
    SyncSlice = 0;
    if(NBcubes == 1)
    {
        IDin[0] = image_ID(IDinname);

        // in single cube mode, optional sync stream drives updates to next slice within cube
        IDsync = image_ID(IDsync_name);
        if(IDsync != -1)
        {
            SyncSlice = 1;
            sync_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDsync], semtrig);
        }
    }
    else
    {
        IDsync = image_ID(IDsync_name);
        sync_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDsync], semtrig);

        for(cubeindex = 0; cubeindex < NBcubes; cubeindex++)
        {
            sprintf(imname, "%s_%03ld", IDinname, cubeindex);
            IDin[cubeindex] = image_ID(imname);
        }
        offsetfr = (long)(0.5 + 1.0 * offsetus / usperiod);

        printf("FRAMES OFFSET = %ld\n", offsetfr);
    }



    printf("SyncSlice = %d\n", SyncSlice);

    printf("Creating / connecting to image stream ...\n");
    fflush(stdout);


    naxis = data.image[IDin[0]].md[0].naxis;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        printf("ERROR: input image %s should be 3D\n", IDinname);
        exit(0);
    }
    arraysize[0] = data.image[IDin[0]].md[0].size[0];
    arraysize[1] = data.image[IDin[0]].md[0].size[1];
    arraysize[2] = data.image[IDin[0]].md[0].size[2];



    datatype = data.image[IDin[0]].md[0].datatype;

    IDout = image_ID(IDoutname);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDoutname, 2, arraysize, datatype, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDoutname, IMAGE_NB_SEMAPHORE);
    }

    cubeindex = 0;
    pcnt = 0;
    if(NBcubes > 1)
    {
        cntsync = data.image[IDsync].md[0].cnt0;
    }

    twait1 = usperiod;
    kk = 0;
    cntDelayMode = 0;



    if(data.processinfo == 1)
    {
        processinfo->loopstat = 1;    // loop running
    }
    int loopOK = 1;
    int loopCTRLexit = 0; // toggles to 1 when loop is set to exit cleanly
    long loopcnt = 0;

    while(loopOK == 1)
    {

        // processinfo control
        if(data.processinfo == 1)
        {
            while(processinfo->CTRLval == 1)  // pause
            {
                usleep(50);
            }

            if(processinfo->CTRLval == 2) // single iteration
            {
                processinfo->CTRLval = 1;
            }

            if(processinfo->CTRLval == 3) // exit loop
            {
                loopCTRLexit = 1;
            }
        }



        if(NBcubes > 1)
        {
            if(cntsync != data.image[IDsync].md[0].cnt0)
            {
                pcnt++;
                cntsync = data.image[IDsync].md[0].cnt0;
            }
            if(pcnt == period)
            {
                pcnt = 0;
                offsetfrcnt = 0;
                cntDelayMode = 1;
            }

            if(cntDelayMode == 1)
            {
                if(offsetfrcnt < offsetfr)
                {
                    offsetfrcnt++;
                }
                else
                {
                    cntDelayMode = 0;
                    cubeindex++;
                    kk = 0;
                }
            }
            if(cubeindex == NBcubes)
            {
                cubeindex = 0;
            }
        }


        switch(datatype)
        {

            case _DATATYPE_INT8:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI8;
                ptr1 = (char *) data.image[IDout].array.SI8;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT8;
                break;

            case _DATATYPE_UINT8:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI8;
                ptr1 = (char *) data.image[IDout].array.UI8;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT8;
                break;

            case _DATATYPE_INT16:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI16;
                ptr1 = (char *) data.image[IDout].array.SI16;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT16;
                break;

            case _DATATYPE_UINT16:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI16;
                ptr1 = (char *) data.image[IDout].array.UI16;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT16;
                break;

            case _DATATYPE_INT32:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI32;
                ptr1 = (char *) data.image[IDout].array.SI32;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT32;
                break;

            case _DATATYPE_UINT32:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI32;
                ptr1 = (char *) data.image[IDout].array.UI32;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT32;
                break;

            case _DATATYPE_INT64:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI64;
                ptr1 = (char *) data.image[IDout].array.SI64;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT64;
                break;

            case _DATATYPE_UINT64:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI64;
                ptr1 = (char *) data.image[IDout].array.UI64;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT64;
                break;


            case _DATATYPE_FLOAT:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.F;
                ptr1 = (char *) data.image[IDout].array.F;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * sizeof(float);
                break;

            case _DATATYPE_DOUBLE:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.D;
                ptr1 = (char *) data.image[IDout].array.D;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * sizeof(double);
                break;

        }




        clock_gettime(CLOCK_REALTIME, &t0);

        ptr0 = ptr0s + kk * framesize;
        data.image[IDout].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, framesize);
        data.image[IDout].md[0].cnt1 = kk;
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

        kk++;
        if(kk == data.image[IDin[0]].md[0].size[2])
        {
            kk = 0;
        }



        if(SyncSlice == 0)
        {
            usleep(twait1);

            clock_gettime(CLOCK_REALTIME, &t1);
            tdiff = timespec_diff(t0, t1);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

            if(tdiffv < 1.0e-6 * usperiod)
            {
                twait1 ++;
            }
            else
            {
                twait1 --;
            }

            if(twait1 < 0)
            {
                twait1 = 0;
            }
            if(twait1 > usperiod)
            {
                twait1 = usperiod;
            }
        }
        else
        {
            sem_wait(data.image[IDsync].semptr[sync_semwaitindex]);
        }

        if(loopCTRLexit == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                struct timespec tstop;
                struct tm *tstoptm;
                char msgstring[200];

                clock_gettime(CLOCK_REALTIME, &tstop);
                tstoptm = gmtime(&tstop.tv_sec);

                sprintf(msgstring, "CTRLexit at %02d:%02d:%02d.%03d", tstoptm->tm_hour,
                        tstoptm->tm_min, tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
                strncpy(processinfo->statusmsg, msgstring, 200);

                processinfo->loopstat = 3; // clean exit
            }
        }

        loopcnt++;
        if(data.processinfo == 1)
        {
            processinfo->loopcnt = loopcnt;
        }
    }

    free(IDin);

    return IDout;
}







// takes a 3Dimage (circular buffer) and writes slices to a 2D image synchronized with an image semaphore
imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(
    const char *IDinname,
    const char *IDoutname,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    __attribute__((unused)) int         timingmode
)
{
    imageID    IDin;
    imageID    IDout;
    imageID    IDsync;

    long       kk;
    long       kk1;

    uint32_t  *arraysize;
    long       naxis;
    uint8_t    datatype;
    char      *ptr0s; // source start 3D array ptr
    char      *ptr0; // source
    char      *ptr1; // dest
    long       framesize;
//    int        semval;

    int        RT_priority = 80; //any number from 0-99
    struct     sched_param schedpar;


    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
#endif


    printf("Creating / connecting to image stream ...\n");
    fflush(stdout);

    IDin = image_ID(IDinname);
    naxis = data.image[IDin].md[0].naxis;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        printf("ERROR: input image %s should be 3D\n", IDinname);
        exit(0);
    }
    arraysize[0] = data.image[IDin].md[0].size[0];
    arraysize[1] = data.image[IDin].md[0].size[1];
    arraysize[2] = data.image[IDin].md[0].size[2];





    datatype = data.image[IDin].md[0].datatype;

    IDout = image_ID(IDoutname);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDoutname, 2, arraysize, datatype, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDoutname, IMAGE_NB_SEMAPHORE);
    }

    switch(datatype)
    {

        case _DATATYPE_INT8:
            ptr0s = (char *) data.image[IDin].array.SI8;
            ptr1 = (char *) data.image[IDout].array.SI8;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT8;
            break;

        case _DATATYPE_UINT8:
            ptr0s = (char *) data.image[IDin].array.UI8;
            ptr1 = (char *) data.image[IDout].array.UI8;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT8;
            break;

        case _DATATYPE_INT16:
            ptr0s = (char *) data.image[IDin].array.SI16;
            ptr1 = (char *) data.image[IDout].array.SI16;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT16;
            break;

        case _DATATYPE_UINT16:
            ptr0s = (char *) data.image[IDin].array.UI16;
            ptr1 = (char *) data.image[IDout].array.UI16;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT16;
            break;

        case _DATATYPE_INT32:
            ptr0s = (char *) data.image[IDin].array.SI32;
            ptr1 = (char *) data.image[IDout].array.SI32;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT32;
            break;

        case _DATATYPE_UINT32:
            ptr0s = (char *) data.image[IDin].array.UI32;
            ptr1 = (char *) data.image[IDout].array.UI32;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT32;
            break;

        case _DATATYPE_INT64:
            ptr0s = (char *) data.image[IDin].array.SI64;
            ptr1 = (char *) data.image[IDout].array.SI64;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT64;
            break;

        case _DATATYPE_UINT64:
            ptr0s = (char *) data.image[IDin].array.UI64;
            ptr1 = (char *) data.image[IDout].array.UI64;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT64;
            break;


        case _DATATYPE_FLOAT:
            ptr0s = (char *) data.image[IDin].array.F;
            ptr1 = (char *) data.image[IDout].array.F;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        sizeof(float);
            break;

        case _DATATYPE_DOUBLE:
            ptr0s = (char *) data.image[IDin].array.D;
            ptr1 = (char *) data.image[IDout].array.D;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        sizeof(double);
            break;
    }




    IDsync = image_ID(IDsync_name);

    kk = 0;
    kk1 = 0;

    int sync_semwaitindex;
    sync_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDin], semtrig);

    while(1)
    {
        sem_wait(data.image[IDsync].semptr[sync_semwaitindex]);

        kk++;
        if(kk == period) // UPDATE
        {
            kk = 0;
            kk1++;
            if(kk1 == data.image[IDin].md[0].size[2])
            {
                kk1 = 0;
            }
            usleep(offsetus);
            ptr0 = ptr0s + kk1 * framesize;
            data.image[IDout].md[0].write = 1;
            memcpy((void *) ptr1, (void *) ptr0, framesize);
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
            data.image[IDout].md[0].cnt0++;
            data.image[IDout].md[0].write = 0;
        }
    }

    // release semaphore
    data.image[IDsync].semReadPID[sync_semwaitindex] = 0;

    return IDout;
}











/**
 * @brief Manages configuration parameters for streamDelay
 *
 * ## Purpose
 *
 * Initializes configuration parameters structure\n
 *
 * ## Arguments
 *
 * @param[in]
 * char*		fpsname
 * 				name of function parameter structure
 *
 * @param[in]
 * uint32_t		CMDmode
 * 				Command mode
 *
 *
 */
errno_t COREMOD_MEMORY_streamDelay_FPCONF(
    char    *fpsname,
    uint32_t CMDmode
)
{

    FPS_SETUP_INIT(fpsname, CMDmode);

    void *pNull = NULL;
    uint64_t FPFLAG;

    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT;
    FPFLAG &= ~FPFLAG_WRITERUN;

    long delayus_default[4] = { 1000, 1, 10000, 1000 };
    long fp_delayus = function_parameter_add_entry(&fps, ".delayus", "Delay [us]",
                      FPTYPE_INT64, FPFLAG, &delayus_default);
    (void) fp_delayus; // suppresses unused parameter compiler warning

    long dtus_default[4] = { 50, 1, 200, 50 };
    long fp_dtus    = function_parameter_add_entry(&fps, ".dtus",
                      "Loop period [us]", FPTYPE_INT64, FPFLAG, &dtus_default);
    (void) fp_dtus; // suppresses unused parameter compiler warning


    FPS_ADDPARAM_STREAM_IN(stream_inname,        ".in_name",     "input stream");
    FPS_ADDPARAM_STREAM_OUT(stream_outname,       ".out_name",    "output stream");

    long timeavemode_default[4] = { 0, 0, 3, 0 };
    FPS_ADDPARAM_INT64_IN(
        option_timeavemode,
        ".option.timeavemode",
        "Enable time window averaging (>0)",
        &timeavemode_default);

    double avedt_default[4] = { 0.001, 0.0001, 1.0, 0.001};
    FPS_ADDPARAM_FLT64_IN(
        option_avedt,
        ".option.avedt",
        "Averaging time window width",
        &avedt_default);

    // status
    FPS_ADDPARAM_INT64_OUT(zsize,        ".status.zsize",     "cube size");
    FPS_ADDPARAM_INT64_OUT(framelog,     ".status.framelag",  "lag in frame unit");
    FPS_ADDPARAM_INT64_OUT(kkin,         ".status.kkin",
                           "input cube slice index");
    FPS_ADDPARAM_INT64_OUT(kkout,        ".status.kkout",
                           "output cube slice index");





    // ==============================================
    // start function parameter conf loop, defined in function_parameter.h
    FPS_CONFLOOP_START
    // ==============================================


    // here goes the logic
    if(fps.parray[fp_option_timeavemode].val.l[0] !=
            0)     // time averaging enabled
    {
        fps.parray[fp_option_avedt].fpflag |= FPFLAG_WRITERUN;
        fps.parray[fp_option_avedt].fpflag |= FPFLAG_USED;
        fps.parray[fp_option_avedt].fpflag |= FPFLAG_VISIBLE;
    }
    else
    {
        fps.parray[fp_option_avedt].fpflag &= ~FPFLAG_WRITERUN;
        fps.parray[fp_option_avedt].fpflag &= ~FPFLAG_USED;
        fps.parray[fp_option_avedt].fpflag &= ~FPFLAG_VISIBLE;
    }


    // ==============================================
    // stop function parameter conf loop, defined in function_parameter.h
    FPS_CONFLOOP_END
    // ==============================================


    return RETURN_SUCCESS;
}














/**
 * @brief Delay image stream by time offset
 *
 * IDout_name is a time-delayed copy of IDin_name
 *
 */

imageID COREMOD_MEMORY_streamDelay_RUN(
    char *fpsname
)
{
    imageID             IDimc;
    imageID             IDin, IDout;
    uint32_t            xsize, ysize, xysize;
//    long                cnt0old;
    long                ii;
    struct timespec    *t0array;
    struct timespec     tnow;
    double              tdiffv;
    struct timespec     tdiff;
    uint32_t           *arraytmp;
    long                cntskip = 0;
    long                kk;


    // ===========================
    /// ### CONNECT TO FPS
    // ===========================
    FPS_CONNECT(fpsname, FPSCONNECT_RUN);


    // ===============================
    /// ### GET FUNCTION PARAMETER VALUES
    // ===============================
    // parameters are addressed by their tag name
    // These parameters are read once, before running the loop
    //
    char IDin_name[200];
    strncpy(IDin_name,  functionparameter_GetParamPtr_STRING(&fps, ".in_name"),
            FUNCTION_PARAMETER_STRMAXLEN);

    char IDout_name[200];
    strncpy(IDout_name, functionparameter_GetParamPtr_STRING(&fps, ".out_name"),
            FUNCTION_PARAMETER_STRMAXLEN);

    long delayus = functionparameter_GetParamValue_INT64(&fps, ".delayus");

    long dtus    = functionparameter_GetParamValue_INT64(&fps, ".dtus");

    int timeavemode = functionparameter_GetParamValue_INT64(&fps,
                      ".option.timeavemode");
    double *avedt   = functionparameter_GetParamPtr_FLOAT64(&fps, ".option.avedt");

    long *zsize    = functionparameter_GetParamPtr_INT64(&fps, ".status.zsize");
    long *framelag = functionparameter_GetParamPtr_INT64(&fps, ".status.framelag");
    long *kkin     = functionparameter_GetParamPtr_INT64(&fps, ".status.kkin");
    long *kkout    = functionparameter_GetParamPtr_INT64(&fps, ".status.kkout");

    DEBUG_TRACEPOINT(" ");

    // ===========================
    /// ### processinfo support
    // ===========================
    PROCESSINFO *processinfo;

    char pinfodescr[200];
    sprintf(pinfodescr, "streamdelay %.10s %.10s", IDin_name, IDout_name);
    processinfo = processinfo_setup(
                      fpsname,                 // re-use fpsname as processinfo name
                      pinfodescr,    // description
                      "startup",  // message on startup
                      __FUNCTION__, __FILE__, __LINE__
                  );

    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority =
        20;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore




    // =============================================
    /// ### OPTIONAL: TESTING CONDITION FOR LOOP ENTRY
    // =============================================
    // Pre-loop testing, anything that would prevent loop from starting should issue message
    int loopOK = 1;


    IDin = image_ID(IDin_name);



    // ERROR HANDLING
    if(IDin == -1)
    {
        struct timespec errtime;
        struct tm *errtm;

        clock_gettime(CLOCK_REALTIME, &errtime);
        errtm = gmtime(&errtime.tv_sec);

        fprintf(stderr,
                "%02d:%02d:%02d.%09ld  ERROR [%s %s %d] Input stream %s does not exist, cannot proceed\n",
                errtm->tm_hour,
                errtm->tm_min,
                errtm->tm_sec,
                errtime.tv_nsec,
                __FILE__,
                __FUNCTION__,
                __LINE__,
                IDin_name);

        char msgstring[200];
        sprintf(msgstring, "Input stream %.20s does not exist", IDin_name);
        processinfo_error(processinfo, msgstring);
        loopOK = 0;
    }


    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    *zsize = (long)(2 * delayus / dtus);
    if(*zsize < 1)
    {
        *zsize = 1;
    }
    xysize = xsize * ysize;

    t0array = (struct timespec *) malloc(sizeof(struct timespec) * *zsize);

    IDimc = create_3Dimage_ID("_tmpc", xsize, ysize, *zsize);



    IDout = image_ID(IDout_name);
    if(IDout == -1)   // CREATE IT
    {
        arraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);
        arraytmp[0] = xsize;
        arraytmp[1] = ysize;
        IDout = create_image_ID(IDout_name, 2, arraytmp, _DATATYPE_FLOAT, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);
        free(arraytmp);
    }


    *kkin = 0;
    *kkout = 0;
//    cnt0old = data.image[IDin].md[0].cnt0;

    float *arraytmpf;
    arraytmpf = (float *) malloc(sizeof(float) * xsize * ysize);

    clock_gettime(CLOCK_REALTIME, &tnow);
    for(kk = 0; kk < *zsize; kk++)
    {
        t0array[kk] = tnow;
    }


    DEBUG_TRACEPOINT(" ");

    // ===========================
    /// ### START LOOP
    // ===========================

    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop

    DEBUG_TRACEPOINT(" ");

    while(loopOK == 1)
    {
        int kkinscan;
        float normframes = 0.0;

        DEBUG_TRACEPOINT(" ");
        loopOK = processinfo_loopstep(processinfo);

        usleep(dtus); // main loop wait

        processinfo_exec_start(processinfo);

        if(processinfo_compute_status(processinfo) == 1)
        {
            DEBUG_TRACEPOINT(" ");

            // has new frame arrived ?
//            cnt0 = data.image[IDin].md[0].cnt0;

//            if(cnt0 != cnt0old) { // new frame
            clock_gettime(CLOCK_REALTIME, &t0array[*kkin]);  // record time of input frame

            DEBUG_TRACEPOINT(" ");
            for(ii = 0; ii < xysize; ii++)
            {
                data.image[IDimc].array.F[(*kkin) * xysize + ii] = data.image[IDin].array.F[ii];
            }
            (*kkin) ++;
            DEBUG_TRACEPOINT(" ");

            if((*kkin) == (*zsize))
            {
                (*kkin) = 0;
            }
            //              cnt0old = cnt0;
            //          }



            clock_gettime(CLOCK_REALTIME, &tnow);
            DEBUG_TRACEPOINT(" ");


            cntskip = 0;
            tdiff = timespec_diff(t0array[*kkout], tnow);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

            DEBUG_TRACEPOINT(" ");


            while((tdiffv > 1.0e-6 * delayus) && (cntskip < *zsize))
            {
                cntskip++;  // advance index until time condition is satisfied
                (*kkout) ++;
                if(*kkout == *zsize)
                {
                    *kkout = 0;
                }
                tdiff = timespec_diff(t0array[*kkout], tnow);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
            }

            DEBUG_TRACEPOINT(" ");

            *framelag = *kkin - *kkout;
            if(*framelag < 0)
            {
                *framelag += *zsize;
            }


            DEBUG_TRACEPOINT(" ");


            switch(timeavemode)
            {


                case 0: // no time averaging - pick more recent frame that matches requirement
                    DEBUG_TRACEPOINT(" ");
                    if(cntskip > 0)
                    {
                        char *ptr; // pointer address

                        data.image[IDout].md[0].write = 1;

                        ptr = (char *) data.image[IDimc].array.F;
                        ptr += SIZEOF_DATATYPE_FLOAT * xysize * *kkout;

                        memcpy(data.image[IDout].array.F, ptr,
                               SIZEOF_DATATYPE_FLOAT * xysize);  // copy time-delayed input to output

                        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
                        data.image[IDout].md[0].cnt0++;
                        data.image[IDout].md[0].write = 0;
                    }
                    break;

                default : // strict time window (note: other modes will be coded in the future)
                    normframes = 0.0;
                    DEBUG_TRACEPOINT(" ");

                    for(ii = 0; ii < xysize; ii++)
                    {
                        arraytmpf[ii] = 0.0;
                    }

                    for(kkinscan = 0; kkinscan < *zsize; kkinscan++)
                    {
                        tdiff = timespec_diff(t0array[kkinscan], tnow);
                        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

                        if((tdiffv > 0) && (fabs(tdiffv - 1.0e-6 * delayus) < *avedt))
                        {
                            float coeff = 1.0;
                            for(ii = 0; ii < xysize; ii++)
                            {
                                arraytmpf[ii] += coeff * data.image[IDimc].array.F[kkinscan * xysize + ii];
                            }
                            normframes += coeff;
                        }
                    }
                    if(normframes < 0.0001)
                    {
                        normframes = 0.0001;    // avoid division by zero
                    }

                    data.image[IDout].md[0].write = 1;
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[ii] = arraytmpf[ii] / normframes;
                    }
                    COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
                    data.image[IDout].md[0].cnt0++;
                    data.image[IDout].md[0].write = 0;

                    break;
            }
            DEBUG_TRACEPOINT(" ");



        }
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);
        DEBUG_TRACEPOINT(" ");
    }

    // ==================================
    /// ### ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);
    function_parameter_RUNexit(&fps);

    DEBUG_TRACEPOINT(" ");

    delete_image_ID("_tmpc");

    free(t0array);
    free(arraytmpf);

    return IDout;
}





























errno_t COREMOD_MEMORY_streamDelay(
    const char *IDin_name,
    const char *IDout_name,
    long        delayus,
    long        dtus
)
{
    char fpsname[200];
    unsigned int pindex = 0;
    FUNCTION_PARAMETER_STRUCT fps;

    // create FPS
    sprintf(fpsname, "%s-%06u", __FUNCTION__, pindex);
    COREMOD_MEMORY_streamDelay_FPCONF(fpsname, FPSCMDCODE_FPSINIT);

    function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_RUN);

    functionparameter_SetParamValue_STRING(&fps, ".instreamname", IDin_name);
    functionparameter_SetParamValue_STRING(&fps, ".outstreamname", IDout_name);

    functionparameter_SetParamValue_INT64(&fps, ".delayus", delayus);
    functionparameter_SetParamValue_INT64(&fps, ".dtus", dtus);

    function_parameter_struct_disconnect(&fps);

    COREMOD_MEMORY_streamDelay_RUN(fpsname);

    return RETURN_SUCCESS;
}














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














//
// pixel decode for unsigned short
// sem0, cnt0 gets updated at each full frame
// sem1 gets updated for each slice
// cnt1 contains the slice index that was just written
//
imageID COREMOD_MEMORY_PixMapDecode_U(
    const char *inputstream_name,
    uint32_t    xsizeim,
    uint32_t    ysizeim,
    const char *NBpix_fname,
    const char *IDmap_name,
    const char *IDout_name,
    const char *IDout_pixslice_fname
)
{
    imageID   IDout = -1;
    imageID   IDin;
    imageID   IDmap;
    long      slice, sliceii;
    long      oldslice = 0;
    long      NBslice;
    long     *nbpixslice;
    uint32_t  xsizein, ysizein;
    FILE     *fp;
    uint32_t *sizearray;
    imageID   IDout_pixslice;
    long      ii;
    unsigned long long      cnt = 0;
    //    int RT_priority = 80; //any number from 0-99

    //    struct sched_param schedpar;
    struct timespec ts;
    long scnt;
    int semval;
    //    long long iter;
    //    int r;
    long tmpl0, tmpl1;
    int semr;

    double *dtarray;
    struct timespec *tarray;
//    long slice1;


    PROCESSINFO *processinfo;

    IDin = image_ID(inputstream_name);
    IDmap = image_ID(IDmap_name);

    xsizein = data.image[IDin].md[0].size[0];
    ysizein = data.image[IDin].md[0].size[1];
    NBslice = data.image[IDin].md[0].size[2];

    char pinfoname[200];  // short name for the processinfo instance
    sprintf(pinfoname, "decode-%s-to-%s", inputstream_name, IDout_name);
    char pinfodescr[200];
    sprintf(pinfodescr, "%ldx%ldx%ld->%ldx%ld", (long) xsizein, (long) ysizein,
            NBslice, (long) xsizeim, (long) ysizeim);
    char msgstring[200];
    sprintf(msgstring, "%s->%s", inputstream_name, IDout_name);

    processinfo = processinfo_setup(
                      pinfoname,             // short name for the processinfo instance, no spaces, no dot, name should be human-readable
                      pinfodescr,    // description
                      msgstring,  // message on startup
                      __FUNCTION__, __FILE__, __LINE__
                  );
    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority =
        20;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore


    int loopOK = 1;

    processinfo_WriteMessage(processinfo, "Allocating memory");

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    int in_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDin], 0);

    if(xsizein != data.image[IDmap].md[0].size[0])
    {
        printf("ERROR: xsize for %s (%d) does not match xsize for %s (%d)\n",
               inputstream_name, xsizein, IDmap_name, data.image[IDmap].md[0].size[0]);
        exit(0);
    }
    if(ysizein != data.image[IDmap].md[0].size[1])
    {
        printf("ERROR: xsize for %s (%d) does not match xsize for %s (%d)\n",
               inputstream_name, ysizein, IDmap_name, data.image[IDmap].md[0].size[1]);
        exit(0);
    }
    sizearray[0] = xsizeim;
    sizearray[1] = ysizeim;
    IDout = create_image_ID(IDout_name, 2, sizearray,
                            data.image[IDin].md[0].datatype, 1, 0);
    COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);
    IDout_pixslice = create_image_ID("outpixsl", 2, sizearray, _DATATYPE_UINT16, 0,
                                     0);



    dtarray = (double *) malloc(sizeof(double) * NBslice);
    tarray = (struct timespec *) malloc(sizeof(struct timespec) * NBslice);


    nbpixslice = (long *) malloc(sizeof(long) * NBslice);
    if((fp = fopen(NBpix_fname, "r")) == NULL)
    {
        printf("ERROR : cannot open file \"%s\"\n", NBpix_fname);
        exit(0);
    }

    for(slice = 0; slice < NBslice; slice++)
    {
        int fscanfcnt = fscanf(fp, "%ld %ld %ld\n", &tmpl0, &nbpixslice[slice], &tmpl1);
        if(fscanfcnt == EOF)
        {
            if(ferror(fp))
            {
                perror("fscanf");
            }
            else
            {
                fprintf(stderr,
                        "Error: fscanf reached end of file, no matching characters, no matching failure\n");
            }
            return RETURN_FAILURE;
        }
        else if(fscanfcnt != 3)
        {
            fprintf(stderr,
                    "Error: fscanf successfully matched and assigned %i input items, 2 expected\n",
                    fscanfcnt);
            return RETURN_FAILURE;
        }


    }
    fclose(fp);

    for(slice = 0; slice < NBslice; slice++)
    {
        printf("Slice %5ld   : %5ld pix\n", slice, nbpixslice[slice]);
    }




    for(slice = 0; slice < NBslice; slice++)
    {
        sliceii = slice * data.image[IDmap].md[0].size[0] *
                  data.image[IDmap].md[0].size[1];
        for(ii = 0; ii < nbpixslice[slice]; ii++)
        {
            data.image[IDout_pixslice].array.UI16[ data.image[IDmap].array.UI16[sliceii +
                                                   ii] ] = (unsigned short) slice;
        }
    }

    save_fits("outpixsl", IDout_pixslice_fname);
    delete_image_ID("outpixsl");

    /*
        if(sigaction(SIGTERM, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGTERM\n");
        }

        if(sigaction(SIGINT, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGINT\n");
        }

        if(sigaction(SIGABRT, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGABRT\n");
        }

        if(sigaction(SIGBUS, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGBUS\n");
        }

        if(sigaction(SIGSEGV, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGSEGV\n");
        }

        if(sigaction(SIGHUP, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGHUP\n");
        }

        if(sigaction(SIGPIPE, &data.sigact, NULL) == -1) {
            printf("\ncan't catch SIGPIPE\n");
        }
    */



    processinfo_WriteMessage(processinfo, "Starting loop");

    // ==================================
    // STARTING LOOP
    // ==================================
    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop


    // long loopcnt = 0;
    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);

        /*
                if(data.processinfo == 1) {
                    while(processinfo->CTRLval == 1) { // pause
                        usleep(50);
                    }

                    if(processinfo->CTRLval == 2) { // single iteration
                        processinfo->CTRLval = 1;
                    }

                    if(processinfo->CTRLval == 3) { // exit loop
                        loopOK = 0;
                    }
                }
        */

        if(data.image[IDin].md[0].sem == 0)
        {
            while(data.image[IDin].md[0].cnt0 == cnt)   // test if new frame exists
            {
                usleep(5);
            }
            cnt = data.image[IDin].md[0].cnt0;
        }
        else
        {
            if(clock_gettime(CLOCK_REALTIME, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 1;
#ifndef __MACH__
            semr = ImageStreamIO_semtimedwait(&data.image[IDin], in_semwaitindex, &ts);
            //semr = sem_timedwait(data.image[IDin].semptr[0], &ts);
#else
            alarm(1);
            semr = ImageStreamIO_semwait(&data.image[IDin], in_semwaitindex);
            //semr = sem_wait(data.image[IDin].semptr[0]);
#endif

            if(processinfo->loopcnt == 0)
            {
                sem_getvalue(data.image[IDin].semptr[in_semwaitindex], &semval);
                for(scnt = 0; scnt < semval; scnt++)
                {
                    sem_trywait(data.image[IDin].semptr[in_semwaitindex]);
                }
            }
        }





        processinfo_exec_start(processinfo);

        if(processinfo_compute_status(processinfo) == 1)
        {
            if(semr == 0)
            {
                slice = data.image[IDin].md[0].cnt1;
                if(slice > oldslice + 1)
                {
                    slice = oldslice + 1;
                }

                if(oldslice == NBslice - 1)
                {
                    slice = 0;
                }

                //   clock_gettime(CLOCK_REALTIME, &tarray[slice]);
                //  dtarray[slice] = 1.0*tarray[slice].tv_sec + 1.0e-9*tarray[slice].tv_nsec;
                data.image[IDout].md[0].write = 1;

                if(slice < NBslice)
                {
                    sliceii = slice * data.image[IDmap].md[0].size[0] *
                              data.image[IDmap].md[0].size[1];
                    for(ii = 0; ii < nbpixslice[slice]; ii++)
                    {
                        data.image[IDout].array.UI16[data.image[IDmap].array.UI16[sliceii + ii] ] =
                            data.image[IDin].array.UI16[sliceii + ii];
                    }
                }
                //     printf("[%ld] ", slice); //TEST

                if(slice == NBslice - 1)   //if(slice<oldslice)
                {
                    COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);


                    data.image[IDout].md[0].cnt0 ++;

                    //     printf("[[ Timimg [us] :   ");
                    //  for(slice1=1;slice1<NBslice;slice1++)
                    //      {
                    //              dtarray[slice1] -= dtarray[0];
                    //           printf("%6ld ", (long) (1.0e6*dtarray[slice1]));
                    //      }
                    // printf("]]");
                    //  printf("\n");//TEST
                    // fflush(stdout);
                }

                data.image[IDout].md[0].cnt1 = slice;

                sem_getvalue(data.image[IDout].semptr[2], &semval);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    sem_post(data.image[IDout].semptr[2]);
                }

                sem_getvalue(data.image[IDout].semptr[3], &semval);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    sem_post(data.image[IDout].semptr[3]);
                }

                data.image[IDout].md[0].write = 0;

                oldslice = slice;
            }
        }


        processinfo_exec_end(processinfo);


        // process signals
        /*
                if(data.signal_TERM == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGTERM);
                    }
                }

                if(data.signal_INT == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGINT);
                    }
                }

                if(data.signal_ABRT == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGABRT);
                    }
                }

                if(data.signal_BUS == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGBUS);
                    }
                }

                if(data.signal_SEGV == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGSEGV);
                    }
                }

                if(data.signal_HUP == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGHUP);
                    }
                }

                if(data.signal_PIPE == 1) {
                    loopOK = 0;
                    if(data.processinfo == 1) {
                        processinfo_SIGexit(processinfo, SIGPIPE);
                    }
                }

                loopcnt++;
                if(data.processinfo == 1) {
                    processinfo->loopcnt = loopcnt;
                }

                //    if((data.signal_INT == 1)||(data.signal_TERM == 1)||(data.signal_ABRT==1)||(data.signal_BUS==1)||(data.signal_SEGV==1)||(data.signal_HUP==1)||(data.signal_PIPE==1))
                //        loopOK = 0;

                //iter++;
                */
    }

    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    /*    if((data.processinfo == 1) && (processinfo->loopstat != 4)) {
            processinfo_cleanExit(processinfo);
        }*/

    free(nbpixslice);
    free(sizearray);
    free(dtarray);
    free(tarray);

    return IDout;
}


































