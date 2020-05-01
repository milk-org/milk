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

//#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"



#include "image_ID.h"
#include "image_keyword.h"
#include "image_set_counters.h"
#include "variable_ID.h"
#include "list_image.h"

#include "create_image.h"
#include "delete_image.h"

#include "image_copy.h"
#include "image_complex.h"

#include "read_shmim.h"
#include "stream_sem.h"
#include "stream_TCP.h"
#include "stream_poke.h"
#include "stream_diff.h"
#include "stream_paste.h"
#include "stream_halfimdiff.h"
#include "stream_ave.h"
#include "stream_updateloop.h"
#include "stream_delay.h"
#include "stream_pixmapdecode.h"
#include "logshmim.h"
#include "saveall.h"
#include "clearall.h"


errno_t COREMOD_MEMORY_testfunc();



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























