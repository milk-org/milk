/**
 * @file    COREMOD_memory.h
 * @brief   Function prototypes for milk memory functions
 *
 * Functions to handle images and streams
 *
 *
 */


#ifndef _COREMODMEMORY_H
#define _COREMODMEMORY_H

/* the number of images in the data structure is kept NB_IMAGES_BUFFER above the number of used images prior to the execution of any function. It means that no function should create more than 100 images. */
#define NB_IMAGES_BUFFER 500
/* when the number of free images in the data structure is below NB_IMAGES_BUFFER, it is increased by  NB_IMAGES_BUFFER */
#define NB_IMAGES_BUFFER_REALLOC 600

/* the number of variables in the data structure is kept NB_VARIABLES_BUFFER above the number of used variables prior to the execution of any function. It means that no function should create more than 100 variables. */
#define NB_VARIABLES_BUFFER 100
/* when the number of free variables in the data structure is below NB_VARIABLES_BUFFER, it is increased by  NB_VARIABLES_BUFFER */
#define NB_VARIABLES_BUFFER_REALLOC 150




/*void print_sys_mem_info();*/


typedef long imageID;











//void __attribute__((constructor)) libinit_COREMOD_memory();



//int ImageCreateSem(IMAGE *image, long NBsem);

//int ImageCreate(IMAGE *image, const char *name, long naxis, uint32_t *size, uint8_t datatype, int shared, int NBkw);


#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/compute_nb_image.h"
#include "COREMOD_memory/image_keyword.h"
#include "COREMOD_memory/compute_image_memory.h"
#include "COREMOD_memory/list_image.h"

#include "COREMOD_memory/variable_ID.h"
#include "COREMOD_memory/compute_nb_variable.h"

#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/delete_image.h"
#include "COREMOD_memory/delete_variable.h"

#include "COREMOD_memory/image_copy.h"
#include "COREMOD_memory/image_complex.h"

#include "COREMOD_memory/read_shmim.h"
#include "COREMOD_memory/stream_sem.h"
#include "COREMOD_memory/stream_TCP.h"
#include "COREMOD_memory/stream_poke.h"
#include "COREMOD_memory/stream_diff.h"
#include "COREMOD_memory/stream_paste.h"
#include "COREMOD_memory/stream_halfimdiff.h"
#include "COREMOD_memory/stream_ave.h"
#include "COREMOD_memory/stream_updateloop.h"
#include "COREMOD_memory/stream_delay.h"
#include "COREMOD_memory/stream_pixmapdecode.h"

#include "COREMOD_memory/logshmim.h"

#include "COREMOD_memory/saveall.h"
#include "COREMOD_memory/clearall.h"

errno_t COREMOD_MEMORY_testfunc();






/* =============================================================================================== */
/* =============================================================================================== */
/** @name 1. MANAGE MEMORY AND IDENTIFIERS
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */




long       compute_variable_memory();



variableID create_variable_long_ID(
    const char *name,
    long value
);

variableID create_variable_string_ID(
    const char *name,
    const char *value
);



///@}







/* =============================================================================================== */
/* =============================================================================================== */
/** @name 5. CREATE VARIABLE
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

variableID create_variable_ID(
    const char *name,
    double      value
);

///@}















/* =============================================================================================== */
/* =============================================================================================== */
/** @name 9. VERIFY SIZE
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

int check_2Dsize(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize
);

int check_3Dsize(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize
);

int COREMOD_MEMORY_check_2Dsize(
    const char *IDname,
    uint32_t    xsize,
    uint32_t    ysize
);

int COREMOD_MEMORY_check_3Dsize(
    const char *IDname,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize
);

///@}




/* =============================================================================================== */
/* =============================================================================================== */
/** @name 10. COORDINATE CHANGE
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

errno_t rotate_cube(const char *ID_name, const char *ID_out_name,
                    int orientation);

///@}














#endif
