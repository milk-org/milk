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
#include "COREMOD_memory/image_keyword.h"
#include "COREMOD_memory/compute_image_memory.h"
#include "COREMOD_memory/list_image.h"

#include "COREMOD_memory/variable_ID.h"

#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/delete_image.h"

#include "COREMOD_memory/image_copy.h"
#include "COREMOD_memory/image_complex.h"

#include "COREMOD_memory/read_shmim.h"
#include "COREMOD_memory/stream_sem.h"
#include "COREMOD_memory/stream_TCP.h"
#include "COREMOD_memory/logshmim.h"


errno_t COREMOD_MEMORY_testfunc();






/* =============================================================================================== */
/* =============================================================================================== */
/** @name 1. MANAGE MEMORY AND IDENTIFIERS
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */


errno_t    memory_monitor(
    const char *termttyname
);

long       compute_nb_image();

long       compute_nb_variable();


long       compute_variable_memory();



errno_t    delete_variable_ID(
    const char *varname
);

variableID create_variable_long_ID(
    const char *name,
    long value
);

variableID create_variable_string_ID(
    const char *name,
    const char *value
);


errno_t    clearall();



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





/* =============================================================================================== */
/* =============================================================================================== */
/** @name 11. SET IMAGE FLAGS / COUNTERS
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

errno_t COREMOD_MEMORY_image_set_status(
    const char *IDname,
    int         status
);

errno_t COREMOD_MEMORY_image_set_cnt0(
    const char *IDname,
    int         cnt0
);

errno_t COREMOD_MEMORY_image_set_cnt1(
    const char *IDname,
    int         cnt1
);

///@}







/* =============================================================================================== */
/* =============================================================================================== */
/** @name 13. SIMPLE OPERATIONS ON STREAMS
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */


/** @brief Poke stream at regular intervals
 */
imageID COREMOD_MEMORY_streamPoke(
    const char *IDstream_name,
    long        usperiod
);


/** @brief Difference between two streams
*/
imageID COREMOD_MEMORY_streamDiff(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreammask_name,
    const char *IDstreamout_name,
    long        semtrig
);


/** @brief Paste two equal size 2D streams into an output 2D stream
*/
imageID COREMOD_MEMORY_streamPaste(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreamout_name,
    long        semtrig0,
    long        semtrig1,
    int         master
);


/** difference between two halves of stream image
*/
imageID COREMOD_MEMORY_stream_halfimDiff(
    const char *IDstream_name,
    const char *IDstreamout_name,
    long        semtrig
);


/** @brief Averages frames in stream
 *
 * @param[in]  IDstream_name        Input stream
 * @param[in]  NBave                Number of consecutive frames to be averaged together
 * @param[in]  mode                 1: Perform average once, exit when completed and write output to local image
 * 									2: Run forever, write output to shared mem stream
 * @param[out] IDout_name           output stream name
 *
 */
imageID COREMOD_MEMORY_streamAve(
    const char *IDstream_name,
    int         NBave,
    int         mode,
    const char *IDout_name
);




/**
 * @brief takes a 3Dimage (circular buffer) and writes slices to a 2D image with time interval specified in us */
imageID COREMOD_MEMORY_image_streamupdateloop(
    const char *IDinname,
    const char *IDoutname,
    long        usperiod,
    long        NBcubes,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    int         timingmode
);


/**
 * @brief takes a 3Dimage (circular buffer) and writes slices to a 2D image synchronized with an image semaphore
 *
 *
 * @param[in]	IDinname  		3D circular buffer of frames to be written
 * @param[out]	IDoutname 		2D output stream
 * @param[in]	period	 		number of semaphore waits required to advance to next slice in the circular buffer
 * @param[in]	offsetus		fixed time offset between trigger stream and output write
 * @param[in]	IDsync_name		trigger stream
 * @param[in]	smmtrig			semaphore index for trigger
 * @param[in]	timingmode
 */
imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(
    const char *IDinname,
    const char *IDoutname,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    int         timingmode
);


errno_t COREMOD_MEMORY_streamDelay_FPCONF(
    char    *fpsname,
    uint32_t CMDmode
);

imageID COREMOD_MEMORY_streamDelay_RUN(
    char *fpsname
);


errno_t COREMOD_MEMORY_streamDelay(
    const char *IDin_name,
    const char *IDout_name,
    long        delayus,
    long        dtus
);



errno_t COREMOD_MEMORY_SaveAll_snapshot(
    const char *dirname
);


errno_t COREMOD_MEMORY_SaveAll_sequ(
    const char *dirname,
    const char *IDtrig_name,
    long        semtrig,
    long        NBframes
);






imageID COREMOD_MEMORY_PixMapDecode_U(
    const char *inputstream_name,
    uint32_t    xsizeim,
    uint32_t    ysizeim,
    const char *NBpix_fname,
    const char *IDmap_name,
    const char *IDout_name,
    const char *IDout_pixslice_fname
);

///@}





/* =============================================================================================== */
/* =============================================================================================== */
/** @name 14. DATA LOGGING
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */


///@}


#endif
