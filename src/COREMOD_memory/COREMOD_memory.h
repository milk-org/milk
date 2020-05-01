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

// The number of images in the data structure is kept NB_IMAGES_BUFFER 
// above the number of used images prior to the execution of any function. 
// It means that no function should create more than 100 images.
#define NB_IMAGES_BUFFER 500

// When the number of free images in the data structure is below NB_IMAGES_BUFFER, 
// it is increased by  NB_IMAGES_BUFFER
#define NB_IMAGES_BUFFER_REALLOC 600


// The number of variables in the data structure is kept NB_VARIABLES_BUFFER 
// above the number of used variables prior to the execution of any function. 
// It means that no function should create more than 100 variables.
#define NB_VARIABLES_BUFFER 100

// When the number of free variables in the data structure is below NB_VARIABLES_BUFFER, 
// it is increased by  NB_VARIABLES_BUFFER
#define NB_VARIABLES_BUFFER_REALLOC 150





#include "COREMOD_memory/clearall.h"
#include "COREMOD_memory/compute_image_memory.h"
#include "COREMOD_memory/compute_nb_image.h"
#include "COREMOD_memory/compute_nb_variable.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/delete_image.h"
#include "COREMOD_memory/delete_variable.h"
#include "COREMOD_memory/image_checksize.h"
#include "COREMOD_memory/image_complex.h"
#include "COREMOD_memory/image_copy.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/image_keyword.h"
#include "COREMOD_memory/image_set_counters.h"
#include "COREMOD_memory/list_image.h"
#include "COREMOD_memory/list_variable.h"
#include "COREMOD_memory/logshmim.h"
#include "COREMOD_memory/read_shmim.h"
#include "COREMOD_memory/saveall.h"
#include "COREMOD_memory/stream_ave.h"
#include "COREMOD_memory/stream_delay.h"
#include "COREMOD_memory/stream_diff.h"
#include "COREMOD_memory/stream_halfimdiff.h"
#include "COREMOD_memory/stream_paste.h"
#include "COREMOD_memory/stream_pixmapdecode.h"
#include "COREMOD_memory/stream_poke.h"
#include "COREMOD_memory/stream_sem.h"
#include "COREMOD_memory/stream_TCP.h"
#include "COREMOD_memory/stream_updateloop.h"
#include "COREMOD_memory/variable_ID.h"




//errno_t rotate_cube(const char *ID_name, const char *ID_out_name,
//                    int orientation);


#endif
