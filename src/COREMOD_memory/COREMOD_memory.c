/**
 * @file    COREMOD_memory.c
 * @brief   milk memory functions
 *
 * Functions to handle images and streams
 *
 */


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

#include "CommandLineInterface/CLIcore.h"

#include "clearall.h"
#include "create_image.h"
#include "create_variable.h"
#include "delete_image.h"
#include "image_ID.h"
#include "image_complex.h"
#include "image_copy.h"
#include "image_keyword.h"
#include "image_set_counters.h"
#include "list_image.h"
#include "list_variable.h"
#include "logshmim.h"
#include "read_shmim.h"
#include "read_shmimall.h"
#include "saveall.h"
#include "shmim_purge.h"
#include "shmim_setowner.h"
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
#include "variable_ID.h"








/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_memory)






static errno_t init_module_CLI()
{
	
	data.MEM_MONITOR = 0; // 1 if memory monitor is on
	
	
	clearall_addCLIcmd();
	list_image_addCLIcmd();   
	delete_image_addCLIcmd();
	
    //KEYWORDS                                                                                     
	image_keyword_addCLIcmd();

    // READ SHARED MEM IMAGE AND SIZE 
	read_shmim_addCLIcmd();

    // CREATE IMAGE
	create_image_addCLIcmd();

    // COPY IMAGE 
	image_copy_addCLIcmd();

	list_variable_addCLIcmd();
	
	
	
    // TYPE CONVERSIONS TO AND FROM COMPLEX
	image_complex_addCLIcmd();
    
    // SET IMAGE FLAGS / COUNTERS 
	image_set_counters_addCLIcmd();

    // MANAGE SEMAPHORES
	stream_sem_addCLIcmd();

	// STREAMS
	read_shmimall_addCLIcmd();
	shmim_purge_addCLIcmd();	
	shmim_setowner_addCLIcmd();
	
	stream_updateloop_addCLIcmd();
	stream_delay_addCLIcmd();
	saveall_addCLIcmd();
	stream__TCP_addCLIcmd();
	stream_pixmapdecode_addCLIcmd();
	stream_poke_addCLIcmd();
	stream_diff_addCLIcmd();
	stream_paste_addCLIcmd();
	stream_halfimdiff_addCLIcmd();
	stream_ave_addCLIcmd();

    // DATA LOGGING
	logshmim_addCLIcmd();


    // add atexit functions here

    return RETURN_SUCCESS;
}

