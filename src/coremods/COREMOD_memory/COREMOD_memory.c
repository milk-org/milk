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
#define MODULE_SHORTNAME_DEFAULT "mem"

// Module short description
#define MODULE_DESCRIPTION "Memory management for images and variables"

/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "CLIcore.h"

#    include "timeutils.h"

#    include "clearall.h"
#    include "create_image.h"
#    include "create_variable.h"

#    include "delete_image.h"
#    include "delete_sharedmem_image.h"

#    include "fps_ID.h"
#    include "fps_list.h"

#    include "im3D_to_stream2D.h"

#    include "image_ID.h"
#    include "image_complex.h"
#    include "image_copy.h"
#    include "image_copy_shm.h"
#    include "image_keyword.h"
#    include "image_keyword_add.h"
#    include "image_keyword_list.h"
#    include "image_make2D.h"
#    include "image_make3D.h"
#    include "image_mk_amph_from_complex.h"
#    include "image_mk_complex_from_amph.h"
#    include "image_mk_complex_from_reim.h"
#    include "image_mk_reim_from_complex.h"
#    include "image_set_counters.h"

#    include "list_image.h"
#    include "list_variable.h"
#    include "logshmim.h"

#    include "read_shmim.h"
#    include "read_shmim_size.h"
#    include "read_shmimall.h"

#    include "logshmim.h"

#    include "saveall.h"
#    include "shmim_purge.h"
#    include "shmim_setowner.h"
#    include "stream_TCP.h"
#    include "stream_UDP.h"
#    include "stream_ave.h"
#    include "stream_copy.h"
#    include "stream_delay.h"
#    include "stream_merge.h"
#    include "stream_diff.h"
#    include "stream_halfimdiff.h"
#    include "stream_monitorlimits.h"
#    include "stream_paste.h"
#    include "stream_pixmapdecode.h"
#    include "stream_poke.h"
#    include "stream_sem.h"
#    include "stream_updateloop.h"

#    include "variable_ID.h"

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_memory)

/**
 * @brief Register COREMOD_memory CLI commands.
 *
 * Called during module loading to add all memory
 * management operations to the CLI command table.
 */
static errno_t init_module_CLI()
{
    CLIADDCMD_COREMOD_memory__clearall();
    CLIADDCMD_COREMOD_memory__list_image();

    //KEYWORDS
    CLIADDCMD_COREMOD_memory__image_keyword();
    CLIADDCMD_COREMOD_memory__image_keyword_list();
    CLIADDCMD_COREMOD_memory__image_keyword_add();

    // READ SHARED MEM IMAGE AND SIZE
    CLIADDCMD_COREMOD_memory__read_sharedmem_image();
    CLIADDCMD_COREMOD_memory__read_sharedmem_image_size();
    CLIADDCMD_COREMOD_memory__read_shmimall();

    // CREATE IMAGE
    CLIADDCMD_COREMOD_memory__mk2Dim();
    CLIADDCMD_COREMOD_memory__mk3Dim();

    // COPY IMAGE
    CLIADDCMD_COREMOD_memory__image_copy();
    CLIADDCMD_COREMOD_memory__image_copy_shm();

    // DELETE IMAGE
    CLIADDCMD_COREMOD_memory__delete_image();
    CLIADDCMD_COREMOD_memory__delete_sharedmem_image();

    CLIADDCMD_COREMOD_memory__list_variable();

    // FPS
    CLIADDCMD_COREMOD_memory__fps_list();

    // TYPE CONVERSIONS TO AND FROM COMPLEX
    CLIADDCMD_COREMOD__mk_complex_from_reim();
    CLIADDCMD_COREMOD__mk_complex_from_amph();
    CLIADDCMD_COREMOD__mk_reim_from_complex();
    CLIADDCMD_COREMOD__mk_amph_from_complex();

    // SET IMAGE FLAGS / COUNTERS
    CLIADDCMD_COREMOD_memory__image_set_counters();

    // MANAGE SEMAPHORES
    CLIADDCMD_COREMOD_memory__stream_sem();

    // STREAMS
    CLIADDCMD_COREMOD_memory__shmim_purge();
    CLIADDCMD_COREMOD_memory__shmim_setowner();

    CLIADDCMD_COREMOD_memory__stream_updateloop();
    CLIADDCMD_COREMOD_memory__streamdelay();
    CLIADDCMD_COREMOD_memory__saveall();
    CLIADDCMD_COREMOD_memory__stream_TCP();
    CLIADDCMD_COREMOD_memory__stream_UDP();
    CLIADDCMD_COREMOD_memory__stream_pixmapdecode();

    CLIADDCMD_COREMOD_memory__stream_copy();
    CLIADDCMD_COREMOD_memory__stream_merge();
    CLIADDCMD_COREMOD_memory__stream_poke();
    CLIADDCMD_COREMOD_memory__im3D_to_stream2D();

    CLIADDCMD_COREMOD_memory__stream_diff();
    CLIADDCMD_COREMOD_memory__stream_paste();
    CLIADDCMD_COREMOD_memory__stream_halfimdiff();

    CLIADDCMD_streamaverage();
    stream_monitorlimits_addCLIcmd();

    // DATA LOGGING
    //logshmim_addCLIcmd();

    //CLIADDCMD_COREMOD_memory__shmimlog(); -- find deletion commit.
    CLIADDCMD_COREMOD_MEMORY__logshmim();

    // add atexit functions here

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
