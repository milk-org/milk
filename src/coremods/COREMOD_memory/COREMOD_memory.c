// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

#include "milk_runnable.h"

#include "timeutils.h"

#include "clearall.h"
#include "create_image.h"
#include "create_variable.h"

#include "delete_image.h"
#include "delete_sharedmem_image.h"

#include "fps_ID.h"
#include "fps_list.h"

#include "im3D_to_stream2D.h"

#include "image_ID.h"
#include "image_complex.h"
#include "image_copy.h"
#include "image_copy_shm.h"
#include "image_keyword.h"
#include "image_keyword_add.h"
#include "image_keyword_list.h"
#include "image_make2D.h"
#include "image_make3D.h"
#include "image_mk_amph_from_complex.h"
#include "image_mk_complex_from_amph.h"
#include "image_mk_complex_from_reim.h"
#include "image_mk_reim_from_complex.h"
#include "image_set_counters.h"

#include "list_image.h"
#include "list_variable.h"
#include "logshmim.h"

#include "read_shmim.h"
#include "read_shmim_size.h"
#include "read_shmimall.h"

#include "saveall.h"
#include "shmim_purge.h"
#include "shmim_setowner.h"
#include "stream_TCP.h"
#include "stream_UDP.h"
#include "stream_ave.h"
#include "stream_copy.h"
#include "stream_delay.h"
#include "stream_merge.h"
#include "stream_diff.h"
#include "stream_halfimdiff.h"
#include "stream_monitorlimits.h"
#include "stream_paste.h"
#include "stream_pixmapdecode.h"
#include "stream_poke.h"
#include "stream_sem.h"
#include "stream_updateloop.h"

#include "variable_ID.h"

/* ================================================================== */
/* ================================================================== */
/*            MODULE REGISTRY                                         */
/* ================================================================== */
/* ================================================================== */

// CLI exposure = membership here; registration order is the order of this array.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_COREMOD_memory_clearall,
    &runnable_COREMOD_memory_listim,

    // KEYWORDS
    &runnable_COREMOD_memory_imlistkw,
    &runnable_COREMOD_memory_imwritekwL,
    &runnable_COREMOD_memory_image_keyword_list,
    &runnable_COREMOD_memory_image_keyword_add,

    // READ SHARED MEM IMAGE AND SIZE
    &runnable_COREMOD_memory_readshmim,
    &runnable_COREMOD_memory_readshmimsize,
    &runnable_COREMOD_memory_readshmimall,

    // CREATE IMAGE
    &runnable_COREMOD_memory_image_make2D,
    &runnable_COREMOD_memory_image_make3D,

    // COPY IMAGE
    &runnable_COREMOD_memory_cp,
    &runnable_COREMOD_memory_mv,
    &runnable_COREMOD_memory_imcp2shm,
    &runnable_COREMOD_memory_image_copy_shm,

    // DELETE IMAGE
    &runnable_COREMOD_memory_delete_image,
    &runnable_COREMOD_memory_delete_sharedmem_image,

    &runnable_COREMOD_memory_listvar,
    &runnable_COREMOD_memory_listvarf,

    // FPS
    &runnable_COREMOD_memory_fps_list,

    // TYPE CONVERSIONS TO AND FROM COMPLEX
    &runnable_COREMOD_memory_image_mk_complex_from_reim,
    &runnable_COREMOD_memory_image_mk_complex_from_amph,
    &runnable_COREMOD_memory_image_mk_reim_from_complex,
    &runnable_COREMOD_memory_image_mk_amph_from_complex,

    // SET IMAGE FLAGS / COUNTERS
    &runnable_COREMOD_memory_imsetstatus,
    &runnable_COREMOD_memory_imsetcnt0,
    &runnable_COREMOD_memory_imsetcnt1,

    // MANAGE SEMAPHORES
    &runnable_COREMOD_memory_imseminfo,
    &runnable_COREMOD_memory_imsetsempost,
    &runnable_COREMOD_memory_imsetsempostl,
    &runnable_COREMOD_memory_imsetsemwait,
    &runnable_COREMOD_memory_imsetsemflush,

    // STREAMS
    &runnable_COREMOD_memory_shmimpurge,
    &runnable_COREMOD_memory_shmimsetowncreator,
    &runnable_COREMOD_memory_shmimsetowncurrent,
    &runnable_COREMOD_memory_shmimsetowninit,

    &runnable_COREMOD_memory_streamburst,
    &runnable_COREMOD_memory_creaimstream,
    &runnable_COREMOD_memory_creaimstreamstrig,
    &runnable_COREMOD_memory_streamdelay,
    &runnable_COREMOD_memory_imsaveallsnap,
    &runnable_COREMOD_memory_imsaveallseq,
    &runnable_COREMOD_memory_imnetwtransmit,
    &runnable_COREMOD_memory_imnetwreceive,
    &runnable_COREMOD_memory_imudptransmit,
    &runnable_COREMOD_memory_imudpreceive,
    &runnable_COREMOD_memory_impixdecodeU,

    &runnable_COREMOD_memory_shmimcopy,
    &runnable_COREMOD_memory_shmimmerge,
    &runnable_COREMOD_memory_shmimpoke,
    &runnable_COREMOD_memory_im3D_to_stream2D,

    &runnable_COREMOD_memory_streamdiff,
    &runnable_COREMOD_memory_streampaste,
    &runnable_COREMOD_memory_streamhalfdiff,

    &runnable_COREMOD_memory_streamave,
    &runnable_COREMOD_memory_streammlim,

    // DATA LOGGING
    &runnable_COREMOD_memory_logshmim,
};

MILK_MODULE_RUNNABLES(COREMOD_memory, NULL, runnables);
