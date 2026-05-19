/**
 * @file CLIcore_datainit.c
 *
 * @brief data structure init
 *
 * Delegates core array allocation to milk_data_init()
 * and syncs data.core = milk_data so both globals share
 * the same image/variable/FPS arrays.
 */

#include <math.h>
#include <sys/time.h>

#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

/*
 * Initialize the "data" structure.
 *
 * Core fields (image, variable, FPS arrays) are
 * allocated by milk_data_init(), then copied into
 * data.core so both DATA.core and milk_data refer
 * to the same storage.
 *
 * Calls milk_data_init() to allocate core arrays
 * (images, variables, FPS), then creates built-in
 * mathematical constants (_PI, _e, _c, etc.) and
 * seeds the random number generator.
 */
errno_t CLI_data_init()
{
    DEBUG_TRACE_FSTART();

    struct timeval t1;

    /* Allocate core arrays (image, variable, FPS) */
    milk_data_init();

    /* Sync: data.core shares milk_data's arrays */
    data.core = milk_data;

    create_variable_ID("_PI", 3.14159265358979323846264338328);
    create_variable_ID("_e", exp(1));
    create_variable_ID("_gamma", 0.5772156649);
    create_variable_ID("_c", 299792458.0);
    create_variable_ID("_h", 6.626075540e-34);
    create_variable_ID("_k", 1.38065812e-23);
    create_variable_ID("_pc", 3.0856776e16);
    create_variable_ID("_ly", 9.460730472e15);
    create_variable_ID("_AU", 1.4959787066e11);

    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
