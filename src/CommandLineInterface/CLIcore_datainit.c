/**
 * @file CLIcore_datainit.c
 *
 * @brief data structure init
 *
 */

#include <math.h>
#include <sys/time.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

/*^-----------------------------------------------------------------------------
|  Initialization the "data" structure
|
|
|
|
+-----------------------------------------------------------------------------*/
errno_t CLI_data_init()
{
    DEBUG_TRACE_FSTART();

    long tmplong;
    //  int i;
    struct timeval t1;

    /* initialization of the data structure
     */
    data.NB_MAX_IMAGE    = STATIC_NB_MAX_IMAGE;
    data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
    data.NB_MAX_FPS      = 100;
    data.INVRANDMAX      = 1.0 / RAND_MAX;

    // do not remove files when delete command on SHM
    data.rmSHMfile = 0;

    // Allocate data.image

#ifdef DATA_STATIC_ALLOC
    // image static allocation mode
    data.NB_MAX_IMAGE = STATIC_NB_MAX_IMAGE;
    printf("STATIC ALLOCATION mode: set data.NB_MAX_IMAGE      = %5ld\n",
           data.NB_MAX_IMAGE);
#else
    data.image = (IMAGE *) calloc(data.NB_MAX_IMAGE, sizeof(IMAGE));
    if (data.image == NULL)
        {
            PRINT_ERROR(
                "Allocation of data.image has failed - exiting program");
            exit(1);
        }
    if (data.Debug > 0)
        {
            printf("Allocation of data.image completed %p\n", data.image);
            fflush(stdout);
        }
#endif

    for (long i = 0; i < data.NB_MAX_IMAGE; i++)
        {
            data.image[i].used      = 0;
            data.image[i].createcnt = 0;
        }

        // Allocate data.variable

#ifdef DATA_STATIC_ALLOC
    // variable static allocation mode
    data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
    printf("STATIC ALLOCATION mode: set data.NB_MAX_VARIABLE   = %5ld\n",
           data.NB_MAX_VARIABLE);
#else
    data.variable = (VARIABLE *) calloc(data.NB_MAX_VARIABLE, sizeof(VARIABLE));
    if (data.variable == NULL)
        {
            PRINT_ERROR(
                "Allocation of data.variable has failed - exiting program");
            exit(1);
        }

    data.image[0].used   = 0;
    data.image[0].shmfd  = -1;
    tmplong              = data.NB_MAX_VARIABLE;
    data.NB_MAX_VARIABLE = data.NB_MAX_VARIABLE + NB_VARIABLES_BUFFER_REALLOC;

    data.variable =
        (VARIABLE *) realloc(data.variable,
                             data.NB_MAX_VARIABLE * sizeof(VARIABLE));
    for (long i = tmplong; i < data.NB_MAX_VARIABLE; i++)
        {
            data.variable[i].used = 0;
            data.variable[i].type = 0; /** defaults to floating point type */
        }

    if (data.variable == NULL)
        {
            PRINT_ERROR(
                "Reallocation of data.variable has failed - exiting program");
            exit(1);
        }
#endif

    // Allocate data.fps
    data.fpsarray = malloc(sizeof(FUNCTION_PARAMETER_STRUCT) * data.NB_MAX_FPS);
    if (data.fpsarray == NULL)
        {
            FUNC_RETURN_FAILURE("memory allocation error");
        }

    // Initialize file descriptors to -1
    //
    for (int fpsindex = 0; fpsindex < data.NB_MAX_FPS; fpsindex++)
        {
            data.fpsarray[fpsindex].SMfd   = -1;
            data.fpsarray[fpsindex].md     = NULL;
            data.fpsarray[fpsindex].parray = NULL;
        }

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
    //	printf("RAND: %ld\n", t1.tv_usec * t1.tv_sec);
    //  srand(time(NULL));

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
