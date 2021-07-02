/**
 * @file    image_complex.c
 * @brief   complex number conversion
 */


#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "create_image.h"
#include "stream_sem.h"
#include "delete_image.h"






errno_t mk_reim_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *re_out_name,
    const char *im_out_name,
    int         sharedmem
)
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(
        mk_complex_from_amph(am_name, ph_name, "Ctmp", 0)
    );

    FUNC_CHECK_RETURN(
        mk_reim_from_complex("Ctmp", re_out_name, im_out_name, sharedmem)
    );

    FUNC_CHECK_RETURN(
        delete_image_ID("Ctmp", DELETE_IMAGE_ERRMODE_WARNING)
    );

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



errno_t mk_amph_from_reim(
    const char *re_name,
    const char *im_name,
    const char *am_out_name,
    const char *ph_out_name,
    int         sharedmem
)
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(
        mk_complex_from_reim(re_name, im_name, "Ctmp", 0)
    );

    FUNC_CHECK_RETURN(
        mk_amph_from_complex("Ctmp", am_out_name, ph_out_name, sharedmem)
    );

    FUNC_CHECK_RETURN(
        delete_image_ID("Ctmp", DELETE_IMAGE_ERRMODE_WARNING)
    );

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




