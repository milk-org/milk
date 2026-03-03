/**
 * @file    fps_local_store.h
 * @brief   In-process (local) FPS instance management
 *
 * Manages local FPS instances that live in process memory
 * rather than shared memory. Local FPS names start with '_'.
 */

#ifndef FPS_LOCAL_STORE_H
#define FPS_LOCAL_STORE_H

#include "fps.h"

/** Maximum concurrent local FPS instances */
#define FPS_LOCAL_MAX 64


/**
 * @brief Find an existing local FPS by name.
 *
 * @param name  FPS name to search for
 * @return      Pointer to the FPS, or NULL if not found
 */
FUNCTION_PARAMETER_STRUCT *fps_local_find(
    const char *name
);


/**
 * @brief Allocate a new local FPS slot.
 *
 * @param name        FPS name
 * @param NBparamMAX  Maximum number of parameters
 * @return            Pointer to the new slot, or NULL if full
 */
FUNCTION_PARAMETER_STRUCT *fps_local_create(
    const char *name,
    long        NBparamMAX
);


/**
 * @brief Find or create a local FPS by name.
 *
 * @param name        FPS name
 * @param NBparamMAX  Max params (used only if creating)
 * @return            Pointer to the FPS, or NULL on failure
 */
FUNCTION_PARAMETER_STRUCT *fps_local_get_or_create(
    const char *name,
    long        NBparamMAX
);


#endif /* FPS_LOCAL_STORE_H */
