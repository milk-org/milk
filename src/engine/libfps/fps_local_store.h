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

/** Max length of creator fps_name tag */
#define FPS_CREATOR_NAME_MAX 200


/**
 * @brief Find an existing local FPS by name.
 *
 * @param name  FPS name to search for
 * @return      Pointer to the FPS, or NULL
 */
FUNCTION_PARAMETER_STRUCT *fps_local_find(
    const char *name
);


/**
 * @brief Allocate a new local FPS slot.
 *
 * @param name        FPS name
 * @param NBparamMAX  Maximum number of parameters
 * @return            Pointer to the new slot, or NULL
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
 * @return            Pointer to the FPS, or NULL
 */
FUNCTION_PARAMETER_STRUCT *fps_local_get_or_create(
    const char *name,
    long        NBparamMAX
);


/**
 * @brief Get the number of local FPS entries.
 */
int fps_local_count_entries(void);


/**
 * @brief Get a local FPS entry by index.
 *
 * @param idx   Index (0-based)
 * @return      Pointer to FPS, or NULL if unused
 */
FUNCTION_PARAMETER_STRUCT *fps_local_get_by_index(
    int idx
);


/**
 * @brief Tag a local FPS with its creator fps_name.
 *
 * @param name          FPS name (local store key)
 * @param creator_name  The app_info->fps_name
 */
void fps_local_set_creator(
    const char *name,
    const char *creator_name
);


/**
 * @brief Get the creator fps_name for a local FPS.
 *
 * @param idx   Index (0-based)
 * @return      Creator string, or "" if unknown
 */
const char *fps_local_get_creator(int idx);


/** Max tracked shared FPS per compute unit */
#define FPS_SHARED_TRACK_MAX 64


/**
 * @brief Record that a shared FPS was used by
 *        a compute unit.
 *
 * @param fps_name     Shared FPS name
 * @param creator_name app_info->fps_name
 */
void fps_shared_record_usage(
    const char *fps_name,
    const char *creator_name
);


/**
 * @brief Check if a shared FPS was used by a
 *        specific compute unit.
 *
 * @param fps_name     Shared FPS name
 * @param creator_name app_info->fps_name
 * @return             1 if yes, 0 if no
 */
int fps_shared_was_used_by(
    const char *fps_name,
    const char *creator_name
);


#endif /* FPS_LOCAL_STORE_H */
