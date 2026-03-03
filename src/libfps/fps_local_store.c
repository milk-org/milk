/**
 * @file    fps_local_store.c
 * @brief   In-process (local) FPS instance management
 *
 * Provides allocation and lookup for local FPS instances
 * that live in process memory (not shared memory).
 * Local FPS names are prefixed with '_'.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fps.h"
#include "fps_local_store.h"


/** Local FPS store: array, usage flags, count */
static int local_fps_count;
static FUNCTION_PARAMETER_STRUCT
    local_fps_array[FPS_LOCAL_MAX];
static int local_fps_used[FPS_LOCAL_MAX];


FUNCTION_PARAMETER_STRUCT *fps_local_find(
    const char *name
)
{
    if (name == NULL) {
        return NULL;
    }
    for (int i = 0; i < local_fps_count; i++) {
        if (local_fps_used[i] &&
            strncmp(local_fps_array[i].md->name,
                    name,
                    FPS_PNAME_STRMAXLEN - 1) == 0)
        {
            return &local_fps_array[i];
        }
    }
    return NULL;
}


FUNCTION_PARAMETER_STRUCT *fps_local_create(
    const char *name,
    long        NBparamMAX
)
{
    if (local_fps_count >= FPS_LOCAL_MAX) {
        fprintf(stderr,
                "ERROR: local FPS store full "
                "(max %d)\n",
                FPS_LOCAL_MAX);
        return NULL;
    }

    int idx = local_fps_count++;
    local_fps_used[idx] = 1;

    FUNCTION_PARAMETER_STRUCT *fps =
        &local_fps_array[idx];

    memset(fps, 0,
           sizeof(FUNCTION_PARAMETER_STRUCT));
    fps->SMfd = -1;

    /* Allocate metadata */
    fps->md = (FUNCTION_PARAMETER_STRUCT_MD *)
        calloc(1,
            sizeof(FUNCTION_PARAMETER_STRUCT_MD));
    if (fps->md == NULL) {
        fprintf(stderr,
                "ERROR: calloc md for '%s'\n",
                name);
        local_fps_used[idx] = 0;
        local_fps_count--;
        return NULL;
    }

    strncpy(fps->md->name, name,
            FPS_PNAME_STRMAXLEN - 1);
    fps->md->NBparamMAX = NBparamMAX;

    /* Allocate parameter array */
    fps->parray = (FUNCTION_PARAMETER *)
        calloc(NBparamMAX,
            sizeof(FUNCTION_PARAMETER));
    if (fps->parray == NULL) {
        fprintf(stderr,
                "ERROR: calloc parray for '%s'\n",
                name);
        free(fps->md);
        fps->md = NULL;
        local_fps_used[idx] = 0;
        local_fps_count--;
        return NULL;
    }

    return fps;
}


FUNCTION_PARAMETER_STRUCT *fps_local_get_or_create(
    const char *name,
    long        NBparamMAX
)
{
    FUNCTION_PARAMETER_STRUCT *fps =
        fps_local_find(name);

    if (fps != NULL) {
        return fps;
    }
    return fps_local_create(name, NBparamMAX);
}
