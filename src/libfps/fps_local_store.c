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

/** Creator fps_name for each slot */
static char
    local_fps_creator[FPS_LOCAL_MAX]
                     [FPS_CREATOR_NAME_MAX];


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
    local_fps_creator[idx][0] = '\0';

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


int fps_local_count_entries(void)
{
    return local_fps_count;
}


FUNCTION_PARAMETER_STRUCT *fps_local_get_by_index(
    int idx
)
{
    if (idx < 0 || idx >= local_fps_count) {
        return NULL;
    }
    if (!local_fps_used[idx]) {
        return NULL;
    }
    return &local_fps_array[idx];
}


void fps_local_set_creator(
    const char *name,
    const char *creator_name
)
{
    if (name == NULL || creator_name == NULL) {
        return;
    }
    for (int i = 0; i < local_fps_count; i++) {
        if (local_fps_used[i] &&
            strncmp(local_fps_array[i].md->name,
                    name,
                    FPS_PNAME_STRMAXLEN - 1) == 0)
        {
            strncpy(local_fps_creator[i],
                    creator_name,
                    FPS_CREATOR_NAME_MAX - 1);
            local_fps_creator[i][
                FPS_CREATOR_NAME_MAX - 1] = '\0';
            return;
        }
    }
}


const char *fps_local_get_creator(int idx)
{
    if (idx < 0 || idx >= local_fps_count) {
        return "";
    }
    if (!local_fps_used[idx]) {
        return "";
    }
    return local_fps_creator[idx];
}


/* ---- Shared FPS usage tracking ---- */

static int shared_track_count;

static char
    shared_track_fps[FPS_SHARED_TRACK_MAX]
                    [FPS_CREATOR_NAME_MAX];
static char
    shared_track_creator[FPS_SHARED_TRACK_MAX]
                        [FPS_CREATOR_NAME_MAX];


void fps_shared_record_usage(
    const char *fps_name,
    const char *creator_name
)
{
    if (fps_name == NULL ||
        creator_name == NULL)
    {
        return;
    }
    /* Check if already recorded */
    for (int i = 0; i < shared_track_count; i++) {
        if (strcmp(shared_track_fps[i],
                  fps_name) == 0 &&
            strcmp(shared_track_creator[i],
                  creator_name) == 0)
        {
            return;
        }
    }
    if (shared_track_count
        >= FPS_SHARED_TRACK_MAX)
    {
        return;
    }
    int idx = shared_track_count++;
    strncpy(shared_track_fps[idx],
            fps_name,
            FPS_CREATOR_NAME_MAX - 1);
    shared_track_fps[idx][
        FPS_CREATOR_NAME_MAX - 1] = '\0';
    strncpy(shared_track_creator[idx],
            creator_name,
            FPS_CREATOR_NAME_MAX - 1);
    shared_track_creator[idx][
        FPS_CREATOR_NAME_MAX - 1] = '\0';
}


int fps_shared_was_used_by(
    const char *fps_name,
    const char *creator_name
)
{
    if (fps_name == NULL ||
        creator_name == NULL)
    {
        return 0;
    }
    for (int i = 0; i < shared_track_count; i++) {
        if (strcmp(shared_track_fps[i],
                  fps_name) == 0 &&
            strcmp(shared_track_creator[i],
                  creator_name) == 0)
        {
            return 1;
        }
    }
    return 0;
}
