/**
 * @file    fps_cli_sync.c
 * @brief   Sync CLI arguments to FPS and local variables
 *
 * Core "unification" logic: takes CLI arguments (either from
 * a standalone executable or from the milk CLI) and syncs them
 * into FPS shared memory, then copies FPS values back into
 * module-local C variables via the binding array.
 */

#include <stdlib.h>
#include <string.h>

#ifndef FPS_STANDALONE
#include "CLIcore.h"
#else
#include "libmilkdata/milkdata.h"
#endif
#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_cli_binding.h"
#include "fps_cli_sync.h"


/* Standalone argc/argv captured by main() */
static int   standalone_argc;
static char **standalone_argv;


void fps_cli_set_standalone_args(
    int    argc,
    char **argv
)
{
    standalone_argc = argc;
    standalone_argv = argv;
}


/**
 * @brief Write a single CLI string value into an FPS
 *        parameter, interpreting it according to the type.
 */
static void set_fps_value_from_string(
    FPS *fps,
    long                       pindex,
    uint64_t                   type,
    const char                *str
)
{
    if (type == FPTYPE_FLOAT64) {
        fps->parray[pindex].val.f64[0] =
            atof(str);
    }
    else if (type == FPTYPE_FLOAT32) {
        fps->parray[pindex].val.f32[0] =
            (float) atof(str);
    }
    else if (type == FPTYPE_INT64) {
        fps->parray[pindex].val.i64[0] =
            atoll(str);
    }
    else if (type == FPTYPE_UINT64) {
        fps->parray[pindex].val.ui64[0] =
            (uint64_t) atoll(str);
    }
    else if (type == FPTYPE_INT32
             || type == FPTYPE_ONOFF) {
        fps->parray[pindex].val.i32[0] =
            atoi(str);
    }
    else if (type == FPTYPE_UINT32) {
        fps->parray[pindex].val.ui32[0] =
            (uint32_t) atoi(str);
    }
    else if (type == FPTYPE_PID) {
        fps->parray[pindex].val.pid[0] =
            (pid_t) atoi(str);
    }
    else if (type == FPTYPE_TIMESPEC) {
        double val = atof(str);
        fps->parray[pindex].val.ts[0].tv_sec =
            (long) val;
        fps->parray[pindex].val.ts[0].tv_nsec =
            (long) ((val - (long) val) * 1e9);
    }
    else if (FPTYPE_IS_STRING(type)) {
        strncpy(
            fps->parray[pindex].val.string[0],
            str,
            FUNCTION_PARAMETER_STRMAXLEN - 1);
    }
}


/**
 * @brief Copy a single FPS parameter value back into
 *        the module-local C variable via the binding.
 */
static void sync_fps_to_local(
    FPS *fps,
    long                       pindex,
    FPS_CLI_BINDING           *b
)
{
    if (b->type == FPTYPE_FLOAT64) {
        *((double *) b->ptr) =
            fps->parray[pindex].val.f64[0];
    }
    else if (b->type == FPTYPE_FLOAT32) {
        *((float *) b->ptr) =
            fps->parray[pindex].val.f32[0];
    }
    else if (b->type == FPTYPE_INT64) {
        *((int64_t *) b->ptr) =
            fps->parray[pindex].val.i64[0];
    }
    else if (b->type == FPTYPE_UINT64) {
        *((uint64_t *) b->ptr) =
            fps->parray[pindex].val.ui64[0];
    }
    else if (b->type == FPTYPE_INT32
             || b->type == FPTYPE_ONOFF) {
        *((int32_t *) b->ptr) =
            fps->parray[pindex].val.i32[0];
    }
    else if (b->type == FPTYPE_UINT32) {
        *((uint32_t *) b->ptr) =
            fps->parray[pindex].val.ui32[0];
    }
    else if (b->type == FPTYPE_PID) {
        *((pid_t *) b->ptr) =
            fps->parray[pindex].val.pid[0];
    }
    else if (b->type == FPTYPE_TIMESPEC) {
        *((struct timespec *) b->ptr) =
            fps->parray[pindex].val.ts[0];
    }
    else if (FPTYPE_IS_STRING(b->type)) {
        strncpy(
            (char *) b->ptr,
            fps->parray[pindex].val.string[0],
            FUNCTION_PARAMETER_STRMAXLEN - 1);
        ((char *) b->ptr)[
            FUNCTION_PARAMETER_STRMAXLEN - 1]
            = '\0';
    }
}


errno_t fps_process_cli_and_sync(
    FPS *fps,
    CLICMDARGDEF              *farg,
    FPS_CLI_BINDING           *bindings,
    int                        nb_b
)
{
    /* ---- Step 1: CLI → FPS ---- */
    if (standalone_argv != NULL) {
        /*
         * MODE B: standalone executable.
         * Find the subcommand position, then map
         * positional args after it to primary bindings.
         */
        int cmd_pos = -1;
        for (int j = 1; j < standalone_argc; j++) {
            /* Extract command part after optional
             * 'name:' prefix */
            const char *arg = standalone_argv[j];
            const char *cmd_part = arg;
            {
                const char *cp = strchr(arg, ':');
                if (cp != NULL)
                    cmd_part = cp + 1;
            }
            if (strcmp(cmd_part, "runstart") == 0 ||
                strcmp(cmd_part, "run") == 0 ||
                strcmp(cmd_part, "exec") == 0 ||
                strcmp(cmd_part, "set") == 0 ||
                strcmp(cmd_part, "confstart") == 0 ||
                strcmp(cmd_part, "confstep") == 0 ||
                strcmp(cmd_part, "fpsinit") == 0)
            {
                cmd_pos = j;
                break;
            }
        }

        /* If no explicit command, try implicit run
         * (first non-flag arg) */
        if (cmd_pos == -1) {
            for (int j = 1;
                 j < standalone_argc; j++)
            {
                if (standalone_argv[j][0] != '-') {
                    cmd_pos = j;
                    break;
                }
            }
        }

        if (cmd_pos != -1) {
            int cli_idx = 0;
            for (int a = cmd_pos + 1;
                 a < standalone_argc; a++)
            {
                /* Skip -flag arguments */
                if (standalone_argv[a][0] == '-')
                    continue;

                /* Map to next primary binding */
                while (cli_idx < nb_b &&
                       !bindings[cli_idx].is_primary)
                    cli_idx++;
                if (cli_idx >= nb_b)
                    break;

                long pindex =
                    functionparameter_GetParamIndex(
                        fps,
                        bindings[cli_idx]
                            .fpskeyword);
                if (pindex != -1 &&
                    strcmp(standalone_argv[a],
                           ".") != 0) {
                    set_fps_value_from_string(
                        fps, pindex,
                        bindings[cli_idx].type,
                        standalone_argv[a]);
                }
                cli_idx++;
            }
        }
    }
    else {
        /*
         * MODE A: running as module in milk CLI.
         * Sync from CLI argdata filled by
         * CLI_checkarg_array.
         */
#ifndef FPS_STANDALONE
        CLIargs_to_FPSparams_setval(
            farg, nb_b, fps);
#else
        (void) farg;
        fprintf(stderr,
            "ERROR: CLI sync not available"
            " in standalone mode\n");
#endif
    }

    /* ---- Step 2: FPS → local C variables ---- */
    for (int i = 0; i < nb_b; i++) {
        long pindex =
            functionparameter_GetParamIndex(
                fps, bindings[i].fpskeyword);

        if (pindex != -1) {
            sync_fps_to_local(
                fps, pindex, &bindings[i]);
        }
    }

    return RETURN_SUCCESS;
}
