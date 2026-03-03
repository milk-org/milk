/**
 * @file    fps_cli_query.c
 * @brief   FPS "?" query handler
 *
 * Prints information about available FPS instances and
 * their parameter values. Used when the user types
 * "command ?" in the milk CLI.
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_init.h"
#include "fps_cli_query.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_local_store.h"
#include "fps_print_info.h"


void fps_print_query_info(
    FPS_APP_INFO    *app_info,
    FPS_CLI_BINDING *bindings,
    int              nb_b
)
{
    printf("\n\033[1;36m=== FPS instances for "
           "%s ===\033[0m\n\n",
           app_info->cmdkey);

    /* ---- Local FPS instances ---- */
    {
        int local_count = 0;

        char pattern[300];
        snprintf(pattern, sizeof(pattern),
                 "_%s", app_info->fps_name);

        FUNCTION_PARAMETER_STRUCT *lfps =
            fps_local_find(pattern);
        if (lfps != NULL && lfps->md != NULL) {
            printf("\033[1;33m  Local FPS "
                   "(in-process memory):\033[0m\n");
            printf("    \033[1;32m%-20s\033[0m  "
                   "%ld params\n",
                   lfps->md->name,
                   lfps->NBparamActive);
            local_count = 1;
        }
        if (local_count == 0) {
            printf("  Local FPS : (none)\n");
        }
    }

    /* ---- Shared FPS in shm ---- */
    {
        char pattern[300];
        snprintf(pattern, sizeof(pattern),
                 "ls %s/*.fps.shm 2>/dev/null",
                 data.shmdir);

        FILE *pp = popen(pattern, "r");
        int shm_count = 0;
        if (pp != NULL) {
            char line[512];
            while (fgets(line, sizeof(line), pp)
                   != NULL)
            {
                line[strcspn(line, "\n")] = '\0';

                char *base = strrchr(line, '/');
                base = base ? base + 1 : line;
                char *dot =
                    strstr(base, ".fps.shm");
                if (dot) {
                    *dot = '\0';
                }

                if (shm_count == 0) {
                    printf("\n\033[1;33m  Shared FPS"
                           " (shm):\033[0m\n");
                }
                printf("    \033[1;32m%-20s\033[0m"
                       "  %s\n",
                       base, line);
                shm_count++;
            }
            pclose(pp);
        }
        if (shm_count == 0) {
            printf("  Shared FPS: (none)\n");
        }
    }

    /* ---- Parameter table ---- */
    printf("\n");

    FUNCTION_PARAMETER_STRUCT *show_fps = NULL;
    int must_disconnect = 0;

    /* Check local FPS */
    {
        char try_local[200];
        snprintf(try_local, sizeof(try_local),
                 "_%s", app_info->fps_name);
        show_fps = fps_local_find(try_local);
    }

    /* Try shared FPS if no local found */
    static FUNCTION_PARAMETER_STRUCT tmp_fps;
    if (show_fps == NULL) {
        char try_name[200];
        if (data.processname[0] != '\0') {
            snprintf(try_name, sizeof(try_name),
                     "%s.%s",
                     app_info->fps_name,
                     data.processname);
        } else {
            strncpy(try_name,
                    app_info->fps_name,
                    sizeof(try_name) - 1);
        }
        if (function_parameter_struct_connect(
                try_name, &tmp_fps,
                FPSCONNECT_SIMPLE) != -1)
        {
            show_fps = &tmp_fps;
            must_disconnect = 1;
        }
    }

    /* Create temporary with defaults */
    if (show_fps == NULL) {
        show_fps = fps_local_create(
            "_defaults",
            FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (show_fps != NULL) {
            fps_init_from_bindings(
                show_fps,
                app_info->cmdkey,
                app_info->description,
                bindings,
                nb_b);
            printf("\033[1;33m  Showing default"
                   " parameter values:\033[0m"
                   "\n\n");
        }
    } else {
        printf("\033[1;33m  Parameters for "
               "'%s':\033[0m\n\n",
               show_fps->md->name);
    }

    if (show_fps != NULL) {
        function_parameter_print_info(
            show_fps, 0, 0);
    }

    if (must_disconnect) {
        function_parameter_struct_disconnect(
            &tmp_fps);
    }
    /*
     * Note: _defaults FPS lives in the local
     * store and should NOT be freed here.
     */

    printf("\n");
}
