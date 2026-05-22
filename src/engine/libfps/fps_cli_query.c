/**
 * @file    fps_cli_query.c
 * @brief   FPS "?" query handler
 *
 * Prints information about available FPS instances and
 * their parameter values. Used when the user types
 * "command ?" in the milk CLI.
 *
 * All listings are scoped to the compute unit
 * (app_info->fps_name) of the queried command.
 */


#include "CLIcore.h"


/**
 * @brief Print FPS query information ("?" mode).
 *
 * Lists all parameters, their types, defaults,
 * and current values. Used when the CLI receives
 * "?" as the FPS name.
 */
void fps_print_query_info(FPS_APP_INFO *app_info, FPS_CLI_BINDING *bindings, int nb_b)
{
    const char *fpsn = app_info->fps_name;

    printf("\n\033[1;36m=== FPS instances for "
           "%s ===\033[0m\n\n",
           app_info->cmdkey);

    /* ---- Local FPS (scoped to compute unit) ---- */
    {
        int local_count = 0;
        int n           = fps_local_count_entries();

        for (int ii = 0; ii < n; ii++)
        {
            FPS *lfps = fps_local_get_by_index(ii);
            if (lfps == NULL || lfps->md == NULL)
            {
                continue;
            }
            if (strcmp(lfps->md->name, "_defaults") == 0)
            {
                continue;
            }
            /* Only show FPS belonging to this
             * compute unit */
            const char *creator = fps_local_get_creator(ii);
            if (creator[0] != '\0' && strcmp(creator, fpsn) != 0)
            {
                continue;
            }
            if (local_count == 0)
            {
                printf("\033[1;33m  Local FPS "
                       "(in-process memory):"
                       "\033[0m\n");
            }
            printf("    \033[1;32m%-20s\033[0m  "
                   "%ld params\n",
                   lfps->md->name, lfps->NBparamActive);
            local_count++;
        }
        if (local_count == 0)
        {
            printf("  Local FPS : (none)\n");
        }
    }

    /* ---- Shared FPS in shm ---- */
    {
        char pattern[1024];
        snprintf(pattern, sizeof(pattern), "ls %s/*.fps.shm 2>/dev/null", dcshmdir);

        FILE *pp        = popen(pattern, "r");
        int   shm_count = 0;
        if (pp != NULL)
        {
            char line[512];
            while (fgets(line, sizeof(line), pp) != NULL)
            {
                line[strcspn(line, "\n")] = '\0';

                char *base = strrchr(line, '/');
                base       = base ? base + 1 : line;
                char *dot  = strstr(base, ".fps.shm");
                if (dot)
                {
                    *dot = '\0';
                }

                /* Only show FPS used by this
                 * compute unit */
                if (!fps_shared_was_used_by(base, fpsn))
                {
                    continue;
                }

                if (shm_count == 0)
                {
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
        if (shm_count == 0)
        {
            printf("  Shared FPS: (none)\n");
        }
    }

    /* ---- Parameter table ---- */
    printf("\n");

    FPS       *show_fps        = NULL;
    int        must_disconnect = 0;
    static FPS tmp_fps;

    /* Try last-used FPS if it belongs to this
     * compute unit */
    if (fps_last_used_name[0] != '\0' && strcmp(fps_last_used_cmdkey, fpsn) == 0)
    {
        if (fps_last_used_name[0] == '_')
        {
            show_fps = fps_local_find(fps_last_used_name);
        }
        else
        {
            if (fps_connect(fps_last_used_name, &tmp_fps, FPSCONNECT_SIMPLE) != -1)
            {
                show_fps        = &tmp_fps;
                must_disconnect = 1;
            }
        }
    }

    /* Fallback: most recent local FPS for this
     * compute unit */
    if (show_fps == NULL)
    {
        int n = fps_local_count_entries();
        for (int ii = n - 1; ii >= 0; ii--)
        {
            FPS *lfps = fps_local_get_by_index(ii);
            if (lfps == NULL || lfps->md == NULL)
            {
                continue;
            }
            if (strcmp(lfps->md->name, "_defaults") == 0)
            {
                continue;
            }
            const char *creator = fps_local_get_creator(ii);
            if (creator[0] != '\0' && strcmp(creator, fpsn) != 0)
            {
                continue;
            }
            show_fps = lfps;
            break;
        }
    }

    /* Try shared FPS if no local found */
    if (show_fps == NULL)
    {
        char try_name[200];
        if (data.processname[0] != '\0')
        {
            snprintf(try_name, sizeof(try_name), "%s.%s", fpsn, data.processname);
        }
        else
        {
            strncpy(try_name, fpsn, sizeof(try_name) - 1);
        }
        if (fps_connect(try_name, &tmp_fps, FPSCONNECT_SIMPLE) != -1)
        {
            show_fps        = &tmp_fps;
            must_disconnect = 1;
        }
    }

    /* Create temporary with defaults */
    if (show_fps == NULL)
    {
        show_fps = fps_local_create("_defaults", FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (show_fps != NULL)
        {
            fps_init_from_bindings(show_fps, app_info->cmdkey, app_info->description, bindings,
                                   nb_b);
            printf("\033[1;33m  Showing default"
                   " parameter values:\033[0m"
                   "\n\n");
        }
    }
    else
    {
        printf("\033[1;33m  Parameters for "
               "'%s':\033[0m\n\n",
               show_fps->md->name);
    }

    if (show_fps != NULL)
    {
        function_parameter_print_info(show_fps, 0, 0);
    }

    if (must_disconnect)
    {
        fps_disconnect(&tmp_fps);
    }
    /*
     * Note: _defaults FPS lives in the local
     * store and should NOT be freed here.
     */

    printf("\n");
}
