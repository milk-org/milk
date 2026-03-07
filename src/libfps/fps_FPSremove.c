/**
 * @file    fps_FPSremove.c
 * @brief   remove FPS
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"

/** @brief remove FPS and associated files
 *
 * Requires CONF and RUN to be off
 *
 */
errno_t functionparameter_FPSremove(FUNCTION_PARAMETER_STRUCT *fps)
{

    // get directory name
    char shmdname[STRINGMAXLEN_DIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    // get FPS shm filename
    char fpsfname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(fpsfname, "%s/%s.fps.shm", shmdname, fps->md->name);

    // delete sym links
    EXECUTE_SYSTEM_COMMAND(
        "find %s -follow -type f -name \"fpslog.*%s\" -exec grep -q \"LOGSTART "
        "%s\" {} \\; -delete",
        shmdname,
        fps->md->name,
        fps->md->name);

    fps->SMfd = -1;
    close(fps->SMfd);

    //    remove(conflogfname);
    int ret     = remove(fpsfname);
    int errcode = errno;
    (void) ret;
    (void) errcode;

    // TEST
    /*
    FILE *fp;
    fp = fopen("rmlist.txt", "a");
    fprintf(fp, "remove %s  %d\n", fpsfname, ret);
    if(ret == -1)
    {
        switch(errcode)
        {

            case EACCES:
                fprintf(fp, "EACCES\n");
                break;

            case EBUSY:
                fprintf(fp, "EBUSY\n");
                break;

            case ENOENT:
                fprintf(fp, "ENOENT\n");
                break;

            case EPERM:
                fprintf(fp, "EPERM\n");
                break;

            case EROFS:
                fprintf(fp, "EROFS\n");
                break;

        }
    }
    fclose(fp);
    */

    /* Terminate tmux session only if it exists.
     * Suppress has-session output; skip silently if absent.
     * 2x exit required: first exits bash-in-bash,
     * second exits tmux.
     */
    {
        char chkcmd[256];
        snprintf(chkcmd, sizeof(chkcmd),
                 "tmux has-session -t %s 2>/dev/null",
                 fps->md->name);
        if (system(chkcmd) == 0)
        {
            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:ctrl"
                " \" exit\" C-m",
                fps->md->name);
            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:ctrl"
                " \" exit\" C-m",
                fps->md->name);

            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:conf"
                " \" exit\" C-m",
                fps->md->name);
            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:conf"
                " \" exit\" C-m",
                fps->md->name);

            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:run"
                " \" exit\" C-m",
                fps->md->name);
            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:run"
                " \" exit\" C-m",
                fps->md->name);
        }
    }

    return RETURN_SUCCESS;
}
