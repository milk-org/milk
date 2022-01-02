/**
 * @file    fps_FPSremove.c
 * @brief   remove FPS
 */

#include "CommandLineInterface/CLIcore.h"





/** @brief remove FPS and associated files
 *
 * Requires CONF and RUN to be off
 *
 */
errno_t functionparameter_FPSremove(
    FUNCTION_PARAMETER_STRUCT *fps
)
{

    // get directory name
    char shmdname[STRINGMAXLEN_DIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    // get FPS shm filename
    char fpsfname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(fpsfname, "%s/%s.fps.shm", shmdname, fps->md->name);

    // delete sym links
    EXECUTE_SYSTEM_COMMAND("find %s -follow -type f -name \"fpslog.*%s\" -exec grep -q \"LOGSTART %s\" {} \\; -delete",
                           shmdname, fps->md->name, fps->md->name);

    fps->SMfd = -1;
    close(fps->SMfd);

    //    remove(conflogfname);
    int ret = remove(fpsfname);
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

    // terminate tmux sessions
    // 2x exit required: first one to exit bash, second one to exit tmux
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \"exit\" C-m",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \"exit\" C-m",
                           fps->md->name);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \"exit\" C-m",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \"exit\" C-m",
                           fps->md->name);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"exit\" C-m",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"exit\" C-m",
                           fps->md->name);

    return RETURN_SUCCESS;
}


