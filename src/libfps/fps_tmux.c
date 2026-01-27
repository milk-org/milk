/**
 * @file    fps_tmux.c
 *
 * @brief   tmux session management
 *
 * tmux wrapper for FPS control
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"




/** @brief Kill FPS tmux sesssion
 *
 */
errno_t functionparameter_FPS_tmux_kill(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    // terminate tmux sessions
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl C-c 2> /dev/null",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND(
        "tmux send-keys -t %s:ctrl \"exit\" C-m 2> /dev/null",
        fps->md->name);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf C-c 2> /dev/null",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND(
        "tmux send-keys -t %s:conf \"exit\" C-m 2> /dev/null",
        fps->md->name);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run C-c 2> /dev/null",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"exit\" C-m 2> /dev/null",
                           fps->md->name);

    EXECUTE_SYSTEM_COMMAND("tmux kill-session -t %s 2> /dev/null",
                           fps->md->name);

    return RETURN_SUCCESS;
}

errno_t functionparameter_FPS_tmux_attach(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    // This should hang until the tmux is detached,
    // and then return to the current fpsCTRL window.
    EXECUTE_SYSTEM_COMMAND("tmux attach -t %s", fps->md->name);
    return RETURN_SUCCESS;
}



/** @brief Initialize FPS tmux sesssion
 *
 */
errno_t functionparameter_FPS_tmux_init(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    int funcstring_maxlen  = 10000;
    int argstring_maxlen   = 1000;
    int mloadstring_maxlen = 2000;

    // delay to allow for tmux commands to be completed
    float tmuxwait = 0.1;

    // terminate tmux sessions
    functionparameter_FPS_tmux_kill(fps);

    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux kill-session -t %s 2> /dev/null",
                           fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux new-session -s %s -d",
                           fps->md->name);


    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux rename-window -t %s:0 ctrl", fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux new-window -t %s -n conf", fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux new-window -t %s -n run", fps->md->name);
    sleep(tmuxwait);


    // Write functions to tmux windows
    //
    char functionstring[funcstring_maxlen];
    char argstring[argstring_maxlen];
    char argstringcp[argstring_maxlen];

    if(fps->md->NBnameindex > 0)
    {
        snprintf(argstring, argstring_maxlen, "%s", fps->md->nameindexW[0]);
    }
    else
    {
        snprintf(argstring, argstring_maxlen, " ");
    }

    for(int i = 1; i < fps->md->NBnameindex; i++)
    {
        snprintf(argstringcp,
                 argstring_maxlen,
                 "%s %s",
                 argstring,
                 fps->md->nameindexW[i]);
        strcpy(argstring, argstringcp);
    }

    // module load string
    char mloadstring[mloadstring_maxlen];
    char mloadstringcp[mloadstring_maxlen];
    snprintf(mloadstring, mloadstring_maxlen, " ");
    for(int m = 0; m < fps->md->NBmodule; m++)
    {
        snprintf(mloadstringcp,
                 mloadstring_maxlen,
                 "%smload %s;",
                 mloadstring,
                 fps->md->modulename[m]);
        strcpy(mloadstring, mloadstringcp);
    }

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \" bash\" C-m",
                           fps->md->name); // This spins a bash-in-bash.
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \" cd %s\" C-m",
                           fps->md->name, fps->md->workdir);

    // source rootdir fpstmuxenv first
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \" source ../fpstmuxenv\" C-m",
                           fps->md->name);
    // then local fpstmuxenv
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \" source fpstmuxenv\" C-m",
                           fps->md->name);


    // confstart
    //
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \" bash\" C-m",
                           fps->md->name); // This spins a bash-in-bash.
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \" cd %s\" C-m",
                           fps->md->name, fps->md->workdir);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \" source ../fpstmuxenv\" C-m",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \" source  fpstmuxenv\" C-m",
                           fps->md->name);


    char progexec[1024];
    if( (strlen(fps->md->execfullpath) > 0) && (strcmp(fps->md->execfullpath, "unknown") != 0) )
    {
        strncpy(progexec, fps->md->execfullpath, 1023);
    }
    else
    {
        snprintf(progexec, 1024, "%s-exec", fps->md->callprogname);
    }

    snprintf(functionstring,
             funcstring_maxlen,
             " function fpsconfstart {\n"
             "echo \"STARTING CONF PROCESS\"\n"
             "MILK_FPSPROCINFO=1 %s -n %s \\\"%s%s _CONFSTART_ %s\\\"\n"
             "}\n",
             progexec,
             fps->md->name,
             mloadstring,
             fps->md->callfuncname,
             argstring);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \" %s\" C-m",
                           fps->md->name,
                           functionstring);

    // runstart
    //
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \" bash\" C-m",
                           fps->md->name); // This spins a bash-in-bash.
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \" cd %s\" C-m",
                           fps->md->name, fps->md->workdir);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \" source ../fpstmuxenv\" C-m",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \" source fpstmuxenv\" C-m",
                           fps->md->name);

    snprintf(functionstring,
             funcstring_maxlen,
             " function fpsrunstart {\n"
             "echo \"STARTING RUN PROCESS\"\n"
             "MILK_FPSPROCINFO=1 %s -n %s \\\"\\${TCSETCMDPREFIX} %s%s "
             "_RUNSTART_ %s\\\"\n"
             "}\n",
             progexec,
             fps->md->name,
             mloadstring,
             fps->md->callfuncname,
             argstring);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"%s\" C-m",
                           fps->md->name,
                           functionstring);

    // runstop
    //
    snprintf(functionstring,
             funcstring_maxlen,
             " function fpsrunstop {\n"
             "echo \"STOPPING RUN PROCESS\"\n"
             "%s -n %s \\\"%s%s _RUNSTOP_ %s\\\"\n"
             "}\n",
             progexec,
             fps->md->name,
             mloadstring,
             fps->md->callfuncname,
             argstring);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"%s\" C-m",
                           fps->md->name,
                           functionstring);

    return RETURN_SUCCESS;
}

/** @brief Ensure FPS tmux sesssion exists
 *
 */
errno_t functionparameter_FPS_tmux_ensure(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s 2>/dev/null", fps->md->name);
    if (system(cmd) != 0) {
        functionparameter_FPS_tmux_init(fps);
    }
    return RETURN_SUCCESS;
}
