/**
 * @file    fps_tmux.c
 *
 * @brief   tmux session management
 *
 * tmux wrapper for FPS control
 */

#include "CommandLineInterface/CLIcore.h"




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

    EXECUTE_SYSTEM_COMMAND("echo \" \" >> tmux.log");
    EXECUTE_SYSTEM_COMMAND("echo \"---------------------------------------------\" >> tmux.log");
    EXECUTE_SYSTEM_COMMAND("echo \">>>>>>>>>>>>>> functionparameter_FPS_tmux_init  %s\">> tmux.log",
                           fps->md->name);
    EXECUTE_SYSTEM_COMMAND("echo \" \" >> tmux.log");
    EXECUTE_SYSTEM_COMMAND("tmux ls >> tmux.log");

    // delay to allow for tmux commands to be completed
    float tmuxwait = 0.1;

    // terminate tmux sessions
    functionparameter_FPS_tmux_kill(fps);
    EXECUTE_SYSTEM_COMMAND("echo \" \" >> tmux.log");
    EXECUTE_SYSTEM_COMMAND("tmux ls >> tmux.log");


    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux kill-session -t %s 2> /dev/null",
                           fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux new-session -s %s -d", fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux rename-window -t %s:0 ctrl", fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux new-window -t %s -n conf", fps->md->name);
    sleep(tmuxwait);
    EXECUTE_SYSTEM_COMMAND("tmux new-window -t %s -n run", fps->md->name);
    sleep(tmuxwait);

    EXECUTE_SYSTEM_COMMAND("echo \" \" >> tmux.log");
    EXECUTE_SYSTEM_COMMAND("tmux ls >> tmux.log");
    EXECUTE_SYSTEM_COMMAND("echo \"---------------------------------------------\" >> tmux.log");

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
        sprintf(argstring, " ");
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
    sprintf(mloadstring, " ");
    for(int m = 0; m < fps->md->NBmodule; m++)
    {
        snprintf(mloadstringcp,
                 mloadstring_maxlen,
                 "%smload %s;",
                 mloadstring,
                 fps->md->modulename[m]);
        strcpy(mloadstring, mloadstringcp);
    }

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \"bash\" C-m",
                           fps->md->name);

    // confstart
    //
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \"bash\" C-m",
                           fps->md->name);

    sprintf(functionstring,
            "function fpsconfstart {\n"
            "echo \"STARTING CONF PROCESS\"\n"
            "MILK_FPSPROCINFO=1 %s-exec -n %s \\\"%s%s _CONFSTART_ %s\\\"\n"
            "}\n",
            fps->md->callprogname,
            fps->md->name,
            mloadstring,
            fps->md->callfuncname,
            argstring);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \"%s\" C-m",
                           fps->md->name,
                           functionstring);

    // runstart
    //
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"bash\" C-m",
                           fps->md->name);

    sprintf(functionstring,
            "function fpsrunstart {\n"
            "echo \"STARTING RUN PROCESS\"\n"
            "MILK_FPSPROCINFO=1 %s-exec -n %s \\\"\\${TCSETCMDPREFIX} %s%s "
            "_RUNSTART_ %s\\\"\n"
            "}\n",
            fps->md->callprogname,
            fps->md->name,
            mloadstring,
            fps->md->callfuncname,
            argstring);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"%s\" C-m",
                           fps->md->name,
                           functionstring);

    // runstop
    //
    sprintf(functionstring,
            "function fpsrunstop {\n"
            "echo \"STOPPING RUN PROCESS\"\n"
            "%s-exec -n %s \\\"%s%s _RUNSTOP_ %s\\\"\n"
            "}\n",
            fps->md->callprogname,
            fps->md->name,
            mloadstring,
            fps->md->callfuncname,
            argstring);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"%s\" C-m",
                           fps->md->name,
                           functionstring);

    return RETURN_SUCCESS;
}
