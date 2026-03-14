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
#include <errno.h>
#include <unistd.h>
#include <string.h>


/** @brief Kill FPS tmux session
 *
 * Sends SIGINT (C-c) to each window for graceful
 * process shutdown, then kills the session.
 */
int functionparameter_FPS_tmux_kill(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    EXECUTE_SYSTEM_COMMAND(
        "tmux send-keys -t %s:ctrl C-c 2>/dev/null",
        fps->md->name);
    EXECUTE_SYSTEM_COMMAND(
        "tmux send-keys -t %s:conf C-c 2>/dev/null",
        fps->md->name);
    EXECUTE_SYSTEM_COMMAND(
        "tmux send-keys -t %s:run C-c 2>/dev/null",
        fps->md->name);

    EXECUTE_SYSTEM_COMMAND(
        "tmux kill-session -t %s 2>/dev/null",
        fps->md->name);

    return RETURN_SUCCESS;
}

int functionparameter_FPS_tmux_attach(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    // This should hang until the tmux is detached,
    // and then return to the current fpsCTRL window.
    EXECUTE_SYSTEM_COMMAND("tmux attach -t %s", fps->md->name);
    return RETURN_SUCCESS;
}


/** @brief Initialize FPS tmux session
 *
 * Creates a tmux session with 3 windows: ctrl, conf, run.
 * Uses atomic tmux command chaining to avoid race conditions.
 */
int functionparameter_FPS_tmux_init(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    int funcstring_maxlen  = 10000;
    int argstring_maxlen   = 1000;
    int mloadstring_maxlen = 2000;

    // Kill any existing session
    functionparameter_FPS_tmux_kill(fps);

    // Create session with all 3 windows atomically
    EXECUTE_SYSTEM_COMMAND(
        "tmux new-session -s %s -d -n ctrl \\;"
        " new-window -n conf \\;"
        " new-window -n run \\;"
        " select-window -t %s:ctrl",
        fps->md->name, fps->md->name);


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
int functionparameter_FPS_tmux_ensure(
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

/** @brief Setup standalone tmux session
 *
 * Creates a tmux session with 3 windows: ctrl, conf, run.
 * No-op if session already exists.
 */
int functionparameter_FPS_tmux_standalone_setup(
    const char *fps_name
)
{
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "tmux has-session -t %s 2>/dev/null",
             fps_name);
    if (system(cmd) == 0)
    {
        return RETURN_SUCCESS;
    }

    EXECUTE_SYSTEM_COMMAND(
        "tmux new-session -s %s -d -n ctrl \\;"
        " new-window -n conf \\;"
        " new-window -n run \\;"
        " select-window -t %s:ctrl",
        fps_name, fps_name);

    return RETURN_SUCCESS;
}

/** @brief Send command to tmux window
 */
int functionparameter_FPS_tmux_send(
    const char *fps_name,
    const char *window,
    const char *cmd_str
)
{
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:%s \"%s\" C-m", fps_name, window, cmd_str);
    return RETURN_SUCCESS;
}

/** @brief Dispatch command to tmux if applicable
 *
 * Returns 0 if handled (sent to tmux), 1 if not handled (should be run locally).
 */
int functionparameter_FPS_tmux_send_dispatch(
    const char *fps_name,
    const char *command,
    const char *exec_path,
    const char *extra_args
)
{
    char cmd_str[2048];
    const char *window = "ctrl";
    int window_index = 0;

    if ( (strcmp(command, "confstart") == 0) || (strcmp(command, "confstep") == 0) ) {
        window = "conf";
        window_index = 1;
    } else if (strcmp(command, "runstart") == 0) {
        window = "run";
        window_index = 2;
    } else if ( (strcmp(command, "confstop") == 0) || (strcmp(command, "runstop") == 0) ) {
        window = "ctrl";
        window_index = 0;
    } else {
        // Only dispatch specific commands to tmux windows
        return 1;
    }

    snprintf(cmd_str, sizeof(cmd_str), "%s%s", exec_path, extra_args);
    functionparameter_FPS_tmux_send(fps_name, window, cmd_str);
    printf("running command %s in tmux window %s:%d\n", cmd_str, fps_name, window_index);
    
    return 0;
}

/** @brief Get path to current executable
 */
char* functionparameter_FPS_get_executable_path(char *buffer, size_t size)
{
    if (!buffer || size == 0) return NULL;
    
    ssize_t len = readlink("/proc/self/exe", buffer, size - 1);
    if (len != -1) {
        buffer[len] = '\0';
        return buffer;
    }
    return NULL;
}