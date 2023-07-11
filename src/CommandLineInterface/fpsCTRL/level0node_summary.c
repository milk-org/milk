#include "CommandLineInterface/CLIcore.h"

#include "TUItools.h"


void fpsCTRLscreen_level0node_summary(
    FUNCTION_PARAMETER_STRUCT *fps,
    int                        fpsindex
)
{
    DEBUG_TRACE_FSTART();
    pid_t pid;

    pid = fps[fpsindex].md->confpid;
    if((getpgid(pid) >= 0) && (pid > 0))
    {
        screenprint_setcolor(2);
        TUI_printfw("C");
        screenprint_unsetcolor(2);
    }
    else // PID not active
    {
        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
        {
            // not clean exit
            screenprint_setcolor(4);
            TUI_printfw("C");
            screenprint_unsetcolor(4);
        }
        else
        {
            // All OK
            TUI_printfw("C");
        }
    }

    if(fps[fpsindex].md->conferrcnt > 0)
    {
        screenprint_setcolor(4);
        TUI_printfw("E");
        screenprint_unsetcolor(4);
    }
    if(fps[fpsindex].md->conferrcnt == 0)
    {
        screenprint_setcolor(2);
        TUI_printfw(">");
        screenprint_unsetcolor(2);
    }

    pid = fps[fpsindex].md->runpid;
    if((getpgid(pid) >= 0) && (pid > 0))
    {
        screenprint_setcolor(2);
        TUI_printfw("R");
        screenprint_unsetcolor(2);
    }
    else
    {
        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
        {
            // not clean exit
            screenprint_setcolor(4);
            TUI_printfw("R");
            screenprint_unsetcolor(4);
        }
        else
        {
            // All OK
            TUI_printfw("R");
        }
    }
    TUI_printfw(" ");

    DEBUG_TRACE_FEXIT();
}
