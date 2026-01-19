#include <stdio.h>
#include "fps.h"
#include "fps_TUI_shim.h"
#include "print_nodeinfo.h"

void fpsCTRLscreen_print_nodeinfo(
    FUNCTION_PARAMETER_STRUCT *fps,
    int nodeSelected,
    int fpsindexSelected,
    long pindexSelected
)
{
    // fps is passed as array base
    TUI_printfw("Node %d : FPS %d pindex %ld\n",
                nodeSelected,
                fpsindexSelected,
                pindexSelected);
    
    // Additional debug info could go here
}