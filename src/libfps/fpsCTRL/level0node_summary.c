#include <stdio.h>
#include <string.h>
#include <ncurses.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_TUI_shim.h"
#include "fpsCTRL_globals.h"

#include "level0node_summary.h"

void fpsCTRLscreen_level0node_summary(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
)
{
    // fpsindex is passed but we also have global fpsarray.
    // If called with global fpsarray, fps[fpsindex] is valid.
    // We assume fps is the array base pointer.

    // Using simplified printing for TUI shim compatibility
    
    // Status flags
    if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw(" C ");
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(COLOR_DIRECTORY);
        TUI_printfw(" C ");
        screenprint_unsetcolor(COLOR_DIRECTORY);
    }

    if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw(" R ");
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(COLOR_DIRECTORY);
        TUI_printfw(" R ");
        screenprint_unsetcolor(COLOR_DIRECTORY);
    }

    if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKERR)
    {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw(" E ");
        screenprint_unsetcolor(COLOR_ERROR);
    }
    else
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw(" E ");
        screenprint_unsetcolor(COLOR_OK);
    }

    if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_SAVE)
    {
        screenprint_setcolor(COLOR_WARNING);
        TUI_printfw(" S ");
        screenprint_unsetcolor(COLOR_WARNING);
    }
    else
    {
        screenprint_setcolor(COLOR_DIRECTORY);
        TUI_printfw(" S ");
        screenprint_unsetcolor(COLOR_DIRECTORY);
    }

    TUI_printfw("  %s\n", fps[fpsindex].md->name);
}