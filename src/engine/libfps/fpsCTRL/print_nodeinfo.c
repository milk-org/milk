/**
 * @file print_nodeinfo.c
 * @brief Print nodeinfo module
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fps.h"
#include "TUItools.h"
#include "print_nodeinfo.h"

void fpsCTRLscreen_print_nodeinfo(
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    int nodeSelected,
    int fpsindexSelected,
    long pindexSelected
)
{
    // fps is passed as array base
    TUI_printfw("======== FPS info ( # %5d) [%ld / %ld params (active/size)]\n", 
                fpsindexSelected, fps[fpsindexSelected].NBparamActive, fps[fpsindexSelected].md->NBparamMAX);

    TUI_printfw("    FPS call              : %s -> %s [",
            fps[fpsindexSelected].md->callprogname,
            fps[fpsindexSelected].md->callfuncname);

    for(int i = 0; i < fps[fpsindexSelected].md->NBnameindex; i++)
    {
        TUI_printfw(" %s", fps[fpsindexSelected].md->nameindexW[i]);
    }
    TUI_printfw(" ]\n");

    TUI_printfw(" FPS descr  : %s\n", fps[fpsindexSelected].md->description);
    TUI_printfw(" Exec path  : %s\n", fps[fpsindexSelected].md->execfullpath);
    TUI_printfw(" FPS source : %s:%d\n",
                fps[fpsindexSelected].md->sourcefname,
                fps[fpsindexSelected].md->sourceline);

    TUI_printfw("   %d libs : ", fps[fpsindexSelected].md->NBmodule);
    for(int m = 0; m < fps[fpsindexSelected].md->NBmodule; m++)
    {
        TUI_printfw(" [%s]", fps[fpsindexSelected].md->modulename[m]);
    }
    TUI_printfw("\n");

    TUI_printfw("    FPS work     directory    : %s\n",
            fps[fpsindexSelected].md->workdir);

    TUI_printfw("    ( FPS output data directory : %s )  ( FPS input conf directory : %s) \n",
            fps[fpsindexSelected].md->datadir,
            fps[fpsindexSelected].md->confdir);

    TUI_printfw("    FPS tmux sessions     :  ");

    char cmd[512];
    
    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s:ctrl 2> /dev/null", fps[fpsindexSelected].md->name);
    if(system(cmd) == 0) {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("%s:ctrl ", fps[fpsindexSelected].md->name);
        screenprint_unsetcolor(COLOR_OK);
    } else {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw("%s:ctrl ", fps[fpsindexSelected].md->name);
        screenprint_unsetcolor(COLOR_ERROR);
    }

    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s:conf 2> /dev/null", fps[fpsindexSelected].md->name);
    if(system(cmd) == 0) {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("%s:conf ", fps[fpsindexSelected].md->name);
        screenprint_unsetcolor(COLOR_OK);
    } else {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw("%s:conf ", fps[fpsindexSelected].md->name);
        screenprint_unsetcolor(COLOR_ERROR);
    }

    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s:run 2> /dev/null", fps[fpsindexSelected].md->name);
    if(system(cmd) == 0) {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("%s:run ", fps[fpsindexSelected].md->name);
        screenprint_unsetcolor(COLOR_OK);
    } else {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw("%s:run ", fps[fpsindexSelected].md->name);
        screenprint_unsetcolor(COLOR_ERROR);
    }
    TUI_printfw("\n");
    
    TUI_printfw("======== NODE info ( # %5d)\n", nodeSelected);
    TUI_printfw("% -30s ", keywnode[nodeSelected].keywordfull);
    
    if(keywnode[nodeSelected].leaf > 0) {
        char typestring[100];
        functionparameter_GetTypeString(
            fps[fpsindexSelected].parray[pindexSelected].type,
            typestring);
        TUI_printfw("type %s\n", typestring);
    } else {
        TUI_printfw("-DIRECTORY-\n");
    }
    TUI_printfw("\n");
}
