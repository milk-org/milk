// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file print_nodeinfo.c
 * @brief Print nodeinfo module
 */

#include "fps.h"
#include "fpsCTRL_TUIcompat.h"

/**
 * @brief Print detailed info for the selected tree node.
 *
 * Displays the node's full keyword path, type,
 * description, and value history.
 */
void fpsCTRLscreen_print_nodeinfo(FPS               *fps,
                                  KEYWORD_TREE_NODE *keywnode,
                                  int                nodeSelected,
                                  int                fpsindexSelected,
                                  long               pindexSelected)
{
    TUI_newline();
    screenprint_setbold();
    screenprint_setcolor(7); // Cyan
    TUI_printfw("  [ FPS DETAILED INFO ]\n");
    screenprint_unsetcolor(7);
    screenprint_unsetbold();

    TUI_printfw("    %-16s: %d\n", "Index", fpsindexSelected);
    TUI_printfw("    %-16s: %ld / %ld (active/max)\n", "Params",
                fps[fpsindexSelected].NBparamActive, fps[fpsindexSelected].md->NBparamMAX);
    TUI_printfw("    %-16s: %s -> %s", "Call", fps[fpsindexSelected].md->callprogname,
                fps[fpsindexSelected].md->callfuncname);

    if (fps[fpsindexSelected].md->NBnameindex > 0)
    {
        TUI_printfw(" [");
        for (int name_idx = 0; name_idx < fps[fpsindexSelected].md->NBnameindex; name_idx++)
        {
            TUI_printfw(" %s", fps[fpsindexSelected].md->nameindexW[name_idx]);
        }
        TUI_printfw(" ]");
    }
    TUI_printfw("\n");

    TUI_printfw("    %-16s: %s\n", "Description", fps[fpsindexSelected].md->description);
    TUI_printfw("    %-16s: %s\n", "Exec Path", fps[fpsindexSelected].md->execfullpath);
    TUI_printfw("    %-16s: %s:%d\n", "Source", fps[fpsindexSelected].md->sourcefname,
                fps[fpsindexSelected].md->sourceline);

    if (fps[fpsindexSelected].md->NBmodule > 0)
    {
        TUI_printfw("    %-16s:", "Libraries");
        for (int mod_idx = 0; mod_idx < fps[fpsindexSelected].md->NBmodule; mod_idx++)
        {
            TUI_printfw(" [%s]", fps[fpsindexSelected].md->modulename[mod_idx]);
        }
        TUI_printfw("\n");
    }

    TUI_printfw("    %-16s: %s\n", "Work Dir", fps[fpsindexSelected].md->workdir);
    TUI_printfw("    %-16s: %s\n", "Data Dir", fps[fpsindexSelected].md->datadir);
    TUI_printfw("    %-16s: %s\n", "Conf Dir", fps[fpsindexSelected].md->confdir);

    TUI_printfw("    %-16s: ", "Sessions");

    char cmd[512];

    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s:ctrl 2> /dev/null",
             fps[fpsindexSelected].md->name);
    if (system(cmd) == 0)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("[ctrl] ");
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(4);
        TUI_printfw(" ctrl  ");
        screenprint_unsetcolor(4);
    }

    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s:conf 2> /dev/null",
             fps[fpsindexSelected].md->name);
    if (system(cmd) == 0)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("[conf] ");
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(4);
        TUI_printfw(" conf  ");
        screenprint_unsetcolor(4);
    }

    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s:run 2> /dev/null",
             fps[fpsindexSelected].md->name);
    if (system(cmd) == 0)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("[run] ");
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(4);
        TUI_printfw(" run  ");
        screenprint_unsetcolor(4);
    }
    TUI_newline();

    TUI_newline();
    screenprint_setbold();
    screenprint_setcolor(7); // Cyan
    TUI_printfw("  [ NODE DETAILED INFO ]\n");
    screenprint_unsetcolor(7);
    screenprint_unsetbold();

    TUI_printfw("    %-16s: %d\n", "Index", nodeSelected);
    TUI_printfw("    %-16s: %s\n", "Keyword", keywnode[nodeSelected].keywordfull);

    if (keywnode[nodeSelected].leaf > 0)
    {
        char typestring[100];
        functionparameter_GetTypeString(fps[fpsindexSelected].parray[pindexSelected].type,
                                        typestring);
        TUI_printfw("    %-16s: %s\n", "Type", typestring);
    }
    else
    {
        TUI_printfw("    %-16s: DIRECTORY\n", "Type");
    }
    TUI_newline();
}
