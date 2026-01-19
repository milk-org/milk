#include <stdio.h>
#include <string.h>
#include <ncurses.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_TUI_shim.h"
#include "fpsCTRL_globals.h"

#include "fpsCTRL_FPSdisplay.h"
#include "print_nodeinfo.h"
#include "level0node_summary.h"

static errno_t fpselem_statusprint_ONOFF(
    int fpsindex,
    int pindex
)
{
    if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_ONOFF)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("  ON ");
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(COLOR_DIRECTORY); // Reusing directory color for neutral/off if NONE not avail
        TUI_printfw(" OFF ");
        screenprint_unsetcolor(COLOR_DIRECTORY);
    }

    return RETURN_SUCCESS;
}

static errno_t fpselem_statusprint_FPSNAME(
    int fpsindex,
    int pindex,
    int isVISIBLE
)
{
    // is FPS connected ?
    int FPSconnected = 1;
    // 0 : not connected, ERR
    // 1 : connected, OK
    // 2 : not connected but not needed, WARN

    if(strlen(fpsarray[fpsindex].parray[pindex].val.string[0]) > 0)
    {
        if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_WRITECONF)
        {
            if(fpsarray[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
            {
                FPSconnected = 1;
            }
            else
            {
                FPSconnected = 0;
            }
        }
        else
        {
            if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_WRITERUN)
            {
                if(fpsarray[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
                {
                    FPSconnected = 1;
                }
                else
                {
                    FPSconnected = 0;
                }
            }
            else
            {
                FPSconnected = 2;
            }
        }
    }
    else
    {
        FPSconnected = 2;
    }

    if(FPSconnected == 1)
    {
        if(isVISIBLE)
        {
            screenprint_setcolor(COLOR_OK);
        }
        TUI_printfw("  CONNECTED    ");
        if(isVISIBLE)
        {
            screenprint_unsetcolor(COLOR_OK);
        }
    }
    else if(FPSconnected == 0)
    {
        if(isVISIBLE)
        {
            screenprint_setcolor(COLOR_ERROR);
        }
        TUI_printfw("  DISCONNECTED ");
        if(isVISIBLE)
        {
            screenprint_unsetcolor(COLOR_ERROR);
        }
    }
    else
    {
        if(isVISIBLE)
        {
            screenprint_setcolor(COLOR_WARNING);
        }
        TUI_printfw("  ---          ");
        if(isVISIBLE)
        {
            screenprint_unsetcolor(COLOR_WARNING);
        }
    }

    return RETURN_SUCCESS;
}

static errno_t fpsCTRLdisplay_FPSerrormsgs(
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    // Display error messages from FPS
    if(fpsarray[fpsCTRLvar->fpsindexSelected].md->status &
            FUNCTION_PARAMETER_STRUCT_STATUS_CHECKERR)
    {
        TUI_printfw(" ERROR(S) [msg %d] [cnt %d] : \n",
                    fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt,
                    fpsarray[fpsCTRLvar->fpsindexSelected].md->conferrcnt);

        int msgi;
        for(msgi = 0;
                msgi < fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt;
                msgi++)
        {
            int pindex = fpsarray[fpsCTRLvar->fpsindexSelected]
                         .md->msgpindex[msgi];
            screenprint_setcolor(COLOR_ERROR);
            TUI_printfw("    %s : %s\n",
                        fpsarray[fpsCTRLvar->fpsindexSelected].parray[pindex].keywordfull,
                        fpsarray[fpsCTRLvar->fpsindexSelected].md->message[msgi]);
            screenprint_unsetcolor(COLOR_ERROR);
        }
    }

    return RETURN_SUCCESS;
}

errno_t fpsCTRL_FPSdisplay(
    KEYWORD_TREE_NODE    *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    // Display logic adapted for standalone
    // Using fpsarray and local includes

    if(fpsCTRLvar->NBfps > 0)
    {
        if(strlen(keywnode[fpsCTRLvar->nodeSelected].keywordfull) < 1)
        {
            fpsCTRLvar->nodeSelected = 1;
        }

        while((strlen(keywnode[fpsCTRLvar->nodeSelected].keywordfull) < 1) &&
                (fpsCTRLvar->nodeSelected < NB_KEYWNODE_MAX))
        {
            if(fpsCTRLvar->nodeSelected < fpsCTRLvar->NBkwn - 1)
            {
                fpsCTRLvar->nodeSelected++;
            }
            else
            {
                break;
            }
        }

        fpsCTRLvar->fpsindexSelected = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        fpsCTRLvar->pindexSelected   = keywnode[fpsCTRLvar->nodeSelected].pindex;

        if(fpsCTRLvar->fpsCTRL_DisplayVerbose == 1)
        {
            fpsCTRLscreen_print_nodeinfo(fpsarray,
                                         fpsCTRLvar->nodeSelected,
                                         fpsCTRLvar->fpsindexSelected,
                                         fpsCTRLvar->pindexSelected);
        }

        // Display tree
        int nodechain[MAXNBLEVELS];
        nodechain[fpsCTRLvar->currentlevel] = fpsCTRLvar->directorynodeSelected;
        if(fpsCTRLvar->currentlevel > 0)
        {
            int level = fpsCTRLvar->currentlevel - 1;
            while(level > -1)
            {
                nodechain[level] = keywnode[nodechain[level + 1]].parent_index;
                level--;
            }
        }

        fpsCTRLvar->currentlevel =
            keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;

        // Header
        screenprint_setreverse();
        TUI_printfw("%s", fpsCTRLvar->fpsnamemask);
        int GUIlineMax = keywnode[fpsCTRLvar->directorynodeSelected].NBchild;
        if(GUIlineMax < 1)
        {
            GUIlineMax = 1;
        }
        for(int level = 0; level < fpsCTRLvar->currentlevel; level++)
        {
            TUI_printfw(".%s", keywnode[nodechain[level + 1]].keyword[level]);
        }
        screenprint_unsetreverse();
        TUI_printfw("  level %d  dirnode %d  NBchild %d\n",
                    fpsCTRLvar->currentlevel,
                    fpsCTRLvar->directorynodeSelected,
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild);

        if(fpsCTRLvar->fpsCTRL_DisplayVerbose == 1)
        {
            TUI_printfw("Node %d level %d dirnode %d NBchild %d\n",
                        fpsCTRLvar->nodeSelected,
                        fpsCTRLvar->currentlevel,
                        fpsCTRLvar->directorynodeSelected,
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild);
            TUI_printfw("   fps %d", fpsCTRLvar->fpsindexSelected);
            TUI_printfw("   pindex %ld\n",
                        keywnode[fpsCTRLvar->nodeSelected].pindex);
            TUI_printfw(
                "DirNode %d  NodeSelected %d  GUIlineSelected %d  NBchild "
                "%d\n",
                fpsCTRLvar->directorynodeSelected,
                fpsCTRLvar->nodeSelected,
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel],
                keywnode[fpsCTRLvar->directorynodeSelected].NBchild);
        }

        // Navigation limits
        if(!(fpsarray[fpsCTRLvar->fpsindexSelected].parray[0].fpflag &
                FPFLAG_VISIBLE))
        {
            // if invisible
            // skip... logic simplified for brevity/porting
        }

        if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
        {
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
        }

        while(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
        {
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]--;
        }

        if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
        {
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
        }

        // Display List
        int child_index[MAXNBLEVELS];
        child_index[0] = fpsCTRLvar->GUIlineSelected[0];
        int knodeindex = keywnode[0].child[child_index[0]];

        for(int level = 0; level < fpsCTRLvar->currentlevel; level++)
        {
            child_index[level + 1] = fpsCTRLvar->GUIlineSelected[level + 1];
            knodeindex = keywnode[knodeindex].child[child_index[level + 1]];
        }

        // Display lines
        for(int i = 0; i < keywnode[fpsCTRLvar->directorynodeSelected].NBchild; i++)
        {
            knodeindex = keywnode[fpsCTRLvar->directorynodeSelected].child[i];
            int fpsindex = keywnode[knodeindex].fpsindex;
            // int pindex   = keywnode[knodeindex].pindex;

            if(i == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
            {
                screenprint_setreverse();
            }

            if(keywnode[knodeindex].leaf == 0)  // DIRECTORY
            {
                screenprint_setcolor(COLOR_DIRECTORY);
                TUI_printfw(" %s ", keywnode[knodeindex].keyword[fpsCTRLvar->currentlevel]);
                screenprint_unsetcolor(COLOR_DIRECTORY);
                if(i == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                {
                    screenprint_unsetreverse();
                }
                TUI_printfw("/\n");
            }
            else // LEAF (Parameter)
            {
                if(fpsCTRLvar->currentlevel == 0)
                {
                    fpsCTRLscreen_level0node_summary(fpsarray, fpsindex);
                }
                else
                {
                    // Print parameter value
                    char valstring[200];
                    functionparameter_PrintParameter_ValueString(
                        &fpsarray[fpsindex].parray[keywnode[knodeindex].pindex],
                        valstring,
                        200);

                    TUI_printfw(" %-20s : %s",
                                keywnode[knodeindex].keyword[fpsCTRLvar->currentlevel],
                                valstring);

                    if(i == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                    {
                        screenprint_unsetreverse();
                    }
                    TUI_printfw("\n");
                }
            }
        }

        fpsCTRLdisplay_FPSerrormsgs(fpsCTRLvar);
    }
    else
    {
        TUI_printfw("NO FPS LOADED\n");
    }

    return RETURN_SUCCESS;
}