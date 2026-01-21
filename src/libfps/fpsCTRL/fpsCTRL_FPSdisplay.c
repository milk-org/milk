#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ncurses.h>
#include <time.h>

#include "fps.h"
#include "fps_internal.h"
#include "TUItools.h"
#include "fpsCTRL_globals.h"

#include "fpsCTRL_FPSdisplay.h"
#include "print_nodeinfo.h"
#include "level0node_summary.h"
#include "fps_GetTypeString.h"
#include "fps_PrintParameterInfo.h"
#include "fps_printparameter_valuestring.h"

static void print_sliding_string(const char *str, int width, int row_index)
{
    int len = strlen(str);
    if (len <= width)
    {
        TUI_printfw("%*s", width, str);
        return;
    }

    // Dynamic sliding logic based on monotonic clock
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    
    // Total cycle time in seconds for one back-and-forth slide
    double cycle_time = 4.0; 
    double time_in_cycle = fmod((double)now.tv_sec + (double)now.tv_nsec * 1e-9 + (double)row_index * 0.5, cycle_time);
    
    int max_offset = len - width;
    int offset;
    
    // Divide cycle into 4 parts: stay at start, slide to end, stay at end, slide to start
    double phase = time_in_cycle / cycle_time;
    if (phase < 0.2) offset = 0;
    else if (phase < 0.5) offset = (int)((phase - 0.2) / 0.3 * max_offset);
    else if (phase < 0.7) offset = max_offset;
    else offset = (int)(max_offset - (phase - 0.7) / 0.3 * max_offset);

    if (offset < 0) offset = 0;
    if (offset > max_offset) offset = max_offset;

    char buf[width + 1];
    strncpy(buf, str + offset, width);
    buf[width] = '\0';
    TUI_printfw("%s", buf);
}

errno_t fpsCTRL_FPSdisplay(
    KEYWORD_TREE_NODE    *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    if(fpsCTRLvar->NBfps > 0)
    {
        DEBUG_TRACEPOINT("Check that selected node is OK");
        if(strlen(keywnode[fpsCTRLvar->nodeSelected].keywordfull) < 1)
        {
            fpsCTRLvar->nodeSelected = 1;
            while((strlen(keywnode[fpsCTRLvar->nodeSelected].keywordfull) < 1)
                    && (fpsCTRLvar->nodeSelected < NB_KEYWNODE_MAX))
            {
                fpsCTRLvar->nodeSelected ++;
            }
        }

        fpsCTRLvar->fpsindexSelected = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        fpsCTRLvar->pindexSelected = keywnode[fpsCTRLvar->nodeSelected].pindex;
        
        if (fpsCTRLvar->fpsCTRL_DisplayVerbose) {
            fpsCTRLscreen_print_nodeinfo(
                fpsarray,
                keywnode,
                fpsCTRLvar->nodeSelected,
                fpsCTRLvar->fpsindexSelected,
                fpsCTRLvar->pindexSelected);
        }

        int nodechain[MAXNBLEVELS];
        nodechain[fpsCTRLvar->currentlevel] = fpsCTRLvar->directorynodeSelected;

        int level = fpsCTRLvar->currentlevel - 1;
        while(level > 0)
        {
            nodechain[level] = keywnode[nodechain[level + 1]].parent_index;
            level --;
        }
        nodechain[0] = 0; // root

        fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
        int GUIlineMax = keywnode[fpsCTRLvar->directorynodeSelected].NBchild;
        for(level = 0; level < fpsCTRLvar->currentlevel; level ++)
        {
            if(keywnode[nodechain[level]].NBchild > GUIlineMax)
            {
                GUIlineMax = keywnode[nodechain[level]].NBchild;
            }
        }

        if(!(fpsarray[fpsCTRLvar->fpsindexSelected].parray[0].fpflag &
                FPFLAG_VISIBLE))      // if invisible
        {
            if(fpsCTRLvar->direction > 0) fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] ++;
            else fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] --;
        }

        while(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
        {
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]--;
        }
        
        if (fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;

        int child_index[MAXNBLEVELS];
        for(level = 0; level < MAXNBLEVELS ;
                level ++)
        {
            child_index[level] = 0;
        }

        // Calculate dynamic widths
        int max_kw_width = 10;
        int max_val_width = 10;
        
        for(int i = 0; i < keywnode[fpsCTRLvar->directorynodeSelected].NBchild; i++)
        {
            int knodeindex = keywnode[fpsCTRLvar->directorynodeSelected].child[i];
            if (keywnode[knodeindex].leaf)
            {
                int fpsindex = keywnode[knodeindex].fpsindex;
                int pindex = keywnode[knodeindex].pindex;
                int kw_len = strlen(fpsarray[fpsindex].parray[pindex].keyword[keywnode[knodeindex].keywordlevel - 1]);
                if (kw_len > max_kw_width) max_kw_width = kw_len;
                
                char valstring[200];
                functionparameter_GetParamValueString(&fpsarray[fpsindex].parray[pindex], valstring, 200);
                int val_len = strlen(valstring);
                if (val_len > max_val_width) max_val_width = val_len;
            }
            else
            {
                int kw_len = strlen(keywnode[knodeindex].keyword[keywnode[knodeindex].keywordlevel - 1]);
                if (kw_len > max_kw_width) max_kw_width = kw_len;
            }
        }

        // Impose constraints based on terminal width
        int reserved_width = (fpsCTRLvar->currentlevel * 12) + 15; 
        int available_width = COLS - reserved_width;
        if (available_width < 30) available_width = 30;
        
        if (max_kw_width > available_width / 3) max_kw_width = available_width / 3;
        if (max_val_width > available_width - max_kw_width - 5) max_val_width = available_width - max_kw_width - 5;

        for(int GUIline = 0; GUIline < GUIlineMax;
                GUIline++)   // GUIline is the line number on GUI display
        {
            for(level = 0; level < fpsCTRLvar->currentlevel; level ++)
            {
                if(GUIline < keywnode[nodechain[level]].NBchild)
                {
                    int snode = 0; // selected node
                    int knodeindex = keywnode[nodechain[level]].child[GUIline];

                    if(level == 0)
                    {
                        int fpsindex = keywnode[knodeindex].fpsindex;
                        fpsCTRLscreen_level0node_summary(fpsarray, fpsindex);
                    }

                    // toggle highlight if node is in the chain
                    int v1 = keywnode[nodechain[level]].child[GUIline];
                    int v2 = nodechain[level + 1];
                    if(v1 == v2)
                    {
                        snode = 1;
                        screenprint_setreverse();
                    }

                    // color node if directory
                    if(keywnode[knodeindex].leaf == 0)
                    {
                        screenprint_setcolor(5);
                    }

                    TUI_printfw("% -10s ", keywnode[knodeindex].keyword[level]);

                    if(keywnode[knodeindex].leaf == 0)   // directory
                    {
                        screenprint_unsetcolor(5);
                    }

                    if(snode == 1)
                    {
                        TUI_printfw("> ");
                        screenprint_unsetreverse();
                    }
                    else
                    {
                        TUI_printfw("  ");
                    }
                    screenprint_setnormal();
                }
                else     // blank space
                {
                    if(level == 0)
                    {
                        TUI_printfw("                    ");
                    }
                    TUI_printfw("             ");
                }
            }

            int knodeindex =
                keywnode[fpsCTRLvar->directorynodeSelected].child[child_index[level]];
            if(knodeindex < fpsCTRLvar->NBkwn)
            {
                int fpsindex = keywnode[knodeindex].fpsindex;
                int pindex = keywnode[knodeindex].pindex;

                if(fpsCTRLvar->currentlevel > 0)
                {
                    screenprint_setreverse();
                    TUI_printfw(" ");
                    screenprint_unsetreverse();
                }

                if(keywnode[knodeindex].leaf == 0)   // If this is a directory
                {
                    if(fpsCTRLvar->currentlevel == 0)   // provide a status summary if at root
                    {
                        fpsindex = keywnode[knodeindex].fpsindex;
                        fpsCTRLscreen_level0node_summary(fpsarray, fpsindex);
                    }

                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                    {
                        screenprint_setreverse();
                        fpsCTRLvar->nodeSelected = knodeindex;
                        fpsCTRLvar->fpsindexSelected = keywnode[knodeindex].fpsindex;
                    }

                    if(child_index[level + 1] < keywnode[fpsCTRLvar->directorynodeSelected].NBchild)
                    {
                        screenprint_setcolor(5);
                        int l = keywnode[knodeindex].keywordlevel;
                        TUI_printfw("% -*.*s", max_kw_width, max_kw_width, keywnode[knodeindex].keyword[l - 1]);
                        screenprint_unsetcolor(5);

                        if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                        {
                            TUI_printfw("> ");
                            screenprint_unsetreverse();
                        }
                        else
                        {
                            TUI_printfw("  ");
                        }
                    }
                    else TUI_printfw("%*s  ", max_kw_width, " ");
                }
                else   // If this is a parameter
                {
                    fpsindex = keywnode[knodeindex].fpsindex;
                    pindex = keywnode[knodeindex].pindex;

                    int isVISIBLE = 1;
                    if(!(fpsarray[fpsindex].parray[pindex].fpflag &
                            FPFLAG_VISIBLE))   // if invisible
                    {
                        isVISIBLE = 0;
                        screenprint_setdim();
                        screenprint_setblink();
                    }

                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                    {
                        fpsCTRLvar->pindexSelected = keywnode[knodeindex].pindex;
                        fpsCTRLvar->fpsindexSelected = keywnode[knodeindex].fpsindex;
                        fpsCTRLvar->nodeSelected = knodeindex;

                        if(isVISIBLE == 1)
                        {
                            screenprint_setcolor(10);
                            screenprint_setbold();
                        }
                    }

                    if(isVISIBLE == 1)
                    {
                        if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_WRITESTATUS)
                        {
                            screenprint_setcolor(10);
                            screenprint_setblink();
                            TUI_printfw("W "); // writable
                            screenprint_unsetcolor(10);
                            screenprint_unsetblink();
                        }
                        else
                        {
                            screenprint_setcolor(4);
                            screenprint_setblink();
                            TUI_printfw("NW"); // non writable
                            screenprint_unsetcolor(4);
                            screenprint_unsetblink();
                        }
                    }
                    else
                    {
                        TUI_printfw("  ");
                    }

                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                        screenprint_setreverse();

                    TUI_printfw(" %-*.*s", max_kw_width, max_kw_width, fpsarray[fpsindex].parray[pindex].keyword[keywnode[knodeindex].keywordlevel - 1]);

                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                    {
                        screenprint_unsetcolor(10);
                        TUI_printfw("> ");
                        screenprint_unsetreverse();
                    }
                    else
                    {
                        TUI_printfw("  ");
                    }
                    TUI_printfw(": ");

                    // VALUE (Synchronized check simplified)
                    int paramsync = 1; // parameter is synchronized

                    if(fpsarray[fpsindex].parray[pindex].fpflag &
                            FPFLAG_ERROR)   // parameter setting error
                    {
                        if(isVISIBLE == 1)
                        {
                            screenprint_setcolor(4);
                        }
                    }

                    // Simple sync check for types that support it
                    if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)
                    {
                        if(fpsarray[fpsindex].parray[pindex].type == FPTYPE_INT64)
                            if(fpsarray[fpsindex].parray[pindex].val.i64[0] != fpsarray[fpsindex].parray[pindex].val.i64[3]) paramsync = 0;
                        if(fpsarray[fpsindex].parray[pindex].type == FPTYPE_FLOAT64)
                            if(fabs(fpsarray[fpsindex].parray[pindex].val.f64[0] - fpsarray[fpsindex].parray[pindex].val.f64[3]) > 1e-6) paramsync = 0;
                    }

                    if(paramsync == 0 && isVISIBLE == 1) screenprint_setcolor(3);

                    char valstring[200];
                    functionparameter_GetParamValueString(&fpsarray[fpsindex].parray[pindex], valstring, 200);
                    
                    print_sliding_string(valstring, max_val_width, GUIline);

                    if(paramsync == 0 && isVISIBLE == 1) screenprint_unsetcolor(3);
                    if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR && isVISIBLE == 1) screenprint_unsetcolor(4);

                    TUI_printfw("    %s", fpsarray[fpsindex].parray[pindex].description);

                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] && isVISIBLE == 1)
                        screenprint_unsetbold();

                    if(isVISIBLE == 0)
                    {
                        screenprint_unsetblink();
                        screenprint_unsetdim();
                    }
                }
                for(int l = 0; l < MAXNBLEVELS ; l++) child_index[l]++;
            }
            TUI_newline();
        }

        fpsCTRLvar->NBindex = GUIlineMax; // Matches number of lines displayed
        if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] > fpsCTRLvar->NBindex - 1)
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = fpsCTRLvar->NBindex - 1;

        TUI_newline();

        if(fpsarray[fpsCTRLvar->fpsindexSelected].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK)
        {
            screenprint_setcolor(2);
            TUI_printfw("[%ld] PARAMETERS OK - RUN function good to go\n", fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt);
            screenprint_unsetcolor(2);
        }
        else
        {
            screenprint_setcolor(4);
            TUI_printfw("[%ld] %d PARAMETER SETTINGS ERROR(s) :\n",
                        fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt,
                        fpsarray[fpsCTRLvar->fpsindexSelected].md->conferrcnt);
            screenprint_unsetcolor(4);
            screenprint_setbold();
            for(int msgi = 0; msgi < fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt; msgi++)
            {
                int pidx = fpsarray[fpsCTRLvar->fpsindexSelected].md->msgpindex[msgi];
                TUI_printfw("% -40s %s\n",
                            fpsarray[fpsCTRLvar->fpsindexSelected].parray[pidx].keywordfull,
                            fpsarray[fpsCTRLvar->fpsindexSelected].md->message[msgi]);
            }
            screenprint_unsetbold();
        }
    }
    else TUI_printfw("NO FPS LOADED\n");

    return RETURN_SUCCESS;
}
