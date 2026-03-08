#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ncurses.h>
#include <time.h>
#include <sys/stat.h>

#include "fps.h"
#include "fps_internal.h"
#include "TUItools.h"
#include "fps_streamname_parse.h"
#include "fpsCTRL_globals.h"

#include "fpsCTRL_FPSdisplay.h"
#include "print_nodeinfo.h"
#include "level0node_summary.h"
#include "fps_GetTypeString.h"
#include "fps_PrintParameterInfo.h"
#include "fps_printparameter_valuestring.h"

#define LEVEL0_SUMMARY_WIDTH 20
#define TREE_LEVEL_WIDTH 12

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
    DEBUG_TRACE_FSTART();

    if(fpsCTRLvar->NBfps > 0)
    {
        DEBUG_TRACEPOINT("Check that selected node is OK");
        if(strlen(keywnode[fpsCTRLvar->nodeSelected].keywordfull) < 1)
        {
            fpsCTRLvar->nodeSelected = 1;
            while((strlen(keywnode[fpsCTRLvar->nodeSelected].keywordfull) < 1)
                    && (fpsCTRLvar->nodeSelected < NB_KEYWNODE_MAX - 1))
            {
                fpsCTRLvar->nodeSelected ++;
            }
        }

        DEBUG_TRACEPOINT("Selected node: %d", fpsCTRLvar->nodeSelected);

        fpsCTRLvar->fpsindexSelected = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        fpsCTRLvar->pindexSelected = keywnode[fpsCTRLvar->nodeSelected].pindex;

        if (fpsarray[fpsCTRLvar->fpsindexSelected].md == NULL)
        {
            return RETURN_SUCCESS;
        }

        if (fpsCTRLvar->currentlevel == -1)
        {
            // Resolve node selected at level 0
            int knodeindex = keywnode[0].child[fpsCTRLvar->GUIlineSelected[0]];
            fpsCTRLvar->nodeSelected = knodeindex;
            fpsCTRLvar->fpsindexSelected = keywnode[knodeindex].fpsindex;

            screenprint_setbold();
            TUI_printfw("Detailed Help for FPS '%s':\n", fpsarray[fpsCTRLvar->fpsindexSelected].md->name);
            TUI_printfw("--------------------------\n");
            screenprint_unsetbold();
            TUI_printfw("%s\n", fpsarray[fpsCTRLvar->fpsindexSelected].md->helptext);
            return RETURN_SUCCESS;
        }
        
        DEBUG_TRACEPOINT("fpsindexSelected: %d, pindexSelected: %d", fpsCTRLvar->fpsindexSelected, fpsCTRLvar->pindexSelected);

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

        DEBUG_TRACEPOINT("GUIlineMax: %d", GUIlineMax);

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

        DEBUG_TRACEPOINT("GUIlineSelected[currentlevel]: %d", fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]);

        int child_index[MAXNBLEVELS];
        for(level = 0; level < MAXNBLEVELS ;
                level ++)
        {
            child_index[level] = 0;
        }

        // Calculate dynamic widths for each level
        int max_kw_width[MAXNBLEVELS];
        for (int l = 0; l < MAXNBLEVELS; l++) max_kw_width[l] = 5; // minimum width

        // Widths for hierarchy columns (levels < currentlevel)
        for (int l = 0; l < fpsCTRLvar->currentlevel; l++)
        {
            int parent_node = nodechain[l];
            for (int i = 0; i < keywnode[parent_node].NBchild; i++)
            {
                int kn = keywnode[parent_node].child[i];
                int len = strlen(keywnode[kn].keyword[l]);
                if (len > max_kw_width[l]) max_kw_width[l] = len;
            }
            if (max_kw_width[l] > 30) max_kw_width[l] = 30;
        }

        // Width for the current level's keyword column
        int max_val_width = 5;
        int cl = fpsCTRLvar->currentlevel;
        for(int i = 0; i < keywnode[fpsCTRLvar->directorynodeSelected].NBchild; i++)
        {
            int knodeindex = keywnode[fpsCTRLvar->directorynodeSelected].child[i];
            int kw_len;
            if (keywnode[knodeindex].leaf)
            {
                int fpsindex = keywnode[knodeindex].fpsindex;
                int pindex = keywnode[knodeindex].pindex;
                kw_len = strlen(fpsarray[fpsindex].parray[pindex].keyword[keywnode[knodeindex].keywordlevel - 1]);
                
                char valstring[200];
                if (fpsarray[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                    snprintf(valstring, 200, "%s", fpsarray[fpsindex].parray[pindex].val.string[0]);
                } else {
                    functionparameter_GetParamValueString(&fpsarray[fpsindex].parray[pindex], valstring, 200);
                }
                int val_len = strlen(valstring);
                if (val_len > max_val_width) max_val_width = val_len;
            }
            else
            {
                kw_len = strlen(keywnode[knodeindex].keyword[keywnode[knodeindex].keywordlevel - 1]);
            }
            if (kw_len > max_kw_width[cl]) max_kw_width[cl] = kw_len;
        }
        if (max_kw_width[cl] > 30) max_kw_width[cl] = 30;
        if (max_val_width > 40) max_val_width = 40;

        DEBUG_TRACEPOINT("max_kw_width[cl]: %d, max_val_width: %d", max_kw_width[cl], max_val_width);

        // Impose constraints based on terminal width
        int reserved_width = 25; 
        for (int l = 0; l < fpsCTRLvar->currentlevel; l++) reserved_width += max_kw_width[l] + 3;
        int available_width = COLS - reserved_width;
        if (available_width < 40) available_width = 40;
        
        if (max_kw_width[cl] > available_width / 2) max_kw_width[cl] = available_width / 2;
        
        // max_val_width is handled by sliding print, but we need enough space for description

        TUI_newline();

        // 1-line summary for selected stream if resolved
        int stream_summary_printed = 0;
        if (keywnode[fpsCTRLvar->nodeSelected].leaf)
        {
            int fpsidx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            int pidx = keywnode[fpsCTRLvar->nodeSelected].pindex;
            if (fpsarray[fpsidx].parray[pidx].type == FPTYPE_STREAMNAME)
            {
                IMAGE tmpimg;
                FPS_STREAMNAME_PARSED sp_sum = fps_streamname_parse(fpsarray[fpsidx].parray[pidx].val.string[0]);
                if (ImageStreamIO_openIm(&tmpimg, sp_sum.name) == IMAGESTREAMIO_SUCCESS)
                {
                    char stream_info[256];
                    char size_str[64];
                    uint32_t naxis = tmpimg.md->naxis;
                    uint32_t xsize = tmpimg.md->size[0];
                    uint32_t ysize = (naxis > 1) ? tmpimg.md->size[1] : 1;
                    uint32_t zsize = (naxis > 2) ? tmpimg.md->size[2] : 1;
                    
                    if (naxis == 1) snprintf(size_str, 64, "%u", xsize);
                    else if (naxis == 2) snprintf(size_str, 64, "%ux%u", xsize, ysize);
                    else snprintf(size_str, 64, "%ux%ux%u", xsize, ysize, zsize);

                    snprintf(stream_info, 256, "STREAM [%s]: %s %s cnt=%lu", 
                             fpsarray[fpsidx].parray[pidx].val.string[0],
                             ImageStreamIO_typename(tmpimg.md->datatype),
                             size_str,
                             tmpimg.md->cnt0);
                    
                    screenprint_setcolor(2);
                    TUI_printfw("%s", stream_info);
                    screenprint_unsetcolor(2);
                    TUI_newline();
                    stream_summary_printed = 1;
                    ImageStreamIO_closeIm(&tmpimg);
                }
            }
        }
        if (!stream_summary_printed)
        {
            TUI_newline();
        }

        for(int GUIline = 0; GUIline < GUIlineMax;
                GUIline++)   // GUIline is the line number on GUI display
        {
            for(level = 0; level < fpsCTRLvar->currentlevel; level ++)
            {
                if(level == 0)
                {
                    if(GUIline < keywnode[nodechain[0]].NBchild)
                    {
                        int knodeindex = keywnode[nodechain[0]].child[GUIline];
                        int fpsindex = keywnode[knodeindex].fpsindex;
                        fpsCTRLscreen_level0node_summary(fpsarray, fpsindex);
                    }
                    else
                    {
                        TUI_printfw("                    ");
                    }
                }

                if(GUIline < keywnode[nodechain[level]].NBchild)
                {
                    int snode = 0; // selected node
                    int knodeindex = keywnode[nodechain[level]].child[GUIline];

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

                    TUI_printfw("%-*.*s ", max_kw_width[level], max_kw_width[level], keywnode[knodeindex].keyword[level]);

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
                    TUI_printfw("%*s ", max_kw_width[level], " ");
                    TUI_printfw("  ");
                }
            }

            if(fpsCTRLvar->currentlevel == 0)
            {
                int knodeindex = keywnode[fpsCTRLvar->directorynodeSelected].child[GUIline];
                int fpsindex = keywnode[knodeindex].fpsindex;
                fpsCTRLscreen_level0node_summary(fpsarray, fpsindex);
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
                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel])
                    {
                        screenprint_setreverse();
                        fpsCTRLvar->nodeSelected = knodeindex;
                        fpsCTRLvar->fpsindexSelected = keywnode[knodeindex].fpsindex;
                    }


                    if(child_index[level] < keywnode[fpsCTRLvar->directorynodeSelected].NBchild)
                    {
                        screenprint_setcolor(5);
                        int l = keywnode[knodeindex].keywordlevel;
                        
                        TUI_printfw("%-*.*s", max_kw_width[cl], max_kw_width[cl], keywnode[knodeindex].keyword[l - 1]);
                        screenprint_unsetcolor(5);

                        if(GUIline == fpsCTRLvar->GUIlineSelected[cl])
                        {
                            TUI_printfw("> ");
                            screenprint_unsetreverse();
                        }
                        else
                        {
                            TUI_printfw("  ");
                        }
                    }
                    else TUI_printfw("%*s  ", max_kw_width[cl], " ");
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
                        if (fpsarray[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                            screenprint_setcolor(COLOR_OK);
                        }

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

                    if(GUIline == fpsCTRLvar->GUIlineSelected[cl])
                        screenprint_setreverse();

                    int is_resolved_stream = 0;
                    if (isVISIBLE == 1 && fpsarray[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                        if (fpsarray[fpsindex].parray[pindex].info.stream.streamID > -1) {
                            is_resolved_stream = 1;
                            screenprint_setcolor(2);
                        }
                    }

                    TUI_printfw(" ");
                    if (fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_PRIMARY_CLI_INPUT) {
                        screenprint_setreverse();
                    }
                    TUI_printfw("%-*.*s", max_kw_width[cl], max_kw_width[cl], fpsarray[fpsindex].parray[pindex].keyword[keywnode[knodeindex].keywordlevel - 1]);
                    if (fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_PRIMARY_CLI_INPUT) {
                        screenprint_unsetreverse();
                    }

                    if (is_resolved_stream) {
                        screenprint_unsetcolor(2);
                    }

                    if (isVISIBLE == 1 && fpsarray[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                        screenprint_unsetcolor(COLOR_OK);
                    }

                    if(GUIline == fpsCTRLvar->GUIlineSelected[cl])
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
                    if (fpsarray[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                        snprintf(valstring, 200, "%s", fpsarray[fpsindex].parray[pindex].val.string[0]);
                    } else {
                        functionparameter_GetParamValueString(&fpsarray[fpsindex].parray[pindex], valstring, 200);
                    }

                    /* Color path-like values green/red */
                    int path_val_color = 0;
                    if (isVISIBLE == 1 &&
                        valstring[0] != '\0')
                    {
                        int ptype = fpsarray[fpsindex]
                            .parray[pindex].type;
                        struct stat st;
                        int do_check = 0;

                        if (ptype == FPTYPE_STREAMNAME)
                        {
                            FPS_STREAMNAME_PARSED sp_v =
                                fps_streamname_parse(
                                    valstring);
                            char shmpath[512];
                            snprintf(shmpath,
                                sizeof(shmpath),
                                "/milk/shm/%s.im.shm",
                                sp_v.name);
                            if (stat(shmpath, &st) == 0)
                                path_val_color = 2;
                            else
                                path_val_color = 4;
                            do_check = 1;
                        }
                        else if (ptype == FPTYPE_DIRNAME)
                        {
                            if (stat(valstring,
                                     &st) == 0 &&
                                S_ISDIR(st.st_mode))
                                path_val_color = 2;
                            else
                                path_val_color = 4;
                            do_check = 1;
                        }
                        else if (ptype
                                 == FPTYPE_EXECFILENAME)
                        {
                            if (stat(valstring,
                                     &st) == 0 &&
                                S_ISREG(st.st_mode) &&
                                (st.st_mode & S_IXUSR))
                                path_val_color = 2;
                            else
                                path_val_color = 4;
                            do_check = 1;
                        }
                        else if (ptype == FPTYPE_FILENAME
                            || ptype
                                == FPTYPE_FITSFILENAME)
                        {
                            if (stat(valstring,
                                     &st) == 0 &&
                                S_ISREG(st.st_mode))
                                path_val_color = 2;
                            else
                                path_val_color = 4;
                            do_check = 1;
                        }

                        if (do_check &&
                            path_val_color != 0)
                        {
                            screenprint_setcolor(
                                path_val_color);
                        }
                    }
                    /* Highlight ONOFF ON state */
                    int onoff_on = 0;
                    if (isVISIBLE == 1 &&
                        fpsarray[fpsindex]
                            .parray[pindex].type
                            == FPTYPE_ONOFF &&
                        fpsarray[fpsindex]
                            .parray[pindex]
                            .val.i32[0])
                    {
                        onoff_on = 1;
                        screenprint_setreverse();
                    }

                    print_sliding_string(valstring, max_val_width, GUIline);

                    if (onoff_on) {
                        screenprint_unsetreverse();
                    }
                    if (path_val_color != 0) {
                        screenprint_unsetcolor(
                            path_val_color);
                    }
                    if(paramsync == 0 && isVISIBLE == 1) screenprint_unsetcolor(3);
                    if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR && isVISIBLE == 1) screenprint_unsetcolor(4);

                    TUI_printfw("    %-s", fpsarray[fpsindex].parray[pindex].description);

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

errno_t fpsCTRL_FPSlog(
    KEYWORD_TREE_NODE    *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    (void) keywnode;
    if (fpsCTRLvar->NBfps > 0)
    {
        int fpsidx = fpsCTRLvar->fpsindexSelected;
        if (fpsarray[fpsidx].md == NULL) {
            return RETURN_SUCCESS;
        }
        char datadir[FPS_DIR_STRLENMAX];
        strncpy(datadir, fpsarray[fpsidx].md->datadir, FPS_DIR_STRLENMAX - 1);

        TUI_printfw("LOG FOR FPS: %s  (directory: %s)\n", fpsarray[fpsidx].md->name, datadir);
        TUI_newline();

        char cmd[1024];
        // Find all .fps files, grep for comment lines, sort them, and take the last 40.
        snprintf(cmd, sizeof(cmd), "grep -h '#' %s/*.fps 2>/dev/null | sort | tail -n 40", datadir);
        
        FILE *fp = popen(cmd, "r");
        if (fp != NULL)
        {
            char line[1024];
            while (fgets(line, sizeof(line), fp) != NULL)
            {
                // Format: value # timestamp cnt0 [pid tid] comment
                // We want to show the timestamp and the comment part.
                TUI_printfw("%s", line);
            }
            pclose(fp);
        }
        else
        {
            TUI_printfw("No log entries found.\n");
        }
    }
    else
    {
        TUI_printfw("NO FPS LOADED\n");
    }

    return RETURN_SUCCESS;
}