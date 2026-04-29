/**
 * @file fpsCTRL_FPSdisplay.c
 * @brief Fpsctrl fpsdisplay module
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include "fps.h"
#include "fps_internal.h"
#include "fpsCTRL_TUIcompat.h"
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
    double time_in_cycle = fmod(
        (double)now.tv_sec + (double)now.tv_nsec * 1e-9 + (double)row_index * 0.5,
        cycle_time);
    
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
            TUI_printfw("Detailed Help for FPS '%s':", fpsarray[fpsCTRLvar->fpsindexSelected].md->name);
            TUI_newline();
            TUI_printfw("--------------------------");
            TUI_newline();
            screenprint_unsetbold();
            TUI_printfw("%s", fpsarray[fpsCTRLvar->fpsindexSelected].md->helptext);
            TUI_newline();
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
                    functionparameter_GetParamValueString(
                        &fpsarray[fpsindex].parray[pindex],
                        valstring,
                        200);
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
        int available_width = sc_term_cols - reserved_width;
        if (available_width < 40) available_width = 40;
        
        if (max_kw_width[cl] > available_width / 2) max_kw_width[cl] = available_width / 2;
        
        // max_val_width is handled by sliding print, but we need enough space for description

        TUI_newline();

        // 1-line summary for selected parameter
        int summary_printed = 0;
        if (keywnode[fpsCTRLvar->nodeSelected].leaf)
        {
            int fpsidx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            int pidx = keywnode[fpsCTRLvar->nodeSelected].pindex;
            if (fpsarray[fpsidx].parray[pidx].type == FPTYPE_STREAMNAME)
            {
                IMAGE tmpimg;
                FPS_STREAMNAME_PARSED sp_sum = fps_streamname_parse(fpsarray[fpsidx].parray[pidx].val.string[0]);
                char shmpath_sum[STRINGMAXLEN_FILE_NAME];
                int shm_sum_ok =
                    (ImageStreamIO_filename(
                         shmpath_sum,
                         sizeof(shmpath_sum),
                         sp_sum.name)
                     == IMAGESTREAMIO_SUCCESS)
                    && (access(shmpath_sum, F_OK) == 0);
                if (shm_sum_ok
                    && ImageStreamIO_openIm(
                           &tmpimg, sp_sum.name)
                        == IMAGESTREAMIO_SUCCESS)
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
                    
                    int sumcolor = 2;
                    if (sp_sum.must_new)
                    {
                        sumcolor = 4;
                    }
                    else if (sp_sum.loc == 'L')
                    {
                        sumcolor = 3;
                    }
                    screenprint_setcolor(sumcolor);
                    TUI_printfw("%s", stream_info);
                    screenprint_unsetcolor(sumcolor);
                    TUI_newline();
                    summary_printed = 1;
                    ImageStreamIO_closeIm(&tmpimg);
                }
            }

            /* Trigger mode description */
            if (!summary_printed)
            {
                const char *kfull =
                    fpsarray[fpsidx]
                        .parray[pidx].keywordfull;
                const char *needle = "triggermode";
                int nlen = strlen(needle);
                int klen = strlen(kfull);

                if (klen >= nlen
                    && strcmp(kfull + klen - nlen,
                             needle) == 0)
                {
                    long tval =
                        fpsarray[fpsidx]
                            .parray[pidx]
                            .val.i64[0];
                    const char *tdesc;
                    switch (tval)
                    {
                    case 0:
                        tdesc = "IMMEDIATE"
                            " -- run without"
                            " waiting";
                        break;
                    case 1:
                        tdesc = "CNT0"
                            " -- wait for cnt0"
                            " increment";
                        break;
                    case 2:
                        tdesc = "CNT1"
                            " -- wait for cnt1"
                            " increment";
                        break;
                    case 3:
                        tdesc = "SEMAPHORE"
                            " -- wait for"
                            " semaphore post";
                        break;
                    case 4:
                        tdesc = "DELAY"
                            " -- wait for"
                            " fixed time delay";
                        break;
                    case 5:
                        tdesc = "SEMAPHORE+"
                            "TIMEOUT"
                            " -- semaphore with"
                            " timeout propagation";
                        break;
                    case 6:
                        tdesc = "CNT2"
                            " -- demand-driven"
                            " (cnt0 < cnt2)";
                        break;
                    default:
                        tdesc = "UNKNOWN";
                        break;
                    }
                    screenprint_setcolor(2);
                    TUI_printfw(
                        "TRIGGER %ld: %s",
                        tval, tdesc);
                    screenprint_unsetcolor(2);
                    TUI_newline();
                    summary_printed = 1;
                }
            }
        }
        short unsigned int wrow=0, wcol=0;
        TUI_get_terminal_size(&wrow, &wcol);

        /* Reserve footer rows: scroll indicator + blank
         * + status line + 1 margin = 4 rows. */
        int footer_rows = 4;
        int dispindexMax = (int) wrow - sc_cursor_row
                           - footer_rows;
        if(dispindexMax < 5) dispindexMax = 5;

        int cl_scroll = fpsCTRLvar->currentlevel;
        if(cl_scroll < 0) cl_scroll = 0;
        
        int pindexActiveSelected = fpsCTRLvar->GUIlineSelected[cl_scroll];
        int doffsetindex = fpsCTRLvar->display_offset[cl_scroll];

        int margin = 2; // Keep at least 2 items visible above/below the cursor
        
        if (pindexActiveSelected < doffsetindex + margin) {
            doffsetindex = pindexActiveSelected - margin;
        }
        if (pindexActiveSelected > doffsetindex + dispindexMax - 1 - margin) {
            doffsetindex = pindexActiveSelected - dispindexMax + 1 + margin;
        }
        
        if (doffsetindex > GUIlineMax - dispindexMax) doffsetindex = GUIlineMax - dispindexMax;
        if (doffsetindex < 0) doffsetindex = 0;
        
        fpsCTRLvar->display_offset[cl_scroll] = doffsetindex;

        int lastindex = doffsetindex + dispindexMax;
        if (lastindex > GUIlineMax) lastindex = GUIlineMax;

        if (!summary_printed)
        {
            TUI_newline();
        }

        int root_offset = fpsCTRLvar->display_offset[0];
        if (root_offset > 0) {
            screenprint_setbold();
            screenprint_setcolor(3);
            TUI_printfw("  ^^^^ (%d more entries above) ^^^^", root_offset);
            screenprint_unsetcolor(3);
            screenprint_unsetbold();
            TUI_newline();
        }

        for (int l = 0; l < MAXNBLEVELS; l++) {
            if (l == fpsCTRLvar->currentlevel) {
                child_index[l] = doffsetindex;
            } else if (l < fpsCTRLvar->currentlevel) {
                int start = fpsCTRLvar->display_offset[l];
                int n_items = keywnode[nodechain[l]].NBchild;
                int sel = fpsCTRLvar->GUIlineSelected[l];
                
                if (sel < start + margin) start = sel - margin;
                if (sel > start + dispindexMax - 1 - margin) start = sel - dispindexMax + 1 + margin;
                if (start > n_items - dispindexMax) start = n_items - dispindexMax;
                if (start < 0) start = 0;
                
                child_index[l] = start;
                fpsCTRLvar->display_offset[l] = start;
            } else {
                child_index[l] = doffsetindex;
            }
        }

        for(int GUIline = doffsetindex; GUIline < lastindex;
                GUIline++)   // GUIline is the line number on GUI display
        {
            for(level = 0; level < fpsCTRLvar->currentlevel; level ++)
            {
                int c_idx = child_index[level];

                /* Determine if this row is the selected
                 * node in the chain for this level. */
                int is_chain_node = 0;
                if(c_idx >= 0
                    && c_idx < keywnode[nodechain[level]].NBchild)
                {
                    int v1 = keywnode[nodechain[level]]
                        .child[c_idx];
                    int v2 = nodechain[level + 1];
                    if(v1 == v2)
                    {
                        is_chain_node = 1;
                    }
                }

                /* Fade non-selected entries so the
                 * active chain stands out. */
                if(!is_chain_node)
                {
                    screenprint_setdim();
                }

                if(level == 0)
                {
                    if(c_idx >= 0 && c_idx < keywnode[nodechain[0]].NBchild)
                    {
                        int knodeindex = keywnode[nodechain[0]].child[c_idx];
                        int fpsindex = keywnode[knodeindex].fpsindex;
                        fpsCTRLscreen_level0node_summary(fpsarray, fpsindex);
                    }
                    else
                    {
                        TUI_printfw("                    ");
                    }
                }

                if(c_idx >= 0 && c_idx < keywnode[nodechain[level]].NBchild)
                {
                    int snode = 0; // selected node
                    int knodeindex = keywnode[nodechain[level]].child[c_idx];

                    // toggle highlight if node is in the chain
                    if(is_chain_node)
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
                    screenprint_setnormal();
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
                    int isTRIGGER = 0;
                    if(!(fpsarray[fpsindex].parray[pindex].fpflag &
                            FPFLAG_VISIBLE))   // if invisible
                    {
                        isVISIBLE = 0;
                        screenprint_setdim();
                        screenprint_setblink();
                    }

                    /* Trigger stream: subtle bg */
                    if (isVISIBLE
                        && (fpsarray[fpsindex]
                            .parray[pindex].fpflag
                            & FPFLAG_TRIGGER_STREAM))
                    {
                        isTRIGGER = 1;
                        screenprint_setcolor(
                            COLOR_TRIGGER_BG);
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
                            if (isTRIGGER)
                                screenprint_setcolor(
                                    COLOR_TRIGGER_BG);
                            screenprint_unsetblink();
                        }
                        else
                        {
                            screenprint_setcolor(4);
                            screenprint_setblink();
                            TUI_printfw("NW"); // non writable
                            screenprint_unsetcolor(4);
                            if (isTRIGGER)
                                screenprint_setcolor(
                                    COLOR_TRIGGER_BG);
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
                        if(GUIline != fpsCTRLvar->GUIlineSelected[cl])
                        {
                            screenprint_setbold();
                        }
                    }
                    TUI_printfw("%-*.*s", max_kw_width[cl], max_kw_width[cl], fpsarray[fpsindex].parray[pindex].keyword[keywnode[knodeindex].keywordlevel - 1]);
                    if (fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_PRIMARY_CLI_INPUT) {
                        if(GUIline != fpsCTRLvar->GUIlineSelected[cl])
                        {
                            screenprint_unsetbold();
                        }
                    }

                    if (is_resolved_stream) {
                        screenprint_unsetcolor(2);
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
                    }

                    if (isVISIBLE == 1 && fpsarray[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                        screenprint_unsetcolor(COLOR_OK);
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
                    }

                    if(GUIline == fpsCTRLvar->GUIlineSelected[cl])
                    {
                        screenprint_unsetcolor(10);
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
                        TUI_printfw("> ");
                        screenprint_unsetreverse();
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
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
                        functionparameter_GetParamValueString(
                            &fpsarray[fpsindex].parray[pindex],
                            valstring,
                            200);
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
                            int exists =
                                (stat(shmpath,
                                      &st) == 0);
                            if (sp_v.loc == 'L')
                            {
                                path_val_color = 3;
                            }
                            else if (sp_v.must_new
                                && exists)
                            {
                                path_val_color = 4;
                            }
                            else if (
                                sp_v.must_exist
                                && !exists)
                            {
                                path_val_color = 4;
                            }
                            else if (exists)
                            {
                                path_val_color = 2;
                            }
                            else if (sp_v.must_new)
                            {
                                path_val_color = 2;
                            }
                            else
                            {
                                path_val_color = 3;
                            }
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
                        if(GUIline == fpsCTRLvar->GUIlineSelected[cl])
                        {
                            screenprint_setreverse();
                        }
                        else
                        {
                            screenprint_setbold();
                        }
                    }

                    print_sliding_string(valstring, max_val_width, GUIline);

                    if (onoff_on) {
                        if(GUIline == fpsCTRLvar->GUIlineSelected[cl])
                        {
                            screenprint_unsetreverse();
                        }
                        else
                        {
                            screenprint_unsetbold();
                        }
                    }
                    if (path_val_color != 0) {
                        screenprint_unsetcolor(
                            path_val_color);
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
                    }
                    if(paramsync == 0 && isVISIBLE == 1)
                    {
                        screenprint_unsetcolor(3);
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
                    }
                    if(fpsarray[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR && isVISIBLE == 1)
                    {
                        screenprint_unsetcolor(4);
                        if (isTRIGGER)
                            screenprint_setcolor(
                                COLOR_TRIGGER_BG);
                    }

                    TUI_printfw("    %-s", fpsarray[fpsindex].parray[pindex].description);

                    if(GUIline == fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] && isVISIBLE == 1)
                        screenprint_unsetbold();

                    if(isVISIBLE == 0)
                    {
                        screenprint_unsetblink();
                        screenprint_unsetdim();
                    }
                    if (isTRIGGER)
                    {
                        screenprint_unsetbgcolor();
                    }
                }
                for(int l = 0; l < MAXNBLEVELS ; l++) child_index[l]++;
            }
            TUI_newline();
        }

        int lastindex_root = fpsCTRLvar->display_offset[0] + dispindexMax;
        if (lastindex_root < GUIlineMax) {
            screenprint_setbold();
            screenprint_setcolor(3);
            TUI_printfw("  vvvv (%d more entries below) vvvv", GUIlineMax - lastindex_root);
            screenprint_unsetcolor(3);
            screenprint_unsetbold();
            TUI_newline();
        }

        fpsCTRLvar->NBindex = GUIlineMax; // Matches number of lines displayed

        if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] > fpsCTRLvar->NBindex - 1)
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = fpsCTRLvar->NBindex - 1;

        TUI_newline();

        if(fpsarray[fpsCTRLvar->fpsindexSelected].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK)
        {
            screenprint_setcolor(2);
            TUI_printfw("[%ld] PARAMETERS OK - RUN function good to go", fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt);
            TUI_newline();
            screenprint_unsetcolor(2);
        }
        else
        {
            screenprint_setcolor(4);
            TUI_printfw("[%ld] %d PARAMETER SETTINGS ERROR(s) :",
                        fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt,
                        fpsarray[fpsCTRLvar->fpsindexSelected].md->conferrcnt);
            TUI_newline();
            screenprint_unsetcolor(4);
            screenprint_setbold();
            for(int msgi = 0; msgi < fpsarray[fpsCTRLvar->fpsindexSelected].md->msgcnt; msgi++)
            {
                int pidx = fpsarray[fpsCTRLvar->fpsindexSelected].md->msgpindex[msgi];
                TUI_printfw("%-40s %s",
                            fpsarray[fpsCTRLvar->fpsindexSelected].parray[pidx].keywordfull,
                            fpsarray[fpsCTRLvar->fpsindexSelected].md->message[msgi]);
                TUI_newline();
            }
            screenprint_unsetbold();
        }
    }
    else
    {
        TUI_newline();
        TUI_newline();
        screenprint_setbold();
        TUI_printfw(
            "  NO FPS LOADED");
        screenprint_unsetbold();
        TUI_newline();
        TUI_newline();
        TUI_printfw(
            "  Waiting for FPS shared"
            " memory files ...");
        TUI_newline();
        TUI_printfw(
            "  Press [s] to rescan,"
            " [x] to exit");
        TUI_newline();
    }

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

        TUI_printfw("LOG FOR FPS: %s  (directory: %s)", fpsarray[fpsidx].md->name, datadir);
        TUI_newline();
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
            TUI_printfw("No log entries found.");
            TUI_newline();
        }
    }
    else
    {
        TUI_printfw("NO FPS LOADED");
        TUI_newline();
    }

    return RETURN_SUCCESS;
}