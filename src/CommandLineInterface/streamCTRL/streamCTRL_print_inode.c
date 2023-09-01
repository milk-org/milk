#include <stdint.h>
#include <sys/types.h>

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include "CLIcore.h"
#include "TUItools.h"
#include "streamCTRL_TUI.h"


int streamCTRL_print_inode(
    ino_t  inode,
    ino_t *upstreaminode,
    int    NBupstreaminode,
    int    downstreamindex
)
{
    int Dispinode_NBchar = 9;
    int is_upstream      = 0;
    int is_downstream    = 0;
    int upstreamindex    = 0;

    for(int i = 0; i < NBupstreaminode; i++)
    {
        if(inode == upstreaminode[i])
        {
            is_upstream   = 1;
            upstreamindex = i;
            break;
        }
    }

    if(downstreamindex < NO_DOWNSTREAM_INDEX)
    {
        is_downstream = 1;
    }

    if(is_upstream || is_downstream)
    {
        int colorcode = 3;
        if(upstreamindex > 0)
        {
            colorcode = 7;
        }

        if(is_upstream)
        {
            //attron(COLOR_PAIR(colorcode));
            screenprint_setcolor(colorcode);
            TUI_printfw("%02d >", upstreamindex);
            //attroff(COLOR_PAIR(colorcode));
            screenprint_unsetcolor(colorcode);
        }
        else
        {
            TUI_printfw("    ");
        }

        TUI_printfw("-");

        if(is_downstream)
        {
            int colorcode = 3;
            if(downstreamindex > 0)
            {
                colorcode = 7;
            }

            //attron(COLOR_PAIR(colorcode));
            screenprint_setcolor(colorcode);
            TUI_printfw("> %02d", downstreamindex);
            //attroff(COLOR_PAIR(colorcode));
            screenprint_unsetcolor(colorcode);
        }
        else
        {
            TUI_printfw("    ");
        }
    }
    else
    {
        TUI_printfw("%*d", Dispinode_NBchar, (int) inode);
    }

    return Dispinode_NBchar;
}
