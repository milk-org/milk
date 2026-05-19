#include "streamCTRL_TUI_render_internal.h"

/**
 * @brief Render the streamCTRL footer status bar.
 */
void streamCTRL__render_footer(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state,
    int                         NBsinfodisp)
{
    /* ---- Scroll indicator footer ---- */
    if(sTUIparam.DisplayMode != DISPLAY_MODE_HELP)
    {
        int above = doffsetindex;
        int below = sTUIparam.NBsindex - (doffsetindex + NBsinfodisp);
        if(below < 0)
        {
            below = 0;
        }

        if(above > 0 || below > 0)
        {
            screenprint_setdim();
            if(above > 0)
            {
                screenprint_setcolor(3); /* yellow */
                TUI_printfw(" \033[1m\xe2\x86\x91\033[22m %d above ", above);
                screenprint_unsetcolor(3);
            }
            else
            {
                TUI_printfw(" -- top -- ");
            }

            TUI_printfw("|");

            if(below > 0)
            {
                screenprint_setcolor(3); /* yellow */
                TUI_printfw(" \033[1m\xe2\x86\x93\033[22m %d below ", below);
                screenprint_unsetcolor(3);
            }
            else
            {
                TUI_printfw(" -- end -- ");
            }
            screenprint_unsetdim();
        } /* if above > 0 || below > 0 */
        /* No trailing TUI_newline(): TUI_cleartobottom() clears
         * the rest of the footer row without risking a scroll. */
    } /* scroll indicator footer */

    TUI_cleartobottom();
    sc_frame_flush();
}
