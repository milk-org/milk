/**
 * @file streamCTRL_find_streams.h
 * @brief Streamctrl find streams module
 */

#ifndef _STREAMCTRL_FINDSTREAMS_H
#define _STREAMCTRL_FINDSTREAMS_H

#include "streamCTRL_TUI.h"


int find_streams(
    STREAMINFO              *streaminfo,
    int                     filter,
    const char * __restrict namefilter);

#endif
