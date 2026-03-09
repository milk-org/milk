#ifndef _STREAMCTRL_PRINT_TRACE_H
#define _STREAMCTRL_PRINT_TRACE_H


errno_t streamCTRL_print_SPTRACE_details(
    IMAGE   *streamCTRLimages,
    imageID  ID,
    pid_t   *upstreamproc,
    int      NBupstreamproc,
    uint32_t print_pid_mode
);

#endif

