#ifndef _STREAMCTRL_PRINT_PROCPID_H
#define _STREAMCTRL_PRINT_PROCPID_H


int streamCTRL_print_procpid(
    int      DispPID_NBchar,
    pid_t    procpid,
    pid_t   *upstreamproc,
    int      NBupstreamproc,
    uint32_t mode
);

#endif
