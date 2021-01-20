/**
 * @file    stream_process_loop_simple.h
 *
 */

#ifndef _STREAM_PROCESS_LOOP_SIMPLE_H
#define _STREAM_PROCESS_LOOP_SIMPLE_H


errno_t stream_process_loop_simple_addCLIcmd();


errno_t milk_module_example__stream_process_loop_simple(
    char *streamA_name,
    char *streamB_name,
    long loopNBiter,
    int semtrig
);

#endif
