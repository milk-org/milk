/**
 * @file    fps_create.h
 */

#ifndef _FPS_CREATE_H
#define FPS_CREATE_H


errno_t fps_create_addCLIcmd();

errno_t function_parameter_struct_create(
    int NBparamMAX,
    const char *name
);

#endif
