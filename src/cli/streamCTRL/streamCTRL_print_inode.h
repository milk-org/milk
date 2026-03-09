#ifndef _STREAMCTRL_PRINT_INODE_H
#define _STREAMCTRL_PRINT_INODE_H



int streamCTRL_print_inode(
    ino_t  inode,
    ino_t *upstreaminode,
    int    NBupstreaminode,
    int    downstreamindex
);

#endif

