/** @file logshmim_types.h
 *
 * data logging of shared memory image stream
 *
 */

#ifndef CLICORE_MEMORY_LOGSHMIM_TYPES_H
#define CLICORE_MEMORY_LOGSHMIM_TYPES_H


typedef struct
{
    char iname[STRINGMAXLEN_IMGNAME];
    char fname[STRINGMAXLEN_FULLFILENAME];
    int  partial;  // 1 if partial cube
    long cubesize; // size of the cube
    float timespan; // execution time for saving

    int saveascii;
    // 0 : Not saving ascii
    // 1 : Saving ascii: arraycnt0, arraycnt1, arraytime
    // 2 : ???
    char compress_string[200];

    char fname_auxFITSheader[STRINGMAXLEN_FULLFILENAME];

    char      fnameascii[STRINGMAXLEN_FULLFILENAME]; // name of frame to be saved
    uint64_t *arrayindex;
    uint64_t *arraycnt0;
    uint64_t *arraycnt1;

    double *arraytime;   // time at which frame has arrived
    double *arrayaqtime; // frame source time, earlier
} STREAMSAVE_THREAD_MESSAGE;




#endif
