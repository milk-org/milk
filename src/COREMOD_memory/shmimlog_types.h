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

    int saveascii;
    // 0 : Not saving ascii
    // 1 : Saving ascii: arraycnt0, arraycnt1, arraytime
    // 2 : ???

    char fname_auxFITSheader[STRINGMAXLEN_FULLFILENAME];

    char      fnameascii[STRINGMAXLEN_FULLFILENAME]; // name of frame to be saved
    uint64_t *arrayindex;
    uint64_t *arraycnt0;
    uint64_t *arraycnt1;

    double *arraytime;   // time at which frame has arrived
    double *arrayaqtime; // frame source time, earlier
} STREAMSAVE_THREAD_MESSAGE;

typedef struct
{
    int       on; /**<  1 if logging, 0 otherwise */
    long long cnt;
    long long filecnt;
    long      interval; /**<  log every x frames (default = 1) */
    int       logexit;  /**<  toggle to 1 when exiting */
    char      fname[STRINGMAXLEN_FULLFILENAME];

    // circular buffer
    uint32_t CBsize;

    uint32_t CBindex; // last frame grabbed
    uint64_t CBcycle; // last frame grabbed
} LOGSHIM_CONF;

#endif
