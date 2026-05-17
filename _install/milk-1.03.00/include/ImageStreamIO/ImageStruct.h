/**
 * @file    ImageStruct.h
 * @brief   Image structure definition
 *
 * The IMAGE structure is defined here
 * Supports shared memory, low latency IPC through semaphores
 *
 * Dynamic allocation within IMAGE:
 * IMAGE includes a pointer to an array of IMAGE_METADATA (usually only one element, >1 element for polymorphism)
 * IMAGE includes a pointer to an array of KEYWORDS
 * IMAGE includes a pointer to a data array
 *
 *
 * @bug No known bugs.
 *
 */

#ifndef _IMAGESTRUCT_H
#define _IMAGESTRUCT_H

// #include "ImageStreamIO_config.h"
// #define IMAGESTRUCT_VERSION "2.00"

#define STRINGMAXLEN_IMAGE_NAME          80
#define STRINGMAXLEN_FILE_NAME          200
#define STRINGMAXLEN_DIR_NAME           800

#define KEYWORD_MAX_STRING  16            /**< maximun size of the keyword's name */
#define KEYWORD_MAX_COMMENT 80            /**< maximun size of a keyword's comment */

// comment if no write history
//#define IMAGESTRUCT_WRITEHISTORY

// number of entries in write history
#define IMAGESTRUCT_FRAMEWRITEMDSIZE 100


#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h> // TEST

#ifdef HAVE_CUDA
// CUDA runtime includes
#include <cuda_runtime_api.h>
#else
// CUDA cudaIpcMemHandle_t is a struct of 64 bytes
// This is needed for for compatibility between ImageStreamIO
// compiled with or without HAVE_CUDA precompiler flag
typedef char cudaIpcMemHandle_t[64];
#endif

#include "ImageStreamIOError.h"


#include <time.h>


#ifdef __cplusplus
extern "C" {
#endif

// comment this line if data should not be packed
// packing data should be use with extreme care, so it is recommended to disable this feature
//#define DATA_PACKED

#define SHAREDMEMDIR    "/milk/shm"    /**< default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR */

#define SEMAPHORE_MAXVAL        10     /**< maximum value for each of the semaphore, mitigates warm-up time when processes catch up with data that has accumulated */
#define SEMAPHORE_INITVAL        0     /**< initial value for each of the semaphore */
#define IMAGE_NB_SEMAPHORE      10     /**< Number of semaphores per image */

#define IMAGE_NB_PROCTRACE      10     /**< Number of STREAM_PROC_TRACE entries per image */

#define GPU_IMAGE_PLACEHOLDER    0     /**< GPU memory placeholder for image info */

// Data types are defined as machine-independent types for portability

#define _DATATYPE_UNINITIALIZED                        0

#define _DATATYPE_UINT8                                1  /**< uint8_t       = char */
#define SIZEOF_DATATYPE_UINT8                          1

#define _DATATYPE_INT8                                 2  /**< int8_t   */
#define SIZEOF_DATATYPE_INT8                           1

#define _DATATYPE_UINT16                               3  /**< uint16_t      usually = unsigned short int */
#define SIZEOF_DATATYPE_UINT16                         2

#define _DATATYPE_INT16                                4  /**< int16_t       usually = short int          */
#define SIZEOF_DATATYPE_INT16                          2

#define _DATATYPE_UINT32                               5  /**< uint32_t      usually = unsigned int       */
#define SIZEOF_DATATYPE_UINT32                         4

#define _DATATYPE_INT32                                6  /**< int32_t       usually = int                */
#define SIZEOF_DATATYPE_INT32                          4

#define _DATATYPE_UINT64                               7  /**< uint64_t      usually = unsigned long      */
#define SIZEOF_DATATYPE_UINT64                         8

#define _DATATYPE_INT64                                8  /**< int64_t       usually = long               */
#define SIZEOF_DATATYPE_INT64                          8

#define _DATATYPE_HALF                                13  /**< IEE 754 half-precision 16-bit (uses uint16_t for storage) */
#define SIZEOF_DATATYPE_HALF                           2

#define _DATATYPE_FLOAT                                9  /**< IEEE 754 single-precision binary floating-point format: binary32 */
#define SIZEOF_DATATYPE_FLOAT                          4

#define _DATATYPE_DOUBLE                              10 /**< IEEE 754 double-precision binary floating-point format: binary64 */
#define SIZEOF_DATATYPE_DOUBLE                         8

#define _DATATYPE_COMPLEX_FLOAT                       11  /**< complex_float  */
#define SIZEOF_DATATYPE_COMPLEX_FLOAT                  8

#define _DATATYPE_COMPLEX_DOUBLE                      12  /**< complex double */
#define SIZEOF_DATATYPE_COMPLEX_DOUBLE                16

#define _DATATYPE_EVENT_UI8_UI8_UI16_UI8              20
#define SIZEOF_DATATYPE_EVENT_UI8_UI8_UI16_UI8         5

#define Dtype                                          9   /**< default data type for floating point */
#define CDtype                                        11   /**< default data type for complex */





// semaphores control
// written by writer to control readers
// IMAGE.semctrl
#define IMAGE_SEMAPHORE_CONTROL_READY          0x00000001 /**< Semaphore ready for semwait. If 0, exit semwait calls until back to 1. This flag is used to notify readers that semaphores are going to be destroyed or re-created, or to pause readers for other reasons */

// semaphores status
// written by readers to communicate real-time status back to stream
// IMAGE.semstatus
#define IMAGE_SEMAPHORE_STATUS_CONNECTED       0x00000001  /**< semaphore is connected to PID */
#define IMAGE_SEMAPHORE_STATUS_SEMWAIT         0x00000002  /**< semaphore is being waited for by PID */
#define IMAGE_SEMAPHORE_STATUS_SEMREADYWAIT    0x00000004  /**< PID waiting for semaphore to be ready */
#define IMAGE_SEMAPHORE_STATUS_SEMTIMEOUT      0x00000008  /**< PID semwait timed out */


// Type of stream

#define CIRCULAR_BUFFER  0x0001  /**< Circular buffer, slice z axis is encoding time -> record writetime array */
#define MATH_DATA        0x0002  /**< Image is mathematical vector or matrix */
#define IMG_RECV         0x0004  /**< Image is stream received from another computer */
#define IMG_SENT         0x0008  /**< Image is stream sent to other computer */

// axis0 definition

#define ZAXIS_UNDEF      0x00000  /**< undefined (default) */
#define ZAXIS_SPACIAL    0x10000  /**< spatial coordinate */
#define ZAXIS_TEMPORAL   0x20000  /**< temporal coordinate */
#define ZAXIS_WAVELENGTH 0x30000  /**< wavelength coordinate */
#define ZAXIS_MAPPING    0x40000  /**< mapping index */

#define MAX_NB_PARTIAL_PACKET 512 /**< max number of partial packet used for partial write in teh SHM */


/** @brief  Keyword
 * The IMAGE_KEYWORD structure includes :
 * 	- name
 * 	- type
 * 	- value
 */
typedef struct
{
    char name[KEYWORD_MAX_STRING]; /**< keyword name                                                   */
    char type;                     /**< N: unused, L: long, D: double, S: 16-char string               */
    uint64_t : 0;                  // align array to 8-byte boundary for speed

    union
    {
        int64_t numl;
        double  numf;
        char    valstr[KEYWORD_MAX_STRING];
    } value;

    uint64_t cnt; // counter, incremented at each keyword write
    char comment[KEYWORD_MAX_COMMENT];
}
IMAGE_KEYWORD;


/** @brief structure holding two 8-byte integers
 *
 * Used in an union with struct timespec to ensure fixed 16 byte length
 */
typedef struct
{
    int64_t firstlong;
    int64_t secondlong;
} TIMESPECFIXED;

typedef struct
{
    float re;
    float im;
} complex_float;

typedef struct
{
    double re;
    double im;
} complex_double;




/** @brief Image metadata
 *
 *
 *
 */
typedef struct
{
    char version[32];
    /** Image structure version.
     *
     * should be equal to IMAGESTRUCT_VERSION
     *
     * Will be tested to ensure current software revision matches data.
     * If does not match, return error message with both versions.
     */

    /** @brief Image Name */
    char name[STRINGMAXLEN_IMAGE_NAME];


    /** @brief Number of axis
     *
     * @warning 1, 2 or 3. Values above 3 not supported.
     */
    uint8_t naxis;


    /** @brief Image size along each axis
     *
     *  If naxis = 1 (1D image), size[1] and size[2] are irrelevant
     */
    uint32_t size[3];


    /** @brief Number of elements in image
     *
     * This is computed upon image creation
     */
    uint64_t nelement;
    //uint64_t imdatamemsize; // image size [bytes] <- End of struct for retro compatibility



    /** @brief Data type
     *
     * Encoded according to data type defines.
     *  -  1: uint8_t
     * 	-  2: int8_t
     * 	-  3: uint16_t
     * 	-  4: int16_t
     * 	-  5: uint32_t
     * 	-  6: int32_t
     * 	-  7: uint64_t
     * 	-  8: int64_t
     * 	-  9: IEEE 754 single-precision binary floating-point format: binary32
     *  - 10: IEEE 754 double-precision binary floating-point format: binary64
     *  - 11: complex_float
     *  - 12: complex double
     *  - 13: half precision floating-point
     *
     */
    uint8_t datatype;





    uint64_t imagetype;              /**< image type */
    /**
     * 0x 0000 0000 0000 0001  Circular buffer, slice z axis is encoding time -> record writetime array
     * 0x 0000 0000 0000 0002  Image is mathematical vector or matrix
     * 0x 0000 0000 0000 0004  Image is stream received from another computer
     * 0x 0000 0000 0000 0008  Image is stream sent to other computer
     *
     * 0x 0000 0000 000X 0000  axis[0] encoding code (0-15):
     *    0: undefined (default)
     *    1: spatial coordinate
     *    2: temporal coordinate
     *    3: wavelength coordinate
     *    4: mapping index
     *
     *
     *
     */




    // relative timers using time relative to process start

    // double creationtime;             /**< Creation / load time of data structure (since process start)  */
    // double lastaccesstime;           /**< last time the image was accessed  (since process start)                      */



    // absolute timers using struct timespec

    struct timespec creationtime;
    struct timespec lastaccesstime;

    // time at which data was acquires/created. This time CAN be copied from input to output
    struct timespec atime;

    // last write time into data array
    struct timespec writetime;


    pid_t creatorPID;  /**< PID of process that created the stream (if shared = 1) */

    pid_t ownerPID;    /**< PID of process owning the stream (if shared = 1) */
    /* May be used to purge stream(s) when a process is completed/dead */
    /* Initialized to 0 */
    /* Set to 1 to indicate the stream does not belong to a process */


    uint8_t  shared;                   /**< stream is in shared memory */

    ino_t    inode;                    /**< inode nummber if shared memory */
    int8_t   location;                 /**< -1 if in CPU memory, >=0 if in GPU memory on `location` device               */
    uint8_t  status;                   /**< 1 to log image (default); 0 : do not log: 2 : stop log (then goes back to 2) */
    uint64_t flag;                     /**< bitmask, encodes read/write permissions.... NOTE: enum instead of defines */

    uint8_t  logflag;                  /**< set to 1 to start logging         */
    uint16_t sem;                      /**< number of semaphores supported, specified at image creation      */
    uint16_t NBproctrace;              /**< number of streamproctrace entries */


    uint64_t : 0; // align array to 8-byte boundary for speed

    uint64_t cnt0;               	/**< counter (incremented if image is updated)                                    */
    uint64_t cnt1;               	/**< in 3D rolling buffer image, this is the last slice written                   */
    uint64_t cnt2;                      /**< in cnt2-based syncronization, proceed until cnt0=cnt2                        */

    /**
     * Cross-process write mutex flag.
     *
     * Writers set to 1 before copying data, then to 0
     * after. Readers spin on it to avoid partial frames.
     *
     * volatile prevents the compiler from caching the
     * value in a register (critical for spin loops).
     * Use SHMIM_WRITE_* macros for proper CPU memory
     * fences via atomic builtins.
     */
    volatile uint8_t  write;


    uint16_t NBkw;                  /**< number of keywords (max: 65536)                                              */

    // fast circular memory buffer
    uint32_t CBsize;    // 0 if no CB allocated
    uint32_t CBindex;   // current index within buffer
    uint64_t CBcycle;   // number of buffer cycles

#ifdef IMAGESTRUCT_WRITEHISTORY
    // write history circ buffer
    uint32_t wCBindex;
    uint64_t wCBcycle;
#endif

    uint64_t imdatamemsize; // image size [bytes]

    cudaIpcMemHandle_t cudaMemHandle;

#ifdef DAO_COMPAT
    // The following fields are used for partial write in the SHM
    uint32_t lastPos;              /**< the 1st positon of the last write                                           */
    uint32_t lastNb;               /**< the number of last write                                                    */
    uint32_t packetNb;             /**< current partial packet write number                                         */
    uint32_t packetTotal;          /**< expected total number of packet                                             */ 
    uint64_t lastNbArray[MAX_NB_PARTIAL_PACKET]; /**< array containing the sub frame number                         */
#endif

} IMAGE_METADATA;



/** @brief STREAM_PROC_TRACE holds trigger and timing info
 *
 * Array of STREAM_PROC_TRACE is held within streams to track history.
 * This information is assembled by a process, and then written to
 * all streams it writes.
 *
 */
typedef struct
{
    int             triggermode;
    pid_t
    procwrite_PID;        /**< PID of process writing stream. 0 if no entry*/
    ino_t           trigger_inode;        /**< trigger stream inode */
    struct timespec ts_procstart;         /**< timestamp process triggered */
    struct timespec ts_streamupdate;      /**< timestamp write this stream */
    int             trigsemindex;         /**< trigger semaphore */
    int             triggerstatus;
    uint64_t cnt0;        /**< trigger stream cnt0 value at trigger */
} STREAM_PROC_TRACE;



/** @brief CBFRAMEMD fast access circular buffer metadata
 */
typedef struct
{
    uint64_t cnt0;
    uint64_t cnt1;
    struct timespec atime;
    struct timespec writetime;
} CBFRAMEMD;


// Write metadata
// keeps track of write times and PIDs
typedef struct
{
    uint64_t cnt0;
    pid_t wpid;       // write process PID
    struct timespec writetime;
} FRAMEWRITEMD;





#define STRINGMAXLEN_SEMFILENAME 200
typedef struct
{
    sem_t semdata;
} SEMFILEDATA;



/** @brief IMAGE structure
 * The IMAGE structure includes :
 *   - an array of IMAGE_KEWORD structures
 *   - an array of IMAGE_METADATA structures (usually only 1 element)
 *
 * IMPORTANT: memory allocations for dynamically allocated arrays need to be
 * included in the memory size computation in ImageStreamIO_createIm_gpu
 *
 */
typedef struct /**< structure used to store data arrays                      */
{
    char name[STRINGMAXLEN_IMAGE_NAME];     /**< local name (can be different from name in shared memory) */

    /** @brief Image usage flag
     *
     * 1 if image is used, 0 otherwise. \n
     * This flag is used when an array of IMAGE type is held in memory as a way to store multiple images. \n
     * When an image is freed, the corresponding memory (in array) is freed and this flag set to zero. \n
     * The active images can be listed by looking for IMAGE[i].used==1 entries.\n
     *
     */
    uint8_t used;

    int64_t createcnt; /**< increments when image is (re)-created */

    int32_t shmfd; /**< if shared memory, file descriptor */

    uint64_t memsize; /**< total size in memory if shared    */

    sem_t *semlog; /**< pointer to semaphore for logging  (8 bytes on 64-bit system) */

    IMAGE_METADATA *md;


    uint64_t : 0; // align array to 8-byte boundary for speed

    /** @brief data storage array
     *
     * The array is declared as a union, so that multiple data types can be supported \n
     *
     * For 2D image with pixel indices ii (x-axis) and jj (y-axis), the pixel values are stored as array.<TYPE>[ jj * md[0].size[0] + ii ] \n
     * image md[0].size[0] is x-axis size, md[0].size[1] is y-axis size
     *
     * For 3D image with pixel indices ii (x-axis), jj (y-axis) and kk (z-axis), the pixel values are stored as array.<TYPE>[ kk * md[0].size[1] * md[0].size[0] + jj * md[0].size[0] + ii ] \n
     * image md[0].size[0] is x-axis size, md[0].size[1] is y-axis size, md[0].size[2] is z-axis size
     *
     * @note Up to this point, all members of the structure have a fixed memory offset to the start point
     */
    union
    {
        void *raw;  // raw pointer

        uint8_t *UI8;  // char
        int8_t *SI8;

        uint16_t *UI16;  // unsigned short
        int16_t *SI16;

        uint32_t *UI32;
        int32_t *SI32;  // int

        uint64_t *UI64;
        int64_t *SI64;  // long

        float *F;
        double *D;

        complex_float *CF;
        complex_double *CD;

#ifdef DAO_COMPAT
        void * V; // add so the cpp interface and reinterpret cast using templaces to required type.
#endif
    } array; /**< pointer to data array */


    // Semaphores

    sem_t **semptr;                    /**< array of pointers to semaphores   (each 8 bytes on 64-bit system) */

    IMAGE_KEYWORD *kw;


    SEMFILEDATA *semfile;

    // PID of process that read shared memory stream
    // Initialized at 0. Otherwise, when process is waiting on semaphore, its PID is written in this array
    // The array can be used to look for available semaphores
    pid_t *semReadPID;

    // PID of the process posting the semaphores
    pid_t *semWritePID;

    // semaphore control, written by writer to control semaphore behavior
    // see SEMAPHORE_CONTROL_XXX defines for details
    uint32_t *semctrl;

    // semaphore status, written by readers to report back to stream what is their current status
    // see SEMAPHORE_STATUS_XXX defines for details
    uint32_t *semstatus;

    // array
    // keeps track of stream history/depedencies
    STREAM_PROC_TRACE *streamproctrace;


    uint64_t *flagarray;               /**<  flag for each slice if needed (depends on imagetype) */
    uint64_t *cntarray;                /**< For circular buffer: counter array for circular buffer, copy of cnt0 onto slice index  */

    // For each slice index: time at which data was acquires/created. This time CAN be copied from input to output
    struct timespec *atimearray;

    // For each slice index: time at which data was written. This time CAN be copied from input to output
    struct timespec *writetimearray;


    // Circular Buffer (CB) option
    // if CBsize>0, recent frames are memcpied in circular buffer
    // recent frames may be accessed in small CB for logging

    CBFRAMEMD * CircBuff_md; // circular buffer metadata
    void * CBimdata;         // data storage for circ buffer

#ifdef IMAGESTRUCT_WRITEHISTORY
    // Write history
    FRAMEWRITEMD *writehist;
#endif

} IMAGE;


/*
 * =========================================================
 * Atomic accessors for IMAGE_METADATA fields
 * =========================================================
 *
 * These macros provide proper CPU memory fences for
 * cross-process synchronization of shared memory fields.
 *
 * On x86-64 (TSO), release/acquire compile to plain
 * mov instructions plus a compiler barrier — zero
 * runtime overhead.  On ARM/other weak-memory
 * architectures, appropriate dmb fences are emitted.
 *
 * Usage:
 *   SHMIM_WRITE_ACQUIRE(md) — set write=1
 *   SHMIM_WRITE_RELEASE(md) — set write=0
 *   SHMIM_WRITE_LOAD(md)    — read write flag
 *   SHMIM_CNT0_INCREMENT(md) — atomically cnt0++
 *   SHMIM_CNT0_LOAD(md)     — read cnt0
 */

#if defined(__STDC_VERSION__) \
    && __STDC_VERSION__ >= 201112L \
    && !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>

#define SHMIM_WRITE_ACQUIRE(md) \
    atomic_store_explicit(      \
        (_Atomic uint8_t *)&(md)->write, \
        1, memory_order_release)

#define SHMIM_WRITE_RELEASE(md) \
    atomic_store_explicit(      \
        (_Atomic uint8_t *)&(md)->write, \
        0, memory_order_release)

#define SHMIM_WRITE_LOAD(md) \
    atomic_load_explicit(    \
        (_Atomic uint8_t *)&(md)->write, \
        memory_order_acquire)

#define SHMIM_CNT0_INCREMENT(md) \
    atomic_fetch_add_explicit(   \
        (_Atomic uint64_t *)&(md)->cnt0, \
        1, memory_order_release)

#define SHMIM_CNT0_LOAD(md) \
    atomic_load_explicit(    \
        (_Atomic uint64_t *)&(md)->cnt0, \
        memory_order_acquire)

#else
/* GCC/Clang __atomic builtins fallback */

#define SHMIM_WRITE_ACQUIRE(md) \
    __atomic_store_n(           \
        &(md)->write, 1, __ATOMIC_RELEASE)

#define SHMIM_WRITE_RELEASE(md) \
    __atomic_store_n(           \
        &(md)->write, 0, __ATOMIC_RELEASE)

#define SHMIM_WRITE_LOAD(md) \
    __atomic_load_n(         \
        &(md)->write, __ATOMIC_ACQUIRE)

#define SHMIM_CNT0_INCREMENT(md) \
    __atomic_add_fetch(          \
        &(md)->cnt0, 1, __ATOMIC_RELEASE)

#define SHMIM_CNT0_LOAD(md) \
    __atomic_load_n(         \
        &(md)->cnt0, __ATOMIC_ACQUIRE)

#endif


#ifdef __cplusplus
}  // extern "C"
#endif

#endif
