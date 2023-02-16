/** @file cudacomp_MVMextractModesLoop.h
 */

#ifdef HAVE_CUDA

errno_t cudacomp_MVMextractModesLoop_addCLIcmd();

int __attribute__((hot)) CUDACOMP_MVMextractModesLoop(
    const char *in_stream,     // input stream
    const char *intot_stream,  // [optional]   input normalization stream
    const char *IDmodes_name,  // Modes matrix
    const char *IDrefin_name,  // [optional] input reference  - to be subtracted
    const char *IDrefout_name, // [optional] output reference - to be added
    const char *IDmodes_val_name, // ouput stream
    int         GPUindex,         // GPU index
    int         PROCESS,          // 1 if postprocessing
    int         TRACEMODE,        // 1 if writing trace
    int         MODENORM,         // 1 if input modes should be normalized
    int         insem,            // input semaphore index
    int         axmode, // 0 for normal mode extraction, 1 for expansion
    long        twait,  // if >0, insert time wait [us] at each iteration
    int         semwarn // 1 if warning when input stream semaphore >1
);

#endif
