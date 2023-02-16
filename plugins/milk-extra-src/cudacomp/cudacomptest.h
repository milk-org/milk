/** @file cudacomptest.h
 */

#ifdef HAVE_CUDA

errno_t cudacomptest_addCLIcmd();

errno_t GPUcomp_test(__attribute__((unused)) long NBact,
                     long                         NBmodes,
                     long                         WFSsize,
                     long                         GPUcnt);

#endif
