/** @file cudacompinit.h
 */

#ifdef HAVE_CUDA

errno_t cudacompinit_addCLIcmd();

int CUDACOMP_init();

void *GPU_scanDevices(void *deviceCount_void_ptr);

#endif
