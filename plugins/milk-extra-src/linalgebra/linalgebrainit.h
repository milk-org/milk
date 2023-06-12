/** @file linalgebrainit.h
 */

#ifdef HAVE_CUDA

errno_t linalgebrainit_addCLIcmd();

int LINALGEBRA_init();

void *GPU_scanDevices(void *deviceCount_void_ptr);

#endif
