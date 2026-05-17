#ifndef IMAGESTREAMIO_ERROR_H
#define IMAGESTREAMIO_ERROR_H

#ifndef IMAGESTREAMIO_SUCCESS

#define IMAGESTREAMIO_SUCCESS        (0) 
#define IMAGESTREAMIO_FAILURE        (1)  // generic error code

#define IMAGESTREAMIO_INVALIDARG    (10)  //For arguments not in range or otherwise invalid
#define IMAGESTREAMIO_NOTIMPL       (20)  //For methods or features not implemented or supported
#define IMAGESTREAMIO_BADALLOC      (30)  //Memory allocation failed
#define IMAGESTREAMIO_FILEOPEN      (40)  //Error opening file
#define IMAGESTREAMIO_FILESEEK      (42)  //error seeking on file
#define IMAGESTREAMIO_FILEWRITE     (44)  //error writing to file
#define IMAGESTREAMIO_FILEEXISTS    (46)  //error existing file
#define IMAGESTREAMIO_INODE         (48)  //error getting inode
#define IMAGESTREAMIO_MMAP          (50)  //mmap or munmap error
#define IMAGESTREAMIO_SEMINIT       (60)  //semaphore initialization error
#define IMAGESTREAMIO_VERSION      (100)  //For when the wrong ImageStreamIO version is found


#endif

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#endif // IMAGESTREAMIO_ERROR_H
