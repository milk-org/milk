/**
 * @file    ImageStreamIO.h
 * @brief   Function prototypes for ImageStreamIO
 *
 *
 */



#ifndef _IMAGESTREAMIO_H
#define _IMAGESTREAMIO_H

//If CLOCK_ISIO is not defined, define it as CLOCK_TAI if available, fallback to CLOCK_REALTIME
#ifndef CLOCK_ISIO
    #ifdef CLOCK_TAI
        #define CLOCK_ISIO CLOCK_TAI
    #else
        #define CLOCK_ISIO CLOCK_REALTIME
    #endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#include "ImageStruct.h"

void __attribute__((constructor)) libinit_ImageStreamIO();

#define ROUND_UP_8(x) (((x) + 7) & (-8))

errno_t init_ImageStreamIO();

/** @brief Set the error reporting function to the default provided by the library.
  *
  * \returns IMAGESTREAMIO_SUCCESS on success
  * \returns IMAGESTREAMIO_FAILURE on an error
  */
errno_t ImageStreamIO_set_default_printError();

/** @brief Set the error reporting function.
  * The new function supplied by the pointer will be called whenever a library function reports an error.
  * Pass `NULL` to turn off error reporting from within the library.
  *
  * \param new_printError is a pointer to the function to use for reporting errors. Can be NULL.
  *
  * \returns IMAGESTREAMIO_SUCCESS on success
  * \returns IMAGESTREAMIO_FAILURE on an error
  */
errno_t ImageStreamIO_set_printError(errno_t (*new_printError)(const char *,
                                     const char *, int, errno_t, char *));


/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 0. Utilities                                                             */
/**@{                                                                                              */
/* =============================================================================================== */
/* =============================================================================================== */

inline uint64_t ImageStreamIO_nbSlices(const IMAGE *image)
{
    return (image->md->naxis == 3 ? image->md->size[2] : 1);
}

inline uint64_t ImageStreamIO_writeIndex(const IMAGE *image)
{
    return (image->md->cnt1 + 1) % ImageStreamIO_nbSlices(image);
}

inline uint64_t ImageStreamIO_readLastWroteIndex(const IMAGE *image)
{
    return (image->md->naxis == 3 ? image->md->cnt1 : 0);
}

/** @brief Get the raw pointer to the beginning of the slice slice_index.
  *
  *
  * ## Purpose
  *
  * Return the raw pointer to the beginning of the slice slice_index
  *
  * ## Arguments
  *
  * @param[in]
  * image	IMAGE*
  * 			pointer to shmim
  *
  * @param[in]
  * indec	const int
  * 			slice_index of the slice to read
  *
  * @param[out]
  * buffer	void**
  * 			pointer to the beginning of the slice
  *
  * \return the error code
  */
errno_t ImageStreamIO_readBufferAt(
    const IMAGE *image,
    const unsigned int slice_index,
    void **buffer
);

/** @brief Get the raw pointer where the producer should write.
  *
  *
  * ## Purpose
  *
  * Return the raw pointer where the producer should write
  *
  * ## Arguments
  *
  * @param[in]
  * image	IMAGE*
  * 			pointer to shmim
  *
  * @param[out]
  * buffer	void**
  * 			raw pointer where the producer should write
  *
  * \return the error code
  */
inline errno_t ImageStreamIO_writeBuffer(
    const IMAGE *image, ///< [in] the name of the shared memory file
    void **buffer ///< [out] raw pointer where the producer should write
)
{
    const uint64_t write_index = ImageStreamIO_writeIndex(image);
    return ImageStreamIO_readBufferAt(image, write_index, buffer);
}


/** @brief Get the raw pointer where the consumer will find the last frame wrote.
  *
  *
  * ## Purpose
  *
  * Return the raw pointer where the consumer will find the last frame wrote
  *
  * ## Arguments
  *
  * @param[in]
  * image	IMAGE*
  * 			pointer to shmim
  *
  * @param[out]
  * buffer	void**
  * 			raw pointer where the consumer will find the last frame wrote
  *
  * \return the error code
  */
inline errno_t ImageStreamIO_readLastWroteBuffer(
    const IMAGE *image, ///< [in] the name of the shared memory file
    void **buffer ///< [out] raw pointer where the consumer will find the last frame wrote
)
{
    const int64_t read_index = ImageStreamIO_readLastWroteIndex(image);
    return ImageStreamIO_readBufferAt(image, read_index, buffer);
}

/** @brief Get the standard stream filename.
  *
  * Fills in the \p file_name string with the standard shared memory image path, e.g.
  *  \code
  *    char file_name[64];
  *    ImageStreamIO_filename(file_name, 64, "image00");
  *    printf("%s\n", file_name);
  *  \endcode
  * produces the output:
  *  \verbatim
  *    /milk/shm/image00.im.shm
  8  \endverbatim
  *
  * \returns IMAGESTREAMIO_SUCCESS on success
  * \returns IMAGESTREAMIO_FAILURE on error
  */
errno_t ImageStreamIO_filename(
    char *file_name,     ///< [out] the file name string to fill in
    size_t ssz,          ///< [in] the allocated size of file_name
    const char *im_name  ///< [in] the image name
);

/** @brief Get the size in bytes from the data type code.
  *
  * \returns the size in bytes of the data type if valid
  * \returns -1 if atype is not valid
  */
int ImageStreamIO_typesize(uint8_t
                           atype  /**< [in] the type code (see ImageStruct.h*/
                          );


const char* ImageStreamIO_typename(
    uint8_t datatype
);
const char* ImageStreamIO_typename_short(
    uint8_t datatype
);
const char* ImageStreamIO_typename_7(
    uint8_t datatype
);


int ImageStreamIO_checktype(uint8_t datatype, int complex_allowed);

/** @brief Get the appropriate floating point type for arithmetic from any type
  *
  * \returns the atype of the matching float type
  * \returns -1 if atype is not valid
  */
int ImageStreamIO_floattype(
    uint8_t datatype
);

/** @brief Get the FITSIO BITPIX from the data type code.
  *
  * \returns the BITPIX if atype valid
  * \returns -1 if atype is not valid
  */
int ImageStreamIO_FITSIObitpix(uint8_t
                        atype
                        );

int ImageStreamIO_FITSIOdatatype(uint8_t datatype);

errno_t ImageStreamIO_check_image_inode(IMAGE* image);
errno_t ImageStreamIO_check_image_endpoint_inode(IMAGE *image);
errno_t ImageStreamIO_autorelink_if_need_if_can(IMAGE *image);
errno_t ImageStreamIO_new_image_compatible(IMAGE* source_image, IMAGE* new_image);


///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 1. READ / WRITE STREAM                                                   */
/**@{                                                                                              */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Create shared memory image stream (legacy API) */
errno_t ImageStreamIO_createIm(
    IMAGE *image,      ///< [out] IMAGE structure which will have its members allocated and initialized.
    const char
    *name,  ///< [in] the name of the shared memory file will be data.tmpfsdir/<name>_im.shm
    long naxis,        ///< [in] number of axes in the image.
    uint32_t *size,    ///< [in] the size of the image along each axis.  Must have naxis elements.
    uint8_t atype,     ///< [in] data type code
    int shared,        ///< [in] if true then a shared memory buffer is allocated.  If false, only local storage is used.
    int NBkw,          ///< [in] the number of keywords to allocate.
    int CBsize         ///< [in] Circular Buffer size

);

/** @brief Create shared memory image stream */
errno_t ImageStreamIO_createIm_gpu(
    IMAGE *image,      ///< [out] IMAGE structure which will have its members allocated and initialized.
    const char
    *name,  ///< [in] the name of the shared memory file will be data.tmpfsdir/<name>_im.shm
    long naxis,        ///< [in] number of axes in the image.
    uint32_t *size,    ///< [in] the size of the image along each axis.  Must have naxis elements.
    uint8_t atype,     ///< [in] data type code
    int8_t location,   ///< [in] if -1 then a CPU memory buffer is allocated. If >=0, GPU memory buffer is allocated on devive `location`.
    int shared,        ///< [in] if true then a shared memory buffer is allocated.  If false, only local storage is used.
    int NBsem,         ///< [in] the number of semaphores to allocate.
    int NBkw,          ///< [in] the number of keywords to allocate.
    uint64_t imagetype,///< [in] type of the stream
    uint32_t CBsize    ///< [in] Number of circ buff frames if shared mem, 0 if unused
);

/** @brief Deallocate and remove an IMAGE structure.
  *
  * For a shared image:
  * Closes all semaphores, deallcoates sem pointers,
  * and removes associated files. Unmaps the shared memory
  * segment, and finally removes the file. Sets the metadata and
  * keyword pointers to NULL.
  *
  * For a non-shred image:
  * Deallocates all arrays and sets pointers to NULL.
  *
  * \returns IMAGESTREAMIO_SUCCESS on success
  * \returns IMAGESTREAMIO_FAILURE on an error (but currently no checks done)
  *
  */
errno_t ImageStreamIO_destroyIm(
    IMAGE *image /**< [in] The IMAGE structure to deallocate and remove from the system.*/
);

/** @brief Connect to an existing shared memory image stream
  *
  * Wrapper for  \ref ImageStreamIO_read_sharedmem_image_toIMAGE
  */
errno_t ImageStreamIO_openIm(
    IMAGE *image,    ///< [out] IMAGE structure which will be attached to the existing IMAGE
    const char
    *name ///< [in] the name of the shared memory file will be data.tmpfsdir/<name>_im.shm
);

void *ImageStreamIO_get_image_d_ptr(IMAGE *image);


/** @brief Read / connect to existing shared memory image stream */
errno_t ImageStreamIO_read_sharedmem_image_toIMAGE(
    const char
    *name, ///< [in] the name of the shared memory file to access, as in data.tmpfsdir/<name>_im.shm
    IMAGE *image      ///< [out] the IMAGE structure to connect to the stream
);


/** @brief Close a shared memmory image stream.
  *
  * For use in clients, detaches and cleans up memory used by non-owner process.
  *
  * \returns IMAGESTREAMIO_SUCCESS on success
  * \returns the appropriate error code otherwise if an error occurs
  *
  */
errno_t ImageStreamIO_closeIm(IMAGE
                              *image /**< [in] A real-time image structure which contains the image data and meta-data.*/);

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 2. MANAGE SEMAPHORES                                                     */
/**@{                                                                                              */
/* =============================================================================================== */
/* =============================================================================================== */


/** @brief Post all shmim semaphores
 *
 * ## Purpose
 *
 * Posts semaphore of a shmim
 * if index < 0, post all semaphores
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			index of semaphore to be posted
 *          if index=-1, post all semaphores
 */
long ImageStreamIO_sempost(
    IMAGE *image,  ///< [in] the name of the shared memory file
    long index     ///< [in] semaphore index
);



/** @brief Post all shmim semaphores except one
 *
 * ## Purpose
 *
 * Posts all semaphores of a shmim except one
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			index of semaphore to be excluded
 */
long ImageStreamIO_sempost_excl(
    IMAGE *image,  ///< [in] the name of the shared memory file
    long index     ///< [in] semaphore index
);



/** @brief Post shmim semaphores at regular time interval
 *
 * ## Purpose
 *
 * Posts all semaphores of a shmim at regular time intervals
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			is =-1, post all semaphores
 *
 * @param[in]
 * dtus     time interval [us]
 *
 */
long ImageStreamIO_sempost_loop(
    IMAGE *image,  ///< [in] the name of the shared memory file
    long index,    ///< [in] semaphore index
    long dtus
);

/** @brief Get available semaphore index
 *
 *
 *
 */
int ImageStreamIO_getsemwaitindex(
    IMAGE *image,  ///< [in] the name of the shared memory file
    int semindexdefault
);


/** @brief Wait for semaphore
 *
 * ## Purpose
 *
 * Wait on a shmim semaphore
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 *
 */
int ImageStreamIO_semwait(
    IMAGE *image,  ///< [in] the name of the shared memory file
    int index      ///< [in] semaphore index
);
int ImageStreamIO_semtrywait(
    IMAGE *image,  ///< [in] the name of the shared memory file
    int index      ///< [in] semaphore index
);
int ImageStreamIO_semtimedwait(
    IMAGE *image,  ///< [in] the name of the shared memory file
    int index,     ///< [in] semaphore index
    const struct timespec *semwts
);


/** @brief Flush all semaphores of a shmim
 *
 * ## Purpose
 *
 * Flush shmim semaphore
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			flush all semaphores if index<0
 *
 */
long ImageStreamIO_semflush(
    IMAGE *image,  ///< [in] the name of the shared memory file
    long index
);


long ImageStreamIO_semvalue(
    IMAGE *image,
    long index); // Warning returns in-band error if semID is bad.



/** @brief Update image metadata and post semaphores after image is updated
 *
 * Increments counters, sets times, sets write flag to zero, and posts semaphores.
 * Should be called each time image content is updated
 *
 */
long ImageStreamIO_UpdateIm_atime( IMAGE *image,           ///< [out] pointer to the IMAGE to update
                                   struct timespec * atime ///< [in] the acquisition time, if NULL writetime is used for atime
                                 );

/** @brief Update image metadata and post semaphores after image is updated
 *
 * Increments counters, sets times, sets write flag to zero, and posts semaphores.
 * Should be called each time image content is updated
 *
 * Acquisition time (atime) will be set to the write time.
 *
 */
long ImageStreamIO_UpdateIm( IMAGE *image /**< [out] pointer to the IMAGE to update*/);



///@}


#ifdef __cplusplus
} //extern "C"
#endif


#endif


