/** @file FITS_to_ushortintbin_lock.c
 */

#include <sys/file.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID IMAGE_FORMAT_FITS_to_ushortintbin_lock(const char *__restrict IDname,
        const char *__restrict fname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_FORMAT_FITS_to_ushortintbin_lock_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) == 0)
    {
        IMAGE_FORMAT_FITS_to_ushortintbin_lock(data.cmdargtoken[1].val.string,
                                               data.cmdargtoken[2].val.string);
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t FITS_to_ushortintbin_lock_addCLIcmd()
{

    RegisterCLIcommand("writeushortintlock",
                       __FILE__,
                       IMAGE_FORMAT_FITS_to_ushortintbin_lock_cli,
                       "write unsigned short int with file locking",
                       "str1 is image, str2 is binary file",
                       "writeushortintlock im im.bin",
                       "long IMAGE_FORMAT_FITS_to_ushortintbin_lock( const "
                       "char *IDname, const char *fname)");

    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_FITS_to_ushortintbin_lock(const char *__restrict IDname,
        const char *__restrict fname)
{
    imageID             ID;
    long                xsize, ysize;
    long                ii;
    int                 fd;
    unsigned short int *valarray;

    ID    = image_ID(IDname);
    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];

    valarray = (unsigned short int *) malloc(sizeof(unsigned short int) *
               xsize * ysize);
    if(valarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if(data.image[ID].md[0].datatype == _DATATYPE_FLOAT)
    {
        printf("float -> unsigned short int array\n");
        for(ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = (unsigned short int) data.image[ID].array.F[ii];
        }
    }
    if(data.image[ID].md[0].datatype == _DATATYPE_DOUBLE)
    {
        printf("double -> unsigned short int array\n");
        for(ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = (unsigned short int) data.image[ID].array.D[ii];
        }
    }

    fd = open(fname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    flock(fd, LOCK_EX);
    if(fd < 0)
    {
        perror("Error opening file");
        printf("Error opening file \"%s\": %s\n", fname, strerror(errno));
    }
    if(write(fd, valarray, sizeof(unsigned short int) * xsize * ysize) < 1)
    {
        PRINT_ERROR("write() returns <1 value");
    }
    flock(fd, LOCK_UN);
    close(fd);

    free(valarray);

    return ID;
}
