/** @file read_binary32f.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID IMAGE_FORMAT_read_binary32f(const char *__restrict fname,
                                    long xsize,
                                    long ysize,
                                    const char *__restrict IDname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_FORMAT_read_binary32f_cli()
{
    if(CLI_checkarg(1, 3) + CLI_checkarg(2, 2) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 3) ==
            0)
    {
        IMAGE_FORMAT_read_binary32f(data.cmdargtoken[1].val.string,
                                    data.cmdargtoken[2].val.numl,
                                    data.cmdargtoken[3].val.numl,
                                    data.cmdargtoken[4].val.string);
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

errno_t read_binary32f_addCLIcmd()
{

    RegisterCLIcommand("readb32fim",
                       __FILE__,
                       IMAGE_FORMAT_read_binary32f_cli,
                       "read 32-bit float RAW image",
                       "<bin file> <xsize> <ysize> <output image>",
                       "readb32fim im.bin xsize ysize im",
                       "long IMAGE_FORMAT_read_binary32f(const char *fname, "
                       "long xsize, long ysize, const char *IDname)");

    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_read_binary32f(const char *__restrict fname,
                                    long xsize,
                                    long ysize,
                                    const char *__restrict IDname)
{
    DEBUG_TRACE_FSTART();

    FILE         *fp;
    float        *buffer;
    unsigned long fileLen;
    long          i, ii, jj;
    imageID       ID;
    //long v1;

    //Open file
    if((fp = fopen(fname, "rb")) == NULL)
    {
        PRINT_ERROR("Cannot open file");
        return (0);
    }

    //Get file length
    fseek(fp, 0, SEEK_END);
    fileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    //Allocate memory
    buffer = (float *) malloc(fileLen + 1);
    if(!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(fp);
        return (0);
    }

    //Read file contents into buffer
    if(fread(buffer, fileLen, 1, fp) < 1)
    {
        PRINT_ERROR("fread() returns <1 value");
    }
    fclose(fp);

    FUNC_CHECK_RETURN(create_2Dimage_ID(IDname, xsize, ysize, &ID));

    i = 0;
    for(jj = 0; jj < ysize; jj++)
        for(ii = 0; ii < xsize; ii++)
        {
            data.image[ID].array.F[jj * xsize + ii] = buffer[i];
            i++;
        }

    free(buffer);

    DEBUG_TRACE_FEXIT();
    return ID;
}
