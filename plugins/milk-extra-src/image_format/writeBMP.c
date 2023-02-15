/** @file writeBMP.c
 */

#include "CommandLineInterface/CLIcore.h"

const int BYTES_PER_PIXEL  = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

// Local variables pointers
static char *BMPfname;
static char *imRname;
static char *imGname;
static char *imBname;

static CLICMDARGDEF farg[] = {{
        CLIARG_STR,
        ".bmp_fname",
        "BMP file name",
        "out.bmp",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &BMPfname,
        NULL
    },
    {
        CLIARG_IMG,
        ".imRname",
        "Red channel image",
        "imR",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imRname,
        NULL
    },
    {
        CLIARG_IMG,
        ".imGname",
        "Green channel image",
        "imG",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imGname,
        NULL
    },
    {
        CLIARG_IMG,
        ".imBname",
        "Blue channel image",
        "imB",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imBname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "mkBMPim", "make BMP image", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Create BMP format image\n");

    return RETURN_SUCCESS;
}

static unsigned char *createBitmapFileHeader(int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] =
    {
        0,
        0, /// signature
        0,
        0,
        0,
        0, /// image file size in bytes
        0,
        0,
        0,
        0, /// reserved
        0,
        0,
        0,
        0, /// start of pixel array
    };

    fileHeader[0]  = (unsigned char)('B');
    fileHeader[1]  = (unsigned char)('M');
    fileHeader[2]  = (unsigned char)(fileSize);
    fileHeader[3]  = (unsigned char)(fileSize >> 8);
    fileHeader[4]  = (unsigned char)(fileSize >> 16);
    fileHeader[5]  = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

static unsigned char *createBitmapInfoHeader(int height, int width)
{
    static unsigned char infoHeader[] =
    {
        0, 0, 0, 0, /// header size
        0, 0, 0, 0, /// image width
        0, 0, 0, 0, /// image height
        0, 0,       /// number of color planes
        0, 0,       /// bits per pixel
        0, 0, 0, 0, /// compression
        0, 0, 0, 0, /// image size
        0, 0, 0, 0, /// horizontal resolution
        0, 0, 0, 0, /// vertical resolution
        0, 0, 0, 0, /// colors in color table
        0, 0, 0, 0, /// important color count
    };

    infoHeader[0]  = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[4]  = (unsigned char)(width);
    infoHeader[5]  = (unsigned char)(width >> 8);
    infoHeader[6]  = (unsigned char)(width >> 16);
    infoHeader[7]  = (unsigned char)(width >> 24);
    infoHeader[8]  = (unsigned char)(height);
    infoHeader[9]  = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return infoHeader;
}

static void generateBitmapImage(unsigned char *image,
                                int            height,
                                int            width,
                                char          *imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3]  = {0, 0, 0};
    int           paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE *imageFile = fopen(imageFileName, "wb");

    unsigned char *fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char *infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for(i = 0; i < height; i++)
    {
        fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

errno_t image_writeBMP(const char *__restrict IDnameR,
                       const char *__restrict IDnameG,
                       const char *__restrict IDnameB,
                       char *__restrict outname)
{
    imageID        IDR, IDG, IDB;
    uint32_t       width;
    uint32_t       height;
    unsigned char *array;
    uint32_t       ii, jj;

    printf("Function %s\n", __FUNCTION__);

    IDR    = image_ID(IDnameR);
    IDG    = image_ID(IDnameG);
    IDB    = image_ID(IDnameB);
    width  = (uint32_t) data.image[IDR].md[0].size[0];
    height = (uint32_t) data.image[IDR].md[0].size[1];

    array =
        (unsigned char *) malloc(sizeof(unsigned char) * width * height * 3);
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(ii = 0; ii < width; ii++)
        for(jj = 0; jj < height; jj++)
        {
            array[(jj * width + ii) * 3] =
                (unsigned char)(data.image[IDB]
                                .array.F[(height - jj - 1) * width + ii]);

            array[(jj * width + ii) * 3 + 1] =
                (unsigned char)(data.image[IDG]
                                .array.F[(height - jj - 1) * width + ii]);

            array[(jj * width + ii) * 3 + 2] =
                (unsigned char)(data.image[IDR]
                                .array.F[(height - jj - 1) * width + ii]);
        }
    generateBitmapImage(array, height, width, outname);

    free(array);

    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_writeBMP(imRname, imGname, imBname, BMPfname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_image_format__mkBMPimage()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
