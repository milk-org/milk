/**
 * @file    image_format.c
 * @brief   Convert between image formats
 *
 * read and write images other than FITS
 *
 */

#define MODULE_SHORTNAME_DEFAULT "imgformat"
#define MODULE_DESCRIPTION       "Conversion between image format, I/O"

#include "CommandLineInterface/CLIcore.h"

#include "CR2toFITS.h"
#include "FITS_to_floatbin_lock.h"
#include "FITS_to_ushortintbin_lock.h"
#include "combineHDR.h"
#include "stream_temporal_stats.h"
#include "extract_RGGBchan.h"
#include "extract_utr.h"
#include "imtoASCII.h"
#include "loadCR2toFITSRGB.h"
#include "read_binary32f.h"
#include "writeBMP.h"



/*typedef struct
{
    int rows;
    int cols;
    unsigned char *data;
} sImage;
*/
/* This pragma is necessary so that the data in the structures is aligned to 2-byte
   boundaries.  Some different compilers have a different syntax for this line.  For
   example, if you're using cc on Solaris, the line should be #pragma pack(2).
*/
//#pragma pack(2)

INIT_MODULE_LIB(image_format)

static errno_t init_module_CLI()
{

    CLIADDCMD_image_format__extractRGGBchan();

    CLIADDCMD_image_format__combineHDR();
    CLIADDCMD_image_format__cred_cds_utr();
    CLIADDCMD_image_format__temporal_stats();

    imtoASCII_addCLIcmd();

    CLIADDCMD_image_format__mkBMPimage();
    //	writeBMP_addCLIcmd();

    CR2toFITS_addCLIcmd();
    loadCR2toFITSRGB_addCLIcmd();
    FITS_to_floatbin_lock_addCLIcmd();
    FITS_to_ushortintbin_lock_addCLIcmd();
    read_binary32f_addCLIcmd();

    // add atexit functions here

    return RETURN_SUCCESS;
}

/*


long getImageInfo(
    FILE *inputFile,
    long  offset,
    int   numberOfChars
)
{
    unsigned char			*ptrC;
    long				value = 0L;
    int				i;
    unsigned char			dummy;

    dummy = '0';
    ptrC = &dummy;

    fseek(inputFile, offset, SEEK_SET);

    for(i = 1; i <= numberOfChars; i++)
    {
        if(fread(ptrC, sizeof(char), 1, inputFile) < 1)
        {
            PRINT_ERROR("fread() returns <1 value");
        }

        // calculate value based on adding bytes
        value = (long)(value + (*ptrC) * (pow(256, (i - 1))));
    }

    return(value);
}





// ASCII format:
// ii jj value
// one line per pixel
imageID read_ASCIIimage(
    const char *filename,
    const char *ID_name,
    long        xsize,
    long        ysize
)
{
    long ID;
    FILE *fp;

    ID = create_2Dimage_ID(ID_name, xsize, ysize);

    fp = fopen(filename, "r");
    if(fp == NULL)
    {
        fprintf(stderr, "ERROR: cannot open file \"%s\"\n", filename);
    }
    else
    {
        long iipix, jjpix;
        float value;


        while((fscanf(fp, "%ld %ld %f\n", &iipix, &jjpix, &value)) == 3)
            if((iipix > -1) && (iipix < xsize) && (jjpix > -1) && (jjpix < ysize))
            {
                data.image[ID].array.F[jjpix * xsize + iipix] = value;
            }
        fclose(fp);
    }

    return(ID);
}




// ASCII format:
// value
imageID read_ASCIIimage1(
    const char *filename,
    const char *ID_name,
    long        xsize,
    long        ysize
)
{
    imageID ID;
    FILE *fp;

    ID = create_2Dimage_ID(ID_name, xsize, ysize);

    fp = fopen(filename, "r");
    if(fp == NULL)
    {
        fprintf(stderr, "ERROR: cannot open file \"%s\"\n", filename);
    }
    else
    {
        long ii, jj;
        double value;

        for(ii = 0; ii < xsize; ii++)
            for(jj = 0; jj < ysize; jj++)
            {
                if(fscanf(fp, "%lf", &value) == 1)
                {
                    data.image[ID].array.F[jj * xsize + ii] = value;
                }
                else
                {
                    PRINT_ERROR("read error");
                    exit(0);
                }
            }
        fclose(fp);
    }

    return(ID);
}




errno_t read_BMPimage(
    char       *filename,
    const char *IDname_R,
    const char *IDname_G,
    const char *IDname_B
)
{
    FILE				*bmpInput, *rasterOutput;
    sImage			originalImage;
    unsigned char			someChar;
    unsigned char			*pChar;
    long				fileSize;
    int				nColors;
    int				r, c;
    unsigned int BlueValue, RedValue, GreenValue;
    long IDR, IDG, IDB;



    //--------INITIALIZE POINTER----------
    someChar = '0';
    pChar = &someChar;

    printf("Reading file %s\n", filename);

    // ----DECLARE INPUT AND OUTPUT FILES----
    bmpInput = fopen(filename, "rb");
    rasterOutput = fopen("data24.txt", "w");

    fseek(bmpInput, 0L, SEEK_END);

    //-----GET BMP INFO-----
    originalImage.cols = (int)getImageInfo(bmpInput, 18, 4) + 1;
    originalImage.rows = (int)getImageInfo(bmpInput, 22, 4);
    fileSize = getImageInfo(bmpInput, 2, 4);
    nColors = getImageInfo(bmpInput, 46, 4);

    //----PRINT BMP INFO TO SCREEN-----
    printf("Width: %d\n", originalImage.cols);
    printf("Height: %d\n", originalImage.rows);
    printf("File size: %ld\n", fileSize);
    printf("Bits/pixel: %ld\n", getImageInfo(bmpInput, 28, 4));
    printf("No. colors: %d\n", nColors);


    IDR = create_2Dimage_ID(IDname_R, (long) originalImage.cols,
                            (long) originalImage.rows);
    IDG = create_2Dimage_ID(IDname_G, (long) originalImage.cols,
                            (long) originalImage.rows);
    IDB = create_2Dimage_ID(IDname_B, (long) originalImage.cols,
                            (long) originalImage.rows);

    // ----FOR 24-BIT BMP, THERE IS NO COLOR TABLE-----
    fseek(bmpInput, 54, SEEK_SET);

    // -----------READ RASTER DATA-----------
    for(r = 0; r <= originalImage.rows - 1; r++)
    {
        for(c = 0; c <= originalImage.cols - 1; c++)
        {

            // ----READ FIRST BYTE TO GET BLUE VALUE-----
            if(fread(pChar, sizeof(char), 1, bmpInput) < 1)
            {
                PRINT_ERROR("fread() returns <1 value");
            }
            BlueValue = *pChar;

            // -----READ NEXT BYTE TO GET GREEN VALUE-----
            if(fread(pChar, sizeof(char), 1, bmpInput) < 1)
            {
                PRINT_ERROR("fread() returns <1 value");
            }
            GreenValue = *pChar;

            // -----READ NEXT BYTE TO GET RED VALUE-----
            if(fread(pChar, sizeof(char), 1, bmpInput) < 1)
            {
                PRINT_ERROR("fread() returns <1 value");
            }
            RedValue = *pChar;

            // ---------WRITE TO FILES ---------
            data.image[IDR].array.F[r * originalImage.cols + c] = 1.0 * RedValue;
            data.image[IDG].array.F[r * originalImage.cols + c] = 1.0 * GreenValue;
            data.image[IDB].array.F[r * originalImage.cols + c] = 1.0 * BlueValue;
        }
    }

    fclose(bmpInput);
    fclose(rasterOutput);

    return RETURN_SUCCESS;
}






imageID loadCR2(
    const char *fnameCR2,
    const char *IDname
)
{
    imageID ID;

    EXECUTE_SYSTEM_COMMAND("dcraw -t 0 -D -4 -c %s > _tmppgm.pgm", fnameCR2);

    ID = read_PGMimage("_tmppgm.pgm", IDname);
    if(system("rm _tmppgm.pgm") != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    return ID;
}




// load all images matching strfilter + .CR2
// return number of images converted
// FITS image name = CR2 image name with .CR2 -> .fits
long CR2toFITS_strfilter(
    const char *strfilter
)
{
    long cnt = 0;
    char fname[STRINGMAXLEN_FULLFILENAME];
    char fname1[STRINGMAXLEN_FULLFILENAME];
    FILE *fp;

    EXECUTE_SYSTEM_COMMAND("ls %s.CR2 > flist.tmp\n", strfilter);

    fp = fopen("flist.tmp", "r");
    while(fgets(fname, STRINGMAXLEN_FULLFILENAME, fp) != NULL)
    {
		int slen = strlen(fname);
		if(slen > STRINGMAXLEN_FULLFILENAME-1) {
			slen = STRINGMAXLEN_FULLFILENAME-1;
		}

        fname[slen - 1] = '\0';

        strncpy(fname1, fname, slen - 4);

        fname1[slen - 4] = '.';
        fname1[slen - 3] = 'f';
        fname1[slen - 2] = 'i';
        fname1[slen - 1] = 't';
        fname1[slen] = 's';
        fname1[slen + 1] = '\0';

        CR2toFITS(fname, fname1);
        printf("File %s  -> file %s\n", fname, fname1);
        cnt++;
    }

    fclose(fp);

    EXECUTE_SYSTEM_COMMAND("rm flist.tmp");

    printf("%ld files converted\n", cnt);

    return(cnt);
}







//
// assembles 4 channels into a single image (inverse operation of routine above)
//
imageID image_format_reconstruct_from_RGGBchan(
    const char *IDr_name,
    const char *IDg1_name,
    const char *IDg2_name,
    const char *IDb_name,
    const char *IDout_name
)
{
    imageID ID;
    imageID IDr, IDg1, IDg2, IDb;
    long xsize1, ysize1, xsize2, ysize2;
    long ii1, jj1;
    int RGBmode = 0;
    imageID ID00, ID01, ID10, ID11;


    IDr = image_ID(IDr_name);
    IDg1 = image_ID(IDg1_name);
    IDg2 = image_ID(IDg2_name);
    IDb = image_ID(IDb_name);
    xsize1 = data.image[IDr].md[0].size[0];
    ysize1 = data.image[IDr].md[0].size[1];

    xsize2 = 2 * xsize1;
    ysize2 = 2 * ysize1;

    if((xsize2 == 4770) && (ysize2 == 3178))
    {
        RGBmode = 1;
    }
    if((xsize2 == 5202) && (ysize2 == 3465))
    {
        RGBmode = 2;
    }

    if(RGBmode == 0)
    {
        PRINT_ERROR("Unknown RGB image mode\n");
        exit(0);
    }

    if(RGBmode == 1) // GBRG
    {
        ID00 = IDg1;
        ID10 = IDb;
        ID01 = IDr;
        ID11 = IDg2;
    }

    if(RGBmode == 2)
    {
        ID00 = IDr;
        ID10 = IDg1;
        ID01 = IDg2;
        ID11 = IDb;
    }

    ID = create_2Dimage_ID(IDout_name, xsize2, ysize2);

    for(ii1 = 0; ii1 < xsize1; ii1++)
        for(jj1 = 0; jj1 < ysize1; jj1++)
        {
            data.image[ID].array.F[(2 * jj1 + 1)*xsize2 + 2 * ii1] =
                data.image[ID01].array.F[jj1 * xsize1 + ii1];
            data.image[ID].array.F[2 * jj1 * xsize2 + 2 * ii1] =
                data.image[ID00].array.F[jj1 * xsize1 + ii1];
            data.image[ID].array.F[(2 * jj1 + 1)*xsize2 + (2 * ii1 + 1)] =
                data.image[ID11].array.F[jj1 * xsize1 + ii1];
            data.image[ID].array.F[2 * jj1 * xsize2 + (2 * ii1 + 1)] =
                data.image[ID10].array.F[jj1 * xsize1 + ii1];
        }

    return(ID);
}













// ron: readout noise in ADU
// gain: in e-/ADU
// alpha: 0 - 1, sets quantization noise at alpha x overall noise
// bias: image bias in ADU
imageID IMAGE_FORMAT_requantize(
    const char *IDin_name,
    const char *IDout_name,
    double      alpha,
    double      RON,
    double      gain,
    double      bias
)
{
    imageID IDin, IDout;
    long ii;
    long xsize, ysize;

    IDin = image_ID(IDin_name);
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];

    IDout = create_2Dimage_ID(IDout_name, xsize, ysize);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        double value;

        value = data.image[IDin].array.F[ii];
        value = value - bias;
        if(value < 0.0)
        {
            value = value / (alpha * RON);
        }
        else
        {
            value = 2.0 / alpha * sqrt(gain) * (sqrt(gain * RON * RON + value) - sqrt(
                                                    gain) * RON);
        }
        data.image[IDout].array.F[ii] = value + 0.5;
    }

    return(IDout);
}







imageID IMAGE_FORMAT_dequantize(
    const char *IDin_name,
    const char *IDout_name,
    double alpha,
    double RON,
    double gain,
    double bias
)
{
    imageID IDin, IDout;
    long ii;
    long xsize, ysize;

    IDin = image_ID(IDin_name);
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];

    IDout = create_2Dimage_ID(IDout_name, xsize, ysize);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        double value;

        value = data.image[IDin].array.F[ii];
        if(value < 0.0)
        {
            value = value * alpha * RON + bias;
        }
        else
        {
            value = alpha / 2.0 * value / sqrt(gain) + RON * sqrt(gain);
            value = value * value;
            value = value - gain * RON * RON + bias;
        }
        data.image[IDout].array.F[ii] = value;
    }

    return(IDout);
}




imageID IMAGE_FORMAT_read_binary16(
    const char *fname,
    long xsize,
    long ysize,
    const char *IDname
)
{
    FILE *fp;
    char *buffer;
    unsigned long fileLen;
    long ii, jj;
    imageID ID = -1;

    //Open file
    if((fp = fopen(fname, "rb")) == NULL)
    {
        PRINT_ERROR("Cannot open file");
        exit(0);
    }


    //Get file length
    fseek(fp, 0, SEEK_END);
    fileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    //Allocate memory
    buffer = (char *)malloc(fileLen + 1);
    if(!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(fp);
        return(0);
    }

    //Read file contents into buffer
    if(fread(buffer, fileLen, 1, fp) < 1)
    {
        PRINT_ERROR("fread() returns <1 value");
    }
    fclose(fp);

    ID = create_2Dimage_ID(IDname, xsize, ysize);

    unsigned long i = 0;
    for(jj = 0; jj < ysize; jj++)
        for(ii = 0; ii < xsize; ii++)
        {
            long v1;

            if(i < fileLen + 1)
            {
                v1 = (long)(((unsigned const char *)buffer)[i]) + (long)(256 * ((
                            unsigned const char *)buffer)[i + 1]);
            }
            data.image[ID].array.F[jj * xsize + ii] = (float) v1;
            i += 2;
        }

    free(buffer);

    return ID;
}



*/
