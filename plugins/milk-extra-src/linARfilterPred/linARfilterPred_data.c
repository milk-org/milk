#include "linARfilterPred_internal.h"

/* =============================================================================================== */
/*                                                                                                 */
/* 1. INITIALIZATION                                                                               */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 2. I/O TOOLS                                                                                    */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

int NBwords(const char sentence[])
{
    int counted = 0; // result

    // state:
    const char *it     = sentence;
    int         inword = 0;

    do
        switch(*it)
        {
            case '\0':
            case ' ':
            case '\t':
            case '\n':
            case '\r':
                if(inword)
                {
                    inword = 0;
                    counted++;
                }
                break;
            default:
                inword = 1;
        }
    while(*it++);

    return counted;
}

/**
 * @brief load ascii file(s) into image cube
 *
 *  resamples sequence(s) of data points
 * INPUT FILES HAVE TO BE NAMED seq000.dat, seq001.dat etc...
 *
 * file starts at tstart, sampling = dt
 * NBpt per file
 * NBfr files
*/

long LINARFILTERPRED_LoadASCIIfiles(
    double tstart, double dt, long NBpt, long NBfr, const char *IDoutname)
{
    FILE       *fp;
    long        NBfiles;
    double      runtime;
    char        fname[200];
    struct stat fstat;
    int         fOK;
    long        NBvarin[200];
    long        fcnt;
    FILE       *fparray[200];
    long        kk;
    size_t      linesiz = 0;
    char       *linebuf = 0;
    //ssize_t linelen=0;
    //int     ret;
    long    vcnt;
    double  ftime0[200];
    double  var0[200][200];
    double  ftime1[200];
    double  var1[200][200];
    double  varC[200][200];
    float   alpha;
    long    nbvar;
    long    fr;
    char    imoutname[200];
    FILE   *fpout;
    imageID IDout[200];
    //int     HPfilt = 1; // high pass filter
    float HPgain = 0.005;

    long ii;
    long kkpt, kkfr;

    runtime = tstart;

    fOK     = 1;
    NBfiles = 0;
    nbvar   = 0;
    while(fOK == 1)
    {
        snprintf(fname, sizeof(fname),
                 "seq%03ld.dat", NBfiles);
        if(stat(fname, &fstat) == 0)
        {
            printf("Found file %s\n", fname);
            fflush(stdout);
            fp = fopen(fname, "r");
            //linelen =
            if(getline(&linebuf, &linesiz, fp) == -1)
            {
                PRINT_ERROR("getline error");
            }
            fclose(fp);
            NBvarin[NBfiles] = NBwords(linebuf) - 1;
            free(linebuf);
            linebuf = NULL;
            printf("   NB variables = %ld\n", NBvarin[NBfiles]);
            nbvar += NBvarin[NBfiles];
            NBfiles++;
        }
        else
        {
            printf("No more files\n");
            fflush(stdout);
            fOK = 0;
        }
    }
    printf("NBfiles = %ld\n", NBfiles);

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        snprintf(fname, sizeof(fname),
                 "seq%03ld.dat", fcnt);
        printf("   %03ld  OPENING FILE %s\n", fcnt, fname);
        fflush(stdout);
        fparray[fcnt] = fopen(fname, "r");
    }

    kk      = 0; // time
    runtime = tstart;

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        if(fscanf(fparray[fcnt], "%lf", &ftime0[fcnt]) != 1)
        {
            PRINT_ERROR("fscanf error");
        }

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            if(fscanf(fparray[fcnt], "%lf", &var0[fcnt][vcnt]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
        }
        if(fscanf(fparray[fcnt], "\n") != 0)
        {
            PRINT_ERROR("fscanf error");
        }

        if(fscanf(fparray[fcnt], "%lf", &ftime1[fcnt]) != 1)
        {
            PRINT_ERROR("fscanf error");
        }

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            if(fscanf(fparray[fcnt], "%lf", &var1[fcnt][vcnt]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
        }
        if(fscanf(fparray[fcnt], "\n") != 0)
        {
            PRINT_ERROR("fscanf error");
        }

        printf("FILE %ld :  \n", fcnt);
        printf(" time :    %20f  %20f\n", ftime0[fcnt], ftime1[fcnt]);
        fflush(stdout);

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            printf("    variable %3ld   :   %20f  %20f\n",
                   vcnt,
                   var0[fcnt][vcnt],
                   var1[fcnt][vcnt]);
            varC[fcnt][vcnt] = var0[fcnt][vcnt];
        }
        printf("\n");
    }

    for(fr = 0; fr < NBfr; fr++)
    {
        snprintf(imoutname, sizeof(imoutname),
                 "%s_%03ld", IDoutname, fr);
        create_3Dimage_ID(imoutname, nbvar, 1, NBpt, &(IDout[fr]));
    }

    fpout = fopen("out.txt", "w");

    kk   = 0;
    kkpt = 0;
    kkfr = 0;
    while(kkfr < NBfr)
    {
        fprintf(fpout, "%20f", runtime);

        ii = 0;
        for(fcnt = 0; fcnt < NBfiles; fcnt++)
        {
            while(ftime1[fcnt] < runtime)
            {
                ftime0[fcnt] = ftime1[fcnt];
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    var0[fcnt][vcnt] = var1[fcnt][vcnt];
                }

                if(fscanf(fparray[fcnt], "%lf", &ftime1[fcnt]) != 1)
                {
                    PRINT_ERROR("fscanf error");
                }
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    if(fscanf(fparray[fcnt], "%lf", &var1[fcnt][vcnt]) != 1)
                    {
                        PRINT_ERROR("fscanf error");
                    }
                }
                if(fscanf(fparray[fcnt], "\n") != 0)
                {
                    PRINT_ERROR("fscanf error");
                }
            }
            if(kk == 0)
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    varC[fcnt][vcnt] = var0[fcnt][vcnt];
                }

            alpha = (runtime - ftime0[fcnt]) / (ftime1[fcnt] - ftime0[fcnt]);
            for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
            {
                fprintf(fpout,
                        " %20f",
                        (1.0 - alpha) * var0[fcnt][vcnt] +
                        alpha * var1[fcnt][vcnt] - varC[fcnt][vcnt]);
                varC[fcnt][vcnt] = (1.0 - HPgain) * varC[fcnt][vcnt] +
                                   HPgain * ((1.0 - alpha) * var0[fcnt][vcnt] +
                                             alpha * var1[fcnt][vcnt]);

                dcimg[IDout[kkfr]].array.F[kkpt * nbvar + ii] =
                    (1.0 - alpha) * var0[fcnt][vcnt] +
                    alpha * var1[fcnt][vcnt] - varC[fcnt][vcnt];
                ii++;
            }
        }

        fprintf(fpout, "\n");

        kk++;
        kkpt++;
        runtime += dt;
        if(kkpt == NBpt)
        {
            kkpt = 0;
            kkfr++;
        }
    }

    fclose(fpout);

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        fclose(fparray[fcnt]);
    }

    return (NBfiles);
}

// select block on first dimension
imageID LINARFILTERPRED_SelectBlock(const char *IDin_name,
                                    const char *IDblknb_name,
                                    long        blkNB,
                                    const char *IDout_name)
{
    imageID IDin;
    imageID IDblknb;
    uint8_t naxis;

    long          m;
    long          NBmodes1;
    uint32_t     *sizearray;
    uint32_t      xsize, ysize, zsize;
    unsigned long cnt;
    imageID       IDout;
    //char imname[200];
    long mmax;

    printf("Selecting block %ld ...\n", blkNB);
    fflush(stdout);

    IDin    = image_ID(IDin_name, dcimg, dcnimg);
    IDblknb = image_ID(IDblknb_name, dcimg, dcnimg);
    naxis   = dcimg[IDin].md[0].naxis;
    mmax    = dcimg[IDblknb].md[0].size[0];

    if(dcimg[IDin].md[0].size[0] != dcimg[IDblknb].md[0].size[0])
    {
        printf(
            "WARNING: block index file and telemetry have different sizes\n");
        fflush(stdout);
        mmax = dcimg[IDin].md[0].size[0];
        if(dcimg[IDblknb].md[0].size[0] < mmax)
        {
            mmax = dcimg[IDblknb].md[0].size[0];
        }
    }

    NBmodes1 = 0;
    for(m = 0; m < mmax; m++)
    {
        if(dcimg[IDblknb].array.UI16[m] == blkNB)
        {
            NBmodes1++;
        }
    }

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(uint8_t axis = 0; axis < naxis; axis++)
    {
        sizearray[axis] = dcimg[IDin].md[0].size[axis];
    }
    sizearray[0] = NBmodes1;

    {
        IMGID imgout_tmp =
            imgid_make_from_name(
                IDout_name);
        imgout_tmp.mdt->naxis = naxis;
        for(uint8_t a = 0; a < naxis;
            a++)
        {
            imgout_tmp.mdt->size[a] =
                sizearray[a];
        }
        imgout_tmp.mdt->datatype =
            _DATATYPE_FLOAT;
        imgout_tmp.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }

    xsize = dcimg[IDin].md[0].size[0];
    if(naxis > 1)
    {
        ysize = dcimg[IDin].md[0].size[1];
    }
    else
    {
        ysize = 1;
    }
    if(naxis > 2)
    {
        zsize = dcimg[IDin].md[0].size[2];
    }
    else
    {
        zsize = 1;
    }

    cnt = 0;

    for(uint32_t jj = 0; jj < ysize; jj++)
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint32_t ii = 0; ii < mmax; ii++)
                if(dcimg[IDblknb].array.UI16[ii] == blkNB)
                {
                    //printf("%ld / %ld   cnt = %8ld / %ld\n", ii, xsize, cnt, NBmodes1*ysize*zsize);
                    //fflush(stdout);
                    dcimg[IDout].array.F[cnt] =
                        dcimg[IDin]
                        .array.F[kk * xsize * ysize + jj * ysize + ii];
                    cnt++;
                }

    free(sizearray);

    return (IDout);
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 3. BUILD PREDICTIVE FILTER                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Expand 2D image/matrix in X direction by repeat and shift
 *
 */
imageID linARfilterPred_repeat_shift_X(const char *IDin_name,
                                       long        NBstep,
                                       const char *IDout_name)
{
    imageID  IDin;
    uint32_t xsize, ysize;

    imageID  IDout;
    uint32_t xsizeout, ysizeout;

    uint32_t *imsizeout;

    IDin     = image_ID(IDin_name, dcimg, dcnimg);
    xsize    = dcimg[IDin].md[0].size[0];
    ysize    = dcimg[IDin].md[0].size[1];
    xsizeout = xsize * NBstep;
    ysizeout = ysize - NBstep;

    imsizeout = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsizeout == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imsizeout[0] = xsizeout;
    imsizeout[1] = ysizeout;
    {
        IMGID imgout_tmp =
            imgid_make_from_name(
                IDout_name);
        imgout_tmp.mdt->naxis = 2;
        imgout_tmp.mdt->size[0] =
            imsizeout[0];
        imgout_tmp.mdt->size[1] =
            imsizeout[1];
        imgout_tmp.mdt->datatype =
            _DATATYPE_FLOAT;
        imgout_tmp.mdt->shared = 1;
        imgout_tmp.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }
    free(imsizeout);

    long step;
    for(step = 0; step < NBstep; step++)
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
        {
            for(uint32_t jjout = 0; jjout < ysize - NBstep; jjout++)
            {
                dcimg[IDout]
                .array.F[jjout * xsizeout + step * xsize + ii] =
                    dcimg[IDin]
                    .array.F[(jjout + NBstep - step - 1) * xsize + ii];
            }
        }
    }

    return IDout;
}
