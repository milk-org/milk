/** @file CR2tomov.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "info/info.h"

#include "loadCR2toFITSRGB.h"
#include "writeBMP.h"

errno_t CR2tomov()
{
    char    configfile[STRINGMAXLEN_FILENAME];
    imageID ID, IDr, IDg, IDb;
    long    ii, i;
    long    cnt = 0;
    long    cntmax;
    char    fname[STRINGMAXLEN_FULLFILENAME];
    char    fnamer[STRINGMAXLEN_FULLFILENAME];
    char    fnameg[STRINGMAXLEN_FULLFILENAME];
    char    fnameb[STRINGMAXLEN_FULLFILENAME];
    char    fnamestat[STRINGMAXLEN_FULLFILENAME];
    char    fnameoutr[STRINGMAXLEN_FULLFILENAME];
    char    fnameoutg[STRINGMAXLEN_FULLFILENAME];
    char    fnameoutb[STRINGMAXLEN_FULLFILENAME];
    char    fnamejpg[STRINGMAXLEN_FULLFILENAME];
    FILE   *fp;
    FILE   *fp1;

    long xsize, ysize;

    imageID IDrtot;
    imageID IDgtot;
    imageID IDbtot;

    //double tot;

    //double alpha = 0.7;

    // conversion from CR2 to FITS RGB
    int  CR2toFITSrgb;
    int  CR2TOFITSRGB_FORCE;
    long maxnbFITSfiles;
    int  binfact;

    // conversion from FITS RGB to JPEG
    int    FITStoJPEG;
    double MINLEVEL;
    double MAXLEVEL;

    int    MAXLEVEL_AUTO;
    double MAXLEVEL_AUTO_FLOOR;
    double MAXLEVEL_AUTO_CEIL;

    double MAX_PERC01_COEFF;
    double MAX_PERC05_COEFF;
    double MAX_PERC10_COEFF;
    double MAX_PERC20_COEFF;
    double MAX_PERC50_COEFF;
    double MAX_PERC80_COEFF;
    double MAX_PERC90_COEFF;
    double MAX_PERC95_COEFF;
    double MAX_PERC99_COEFF;
    double MAX_PERC995_COEFF;
    double MAX_PERC998_COEFF;
    double MAX_PERC999_COEFF;

    double RGBM_RR;
    double RGBM_RG;
    double RGBM_RB;
    double RGBM_GR;
    double RGBM_GG;
    double RGBM_GB;
    double RGBM_BR;
    double RGBM_BG;
    double RGBM_BB;
    double LUMR, LUMG, LUMB; // luminance vector
    double COLORSAT;

    double ALPHA;

    double vp01, vp05, vp10, vp20, vp50, vp80, vp90, vp95, vp99, vp995, vp998,
           vp999;
    double vp01r, vp05r, vp10r, vp20r, vp50r, vp80r, vp90r, vp95r, vp99r,
           vp995r, vp998r, vp999r;
    double vp01g, vp05g, vp10g, vp20g, vp50g, vp80g, vp90g, vp95g, vp99g,
           vp995g, vp998g, vp999g;
    double vp01b, vp05b, vp10b, vp20b, vp50b, vp80b, vp90b, vp95b, vp99b,
           vp995b, vp998b, vp999b;
    double *maxlevel;
    double *maxlevel1;
    double *array;
    double  value, valuecnt;
    long    boxsize;
    double  sigma;
    long    j;
    long    jstart;
    //long jend;
    long j1;

    //int NLCONV;
    //double  NLCONV_OFFSET;
    //double  NLCONV_LIMIT;
    //double  NLCONV_FACT;
    //double  NLCONV_POW;
    //double  NLCONV_SIGMA;
    //imageID IDr1, IDg1, IDb1, IDrp, IDgp, IDbp;

    long SKIP, SKIPcnt;
    long SKIP_FITStoJPEG, SKIPcnt_FITStoJPEG;

    int MKim;

    {
        // code block write image name
        int slen =
            snprintf(configfile, STRINGMAXLEN_FILENAME, "cr2tojpegconf.txt");
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_FILENAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    } // end code block

    CR2toFITSrgb = read_config_parameter_int(configfile, "CR2TOFITSRGB");
    CR2TOFITSRGB_FORCE =
        read_config_parameter_int(configfile, "CR2TOFITSRGB_FORCE");
    maxnbFITSfiles =
        read_config_parameter_long(configfile, "CR2TOFITS_MAXNBFILE");
    binfact = read_config_parameter_int(configfile, "CR2TOFITSBIN");

    maxlevel = (double *) malloc(sizeof(double) * maxnbFITSfiles);

    FITStoJPEG = read_config_parameter_int(configfile, "FITStoJPEG");
    MINLEVEL   = read_config_parameter_float(configfile, "MINLEVEL");
    MAXLEVEL   = read_config_parameter_float(configfile, "MAXLEVEL");

    MAXLEVEL_AUTO = read_config_parameter_int(configfile, "MAXLEVEL_AUTO");
    MAXLEVEL_AUTO_FLOOR =
        read_config_parameter_float(configfile, "MAXLEVEL_AUTO_FLOOR");

    if(read_config_parameter_exists(configfile, "MAXLEVEL_AUTO_CEIL") == 1)
    {
        MAXLEVEL_AUTO_CEIL =
            read_config_parameter_float(configfile, "MAXLEVEL_AUTO_CEIL");
    }
    else
    {
        MAXLEVEL_AUTO_CEIL = 100000.0;
    }

    MAX_PERC01_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC01_COEFF");
    MAX_PERC05_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC05_COEFF");
    MAX_PERC10_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC10_COEFF");
    MAX_PERC20_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC20_COEFF");
    MAX_PERC50_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC50_COEFF");
    MAX_PERC80_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC80_COEFF");
    MAX_PERC90_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC90_COEFF");
    MAX_PERC95_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC95_COEFF");
    MAX_PERC99_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC99_COEFF");
    MAX_PERC995_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC995_COEFF");
    MAX_PERC998_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC998_COEFF");
    MAX_PERC999_COEFF =
        read_config_parameter_float(configfile, "MAX_PERC999_COEFF");

    RGBM_RR  = read_config_parameter_float(configfile, "RGBM_RR");
    RGBM_RG  = read_config_parameter_float(configfile, "RGBM_RG");
    RGBM_RB  = read_config_parameter_float(configfile, "RGBM_RB");
    RGBM_GR  = read_config_parameter_float(configfile, "RGBM_GR");
    RGBM_GG  = read_config_parameter_float(configfile, "RGBM_GG");
    RGBM_GB  = read_config_parameter_float(configfile, "RGBM_GB");
    RGBM_BR  = read_config_parameter_float(configfile, "RGBM_BR");
    RGBM_BG  = read_config_parameter_float(configfile, "RGBM_BG");
    RGBM_BB  = read_config_parameter_float(configfile, "RGBM_BB");
    LUMR     = read_config_parameter_float(configfile, "LUMR");
    LUMG     = read_config_parameter_float(configfile, "LUMG");
    LUMB     = read_config_parameter_float(configfile, "LUMB");
    COLORSAT = read_config_parameter_float(configfile, "COLORSAT");

    //NLCONV = 0;
    /*if(read_config_parameter_exists(configfile,"NLCONV")==1)
    {
        NLCONV = read_config_parameter_int(configfile,"NLCONV");
        NLCONV_OFFSET = read_config_parameter_float(configfile,"NLCONV_OFFSET");
        NLCONV_LIMIT = read_config_parameter_float(configfile,"NLCONV_LIMIT");
        NLCONV_FACT = read_config_parameter_float(configfile,"NLCONV_FACT");
        NLCONV_POW = read_config_parameter_float(configfile,"NLCONV_POW");
        NLCONV_SIGMA = read_config_parameter_float(configfile,"NLCONV_SIGMA");
    }*/

    ALPHA = read_config_parameter_float(configfile, "ALPHA");

    SKIP = 0;

    ID = variable_ID("SKIP");
    if(ID != 1)
    {
        SKIP = (long)(data.variable[ID].value.f + 0.1);
    }
    printf("SKIP = %ld\n", SKIP);

    if(CR2toFITSrgb == 1)
    {
        load_fits("bias.fits", "bias", 1, NULL);
        load_fits("dark.fits", "dark", 1, NULL);
        load_fits("badpix.fits", "badpix", 1, NULL);
        load_fits("flat.fits", "flat", 1, NULL);

        EXECUTE_SYSTEM_COMMAND("ls ./CR2/*.CR2 > flist.tmp");

        if((fp = fopen("flist.tmp", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        SKIPcnt = 0;
        while((fgets(fname, 200, fp) != NULL) && (cnt < maxnbFITSfiles))
        {
            WRITE_FULLFILENAME(fnamestat, "./FITS/imgstats.%05ld.txt", cnt);
            WRITE_FULLFILENAME(fnameoutr, "./FITS/imr%05ld.fits", cnt);
            WRITE_FULLFILENAME(fnameoutg, "./FITS/img%05ld.fits", cnt);
            WRITE_FULLFILENAME(fnameoutb, "./FITS/imb%05ld.fits", cnt);

            MKim = 0;
            if((file_exists(fnameoutr) == 1) &&
                    (file_exists(fnameoutg) == 1) &&
                    (file_exists(fnameoutb) == 1) && (CR2TOFITSRGB_FORCE == 0))
            {
                printf("Files %s %s %s exist, no need to recreate\n",
                       fnameoutr,
                       fnameoutg,
                       fnameoutb);
            }
            else
            {
                if(SKIPcnt == 0)
                {
                    MKim = 1;
                    printf("[%ld] working on file %s\n", cnt, fname);
                    fname[strlen(fname) - 1] = '\0';
                    loadCR2toFITSRGB(fname, "imr", "img", "imb");
                    /*		  if(binfact!=1)
                      {
                        basic_contract("imr","imrc",binfact,binfact);
                        delete_image_ID("imr", DELETE_IMAGE_ERRMODE_WARNING);
                        chname_image_ID("imrc","imr");
                        basic_contract("img","imgc",binfact,binfact);
                        delete_image_ID("img", DELETE_IMAGE_ERRMODE_WARNING);
                        chname_image_ID("imgc","img");
                        basic_contract("imb","imbc",binfact,binfact);
                        delete_image_ID("imb", DELETE_IMAGE_ERRMODE_WARNING);
                        chname_image_ID("imbc","imb");
                        }*/
                    ID    = image_ID("imr");
                    xsize = data.image[ID].md[0].size[0];
                    ysize = data.image[ID].md[0].size[1];

                    IDrtot = image_ID("imrtot");
                    if(IDrtot == -1)
                    {
                        create_2Dimage_ID("imrtot", xsize, ysize, &IDrtot);
                        create_2Dimage_ID("imgtot", xsize, ysize, &IDgtot);
                        create_2Dimage_ID("imbtot", xsize, ysize, &IDbtot);
                    }

                    IDr = image_ID("imr");
                    IDg = image_ID("img");
                    IDb = image_ID("imb");

                    for(ii = 0; ii < xsize * ysize; ii++)
                    {
                        data.image[IDr].array.F[ii] /= binfact * binfact;
                        data.image[IDg].array.F[ii] /= binfact * binfact;
                        data.image[IDb].array.F[ii] /= binfact * binfact;

                        data.image[IDrtot].array.F[ii] +=
                            data.image[IDr].array.F[ii];
                        data.image[IDgtot].array.F[ii] +=
                            data.image[IDg].array.F[ii];
                        data.image[IDbtot].array.F[ii] +=
                            data.image[IDb].array.F[ii];
                    }
                    save_fl_fits("imrtot", "imrtot.fits");
                    save_fl_fits("imgtot", "imgtot.fits");
                    save_fl_fits("imbtot", "imbtot.fits");

                    WRITE_FULLFILENAME(fnameoutr, "./FITS/imr%05ld.fits", cnt);
                    save_fl_fits("imr", fnameoutr);
                    WRITE_FULLFILENAME(fnameoutg, "./FITS/img%05ld.fits", cnt);
                    save_fl_fits("img", fnameoutg);
                    WRITE_FULLFILENAME(fnameoutb, "./FITS/imb%05ld.fits", cnt);
                    save_fl_fits("imb", fnameoutb);
                }
            }

            if(((MKim == 1) || (file_exists(fnamestat) == 0)) &&
                    (SKIPcnt == 0))
            {
                printf("[%ld] working on file %s (statistics)\n", cnt, fname);
                if(MKim == 0)
                {
                    WRITE_FULLFILENAME(fnameoutr, "./FITS/imr%05ld.fits", cnt);
                    WRITE_FULLFILENAME(fnameoutg, "./FITS/img%05ld.fits", cnt);
                    WRITE_FULLFILENAME(fnameoutb, "./FITS/imb%05ld.fits", cnt);
                    load_fits(fnameoutr, "imr", 1, NULL);
                    load_fits(fnameoutg, "img", 1, NULL);
                    load_fits(fnameoutb, "imb", 1, NULL);
                }

                info_image_stats("imr", "");
                ID     = variable_ID("vp01");
                vp01r  = data.variable[ID].value.f;
                ID     = variable_ID("vp05");
                vp05r  = data.variable[ID].value.f;
                ID     = variable_ID("vp10");
                vp10r  = data.variable[ID].value.f;
                ID     = variable_ID("vp20");
                vp20r  = data.variable[ID].value.f;
                ID     = variable_ID("vp50");
                vp50r  = data.variable[ID].value.f;
                ID     = variable_ID("vp80");
                vp80r  = data.variable[ID].value.f;
                ID     = variable_ID("vp90");
                vp90r  = data.variable[ID].value.f;
                ID     = variable_ID("vp95");
                vp95r  = data.variable[ID].value.f;
                ID     = variable_ID("vp99");
                vp99r  = data.variable[ID].value.f;
                ID     = variable_ID("vp995");
                vp995r = data.variable[ID].value.f;
                ID     = variable_ID("vp998");
                vp998r = data.variable[ID].value.f;
                ID     = variable_ID("vp999");
                vp999r = data.variable[ID].value.f;
                delete_image_ID("imr", DELETE_IMAGE_ERRMODE_WARNING);

                info_image_stats("img", "");
                ID     = variable_ID("vp01");
                vp01g  = data.variable[ID].value.f;
                ID     = variable_ID("vp05");
                vp05g  = data.variable[ID].value.f;
                ID     = variable_ID("vp10");
                vp10g  = data.variable[ID].value.f;
                ID     = variable_ID("vp20");
                vp20g  = data.variable[ID].value.f;
                ID     = variable_ID("vp50");
                vp50g  = data.variable[ID].value.f;
                ID     = variable_ID("vp80");
                vp80g  = data.variable[ID].value.f;
                ID     = variable_ID("vp90");
                vp90g  = data.variable[ID].value.f;
                ID     = variable_ID("vp95");
                vp95g  = data.variable[ID].value.f;
                ID     = variable_ID("vp99");
                vp99g  = data.variable[ID].value.f;
                ID     = variable_ID("vp995");
                vp995g = data.variable[ID].value.f;
                ID     = variable_ID("vp998");
                vp998g = data.variable[ID].value.f;
                ID     = variable_ID("vp999");
                vp999g = data.variable[ID].value.f;
                delete_image_ID("img", DELETE_IMAGE_ERRMODE_WARNING);

                info_image_stats("imb", "");
                ID     = variable_ID("vp01");
                vp01b  = data.variable[ID].value.f;
                ID     = variable_ID("vp05");
                vp05b  = data.variable[ID].value.f;
                ID     = variable_ID("vp10");
                vp10b  = data.variable[ID].value.f;
                ID     = variable_ID("vp20");
                vp20b  = data.variable[ID].value.f;
                ID     = variable_ID("vp50");
                vp50b  = data.variable[ID].value.f;
                ID     = variable_ID("vp80");
                vp80b  = data.variable[ID].value.f;
                ID     = variable_ID("vp90");
                vp90b  = data.variable[ID].value.f;
                ID     = variable_ID("vp95");
                vp95b  = data.variable[ID].value.f;
                ID     = variable_ID("vp99");
                vp99b  = data.variable[ID].value.f;
                ID     = variable_ID("vp995");
                vp995b = data.variable[ID].value.f;
                ID     = variable_ID("vp998");
                vp998b = data.variable[ID].value.f;
                ID     = variable_ID("vp999");
                vp999b = data.variable[ID].value.f;
                delete_image_ID("imb", DELETE_IMAGE_ERRMODE_WARNING);

                fp1 = fopen(fnamestat, "w");
                fprintf(fp1,
                        "%05ld %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g "
                        "%g %g %g %g %g %g %g %g %g %g %g %g %g "
                        "%g %g %g %g %g %g %g\n",
                        cnt,
                        vp01r,
                        vp05r,
                        vp10r,
                        vp20r,
                        vp50r,
                        vp80r,
                        vp90r,
                        vp95r,
                        vp99r,
                        vp995r,
                        vp998r,
                        vp999r,
                        vp01g,
                        vp05g,
                        vp10g,
                        vp20g,
                        vp50g,
                        vp80g,
                        vp90g,
                        vp95g,
                        vp99g,
                        vp995g,
                        vp998g,
                        vp999g,
                        vp01b,
                        vp05b,
                        vp10b,
                        vp20b,
                        vp50b,
                        vp80b,
                        vp90b,
                        vp95b,
                        vp99b,
                        vp995b,
                        vp998b,
                        vp999b);
                fclose(fp1);

                if(MKim == 0)
                {
                    delete_image_ID("imr", DELETE_IMAGE_ERRMODE_WARNING);
                    delete_image_ID("img", DELETE_IMAGE_ERRMODE_WARNING);
                    delete_image_ID("imb", DELETE_IMAGE_ERRMODE_WARNING);
                }
            }

            if(MKim == 1)
            {
                delete_image_ID("imr", DELETE_IMAGE_ERRMODE_WARNING);
                delete_image_ID("img", DELETE_IMAGE_ERRMODE_WARNING);
                delete_image_ID("imb", DELETE_IMAGE_ERRMODE_WARNING);
            }

            SKIPcnt++;
            if(SKIPcnt > SKIP - 1)
            {
                SKIPcnt = 0;
            }

            cnt++;
        }
        fclose(fp);
        if(system("rm flist.tmp") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }

        printf("%ld images processed\n", cnt);
    }

    if(system("rm imgstats.txt") != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    if(system("cat ./FITS/imgstats.*.txt > imgstats.txt") != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    if(MAXLEVEL_AUTO == 1)
    {
        if((fp = fopen("imgstats.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        while(
            fscanf(fp,
                   "%05ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                   "%lf %lf %lf %lf %lf %lf %lf %lf %lf "
                   "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                   &cnt,
                   &vp01r,
                   &vp05r,
                   &vp10r,
                   &vp20r,
                   &vp50r,
                   &vp80r,
                   &vp90r,
                   &vp95r,
                   &vp99r,
                   &vp995r,
                   &vp998r,
                   &vp999r,
                   &vp01g,
                   &vp05g,
                   &vp10g,
                   &vp20g,
                   &vp50g,
                   &vp80g,
                   &vp90g,
                   &vp95g,
                   &vp99g,
                   &vp995g,
                   &vp998g,
                   &vp999g,
                   &vp01b,
                   &vp05b,
                   &vp10b,
                   &vp20b,
                   &vp50b,
                   &vp80b,
                   &vp90b,
                   &vp95b,
                   &vp99b,
                   &vp995b,
                   &vp998b,
                   &vp999b) == 37)
        {
            vp01  = (vp01r + vp01g + vp01b) / 3.0;
            vp05  = (vp05r + vp05g + vp05b) / 3.0;
            vp10  = (vp10r + vp10g + vp10b) / 3.0;
            vp20  = (vp20r + vp20g + vp20b) / 3.0;
            vp50  = (vp50r + vp50g + vp50b) / 3.0;
            vp80  = (vp80r + vp80g + vp80b) / 3.0;
            vp90  = (vp90r + vp90g + vp90b) / 3.0;
            vp95  = (vp95r + vp95g + vp95b) / 3.0;
            vp99  = (vp99r + vp99g + vp99b) / 3.0;
            vp995 = (vp995r + vp995g + vp995b) / 3.0;
            vp998 = (vp998r + vp998g + vp998b) / 3.0;
            vp999 = (vp999r + vp999g + vp999b) / 3.0;
            if(cnt < maxnbFITSfiles)
            {
                maxlevel[cnt] =
                    vp01 * MAX_PERC01_COEFF + vp05 * MAX_PERC05_COEFF +
                    vp10 * MAX_PERC10_COEFF + vp20 * MAX_PERC20_COEFF +
                    vp50 * MAX_PERC50_COEFF + vp80 * MAX_PERC80_COEFF +
                    vp90 * MAX_PERC90_COEFF + vp95 * MAX_PERC95_COEFF +
                    vp99 * MAX_PERC99_COEFF + vp995 * MAX_PERC995_COEFF +
                    vp998 * MAX_PERC998_COEFF + vp999 * MAX_PERC999_COEFF;
                printf("[%ld %g]   ", cnt, maxlevel[cnt]);
                maxlevel[cnt] = sqrt(maxlevel[cnt] * maxlevel[cnt] +
                                     MAXLEVEL_AUTO_FLOOR * MAXLEVEL_AUTO_FLOOR);
                if(maxlevel[cnt] > MAXLEVEL_AUTO_CEIL)
                {
                    maxlevel[cnt] = MAXLEVEL_AUTO_CEIL;
                }
                printf("%ld -> %g\n", cnt, maxlevel[cnt]);
            }
        }
        fclose(fp);

        if(0)
        {
            // smooth the maxlevel in time
            // scheme employed is running median/average
            cntmax = maxnbFITSfiles;
            if(cntmax > cnt + 1)
            {
                cntmax = cnt + 1;
            }

            printf("cntmax = %ld\n", cntmax);
            boxsize = 100;
            if(boxsize > 0.1 * cntmax)
            {
                boxsize = (long)(0.1 * cntmax);
            }
            sigma = 0.2 * boxsize;

            if(boxsize == 0)
            {
                boxsize = 1;
            }
            printf("boxsize = %ld\n", boxsize);

            array     = (double *) malloc(sizeof(double) * (2 * boxsize + 1));
            maxlevel1 = (double *) malloc(sizeof(double) * cntmax);
            for(i = 0; i < cntmax; i++)
            {
                jstart = i - boxsize;
                //jend = i+boxsize+1;
                /*	  jcent = 0;

                while(jstart<0)
                {
                jstart++;
                jend++;
                jcent++;
                }
                while(jend>cntmax-1)
                {
                jstart--;
                jend--;
                jcent--;
                }
                */

                for(j = 0; j < 2 * boxsize + 1; j++)
                {
                    j1 = j + jstart;
                    if(j1 < 0)
                    {
                        j1 = 0;
                    }
                    if(j1 > cntmax - 1)
                    {
                        j1 = cntmax - 1;
                    }

                    array[j] = maxlevel[j1];
                }

                quick_sort_double(array, 2 * boxsize + 1);

                value    = 0.0;
                valuecnt = 0.0;
                for(ii = 0; ii < 2 * boxsize + 1; ii++)
                {
                    double tmp1;

                    tmp1 = 1.0 * (ii - boxsize);
                    value +=
                        log10(array[ii]) * exp(-tmp1 * tmp1 / sigma / sigma);
                    valuecnt += exp(-tmp1 * tmp1 / sigma / sigma);
                }

                maxlevel1[i] = pow(10.0, value / valuecnt);
            }
            free(array);

            fp = fopen("maxlevel.log", "w");
            for(i = 0; i < cntmax; i++)
            {
                printf("%ld MAXLEVEL : %g ---> %g\n",
                       i,
                       maxlevel[i],
                       maxlevel1[i]);
                fprintf(fp, "%ld %g %g\n", i, maxlevel[i], maxlevel1[i]);
                maxlevel[i] = maxlevel1[i];
            }
            fclose(fp);
            free(maxlevel1);
        }
    }

    if(FITStoJPEG == 1)
    {
        printf("FITS to JPEG\n");

        SKIP_FITStoJPEG = 0;

        ID = variable_ID("SKIP_FITStoJPEG");
        if(ID != 1)
        {
            SKIP_FITStoJPEG = (long)(data.variable[ID].value.f + 0.1);
        }
        printf("SKIP FITS to JPEG = %ld\n", SKIP_FITStoJPEG);

        SKIPcnt_FITStoJPEG = 0;

        for(i = 0; i < maxnbFITSfiles; i++)
        {
            WRITE_FULLFILENAME(fnamejpg, "./JPEG/im%05ld.jpg", i);
            if(file_exists(fnamejpg) == 1)
            {
                printf("Files %s exists, no need to recreate\n", fnamejpg);
            }
            else
            {
                WRITE_FULLFILENAME(fnamer, "./FITS/imr%05ld.fits", i);
                if(file_exists(fnamer) == 1)
                {
                    if(SKIPcnt_FITStoJPEG == 0)
                    {
                        printf("file %s exists\n", fnamer);

                        WRITE_FULLFILENAME(fnamer, "./FITS/imr%05ld.f.fits", i);
                        if(file_exists(fnamer) == 1)
                        {
                            load_fits(fnamer, "imr", 1, &IDr);
                        }
                        else
                        {
                            WRITE_FULLFILENAME(fnamer,
                                               "./FITS/imr%05ld.fits",
                                               i);
                            load_fits(fnamer, "imr", 1, &IDr);
                        }

                        WRITE_FULLFILENAME(fnameg, "./FITS/img%05ld.f.fits", i);
                        if(file_exists(fnameg) == 1)
                        {
                            load_fits(fnameg, "img", 1, &IDg);
                        }
                        else
                        {
                            WRITE_FULLFILENAME(fnameg,
                                               "./FITS/img%05ld.fits",
                                               i);
                            load_fits(fnameg, "img", 1, &IDg);
                        }

                        WRITE_FULLFILENAME(fnameb, "./FITS/imb%05ld.f.fits", i);
                        if(file_exists(fnameb) == 1)
                        {
                            load_fits(fnameb, "imb", 1, &IDb);
                        }
                        else
                        {
                            WRITE_FULLFILENAME(fnameb,
                                               "./FITS/imb%05ld.fits",
                                               i);
                            load_fits(fnameb, "imb", 1, &IDb);
                        }

                        xsize = data.image[IDr].md[0].size[0];
                        ysize = data.image[IDr].md[0].size[1];

                        if(MAXLEVEL_AUTO == 1)
                        {
                            MAXLEVEL = maxlevel[i];
                        }

                        for(ii = 0; ii < xsize * ysize; ii++)
                        {
                            double r0, g0, b0;
                            double tmpr, tmpg, tmpb, tmpr1, tmpg1, tmpb1;

                            r0 = data.image[IDr].array.F[ii];
                            g0 = data.image[IDg].array.F[ii];
                            b0 = data.image[IDb].array.F[ii];

                            r0 = (r0 - MINLEVEL) / (MAXLEVEL - MINLEVEL);
                            g0 = (g0 - MINLEVEL) / (MAXLEVEL - MINLEVEL);
                            b0 = (b0 - MINLEVEL) / (MAXLEVEL - MINLEVEL);

                            tmpr = r0 * RGBM_RR + g0 * RGBM_RG + b0 * RGBM_RB;
                            tmpg = r0 * RGBM_GR + g0 * RGBM_GG + b0 * RGBM_GB;
                            tmpb = r0 * RGBM_BR + g0 * RGBM_BG + b0 * RGBM_BB;

                            tmpr1 =
                                tmpr * ((1.0 - COLORSAT) * LUMR + COLORSAT) +
                                tmpg * ((1.0 - COLORSAT) * LUMG) +
                                tmpb * ((1.0 - COLORSAT) * LUMB);
                            tmpg1 =
                                tmpr * ((1.0 - COLORSAT) * LUMR) +
                                tmpg * ((1.0 - COLORSAT) * LUMG + COLORSAT) +
                                tmpb * ((1.0 - COLORSAT) * LUMB);
                            tmpb1 = tmpr * ((1.0 - COLORSAT) * LUMR) +
                                    tmpg * ((1.0 - COLORSAT) * LUMG) +
                                    tmpb * ((1.0 - COLORSAT) * LUMB + COLORSAT);

                            data.image[IDr].array.F[ii] = tmpr1;
                            data.image[IDg].array.F[ii] = tmpg1;
                            data.image[IDb].array.F[ii] = tmpb1;
                        }

                        for(ii = 0; ii < xsize * ysize; ii++)
                        {
                            double vr, vg, vb;

                            vr = data.image[IDr].array.F[ii];
                            vg = data.image[IDg].array.F[ii];
                            vb = data.image[IDb].array.F[ii];

                            if(vr < 0.0)
                            {
                                vr = 0.0;
                            }
                            if(vg < 0.0)
                            {
                                vg = 0.0;
                            }
                            if(vb < 0.0)
                            {
                                vb = 0.0;
                            }
                        }

                        // non-linear convolution

                        /*	      if(NLCONV==1)
                        {
                          printf("NLCONV_OFFSET = %f\n",NLCONV_OFFSET);
                          printf("NLCONV_LIMIT = %f\n",NLCONV_LIMIT);
                          printf("NLCONV_FACT = %f\n",NLCONV_FACT);
                          printf("NLCONV_POW = %f\n",NLCONV_POW);
                          printf("NLCONV_SIGMA = %f\n",NLCONV_SIGMA);

                          IDrp = create_2Dimage_ID("imrp",xsize,ysize);
                          IDgp = create_2Dimage_ID("imgp",xsize,ysize);
                          IDbp = create_2Dimage_ID("imbp",xsize,ysize);

                          for(ii=0;ii<xsize*ysize;ii++)
                            {
                              if(data.image[IDr].array.F[ii]>NLCONV_LIMIT)
                        	data.image[IDr].array.F[ii] = NLCONV_LIMIT;
                              if(data.image[IDg].array.F[ii]>NLCONV_LIMIT)
                        	data.image[IDg].array.F[ii] = NLCONV_LIMIT;
                              if(data.image[IDb].array.F[ii]>NLCONV_LIMIT)
                        	data.image[IDb].array.F[ii] = NLCONV_LIMIT;


                              if(data.image[IDr].array.F[ii]>NLCONV_OFFSET)
                        	data.image[IDrp].array.F[ii] = pow((data.image[IDr].array.F[ii]-NLCONV_OFFSET)*NLCONV_FACT,NLCONV_POW);
                              if(data.image[IDg].array.F[ii]>NLCONV_OFFSET)
                        	data.image[IDgp].array.F[ii] = pow((data.image[IDg].array.F[ii]-NLCONV_OFFSET)*NLCONV_FACT,NLCONV_POW);
                              if(data.image[IDb].array.F[ii]>NLCONV_OFFSET)
                        	data.image[IDbp].array.F[ii] = pow((data.image[IDb].array.F[ii]-NLCONV_OFFSET)*NLCONV_FACT,NLCONV_POW);
                            }
                          make_gauss("kerg",xsize,ysize,NLCONV_SIGMA,1.0);
                          tot = arith_image_total("kerg");
                          arith_image_cstmult_inplace("kerg",1.0/tot);

                          fconvolve_padd("imrp","kerg",(long) (10.0*NLCONV_SIGMA+2.0),"imr_c");
                          fconvolve_padd("imgp","kerg",(long) (10.0*NLCONV_SIGMA+2.0),"img_c");
                          fconvolve_padd("imbp","kerg",(long) (10.0*NLCONV_SIGMA+2.0),"imb_c");
                          delete_image_ID("kerg");
                          delete_image_ID("imrp");
                          delete_image_ID("imgp");
                          delete_image_ID("imbp");
                          IDr1 = image_ID("imr_c");
                          IDg1 = image_ID("img_c");
                          IDb1 = image_ID("imb_c");
                          for(ii=0;ii<xsize*ysize;ii++)
                            {
                              data.image[IDr].array.F[ii] += data.image[IDr1].array.F[ii];
                              data.image[IDg].array.F[ii] += data.image[IDg1].array.F[ii];
                              data.image[IDb].array.F[ii] += data.image[IDb1].array.F[ii];
                            }
                          delete_image_ID("imr_c");
                          delete_image_ID("img_c");
                          delete_image_ID("imb_c");
                          }*/

                        for(ii = 0; ii < xsize * ysize; ii++)
                        {
                            double vr, vg, vb;

                            vr = data.image[IDr].array.F[ii];
                            vg = data.image[IDg].array.F[ii];
                            vb = data.image[IDb].array.F[ii];

                            if(vr < 0.0)
                            {
                                vr = 0.0;
                            }
                            if(vg < 0.0)
                            {
                                vg = 0.0;
                            }
                            if(vb < 0.0)
                            {
                                vb = 0.0;
                            }

                            if(vr > 1.0)
                            {
                                vr = 1.0;
                            }

                            if(vg > 1.0)
                            {
                                vg = 1.0;
                            }

                            if(vb > 1.0)
                            {
                                vb = 1.0;
                            }

                            vr = 255.0 * pow(vr, ALPHA);
                            vg = 255.0 * pow(vg, ALPHA);
                            vb = 255.0 * pow(vb, ALPHA);

                            data.image[IDr].array.F[ii] = vr;
                            data.image[IDg].array.F[ii] = vg;
                            data.image[IDb].array.F[ii] = vb;
                        }

                        image_writeBMP("imr", "img", "imb", "imrgb.bmp");
                        delete_image_ID("imr", DELETE_IMAGE_ERRMODE_WARNING);
                        delete_image_ID("img", DELETE_IMAGE_ERRMODE_WARNING);
                        delete_image_ID("imb", DELETE_IMAGE_ERRMODE_WARNING);
                        //		  WRITE_FULLFILENAME(fnamejpg,"./JPEG/im%05ld.jpg",i);

                        EXECUTE_SYSTEM_COMMAND(
                            "bmptoppm imrgb.bmp | ppmtojpeg --quality 95 > "
                            "_tmpjpeg.jpg; mv _tmpjpeg.jpg %s",
                            fnamejpg);
                        EXECUTE_SYSTEM_COMMAND("rm imrgb.bmp");
                    }
                    SKIPcnt_FITStoJPEG++;
                    if(SKIPcnt_FITStoJPEG > SKIP_FITStoJPEG - 1)
                    {
                        SKIPcnt_FITStoJPEG = 0;
                    }
                }
            }
        }
    }
    free(maxlevel);

    return RETURN_SUCCESS;
}
