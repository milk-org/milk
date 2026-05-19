#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_tools/COREMOD_tools.h" 
#include "statistic/statistic.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "image_gen.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>


imageID make_jacquinot_pupil(const char *ID_name,
                             uint32_t    l1,
                             uint32_t    l2,
                             double      x_center,
                             double      y_center,
                             double      width,
                             double      height)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if((fabs(jj - y_center) / height) <
                    exp(-((ii - x_center) * (ii - x_center) / width / width)))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }
        }

    return (ID);
}

imageID make_sectors(const char *ID_name,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      step,
                     long        NB_sectors)
{
    imageID  ID;
    uint32_t naxes[2];
    double   theta;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            theta = atan2((ii - x_center), (jj - y_center));
            if(theta < 0.0)
            {
                theta += 2.0 * PI;
            }
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                step * ((long)(theta / 2.0 / PI * NB_sectors));
        }

    return (ID);
}

imageID
make_rnd(const char *ID_name, uint32_t l1, uint32_t l2, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    int      distrib;
    uint64_t nelement;

    distrib = 0; /* uniform */
    if(strstr(options, "gauss") != NULL)
    {
        distrib = 1; /* gauss */
        printf("gaussian distribution\n");
    }

    if(strstr(options, "trgauss") != NULL)
    {
        distrib = 2; /* truncated gauss */
        printf("truncated gaussian distribution\n");
    }

    if(dcdebug > 1)
    {
        fprintf(stdout, "Image size = %u %u\n", l1, l2);
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    nelement = naxes[0] * naxes[1];

    // openMP is slow when calling gsl random number generator : do not use openMP here
    if(distrib == 0)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.F[ii] = (double) ran1();
        }
    }
    if(distrib == 1)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.F[ii] = (double) gauss();
        }
    }
    if(distrib == 2)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.F[ii] = (double) gauss_trc();
        }
    }

    return (ID);
}

imageID make_rnd_double(const char *ID_name,
                        uint32_t    l1,
                        uint32_t    l2,
                        const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    int      distrib;
    uint64_t nelement;

    distrib = 0; /* uniform */
    if(strstr(options, "gauss") != NULL)
    {
        distrib = 1; /* gauss */
        printf("gaussian distribution\n");
    }

    if(strstr(options, "trgauss") != NULL)
    {
        distrib = 2; /* truncated gauss */
        printf("truncated gaussian distribution\n");
    }

    if(dcdebug > 1)
    {
        fprintf(stdout, "Image size = %u %u\n", l1, l2);
    }

    create_2Dimage_ID_double(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    nelement = naxes[0] * naxes[1];

    // openMP is slow when calling gsl random number generator : do not use openMP here
    if(distrib == 0)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.D[ii] = (double) ran1();
        }
    }
    if(distrib == 1)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.D[ii] = (double) gauss();
        }
    }
    if(distrib == 2)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.D[ii] = (double) gauss_trc();
        }
    }

    return (ID);
}

/*
int make_rnd1(const char *ID_name, long l1, long l2, const char *options)
{
  int ID;
  long naxes[2];
  int distrib;
  long nelements;
  struct prng *g;

  distrib = 0;
  if (strstr(options,"-gauss")!=NULL)
    {
      distrib = 1;
    }

  if (strstr(options,"-trgauss")!=NULL)
    {
      distrib = 2;
      printf("truncated gaussian distribution\n");
   }

  g = prng_new("eicg(2147483647,111,1,0)");

   if (g == NULL)
   {
      fprintf(stderr,"Initialisation of generator failed.\n");
      exit (-1);
   }

   printf("Short name: %s\n",prng_short_name(g));

   printf("Expanded name: %s\n",prng_long_name(g));


   create_2Dimage_ID(ID_name,l1,l2);
   ID = image_ID(ID_name, dcimg, dcnimg);
   naxes[0] = dcimg[ID].md[0].size[0];
   naxes[1] = dcimg[ID].md[0].size[1];
   nelements=naxes[0]*naxes[1];

   prng_get_array(g,dcimg[ID].array.F,nelements);
   prng_reset(g);
   prng_free(g);


   return(0);
}
*/

imageID
make_gauss(const char *ID_name, uint32_t l1, uint32_t l2, double a, double A)
{
    imageID  ID;
    uint32_t naxes[2];
    double   distsq;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distsq = (ii - naxes[0] / 2) * (ii - naxes[0] / 2) +
                     (jj - naxes[1] / 2) * (jj - naxes[1] / 2);
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (double) A * exp(-distsq / a / a);
        }
    /*  printf("FWHM = %f\n",2.0*a*sqrt(log(2)));*/
    return (ID);
}

imageID make_FiberCouplingOverlap(const char *ID_name)
{
    imageID  ID;
    uint32_t naxes[2];
    uint32_t size = 128;

    create_2Dimage_ID(ID_name, size, size, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    float TTcoeff = 0.2;

    float puprad = 0.1 * size;
    float xcent  = 1.32;
    float ycent  = 0.0;

    // compute TEM00 map
    imageID IDtem00;
    create_2Dimage_ID("tem00", size, size, &IDtem00);

    double fluxtot = 0.0;
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            float x     = 1.0 * (1.0 * ii - 0.5 * naxes[0]) / puprad;
            float y     = 1.0 * (1.0 * jj - 0.5 * naxes[1]) / puprad;
            float r0    = sqrtf(x * x + y * y);
            float TEM00 = expf(-r0 * r0);

            fluxtot += TEM00 * TEM00;
            dcimg[IDtem00].array.F[jj * naxes[0] + ii] = TEM00;
        }
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[IDtem00].array.F[jj * naxes[0] + ii] /= sqrt(fluxtot);
        }
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            double totre = 0.0;
            double totim = 0.0;

            float TTx = 1.0 * (1.0 * ii - 0.5 * naxes[0]) * TTcoeff;
            float TTy = 1.0 * (1.0 * jj - 0.5 * naxes[1]) * TTcoeff;

            fluxtot = 0.0;
            for(uint32_t jj0 = 0; jj0 < naxes[1]; jj0++)
            {
                for(uint32_t ii0 = 0; ii0 < naxes[0]; ii0++)
                {
                    float pup_ampl;
                    float pup_pha;

                    // pupil coord x, y

                    float x  = 1.0 * (1.0 * ii0 - 0.5 * naxes[0]) / puprad;
                    float y  = 1.0 * (1.0 * jj0 - 0.5 * naxes[1]) / puprad;
                    float dx = x - xcent;
                    float dy = y - ycent;

                    float r = sqrtf(dx * dx + dy * dy);

                    float TEM00 =
                        dcimg[IDtem00].array.F[jj0 * naxes[0] + ii0];

                    //dcimg[ID].array.F[jj * naxes[0] + ii] = -r;

                    if((r < 1.0) && (r > 0.3))
                    {
                        pup_ampl =
                            1.0f; //dcimg[IDtem00].array.F[jj0 * naxes[0] + ii0];
                        pup_pha = x * TTx + y * TTy;

                        fluxtot += pup_ampl * pup_ampl;

                        totre += TEM00 * (pup_ampl * cos(pup_pha));
                        totim += TEM00 * (pup_ampl * sin(pup_pha));
                    }
                    else
                    {
                        dcimg[ID].array.F[jj * naxes[0] + ii] = 0.0f;
                        pup_ampl                                   = 0.0;
                    }
                }
            }

            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (totre * totre + totim * totim) / sqrt(fluxtot);
        }
    }

    return ID;
}

imageID make_2axis_gauss(const char *ID_name,
                         uint32_t    l1,
                         uint32_t    l2,
                         double      a,
                         double      A,
                         double      E,
                         double      PA)
{
    imageID  ID;
    uint32_t naxes[2];
    double   distsq;
    double   iin, jjn;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            iin = 1.0 * (ii - naxes[0] / 2) * cos(PA) +
                  1.0 * (jj - naxes[1] / 2) * sin(PA);
            jjn = 1.0 * (jj - naxes[1] / 2) * cos(PA) -
                  1.0 * (ii - naxes[0] / 2) * sin(PA);
            distsq = iin * iin + (1.0 / (1.0 + E)) * jjn * jjn;
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (double) A * exp(-distsq / a / a);
        }

    return (ID);
}

imageID
make_cluster(const char *ID_name, uint32_t l1, uint32_t l2, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    long     nb_star       = 3000;
    double   cluster_size  = 0.1; /* relative to the FOV */
    double   concentration = 1.0;
    long     i;
    double   tmp, dist, angle;
    char     input[50];
    int      str_pos;
    int      sim = 0;
    long     lii, ljj, hii, hjj;

    if(strstr(options, "-nbstars ") != NULL)
    {
        str_pos = strstr(options, "-nbstars ") - options;
        str_pos = str_pos + strlen("-nbstars ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        nb_star  = atol(input);
        printf("number of stars is %ld\n", nb_star);
    }

    if(strstr(options, "-conc ") != NULL)
    {
        str_pos = strstr(options, "-conc ") - options;
        str_pos = str_pos + strlen("-conc ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]      = '\0';
        concentration = atof(input);
        printf("concentration is %f\n", concentration);
    }

    if(strstr(options, "-size ") != NULL)
    {
        str_pos = strstr(options, "-size ") - options;
        str_pos = str_pos + strlen("-size ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]     = '\0';
        cluster_size = atof(input);
        printf("cluster size is %f\n", cluster_size);
    }

    if(strstr(options, "-sim") != NULL)
    {
        printf("all sources in the central half array \n");
        sim = 1;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    if(sim == 0)
    {
        lii = 0;
        ljj = 0;
        hii = naxes[0];
        hjj = naxes[1];
    }
    else
    {
        lii = naxes[0] / 4;
        ljj = naxes[1] / 4;
        hii = 3 * naxes[0] / 4;
        hjj = 3 * naxes[1] / 4;
    }

    i = 0;
    while(i < nb_star)
    {
        dist        = gauss();
        dist        = sqrt(sqrt(dist * dist));
        dist        = powf(dist, concentration);
        angle       = 2 * PI * ran1();
        uint32_t ii = (uint32_t)(naxes[0] / 2 + (cluster_size * naxes[0] / 2) *
                                 dist * cos(angle));
        uint32_t jj = (uint32_t)(naxes[1] / 2 + (cluster_size * naxes[1] / 2) *
                                 dist * sin(angle));

        if((ii > lii) && (jj > ljj) && (ii < hii) && (jj < hjj))
        {
            tmp = gauss();
            dcimg[ID].array.F[jj * naxes[0] + ii] += tmp * tmp;
            i++;
        }
    }

    return (ID);
}

imageID make_galaxy(const char *ID_name,
                    uint32_t    l1,
                    uint32_t    l2,
                    double      S_radius,
                    double      S_L0,
                    double      S_ell,
                    double      S_PA,
                    double      E_radius,
                    double      E_L0,
                    double      E_ell,
                    double      E_PA)
{
    imageID  ID;
    uint32_t naxes[2];
    double   x, y, r;
    double   aob, boa; /* a over b and b over a */
    double   total = 0;

    /* E = 1-b/a */

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = l1;
    naxes[1] = l2;

    /* Spiral component */
    aob = 1.0 / (1.0 - S_ell);
    boa = 1.0 - S_ell;

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1] / 2 + 1; jj++)
        {
            x = cos(S_PA) * (ii - naxes[0] / 2) +
                sin(S_PA) * (jj - naxes[1] / 2);
            y = -sin(S_PA) * (ii - naxes[0] / 2) +
                cos(S_PA) * (jj - naxes[1] / 2);
            r = sqrt(aob * x * x + boa * y * y);
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                S_L0 * exp(-r / S_radius);
        }

    /* Elliptical component */
    aob = 1.0 / (1.0 - E_ell);
    boa = 1.0 - E_ell;

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1] / 2 + 1; jj++)
        {
            x = cos(E_PA) * (ii - naxes[0] / 2) +
                sin(E_PA) * (jj - naxes[1] / 2);
            y = -sin(E_PA) * (ii - naxes[0] / 2) +
                cos(E_PA) * (jj - naxes[1] / 2);
            r = sqrt(aob * x * x + boa * y * y);
            dcimg[ID].array.F[jj * naxes[0] + ii] +=
                E_L0 * powf(10.0f, (-3.3307f * (powf((r / E_radius), 0.25f) - 1.0f)));
        }

    /* filling other half */
    for(uint32_t ii = 1; ii < naxes[0]; ii++)
        for(uint32_t jj = 1; jj < naxes[1] / 2; jj++)
        {
            dcimg[ID]
            .array.F[(naxes[1] - jj) * naxes[0] + (naxes[0] - ii)] =
                dcimg[ID].array.F[jj * naxes[0] + ii];
        }
    uint32_t ii = 0;
    for(uint32_t jj = naxes[1] / 2; jj < naxes[1]; jj++)
    {
        aob = 1.0 / (1.0 - S_ell);
        boa = 1.0 - S_ell;
        x   = cos(S_PA) * (ii - naxes[0] / 2) + sin(S_PA) * (jj - naxes[1] / 2);
        y = -sin(S_PA) * (ii - naxes[0] / 2) + cos(S_PA) * (jj - naxes[1] / 2);
        r = sqrt(aob * x * x + boa * y * y);
        dcimg[ID].array.F[jj * naxes[0] + ii] = S_L0 * expf(-r / S_radius);
        aob                                        = 1.0 / (1.0 - E_ell);
        boa                                        = 1.0 - E_ell;
        x = cos(E_PA) * (ii - naxes[0] / 2) + sin(E_PA) * (jj - naxes[1] / 2);
        y = -sin(E_PA) * (ii - naxes[0] / 2) + cos(E_PA) * (jj - naxes[1] / 2);
        r = sqrt(aob * x * x + boa * y * y);
        dcimg[ID].array.F[jj * naxes[0] + ii] +=
            E_L0 * powf(10.0f, (-3.3307f * (powf((r / E_radius), 0.25f) - 1.0f)));
    }

    total = 2.0 * PI * S_L0 * S_radius * S_radius +
            23.02 * E_L0 * E_radius * E_radius;
    printf("total : %f (%f)\n", arith_image_total(ID_name), total);

    return (ID);
}

imageID
make_Egalaxy(const char *ID_name, uint32_t l1, uint32_t l2, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    double   galaxy_size   = 0.1; /* relative to the FOV */
    double   concentration = 1.0;
    long     i;
    double   PA   = 0;
    double   E    = 0.3; /* position angle and ellipticity */
    double   peak = 1;   /* maximum value */
    char     input[50];
    int      str_pos;
    int      sim = 0;
    long     lii, ljj, hii, hjj;
    double   x, y, xcenter, ycenter, distsq;

    if(strstr(options, "-conc ") != NULL)
    {
        str_pos = strstr(options, "-conc ") - options;
        str_pos = str_pos + strlen("-conc ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]      = '\0';
        concentration = atof(input);
        printf("concentration is %f\n", concentration);
    }

    if(strstr(options, "-size ") != NULL)
    {
        str_pos = strstr(options, "-size ") - options;
        str_pos = str_pos + strlen("-size ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]    = '\0';
        galaxy_size = atof(input);
        printf("size is %f\n", galaxy_size);
    }

    if(strstr(options, "-pa ") != NULL)
    {
        str_pos = strstr(options, "-pa ") - options;
        str_pos = str_pos + strlen("-pa ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        PA       = atof(input);
        printf("galaxy pa size is %f radians \n", PA);
    }

    if(strstr(options, "-e ") != NULL)
    {
        str_pos = strstr(options, "-e ") - options;
        str_pos = str_pos + strlen("-e ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        E        = atof(input);
        printf("galaxy elipticity is %f \n", E);
    }

    if(strstr(options, "-sim") != NULL)
    {
        printf("all sources in the central half array \n");
        sim = 1;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    xcenter  = naxes[0] / 2;
    ycenter  = naxes[1] / 2;

    if(sim == 0)
    {
        lii = 0;
        ljj = 0;
        hii = naxes[0];
        hjj = naxes[1];
    }
    else
    {
        lii = naxes[0] / 4;
        ljj = naxes[1] / 4;
        hii = 3 * naxes[0] / 4;
        hjj = 3 * naxes[1] / 4;
    }

    for(uint32_t jj = ljj; jj < hjj; jj++)
        for(uint32_t ii = lii; ii < hii; ii++)
        {
            x = cos(PA) * (ii - xcenter) + sin(PA) * (jj - ycenter);
            y = -sin(PA) * (ii - xcenter) + cos(PA) * (jj - ycenter);
            /* E = sqrt(a*a-b*b)/a */
            /* a = 1 */
            x      = x;
            y      = y / sqrt(1 - E * E);
            distsq = (x * x + y * y) /
                     (naxes[0] * naxes[0] + naxes[1] * naxes[1]) / galaxy_size /
                     galaxy_size;
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (double) peak * exp(-concentration * distsq);
        }

    return (ID);
}

// for sol system, index ~2.4 with local zodi
imageID gen_image_EZdisk(const char *ID_name,
                         uint32_t    size,
                         double      InnerEdge,
                         double      Index,
                         double      Incl)
{
    imageID ID;
    double  x, y, r, r0;
    double  value;

    create_2Dimage_ID(ID_name, size, size, &ID);
    r0 = 6.0;
    for(uint32_t ii = 0; ii < size; ii++)
        for(uint32_t jj = 0; jj < size; jj++)
        {
            x = 1.0 * (ii + 0.5) - size / 2;
            y = 1.0 * (jj + 0.5) - size / 2;
            y /= cos(Incl);
            r = sqrt(x * x + y * y);
            if(r < InnerEdge)
            {
                value = 0.0;
            }
            else
            {
                value = powf(r, -Index);
            }
            value /= cos(Incl);

            value += powf(r0, -Index);
            dcimg[ID].array.F[jj * size + ii] = value;
        }

    return (ID);
}

imageID make_slopexy(
    const char *ID_name, uint32_t l1, uint32_t l2, double sx, double sy)
{
    imageID  ID;
    uint32_t naxes[2];
    double   coeff;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    coeff = sx * (naxes[0] / 2) + sy * (naxes[1] / 2);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                sx * ii + sy * jj - coeff;
        }

    return (ID);
}

imageID
make_dist(const char *ID_name, uint32_t l1, uint32_t l2, double f1, double f2)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                sqrt((f1 - ii) * (f1 - ii) + (f2 - jj) * (f2 - jj));
        }

    return (ID);
}

imageID make_PosAngle(
    const char *ID_name, uint32_t l1, uint32_t l2, double f1, double f2)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            double x, y;
            x                                          = 1.0 * ii - f1;
            y                                          = 1.0 * jj - f2;
            dcimg[ID].array.F[jj * naxes[0] + ii] = atan2(y, x);
        }

    return (ID);
}
