/**
 * @file zernike_value.c
 *
 */


#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "CommandLineInterface/CLIcore.h"
#include "zernike.h"

#include "COREMOD_tools/COREMOD_tools.h"


static ZERNIKE Zernike;
static int zermax = 5000;


double fact(int n)
{
    int    i;
    double value;

    value = 1;
    for(i = 1; i < n + 1; i++)
    {
        value = value * i;
    }

    return (value);
}



int zernike_init()
{
    static int zernikeinit = 0;

    if(zernikeinit == 0)
    {
        long j, n, m, s;
        long ii, jj;

        Zernike.ZERMAX = zermax;

        printf("ZERMAX= %ld\n", Zernike.ZERMAX);
        fflush(stdout);

        Zernike.Zer_n = (long *) malloc(Zernike.ZERMAX * sizeof(long));
        if(Zernike.Zer_n == NULL)
        {
            printf("malloc returns NULL pointer");
            abort();
        }

        Zernike.Zer_m = (long *) malloc(Zernike.ZERMAX * sizeof(long));
        if(Zernike.Zer_m == NULL)
        {
            printf("malloc returns NULL pointer");
            abort();
        }

        Zernike.R_array =
            (double *) malloc(Zernike.ZERMAX * Zernike.ZERMAX * sizeof(double));
        if(Zernike.R_array == NULL)
        {
            printf("malloc returns NULL pointer");
            abort();
        }


        Zernike.Zer_Nollindex = (long *) malloc(Zernike.ZERMAX * sizeof(long));
        if(Zernike.Zer_Nollindex == NULL)
        {
            printf("malloc returns NULL pointer");
            abort();
        }

        Zernike.Zer_reverseNollindex = (long *) malloc(Zernike.ZERMAX * sizeof(long));
        if(Zernike.Zer_reverseNollindex == NULL)
        {
            printf("malloc returns NULL pointer");
            abort();
        }



        /* Zer_n and Zer_m are initialised to 0 */
        for(ii = 0; ii < Zernike.ZERMAX; ii++)
        {
            Zernike.Zer_n[ii] = 0;
            Zernike.Zer_m[ii] = 0;
        }

        /* Zer_n and Zer_m are computed */
        j = 0;
        n = 0;
        m = 0;

        double *Nolldouble = (double *) malloc(sizeof(double) * Zernike.ZERMAX);
        long *index_Nollsort = (long *) malloc(sizeof(double) * Zernike.ZERMAX);
        Zernike.Zer_Nollindex[0] = 1;

        while(j < Zernike.ZERMAX)
        {
            Zernike.Zer_n[j] = n;
            Zernike.Zer_m[j] = m;
            j++;
            m += 2;
            if(m > n)
            {
                n++;
                m = -n;
            }
            long Noll_n = n * (n + 1) / 2;
            int nmod = n % 4;
            Zernike.Zer_Nollindex[j] = Noll_n + abs(m);
            if((nmod == 0) || (nmod == 1))
            {
                if(m <= 0)
                {
                    Zernike.Zer_Nollindex[j]++;
                }
            }
            else
            {
                if(m >= 0)
                {
                    Zernike.Zer_Nollindex[j]++;
                }
            }
            Nolldouble[j] = 1.0 * Zernike.Zer_Nollindex[j];
            index_Nollsort[j] = j;
        }

        /* R_array is initialised */
        for(ii = 0; ii < Zernike.ZERMAX; ii++)
            for(jj = 0; jj < Zernike.ZERMAX; jj++)
            {
                Zernike.R_array[jj * Zernike.ZERMAX + ii] = 0;
            }

        /* now the R_array is computed */
        for(j = 1; j < Zernike.ZERMAX; j++)
        {
            m = labs(Zernike.Zer_m[j]);
            for(s = 0; s < ((int)(0.5 * (Zernike.Zer_n[j] - m) + 1)); s++)
            {
                Zernike.R_array[j * Zernike.ZERMAX + Zernike.Zer_n[j] - 2 * s] =
                    pow(-1, s) * fact(Zernike.Zer_n[j] - s) / fact(s) /
                    fact((Zernike.Zer_n[j] + m) / 2 - s) /
                    fact((Zernike.Zer_n[j] - m) / 2 - s);
            }
        }

        for(ii = 0; ii < Zernike.ZERMAX; ii++)
            for(jj = 0; jj < Zernike.ZERMAX; jj++)
            {
                Zernike.R_array[jj * Zernike.ZERMAX + ii] *=
                    sqrt(Zernike.Zer_n[jj] + 1);
            }

        /* the zernikes index are computed */

        quick_sort2l(Nolldouble, index_Nollsort, Zernike.ZERMAX);

        for(int zi = 0; zi < Zernike.ZERMAX; zi++)
        {
            Zernike.Zer_reverseNollindex[zi] = index_Nollsort[zi];
        }

        // test
        /*{
            FILE *fp = fopen("zern.txt", "w");
            for(int zi=0; zi<Zernike.ZERMAX; zi++)
            {
                fprintf(fp, "%5d  %5ld  %5ld   %5ld  %5ld\n", zi, Zernike.Zer_n[zi], Zernike.Zer_m[zi], Zernike.Zer_Nollindex[zi], Zernike.Zer_reverseNollindex[zi] );
            }
            fclose(fp);
        }*/



        free(Nolldouble);
        free(index_Nollsort);


        zernikeinit = 1;
    }

    return (0);
}





long Zernike_n(long i)
{
    return (Zernike.Zer_n[i]);
}



long Zernike_m(long i)
{
    return (Zernike.Zer_m[i]);
}


// Noll numbering
// 0: Piston
// 1,2: TT
// 3: Focus
//
double Zernike_value(
    long j,
    double r,
    double PA
)
{
    long   i;
    double value = 0.0;
    double tmp, s2;
    long   n, m;
    long zi = Zernike.Zer_reverseNollindex[j];

    n  = Zernike.Zer_n[zi] + 1;
    m  = Zernike.Zer_m[zi];
    s2 = sqrt(2.0);

    if(m == 0)
    {
        for(i = 0; i < n; i++)
        {
            tmp = Zernike.R_array[zi * Zernike.ZERMAX + i];
            if(tmp != 0)
            {
                value += pow(r, i) * tmp;
            }
        }
    }
    else
    {
        for(i = 0; i < n; i++)
        {
            tmp = Zernike.R_array[zi * Zernike.ZERMAX + i];
            if(tmp != 0)
            {
                if(m < 0)
                {
                    value -= tmp * s2 * pow(r, i) * sin(m * PA);
                }
                else
                {
                    value += tmp * s2 * pow(r, i) * cos(m * PA);
                }
            }
        }
    }

    return (value);
}
