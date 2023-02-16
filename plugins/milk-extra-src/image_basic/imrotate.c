/** @file imrotate.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID basic_rotate(const char *__restrict ID_name,
                     const char *__restrict IDout_name,
                     float angle);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_rotate_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) == 0)
    {
        basic_rotate(data.cmdargtoken[1].val.string,
                     data.cmdargtoken[2].val.string,
                     data.cmdargtoken[3].val.numf);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t imrotate_addCLIcmd()
{
    RegisterCLIcommand("rotateim",
                       __FILE__,
                       image_basic_rotate_cli,
                       "rotate 2D image",
                       "<image in> <output image> <angle>",
                       "rotateim imin imout 230",
                       "long basic_rotate(const char *ID_name, const char "
                       "*ID_out_name, float angle)");

    return RETURN_SUCCESS;
}

imageID basic_rotate(const char *__restrict ID_name,
                     const char *__restrict IDout_name,
                     float angle)
{
    imageID  ID, IDout;
    uint32_t naxes[2];

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    create_2Dimage_ID(IDout_name, naxes[0], naxes[1], &IDout);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            long iis = (long)(naxes[0] / 2 + (ii - naxes[0] / 2) * cos(angle) +
                              (jj - naxes[1] / 2) * sin(angle));
            long jjs = (long)(naxes[1] / 2 - (ii - naxes[0] / 2) * sin(angle) +
                              (jj - naxes[1] / 2) * cos(angle));
            if((iis > 0) && (jjs > 0) && (iis < naxes[0]) && (jjs < naxes[1]))
            {
                data.image[IDout].array.F[jj * naxes[0] + ii] =
                    data.image[ID].array.F[jjs * naxes[0] + iis];
            }
        }

    return (IDout);
}

imageID basic_rotate90(const char *__restrict ID_name,
                       const char *__restrict ID_out_name)
{
    imageID  ID;
    imageID  IDout;
    uint32_t naxes[2];

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    create_2Dimage_ID(ID_out_name, naxes[1], naxes[0], &IDout);

    for(uint32_t jj = 0; jj < naxes[0]; jj++)
        for(uint32_t ii = 0; ii < naxes[1]; ii++)
        {
            uint32_t iis = jj;
            uint32_t jjs = naxes[1] - ii - 1;
            data.image[IDout].array.F[jj * naxes[1] + ii] =
                data.image[ID].array.F[jjs * naxes[0] + iis];
        }

    return IDout;
}

imageID basic_rotate_int(const char *__restrict ID_name,
                         const char *__restrict ID_out_name,
                         long nbstep)
{
    float    angle;
    imageID  ID;
    imageID  IDout;
    uint32_t naxes[2];

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    create_2Dimage_ID(ID_out_name, naxes[0], naxes[1], &IDout);

    for(int i = 0; i < nbstep; i++)
    {
        angle = M_PI * i / nbstep;
        for(uint32_t jj = 0; jj < naxes[1]; jj++)
            for(uint32_t ii = 0; ii < naxes[0]; ii++)
            {
                long iis =
                    (long)(naxes[0] / 2 + (ii - naxes[0] / 2) * cos(angle) +
                           (jj - naxes[1] / 2) * sin(angle));
                long jjs =
                    (long)(naxes[1] / 2 + (ii - naxes[0] / 2) * sin(angle) -
                           (jj - naxes[1] / 2) * cos(angle));
                if((iis > 0) && (jjs > 0) && (iis < naxes[0]) &&
                        (jjs < naxes[1]))
                {
                    data.image[IDout].array.F[jj * naxes[0] + ii] +=
                        data.image[ID].array.F[jjs * naxes[0] + iis];
                }
            }
    }

    return IDout;
}

/* rotation that keeps photometry - angle is in radians */
imageID basic_rotate2(const char *__restrict ID_name_in,
                      const char *__restrict ID_name_out,
                      float angle)
{
    imageID  ID_in;
    imageID  ID_out, ID_wout;
    uint32_t naxes[2];
    uint32_t naxes2[2];
    uint64_t nelements;
    float    rotangle;
    float   *pixcorner_x;
    float   *pixcorner_y;
    float    x, y;
    uint32_t NB_step = 20;
    uint32_t i, j;
    float    f1, f2, f3, f4, f5, f6, f7, f8, f9;
    /*  float *f1a;
        float *f2a;
        float *f3a;
        float *f4a;
        float *f5a;
        float *f6a;
        float *f7a;
        float *f8a;
        float *f9a;*/
    int  *f1a;
    int  *f2a;
    int  *f3a;
    int  *f4a;
    int  *f5a;
    int  *f6a;
    int  *f7a;
    int  *f8a;
    int  *f9a;
    float pixcx, pixcy;
    float total;
    int   xint, yint;
    float ccos, ssin;

    printf("rotating %s by %f radians ...\n", ID_name_in, angle);
    fflush(stdout);
    rotangle = angle;
    while(rotangle < 0)
    {
        rotangle += 2.0 * M_PI;
    }
    while(rotangle > 2 * M_PI)
    {
        rotangle -= 2.0 * M_PI;
    }
    /* now the angle is between 0 and 2*PI */
    while(rotangle > (M_PI / 2))
    {
        basic_rotate90(ID_name_in, "tmprot");
        delete_image_ID(ID_name_in, DELETE_IMAGE_ERRMODE_WARNING);
        copy_image_ID("tmprot", ID_name_in, 0);
        delete_image_ID("tmprot", DELETE_IMAGE_ERRMODE_WARNING);
        rotangle -= M_PI / 2.0;
    }

    /* now the angle is between 0 and PI/2 */

    ID_in     = image_ID(ID_name_in);
    naxes[0]  = data.image[ID_in].md[0].size[0];
    naxes[1]  = data.image[ID_in].md[0].size[1];
    nelements = naxes[0] * naxes[1];
    printf("creating temporary arrays\n");
    fflush(stdout);

    f1a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f1a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f2a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f2a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f3a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f3a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f4a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f4a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f5a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f5a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f6a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f6a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f7a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f7a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f8a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f8a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    f9a = (int *) calloc(NB_step * NB_step, sizeof(int));
    if(f9a == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        abort();
    }

    printf("filling up calibration array ... ");
    fflush(stdout);
    total = 1.0 / NB_step / NB_step;
    ccos  = cos(rotangle);
    ssin  = sin(rotangle);

    for(uint32_t ii = 0; ii < NB_step; ii++)
        for(uint32_t jj = 0; jj < NB_step; jj++)
        {
            pixcx = 1.0 * ii / NB_step;
            pixcy = 1.0 * jj / NB_step;

            for(i = 0; i < NB_step; i++)
                for(j = 0; j < NB_step; j++)
                {
                    x = pixcx + 1.0 * (0.5 + i) / NB_step * ccos -
                        1.0 * (0.5 + j) / NB_step * ssin;
                    y = pixcy + 1.0 * (0.5 + i) / NB_step * ssin +
                        1.0 * (0.5 + j) / NB_step * ccos;
                    if(x < 0)
                    {
                        if(y < 1)
                        {
                            f1a[jj * NB_step + ii]++;
                        }
                        else
                        {
                            if(y > 2)
                            {
                                f7a[jj * NB_step + ii]++;
                            }
                            else
                            {
                                f4a[jj * NB_step + ii]++;
                            }
                        }
                    }
                    else
                    {
                        if(x > 1)
                        {
                            if(y < 1)
                            {
                                f3a[jj * NB_step + ii]++;
                            }
                            else
                            {
                                if(y > 2)
                                {
                                    f9a[jj * NB_step + ii]++;
                                }
                                else
                                {
                                    f6a[jj * NB_step + ii]++;
                                }
                            }
                        }
                        else
                        {
                            if(y < 1)
                            {
                                f2a[jj * NB_step + ii]++;
                            }
                            else
                            {
                                if(y > 2)
                                {
                                    f8a[jj * NB_step + ii]++;
                                }
                                else
                                {
                                    f5a[jj * NB_step + ii]++;
                                }
                            }
                        }
                    }
                }
        }
    printf("done\n");
    fflush(stdout);

    pixcorner_x = (float *) malloc(sizeof(float) * nelements);
    if(pixcorner_x == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pixcorner_y = (float *) malloc(sizeof(float) * nelements);
    if(pixcorner_y == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            pixcorner_x[jj * naxes[0] + ii] =
                ii * ccos - jj * ssin + ssin * naxes[1] + 1.0;
            pixcorner_y[jj * naxes[0] + ii] = ii * ssin + jj * ccos;
        }

    naxes2[0] =
        (long)(sin(rotangle) * naxes[1] + cos(rotangle) * naxes[0] + 2.0);
    naxes2[1] =
        (long)(cos(rotangle) * naxes[1] + sin(rotangle) * naxes[0] + 2.0);

    create_2Dimage_ID(ID_name_out, naxes2[0], naxes2[1], &ID_out);
    create_2Dimage_ID("wtmp", naxes2[0], naxes2[1], &ID_wout);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            x = pixcorner_x[jj * naxes[0] + ii];
            y = pixcorner_y[jj * naxes[0] + ii];
            /*printf("%ld %ld %d %d %f %f %f %f\n",ii,jj,i,j,pixcorner_x[jj*naxes[0]+ii],pixcorner_y[jj*naxes[0]+ii],x,y);*/
            xint = (int)(((x + 3 * naxes[0]) - (int)(x + 3 * naxes[0])) *
                         NB_step);
            yint = (int)((y - (int) y) * NB_step);
            /*	printf("%d %d\n",xint,yint);*/
            f1 = 1.0 * f1a[yint * NB_step + xint];
            f2 = 1.0 * f2a[yint * NB_step + xint];
            f3 = 1.0 * f3a[yint * NB_step + xint];
            f4 = 1.0 * f4a[yint * NB_step + xint];
            f5 = 1.0 * f5a[yint * NB_step + xint];
            f6 = 1.0 * f6a[yint * NB_step + xint];
            f7 = 1.0 * f7a[yint * NB_step + xint];
            f8 = 1.0 * f8a[yint * NB_step + xint];
            f9 = 1.0 * f9a[yint * NB_step + xint];

            data.image[ID_out]
            .array.F[((long) y) * naxes2[0] + ((long)(x)) - 1] +=
                f1 * data.image[ID_in].array.F[jj * naxes[0] + ii];
            data.image[ID_out].array.F[((long) y) * naxes2[0] + ((long)(x))] +=
                f2 * data.image[ID_in].array.F[jj * naxes[0] + ii];
            data.image[ID_out]
            .array.F[((long) y) * naxes2[0] + ((long)(x)) + 1] +=
                f3 * data.image[ID_in].array.F[jj * naxes[0] + ii];

            data.image[ID_out]
            .array.F[((long) y + 1) * naxes2[0] + ((long)(x)) - 1] +=
                f4 * data.image[ID_in].array.F[jj * naxes[0] + ii];
            data.image[ID_out]
            .array.F[((long) y + 1) * naxes2[0] + ((long)(x))] +=
                f5 * data.image[ID_in].array.F[jj * naxes[0] + ii];
            data.image[ID_out]
            .array.F[((long) y + 1) * naxes2[0] + ((long)(x)) + 1] +=
                f6 * data.image[ID_in].array.F[jj * naxes[0] + ii];

            data.image[ID_out]
            .array.F[((long) y + 2) * naxes2[0] + ((long)(x)) - 1] +=
                f7 * data.image[ID_in].array.F[jj * naxes[0] + ii];
            data.image[ID_out]
            .array.F[((long) y + 2) * naxes2[0] + ((long)(x))] +=
                f8 * data.image[ID_in].array.F[jj * naxes[0] + ii];
            data.image[ID_out]
            .array.F[((long) y + 2) * naxes2[0] + ((long)(x)) + 1] +=
                f9 * data.image[ID_in].array.F[jj * naxes[0] + ii];

            data.image[ID_wout]
            .array.F[((long) y) * naxes2[0] + ((long)(x)) - 1] += f1;
            data.image[ID_wout]
            .array.F[((long) y) * naxes2[0] + ((long)(x))] += f2;
            data.image[ID_wout]
            .array.F[((long) y) * naxes2[0] + ((long)(x)) + 1] += f3;

            data.image[ID_wout]
            .array.F[((long) y + 1) * naxes2[0] + ((long)(x)) - 1] += f4;
            data.image[ID_wout]
            .array.F[((long) y + 1) * naxes2[0] + ((long)(x))] += f5;
            data.image[ID_wout]
            .array.F[((long) y + 1) * naxes2[0] + ((long)(x)) + 1] += f6;

            data.image[ID_wout]
            .array.F[((long) y + 2) * naxes2[0] + ((long)(x)) - 1] += f7;
            data.image[ID_wout]
            .array.F[((long) y + 2) * naxes2[0] + ((long)(x))] += f8;
            data.image[ID_wout]
            .array.F[((long) y + 2) * naxes2[0] + ((long)(x)) + 1] += f9;
        }

    for(uint32_t jj = 0; jj < naxes2[1]; jj++)
        for(uint32_t ii = 0; ii < naxes2[0]; ii++)
        {
            if(data.image[ID_wout].array.F[jj * naxes2[0] + ii] >
                    (0.9 * NB_step * NB_step))
            {
                data.image[ID_out].array.F[jj * naxes2[0] + ii] /=
                    data.image[ID_wout].array.F[jj * naxes2[0] + ii];
            }
            else
            {
                data.image[ID_out].array.F[jj * naxes2[0] + ii] *= total;
            }
        }

    delete_image_ID("wtmp", DELETE_IMAGE_ERRMODE_WARNING);

    free(pixcorner_x);
    free(pixcorner_y);

    free(f1a);
    free(f2a);
    free(f3a);
    free(f4a);
    free(f5a);
    free(f6a);
    free(f7a);
    free(f8a);
    free(f9a);

    printf("done\n");
    fflush(stdout);

    return ID_out;
}
