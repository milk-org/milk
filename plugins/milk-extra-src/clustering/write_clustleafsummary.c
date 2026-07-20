// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "CommandLineInterface/CLIcore.h"

#include "clustering_defs.h"

errno_t write_clustleafsummary(CLUSTERTREE *ctree,
                               IMGID        img,
                               long        *pixmap,
                               double      *pixgain,
                               long        *frameleafCFindex,
                               long         NBframe,
                               const char *__restrict outdname)
{
    DEBUG_TRACE_FSTART();

    uint32_t xsize = img.md->size[0];
    uint32_t ysize = img.md->size[1];
    uint32_t zsize = img.md->size[2];

    uint64_t xysize = xsize;
    xysize *= ysize;

    if (zsize == 0)
    {
        // if 2D image, assume ysize is number of samples
        xysize = xsize;
        zsize  = ysize;
    }

    char fname[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fname, "%s/clust.leafsummary.dat", outdname);
    DEBUG_TRACEPOINT("writing %s", fname);

    FILE *fp = fopen(fname, "w");

    fprintf(fp, "# col1   leaf CF index\n");
    fprintf(fp, "# col2   Number of point within leaf CF\n");
    fprintf(fp, "# col3   datasq\n");
    fprintf(fp, "# col4   radius2\n");
    fprintf(fp, "# col5   radius3/threshold\n");

    long NBLFcluster = 0;
    for (long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        if (ctree->CFarray[CFindex].type == CLUSTER_CF_TYPE_LEAF)
        {
            NBLFcluster++;

            fprintf(fp, "%5ld %5ld %16g %16g    %6.4f\n", CFindex, ctree->CFarray[CFindex].N,
                    (double) ctree->CFarray[CFindex].datassq,
                    (double) sqrt(ctree->CFarray[CFindex].radius2),
                    (double) sqrt(ctree->CFarray[CFindex].radius2) / ctree->T);

            {
                char fleafname[STRINGMAXLEN_FILENAME];
                WRITE_FILENAME(fleafname, "%s/leaf%05ld.dat", outdname, CFindex);

                FILE *fpleaf = fopen(fleafname, "w");

                fprintf(fpleaf, "# CFindex  %5ld\n", CFindex);
                fprintf(fpleaf, "# level    %5d\n", ctree->CFarray[CFindex].level);
                fprintf(fpleaf, "# N        %5ld\n", ctree->CFarray[CFindex].N);
                fprintf(fpleaf, "# datassq %16g\n", (double) ctree->CFarray[CFindex].datassq);
                fprintf(fpleaf, "# radius2 %16g\n", (double) ctree->CFarray[CFindex].radius2);

                for (long frame = 0; frame < NBframe; frame++)
                {
                    if (frameleafCFindex[frame] == CFindex)
                    {
                        fprintf(fpleaf, "%05ld", frame);
                        double dist2ave = 0.0;
                        double dist2pos = 0.0;

                        if (img.im->md->datatype == _DATATYPE_FLOAT)
                        {
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].datasumvec[ii] /
                                              ctree->CFarray[CFindex].N;
                                dval -= pixgain[ii] * img.im->array.F[frame * xysize + pixmap[ii]];
                                dist2ave += dval * dval;
                            }
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].dataposvec[ii];
                                dval -= pixgain[ii] * img.im->array.F[frame * xysize + pixmap[ii]];
                                dist2pos += dval * dval;
                            }
                        }
                        else if (img.im->md->datatype == _DATATYPE_DOUBLE)
                        {
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].datasumvec[ii] /
                                              ctree->CFarray[CFindex].N;
                                dval -= pixgain[ii] * img.im->array.D[frame * xysize + pixmap[ii]];
                                dist2ave += dval * dval;
                            }
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].dataposvec[ii];
                                dval -= pixgain[ii] * img.im->array.D[frame * xysize + pixmap[ii]];
                                dist2pos += dval * dval;
                            }
                        }
                        else if (img.im->md->datatype == _DATATYPE_INT16)
                        {
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].datasumvec[ii] /
                                              ctree->CFarray[CFindex].N;
                                dval -=
                                    pixgain[ii] * img.im->array.SI16[frame * xysize + pixmap[ii]];
                                dist2ave += dval * dval;
                            }
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].dataposvec[ii];
                                dval -=
                                    pixgain[ii] * img.im->array.SI16[frame * xysize + pixmap[ii]];
                                dist2pos += dval * dval;
                            }
                        }
                        else if (img.im->md->datatype == _DATATYPE_INT32)
                        {
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].datasumvec[ii] /
                                              ctree->CFarray[CFindex].N;
                                dval -=
                                    pixgain[ii] * img.im->array.SI32[frame * xysize + pixmap[ii]];
                                dist2ave += dval * dval;
                            }
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].dataposvec[ii];
                                dval -=
                                    pixgain[ii] * img.im->array.SI32[frame * xysize + pixmap[ii]];
                                dist2pos += dval * dval;
                            }
                        }
                        else if (img.im->md->datatype == _DATATYPE_UINT16)
                        {
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].datasumvec[ii] /
                                              ctree->CFarray[CFindex].N;
                                dval -=
                                    pixgain[ii] * img.im->array.UI16[frame * xysize + pixmap[ii]];
                                dist2ave += dval * dval;
                            }
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].dataposvec[ii];
                                dval -=
                                    pixgain[ii] * img.im->array.UI16[frame * xysize + pixmap[ii]];
                                dist2pos += dval * dval;
                            }
                        }
                        else if (img.im->md->datatype == _DATATYPE_UINT32)
                        {
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].datasumvec[ii] /
                                              ctree->CFarray[CFindex].N;
                                dval -=
                                    pixgain[ii] * img.im->array.UI32[frame * xysize + pixmap[ii]];
                                dist2ave += dval * dval;
                            }
                            for (long ii = 0; ii < ctree->npix; ii++)
                            {
                                double dval = ctree->CFarray[CFindex].dataposvec[ii];
                                dval -=
                                    pixgain[ii] * img.im->array.UI32[frame * xysize + pixmap[ii]];
                                dist2pos += dval * dval;
                            }
                        }

                        fprintf(fpleaf, " %20.3f", sqrt(dist2pos));
                        fprintf(fpleaf, " %20.3f", sqrt(dist2ave));

                        fprintf(fpleaf, "\n");
                    }
                }

                fclose(fpleaf);
            }
        }
    }

    fclose(fp);

    printf("NB leaf cluster = %ld\n", NBLFcluster);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
