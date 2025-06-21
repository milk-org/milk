#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "clustering_defs.h"


errno_t write_clustCFdat(
    CLUSTERTREE *ctree,
    const char *__restrict fname
)
{
    DEBUG_TRACE_FSTART();


    FILE *fp = fopen(fname, "w");

    fprintf(fp,"# col1   CF index\n");
    fprintf(fp,"# col2   CF type (2: node, 3: leaf cluster)\n");
    fprintf(fp,"# col3   CF level\n");
    fprintf(fp,"# col4   Number of point within CF\n");
    fprintf(fp,"# col5   NBchild\n");
    fprintf(fp,"# col6   parent index\n");
    fprintf(fp,"# col7   datasq\n");
    fprintf(fp,"# col8   radius2 (norm2)\n");
    fprintf(fp,"# col9   radius2 (norm2) / threshold\n");
    fprintf(fp,"# col10  radius  (norm inf)\n");
    fprintf(fp,"# col11  radius  (norm inf) / threshold\n");
    fprintf(fp,"# col12  pathcnt\n");
    fprintf(fp,"# col13  children\n");

    for(long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        if(ctree->CFarray[CFindex].type != CLUSTER_CF_TYPE_UNUSED)
        {
            fprintf(fp,
                    "%5ld  %1d %5d  %6ld %5d %5ld  %16.3g    %16.3g %6.4f   %16.3g %6.4f   %12.6g",
                    CFindex,
                    ctree->CFarray[CFindex].type,
                    ctree->CFarray[CFindex].level,
                    ctree->CFarray[CFindex].N,
                    ctree->CFarray[CFindex].NBchild,
                    ctree->CFarray[CFindex].parentindex,
                    (double) ctree->CFarray[CFindex].datassq,
                    (double) sqrt(ctree->CFarray[CFindex].radius2),
                    (double) sqrt(ctree->CFarray[CFindex].radius2)/ctree->T,
                    (double) ctree->CFarray[CFindex].radius,
                    (double) ctree->CFarray[CFindex].radius/ctree->T,
                    ctree->CFarray[CFindex].pathcnt/ctree->CFarray[ctree->rootindex].pathcnt
                   );

            fprintf(fp, "  ");
            for(int chi = 0; chi < ctree->CFarray[CFindex].NBchild; chi++)
            {
                long chicfi = ctree->CFarray[CFindex].childindex[chi];
                fprintf(fp,"%ld[%d],", chicfi, ctree->CFarray[chicfi].type);
            }


            fprintf(fp, "\n");
        }

    }

    fclose(fp);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}