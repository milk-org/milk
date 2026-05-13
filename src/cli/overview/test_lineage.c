/**
 * Quick debug tool: dump lineage for a stream.
 * Usage: ./test_lineage <stream_name>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "overview_data.h"
#include "stream_graph.h"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr,
            "Usage: %s <stream_name>\n",
            argv[0]);
        return 1;
    }

    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    ov_scan_streams(&m);
    ov_scan_fps(&m);
    ov_scan_procs(&m);
    ov_build_graph(&m);

    printf("Model: %d streams, %d procs, "
           "%d fps, %d nodes, %d edges\n",
        m.nb_streams, m.nb_procs,
        m.nb_fps, m.nb_nodes, m.nb_edges);

    /* Dump all edges */
    printf("\n--- Edges ---\n");
    for (int i = 0; i < m.nb_edges; i++)
    {
        const OV_EDGE *e = &m.edges[i];
        const char *sn =
            (e->src_node >= 0
             && e->src_node < m.nb_nodes)
            ? m.nodes[e->src_node].name
            : "??";
        const char *tn =
            (e->tgt_node >= 0
             && e->tgt_node < m.nb_nodes)
            ? m.nodes[e->tgt_node].name
            : "??";
        const char *tnames[] = {
            "PROC_WRITES_STREAM",
            "STREAM_TRIGGERS_PROC",
            "FPS_RUNS_PROC",
            "FPS_INPUT_STREAM",
            "FPS_OUTPUT_STREAM",
            "PROC_TRIGGER_STREAM",
            "STREAM_READ_BY_PROC",
        };
        const char *tn2 = (e->type <= 6)
            ? tnames[e->type] : "??";
        printf("  [%2d] %-20s -> %-20s "
               "%s\n",
            i, sn, tn, tn2);
    }

    int si = ov_find_stream_by_name(
        &m, argv[1]);
    if (si < 0)
    {
        PRINT_ERROR("stream '%s' not found",
                    argv[1]);
        return 1;
    }
    printf("\nStream '%s' -> model idx %d, "
           "node_idx %d\n",
        argv[1], si,
        m.streams[si].node_idx);

    SG_LINEAGE lin;
    sg_compute_lineage(
        &m, si, SG_MODE_FULL, &lin);

    printf("\nAncestors (%d):\n",
        lin.nb_ancestors);
    for (int a = 0;
         a < lin.nb_ancestors; a++)
    {
        printf("  depth=%d stream[%d]=%s "
               "via=%s loop=%d\n",
            lin.ancestors[a].depth,
            lin.ancestors[a].stream_idx,
            m.streams[
                lin.ancestors[a].stream_idx
            ].name,
            lin.ancestors[a].via_name,
            lin.ancestors[a].is_loop);
    }

    printf("\nDescendants (%d):\n",
        lin.nb_descendants);
    for (int d = 0;
         d < lin.nb_descendants; d++)
    {
        printf("  depth=%d stream[%d]=%s "
               "via=%s loop=%d\n",
            lin.descendants[d].depth,
            lin.descendants[d].stream_idx,
            m.streams[
                lin.descendants[d].stream_idx
            ].name,
            lin.descendants[d].via_name,
            lin.descendants[d].is_loop);
    }

    return 0;
}
