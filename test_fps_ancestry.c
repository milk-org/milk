/**
 * Quick diagnostic: test FPS ancestry BFS
 *
 * Build:
 *   gcc -I../src/cli/overview -I../src/engine/libfps
 *       -I../src/engine/ImageStreamIO
 *       -I../src/engine/libprocessinfo
 *       -I../src/engine/libmilkdata
 *       ... (or just compile via cmake)
 */
#include <stdio.h>
#include <string.h>
#include "overview_data.h"
#include "stream_graph.h"

int main(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));

    /* Scan real system */
    ov_model_full_scan(&m);

    printf("=== FPS ancestry diagnostic ===\n");
    printf("Streams: %d  FPS: %d  Procs: %d  "
           "Nodes: %d  Edges: %d\n\n",
           m.nb_streams, m.nb_fps, m.nb_procs,
           m.nb_nodes, m.nb_edges);

    /* Show all edges */
    printf("--- Edges ---\n");
    for (int i = 0; i < m.nb_edges; i++)
    {
        const OV_EDGE *e = &m.edges[i];
        if (e->src_node >= 0
            && e->src_node < m.nb_nodes
            && e->tgt_node >= 0
            && e->tgt_node < m.nb_nodes)
        {
            printf("  [%2d] %-20s -> %-20s  "
                   "type=%d (%s)\n",
                   i, m.nodes[e->src_node].name,
                   m.nodes[e->tgt_node].name,
                   e->type, e->label);
        }
    }

    printf("\n--- FPS nodes ---\n");
    for (int fi = 0; fi < m.nb_fps; fi++)
    {
        printf("  fps[%d] %-20s node_idx=%d\n",
               fi, m.fps[fi].name,
               m.fps[fi].node_idx);
    }

    /* Test BFS from each FPS */
    printf("\n--- BFS from each FPS ---\n");
    for (int fi = 0; fi < m.nb_fps; fi++)
    {
        int node = m.fps[fi].node_idx;
        if (node < 0) continue;

        int8_t depths[OV_MAX_NODES];
        sg_compute_node_depths(
            &m, node, SG_MODE_FULL, depths);

        printf("From FPS '%s' (node %d):\n",
               m.fps[fi].name, node);
        for (int fj = 0; fj < m.nb_fps; fj++)
        {
            int n = m.fps[fj].node_idx;
            if (n >= 0 && depths[n] != 127)
            {
                printf("  -> FPS '%s' depth=%d\n",
                       m.fps[fj].name,
                       (int)depths[n]);
            }
        }
    }

    return 0;
}
