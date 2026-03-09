# Module: clustering

Fast distance-based clustering of streams.

## Source Files

| File | Description |
|------|-------------|
| `CFmeminit.c` | No description available. |
| `CFtree_rebuild.c` | No description available. |
| `addCF_to_CF.c` | Combine Cluster Feature with existing CF and compute stats |
| `compute_imdistance_double.c` | No description available. |
| `condense.c` | Condense single node if possible |
| `create_new_leaf.c` | Create a new leaf |
| `ctree_init.c` | Initialize CF tree with first vector |
| `ctree_memallocate.c` | No description available. |
| `ctree_memfree.c` | No description available. |
| `cubecluster.c` | No description available. |
| `droptree.c` | No description available. |
| `get_availableCFindex.c` | No description available. |
| `leaf_addentry.c` | No description available. |
| `leafnode_attachleaf.c` | Attach EXISITNG leaf to node |
| `mindiffscan.c` | No description available. |
| `node_attachnode.c` | No description available. |
| `printCFtree.c` | No description available. |
| `split_CF_node.c` | Split CF node |
| `update_level.c` | No description available. |
| `write_clustCFave.c` | No description available. |
| `write_clustCFdat.c` | No description available. |
| `write_clustleafsummary.c` | No description available. |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-clustering-cubeclust` | `cubecluster.c` | No description available. |
| `milk-fpsexec-clustering-mindiffscan` | `mindiffscan.c` | No description available. |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
