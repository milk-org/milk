# Module: clustering

Fast distance-based clustering of streams.

## Source Files

| File                          | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| `CFmeminit.c`                 | Initialize Cluster Feature memory pools                 |
| `CFtree_rebuild.c`            | Rebuild CF tree from existing state                     |
| `addCF_to_CF.c`               | Merge Cluster Feature into existing CF and update stats |
| `compute_imdistance_double.c` | Compute pairwise image distance (double precision)      |
| `condense.c`                  | Condense single node when below threshold               |
| `create_new_leaf.c`           | Create a new leaf node in the CF tree                   |
| `ctree_init.c`                | Initialize CF tree with first data vector               |
| `ctree_memallocate.c`         | Allocate CF tree memory structures                      |
| `ctree_memfree.c`             | Free CF tree memory                                     |
| `cubecluster.c`               | Cluster image cube slices                               |
| `droptree.c`                  | Destroy and free entire CF tree                         |
| `get_availableCFindex.c`      | Get next available CF index slot                        |
| `leaf_addentry.c`             | Add data entry to an existing leaf                      |
| `leafnode_attachleaf.c`       | Attach existing leaf to a node                          |
| `mindiffscan.c`               | Minimum-difference scan across data                     |
| `node_attachnode.c`           | Attach child node to parent node                        |
| `printCFtree.c`               | Print CF tree structure to stdout                       |
| `split_CF_node.c`             | Split an overfull CF node                               |
| `update_level.c`              | Update tree level after insertions                      |
| `write_clustCFave.c`          | Write cluster CF averages to file                       |
| `write_clustCFdat.c`          | Write cluster CF data to file                           |
| `write_clustleafsummary.c`    | Write cluster leaf summary to file                      |

## Standalone Executables

| Executable                            | Source File     | Description                         |
| ------------------------------------- | --------------- | ----------------------------------- |
| `milk-fpsexec-clustering-cubeclust`   | `cubecluster.c` | Cluster image cube slices           |
| `milk-fpsexec-clustering-mindiffscan` | `mindiffscan.c` | Minimum-difference scan across data |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
