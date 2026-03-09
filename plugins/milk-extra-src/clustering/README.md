# Module: clustering

Fast distance-based clustering of streams.

## Source Files

| File | Description |
|------|-------------|
| `CFmeminit.c` | Cfmeminit module |
| `CFtree_rebuild.c` | Cftree rebuild module |
| `addCF_to_CF.c` | Combine Cluster Feature with existing CF and compute stats |
| `compute_imdistance_double.c` | Compute imdistance double module |
| `condense.c` | Condense single node if possible |
| `create_new_leaf.c` | Create a new leaf |
| `ctree_init.c` | Initialize CF tree with first vector |
| `ctree_memallocate.c` | Ctree memallocate module |
| `ctree_memfree.c` | Ctree memfree module |
| `cubecluster.c` | Cubecluster module |
| `droptree.c` | Droptree module |
| `get_availableCFindex.c` | Get availablecfindex module |
| `leaf_addentry.c` | log all debug trace points to file |
| `leafnode_attachleaf.c` | Attach EXISITNG leaf to node |
| `mindiffscan.c` | Mindiffscan module |
| `node_attachnode.c` | attach node CFindex to CFindexupnode |
| `printCFtree.c` | Printcftree module |
| `split_CF_node.c` | Split CF node |
| `update_level.c` | Update level module |
| `write_clustCFave.c` | Write clustcfave module |
| `write_clustCFdat.c` | Write clustcfdat module |
| `write_clustleafsummary.c` | Write clustleafsummary module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-clustering-cubeclust` | `cubecluster.c` | Cubecluster module |
| `milk-fpsexec-clustering-mindiffscan` | `mindiffscan.c` | Mindiffscan module |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
