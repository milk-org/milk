// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_data_loops.c
 * @brief Directed cycle detection and loop analysis for milk-CTRL
 *
 * Implements global directed cycle detection across streams, processes, and FPS.
 * Computes canonical cycle signatures, evaluates shared and exclusive resource
 * overlaps, aggregates loop health/rate metrics, and provides persistent loop naming.
 */

#include "overview_data_loops.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* =========================================================
 * Persistent Name Storage Table
 * ========================================================= */

#define OV_MAX_SAVED_NAMES 128

typedef struct
{
    uint64_t hash;
    char     name[OV_LOOP_NAME_LEN];
} ov_saved_loop_name_t;

static pthread_mutex_t      s_loop_names_mutex = PTHREAD_MUTEX_INITIALIZER;
static ov_saved_loop_name_t s_saved_names[OV_MAX_SAVED_NAMES];
static int                  s_nb_saved_names = 0;
static int                  s_names_loaded   = 0;

/**
 * get_config_filepath - Resolve config path (~/.milk_loop_names.conf).
 * @buf: Output buffer
 * @sz:  Buffer capacity
 */
static void get_config_filepath(char *buf, size_t sz)
{
    const char *home = getenv("HOME");
    if (home != NULL && home[0] != '\0')
    {
        snprintf(buf, sz, "%s/.milk_loop_names.conf", home);
        return;
    }

    snprintf(buf, sz, "/tmp/milk_loop_names.conf");
}

/**
 * @brief Load custom loop names from disk into memory table.
 * @param[in,out] model System model containing detected loops
 */
void ov_loop_names_load(OV_MODEL *model)
{
    pthread_mutex_lock(&s_loop_names_mutex);
    char path[256];
    get_config_filepath(path, sizeof(path));

    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        /* Fallback check if user previously had ~/.milk/milk-CTRL_loops.conf */
        const char *home = getenv("HOME");
        if (home != NULL && home[0] != '\0')
        {
            char legacy_path[256];
            snprintf(legacy_path, sizeof(legacy_path), "%s/.milk/milk-CTRL_loops.conf", home);
            fp = fopen(legacy_path, "r");
        }
    }

    if (fp != NULL)
    {
        s_nb_saved_names = 0;
        char line[512];
        while (fgets(line, sizeof(line), fp) != NULL)
        {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            {
                continue;
            }

            uint64_t h = 0;
            char     nm[OV_LOOP_NAME_LEN];
            nm[0] = '\0';

            /* Parse: <hash_hex> <name> */
            if (sscanf(line, "%" PRIx64 " %47[^\r\n]", &h, nm) == 2)
            {
                if (s_nb_saved_names < OV_MAX_SAVED_NAMES)
                {
                    s_saved_names[s_nb_saved_names].hash = h;
                    strncpy(s_saved_names[s_nb_saved_names].name, nm, OV_LOOP_NAME_LEN - 1);
                    s_saved_names[s_nb_saved_names].name[OV_LOOP_NAME_LEN - 1] = '\0';
                    s_nb_saved_names++;
                }
            }
        }
        fclose(fp);
    }
    s_names_loaded = 1;

    /* Apply loaded names to existing model loops */
    if (model != NULL)
    {
        for (int i = 0; i < model->nb_loops; i++)
        {
            OV_LOOP *lp = &model->loops[i];
            for (int k = 0; k < s_nb_saved_names; k++)
            {
                if (s_saved_names[k].hash == lp->signature_hash)
                {
                    strncpy(lp->custom_name, s_saved_names[k].name, sizeof(lp->custom_name) - 1);
                    lp->custom_name[sizeof(lp->custom_name) - 1] = '\0';
                    lp->has_custom_name                          = 1;
                    strncpy(lp->name, lp->custom_name, sizeof(lp->name) - 1);
                    lp->name[sizeof(lp->name) - 1] = '\0';
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&s_loop_names_mutex);
}

/**
 * @brief Save persistent custom loop names to disk (must hold s_loop_names_mutex).
 */
static void ov_loop_names_save_locked(void)
{
    char path[256];
    get_config_filepath(path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        return;
    }

    fprintf(fp, "# milk-CTRL loop custom names configuration\n");
    fprintf(fp, "# Format: <canonical_signature_hash_hex> <custom_name>\n");
    for (int i = 0; i < s_nb_saved_names; i++)
    {
        fprintf(fp, "%016" PRIx64 " %s\n", s_saved_names[i].hash, s_saved_names[i].name);
    }
    fclose(fp);
}

/**
 * @brief Save persistent custom loop names to disk.
 * @param[in] model System model containing detected loops (unused)
 */
void ov_loop_names_save(const OV_MODEL *model)
{
    (void) model;
    pthread_mutex_lock(&s_loop_names_mutex);
    ov_loop_names_save_locked();
    pthread_mutex_unlock(&s_loop_names_mutex);
}

/**
 * @brief Set a custom name for a loop and persist it.
 * @param[in,out] model    System model
 * @param[in]     loop_idx Index in model->loops[] (0..nb_loops-1)
 * @param[in]     new_name New human-readable name string
 * @return 0 on success, non-zero on error.
 */
int ov_loop_rename(OV_MODEL *model, int loop_idx, const char *new_name)
{
    if (model == NULL || loop_idx < 0 || loop_idx >= model->nb_loops || new_name == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&s_loop_names_mutex);
    OV_LOOP *lp = &model->loops[loop_idx];
    if (new_name[0] == '\0')
    {
        /* Clear custom name, restore auto_name */
        lp->has_custom_name = 0;
        lp->custom_name[0]  = '\0';
        strncpy(lp->name, lp->auto_name, sizeof(lp->name) - 1);
        lp->name[sizeof(lp->name) - 1] = '\0';

        /* Remove from saved names */
        for (int i = 0; i < s_nb_saved_names; i++)
        {
            if (s_saved_names[i].hash == lp->signature_hash)
            {
                for (int j = i; j < s_nb_saved_names - 1; j++)
                {
                    s_saved_names[j] = s_saved_names[j + 1];
                }
                s_nb_saved_names--;
                break;
            }
        }
    }
    else
    {
        strncpy(lp->custom_name, new_name, sizeof(lp->custom_name) - 1);
        lp->custom_name[sizeof(lp->custom_name) - 1] = '\0';
        lp->has_custom_name                          = 1;
        strncpy(lp->name, lp->custom_name, sizeof(lp->name) - 1);
        lp->name[sizeof(lp->name) - 1] = '\0';

        /* Update or add in saved table */
        int found = 0;
        for (int i = 0; i < s_nb_saved_names; i++)
        {
            if (s_saved_names[i].hash == lp->signature_hash)
            {
                strncpy(s_saved_names[i].name, lp->custom_name, OV_LOOP_NAME_LEN - 1);
                s_saved_names[i].name[OV_LOOP_NAME_LEN - 1] = '\0';
                found                                       = 1;
                break;
            }
        }
        if (!found && s_nb_saved_names < OV_MAX_SAVED_NAMES)
        {
            s_saved_names[s_nb_saved_names].hash = lp->signature_hash;
            strncpy(s_saved_names[s_nb_saved_names].name, lp->custom_name, OV_LOOP_NAME_LEN - 1);
            s_saved_names[s_nb_saved_names].name[OV_LOOP_NAME_LEN - 1] = '\0';
            s_nb_saved_names++;
        }
    }

    ov_loop_names_save_locked();
    pthread_mutex_unlock(&s_loop_names_mutex);
    return 0;
}

/**
 * ov_get_loop_name - Retrieve display name for a loop ID.
 * @model:   System model
 * @loop_id: 1-based loop ID
 *
 * Return: Display name string or fallback label.
 */
const char *ov_get_loop_name(const OV_MODEL *model, int loop_id)
{
    if (model == NULL || loop_id <= 0)
    {
        return "";
    }
    int idx = loop_id - 1;
    if (idx >= 0 && idx < model->nb_loops)
    {
        return model->loops[idx].name;
    }
    return "";
}

/**
 * ov_find_loop_by_id - Find loop array index from 1-based loop ID.
 * @model:   System model
 * @loop_id: 1-based loop ID
 *
 * Return: Array index in model->loops[], or -1 if not found.
 */
int ov_find_loop_by_id(const OV_MODEL *model, int loop_id)
{
    if (model == NULL || loop_id <= 0)
    {
        return -1;
    }
    for (int i = 0; i < model->nb_loops; i++)
    {
        if (model->loops[i].loop_id == loop_id)
        {
            return i;
        }
    }
    return -1;
}

/* =========================================================
 * FNV-1a 64-bit Hash Helper
 * ========================================================= */

static uint64_t fnv1a_hash(const char *str)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    while (*str)
    {
        hash ^= (uint8_t) (*str++);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

/* =========================================================
 * Cycle Edge Validator
 * ========================================================= */

static int is_valid_loop_edge(const OV_MODEL *m, const OV_EDGE *e, sg_mode_t mode)
{
    if (!e->active)
    {
        return 0;
    }
    int src = e->src_node;
    int tgt = e->tgt_node;
    if (src < 0 || src >= m->nb_nodes || tgt < 0 || tgt >= m->nb_nodes)
    {
        return 0;
    }

    ov_node_type_t stype = m->nodes[src].type;
    ov_node_type_t ttype = m->nodes[tgt].type;

    /* Stream -> Proc / FPS */
    if (stype == OV_NODE_STREAM && (ttype == OV_NODE_PROC || ttype == OV_NODE_FPS))
    {
        if (mode == SG_MODE_TRIGGER)
        {
            return (e->type == OV_EDGE_STREAM_TRIGGERS_PROC ||
                    e->type == OV_EDGE_PROC_TRIGGER_STREAM || e->type == OV_EDGE_FPS_INPUT_STREAM);
        }
        else
        {
            return (e->type == OV_EDGE_STREAM_TRIGGERS_PROC ||
                    e->type == OV_EDGE_PROC_TRIGGER_STREAM ||
                    e->type == OV_EDGE_STREAM_READ_BY_PROC || e->type == OV_EDGE_FPS_INPUT_STREAM);
        }
    }
    /* Proc / FPS -> Stream */
    else if ((stype == OV_NODE_PROC || stype == OV_NODE_FPS) && ttype == OV_NODE_STREAM)
    {
        return (e->type == OV_EDGE_PROC_WRITES_STREAM || e->type == OV_EDGE_FPS_OUTPUT_STREAM);
    }

    return 0;
}

/* =========================================================
 * Cycle Canonicalization & Registration
 * ========================================================= */

static int register_cycle(OV_MODEL *model, const int *path, int path_len)
{
    if (path_len < 2 || model->nb_loops >= OV_MAX_LOOPS)
    {
        return 0;
    }

    /* Find stream node with lexicographically smallest name to anchor representation */
    int         min_stream_pos = -1;
    const char *min_name       = NULL;

    for (int i = 0; i < path_len; i++)
    {
        int ni = path[i];
        if (model->nodes[ni].type == OV_NODE_STREAM)
        {
            const char *curr_name = model->nodes[ni].name;
            if (min_name == NULL || strcmp(curr_name, min_name) < 0)
            {
                min_name       = curr_name;
                min_stream_pos = i;
            }
        }
    }

    if (min_stream_pos < 0)
    {
        return 0;
    }

    /* Rotate cycle so min_stream_pos is at index 0 */
    int rot_path[OV_MAX_LOOP_NODES];
    for (int i = 0; i < path_len; i++)
    {
        rot_path[i] = path[(min_stream_pos + i) % path_len];
    }

    /* Build canonical signature string: s:<name>->p:<name>->... */
    char sig[256];
    sig[0] = '\0';
    for (int i = 0; i < path_len; i++)
    {
        int            ni = rot_path[i];
        const OV_NODE *n  = &model->nodes[ni];
        char           seg[80];
        snprintf(seg, sizeof(seg), "%s%s:%s", (i == 0) ? "" : "->",
                 (n->type == OV_NODE_STREAM) ? "s" : "p", n->name);
        strncat(sig, seg, sizeof(sig) - strlen(sig) - 1);
    }

    uint64_t sig_hash = fnv1a_hash(sig);

    /* Check if already discovered */
    for (int i = 0; i < model->nb_loops; i++)
    {
        if (model->loops[i].signature_hash == sig_hash)
        {
            return 0; /* Duplicate */
        }
    }

    /* Register new loop */
    int      lidx = model->nb_loops;
    OV_LOOP *lp   = &model->loops[lidx];
    memset(lp, 0, sizeof(*lp));

    lp->loop_id        = lidx + 1;
    lp->signature_hash = sig_hash;
    strncpy(lp->signature, sig, sizeof(lp->signature) - 1);
    lp->nb_nodes = path_len;

    for (int i = 0; i < path_len; i++)
    {
        int ni              = rot_path[i];
        lp->node_indices[i] = ni;
        const OV_NODE *n    = &model->nodes[ni];

        if (n->type == OV_NODE_STREAM)
        {
            if (lp->nb_streams < (int) (sizeof(lp->stream_indices) / sizeof(lp->stream_indices[0])))
            {
                lp->stream_indices[lp->nb_streams++] = n->index;
            }
        }
        else if (n->type == OV_NODE_PROC)
        {
            if (lp->nb_procs < (int) (sizeof(lp->proc_indices) / sizeof(lp->proc_indices[0])))
            {
                lp->proc_indices[lp->nb_procs++] = n->index;
            }
        }
        else if (n->type == OV_NODE_FPS)
        {
            if (lp->nb_fps < (int) (sizeof(lp->fps_indices) / sizeof(lp->fps_indices[0])))
            {
                lp->fps_indices[lp->nb_fps++] = n->index;
            }
        }
    }

    /* Build auto-name */
    if (lp->nb_streams == 1)
    {
        snprintf(lp->auto_name, sizeof(lp->auto_name), "L%02d [%s]", lp->loop_id,
                 model->streams[lp->stream_indices[0]].name);
    }
    else if (lp->nb_streams >= 2)
    {
        snprintf(lp->auto_name, sizeof(lp->auto_name), "L%02d [%s->%s]", lp->loop_id,
                 model->streams[lp->stream_indices[0]].name,
                 model->streams[lp->stream_indices[lp->nb_streams - 1]].name);
    }
    else
    {
        snprintf(lp->auto_name, sizeof(lp->auto_name), "Loop %02d", lp->loop_id);
    }

    /* Check saved custom names */
    lp->has_custom_name = 0;
    pthread_mutex_lock(&s_loop_names_mutex);
    for (int k = 0; k < s_nb_saved_names; k++)
    {
        if (s_saved_names[k].hash == sig_hash)
        {
            strncpy(lp->custom_name, s_saved_names[k].name, sizeof(lp->custom_name) - 1);
            lp->custom_name[sizeof(lp->custom_name) - 1] = '\0';
            lp->has_custom_name                          = 1;
            break;
        }
    }
    pthread_mutex_unlock(&s_loop_names_mutex);

    if (lp->has_custom_name)
    {
        strncpy(lp->name, lp->custom_name, sizeof(lp->name) - 1);
    }
    else
    {
        strncpy(lp->name, lp->auto_name, sizeof(lp->name) - 1);
    }
    lp->name[sizeof(lp->name) - 1] = '\0';

    model->nb_loops++;
    return 1;
}

/* =========================================================
 * DFS Cycle Search
 * ========================================================= */

static void dfs_search_cycles(OV_MODEL *model,
                              int       start_node,
                              int       curr_node,
                              int       depth,
                              int      *path,
                              uint8_t  *in_path,
                              const int adj[OV_MAX_NODES][64],
                              const int adj_cnt[OV_MAX_NODES],
                              int      *total_explored)
{
    if (model->nb_loops >= OV_MAX_LOOPS || *total_explored > 4096)
    {
        return;
    }
    (*total_explored)++;

    if (depth >= OV_MAX_LOOP_NODES)
    {
        return;
    }

    path[depth]        = curr_node;
    in_path[curr_node] = 1;

    for (int i = 0; i < adj_cnt[curr_node]; i++)
    {
        int next = adj[curr_node][i];

        if (next == start_node && depth >= 1)
        {
            /* Cycle found */
            register_cycle(model, path, depth + 1);
            if (model->nb_loops >= OV_MAX_LOOPS)
            {
                in_path[curr_node] = 0;
                return;
            }
        }
        else if (!in_path[next])
        {
            /* Avoid visiting nodes with index < start_node to reduce redundant traversals */
            if (next >= start_node)
            {
                dfs_search_cycles(model, start_node, next, depth + 1, path, in_path, adj, adj_cnt,
                                  total_explored);
            }
        }
    }

    in_path[curr_node] = 0;
}

/* =========================================================
 * Loop Detection & Metric Computation Engine
 * ========================================================= */

/**
 * ov_detect_loops - Detect all elementary directed cycles in the graph.
 * @model: System model containing streams, processes, FPS, nodes, and edges
 * @mode:  Traversal mode (trigger, input, full)
 */
void ov_detect_loops(OV_MODEL *model, sg_mode_t mode)
{
    if (model == NULL || model->nb_nodes <= 0)
    {
        return;
    }

    if (!s_names_loaded)
    {
        ov_loop_names_load(model);
    }

    /* Reset loop counts on entities */
    for (int i = 0; i < model->nb_streams; i++)
    {
        model->streams[i].loop_mask       = 0;
        model->streams[i].nb_loops        = 0;
        model->streams[i].primary_loop_id = 0;
    }
    for (int i = 0; i < model->nb_procs; i++)
    {
        model->procs[i].loop_mask       = 0;
        model->procs[i].nb_loops        = 0;
        model->procs[i].primary_loop_id = 0;
    }
    for (int i = 0; i < model->nb_fps; i++)
    {
        model->fps[i].loop_mask       = 0;
        model->fps[i].nb_loops        = 0;
        model->fps[i].primary_loop_id = 0;
    }

    model->nb_loops = 0;

    /* Build adjacency lists */
    static int adj[OV_MAX_NODES][64];
    static int adj_cnt[OV_MAX_NODES];
    memset(adj_cnt, 0, sizeof(adj_cnt));

    for (int ei = 0; ei < model->nb_edges; ei++)
    {
        const OV_EDGE *e = &model->edges[ei];
        if (is_valid_loop_edge(model, e, mode))
        {
            int u = e->src_node;
            int v = e->tgt_node;
            if (adj_cnt[u] < 64)
            {
                int dup = 0;
                for (int k = 0; k < adj_cnt[u]; k++)
                {
                    if (adj[u][k] == v)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (!dup)
                {
                    adj[u][adj_cnt[u]++] = v;
                }
            }
        }
    }

    /* Search for cycles starting from each stream node */
    int     path[OV_MAX_LOOP_NODES];
    uint8_t in_path[OV_MAX_NODES];
    memset(in_path, 0, sizeof(in_path));

    for (int ni = 0; ni < model->nb_nodes; ni++)
    {
        if (model->nodes[ni].type == OV_NODE_STREAM)
        {
            int total_explored = 0;
            dfs_search_cycles(model, ni, ni, 0, path, in_path, adj, adj_cnt, &total_explored);
            if (model->nb_loops >= OV_MAX_LOOPS)
            {
                break;
            }
        }
    }

    /* Tag member entities and aggregate health/rate metrics */
    for (int l = 0; l < model->nb_loops; l++)
    {
        OV_LOOP *lp   = &model->loops[l];
        uint32_t mask = (UINT32_C(1) << l);

        lp->is_running   = 1;
        lp->is_paused    = 0;
        lp->is_stale     = 0;
        lp->is_error     = 0;
        lp->min_hz       = 1e9;
        lp->max_hz       = 0.0;
        int active_rates = 0;

        /* Streams */
        for (int i = 0; i < lp->nb_streams; i++)
        {
            int si = lp->stream_indices[i];
            if (si >= 0 && si < model->nb_streams)
            {
                model->streams[si].loop_mask |= mask;
                model->streams[si].nb_loops++;
                if (model->streams[si].primary_loop_id == 0)
                {
                    model->streams[si].primary_loop_id = lp->loop_id;
                }
                if (model->streams[si].update_hz > 0.0)
                {
                    if (model->streams[si].update_hz < lp->min_hz)
                    {
                        lp->min_hz = model->streams[si].update_hz;
                    }
                    if (model->streams[si].update_hz > lp->max_hz)
                    {
                        lp->max_hz = model->streams[si].update_hz;
                    }
                    active_rates++;
                }
            }
        }

        /* Processes */
        for (int i = 0; i < lp->nb_procs; i++)
        {
            int pi = lp->proc_indices[i];
            if (pi >= 0 && pi < model->nb_procs)
            {
                model->procs[pi].loop_mask |= mask;
                model->procs[pi].nb_loops++;
                if (model->procs[pi].primary_loop_id == 0)
                {
                    model->procs[pi].primary_loop_id = lp->loop_id;
                }

                const OV_PROC *pr = &model->procs[pi];
                if (pr->loopstat != 1)
                {
                    lp->is_running = 0;
                }
                if (pr->loopstat == 2)
                {
                    lp->is_paused = 1;
                }
                if (pr->loopstat == 4)
                {
                    lp->is_error = 1;
                }
                if (pr->stale_count > 2)
                {
                    lp->is_stale = 1;
                }

                if (pr->loop_hz > 0.0)
                {
                    if (pr->loop_hz < lp->min_hz)
                    {
                        lp->min_hz = pr->loop_hz;
                    }
                    if (pr->loop_hz > lp->max_hz)
                    {
                        lp->max_hz = pr->loop_hz;
                    }
                    active_rates++;
                }
            }
        }

        /* FPS */
        for (int i = 0; i < lp->nb_fps; i++)
        {
            int fi = lp->fps_indices[i];
            if (fi >= 0 && fi < model->nb_fps)
            {
                model->fps[fi].loop_mask |= mask;
                model->fps[fi].nb_loops++;
                if (model->fps[fi].primary_loop_id == 0)
                {
                    model->fps[fi].primary_loop_id = lp->loop_id;
                }
            }
        }

        if (active_rates == 0)
        {
            lp->min_hz = 0.0;
        }
    }

    /* Compute pairwise overlaps and shared vs exclusive component counts */
    for (int i = 0; i < model->nb_loops; i++)
    {
        OV_LOOP *lp1         = &model->loops[i];
        lp1->overlap_mask    = 0;
        lp1->nb_shared_nodes = 0;

        for (int k = 0; k < lp1->nb_nodes; k++)
        {
            int            ni      = lp1->node_indices[k];
            const OV_NODE *n       = &model->nodes[ni];
            int            n_loops = 0;

            if (n->type == OV_NODE_STREAM && n->index >= 0 && n->index < model->nb_streams)
            {
                n_loops = model->streams[n->index].nb_loops;
            }
            else if (n->type == OV_NODE_PROC && n->index >= 0 && n->index < model->nb_procs)
            {
                n_loops = model->procs[n->index].nb_loops;
            }
            else if (n->type == OV_NODE_FPS && n->index >= 0 && n->index < model->nb_fps)
            {
                n_loops = model->fps[n->index].nb_loops;
            }

            if (n_loops > 1)
            {
                lp1->nb_shared_nodes++;
            }
        }

        lp1->nb_exclusive_nodes = lp1->nb_nodes - lp1->nb_shared_nodes;

        for (int j = 0; j < model->nb_loops; j++)
        {
            if (i == j)
            {
                continue;
            }
            const OV_LOOP *lp2    = &model->loops[j];
            int            shares = 0;

            for (int k1 = 0; k1 < lp1->nb_nodes; k1++)
            {
                for (int k2 = 0; k2 < lp2->nb_nodes; k2++)
                {
                    if (lp1->node_indices[k1] == lp2->node_indices[k2])
                    {
                        shares = 1;
                        break;
                    }
                }
                if (shares)
                {
                    break;
                }
            }

            if (shares)
            {
                lp1->overlap_mask |= (UINT32_C(1) << j);
            }
        }
    }
}
