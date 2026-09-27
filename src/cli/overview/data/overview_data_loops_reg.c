// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_loops_internal.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* =========================================================
 * FNV-1a 64-bit Hash Helper
 * ========================================================= */

/**
 * fnv1a_hash - Compute 64-bit FNV-1a hash of a null-terminated string
 * @str: Input string
 *
 * Return: 64-bit unsigned hash value.
 */
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

/**
 * is_valid_loop_edge - Check if an edge is eligible to form a causal feedback loop
 * @m:    Pointer to data model
 * @e:    Pointer to edge to validate
 * @mode: Stream graph traversal mode (trigger vs full)
 *
 * Return: 1 if edge is a valid loop edge, 0 otherwise.
 */
int is_valid_loop_edge(const OV_MODEL *m, const OV_EDGE *e, sg_mode_t mode)
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

/**
 * register_cycle - Canonicalize a detected cycle and register it in model
 * @model:    Pointer to data model
 * @path:     Array of node indices forming the cycle
 * @path_len: Number of nodes in @path
 *
 * Return: 1 if new cycle registered, 0 if duplicate or full.
 */
int register_cycle(OV_MODEL *model, const int *path, int path_len)
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
