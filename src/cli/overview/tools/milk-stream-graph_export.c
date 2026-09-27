// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-stream-graph_export.c
 * @brief Output formatting (text, JSON, pretty TrueColor) for stream graphs.
 */

#include "milk-stream-graph.h"
#include <stdio.h>

/**
 * sg_print_text - Output plain-text machine-readable lineage report
 * @m:           Pointer to data model
 * @stream_name: Root stream name
 * @mode:        Active traversal mode
 * @lin:         Computed lineage data
 */
void sg_print_text(const OV_MODEL   *m,
                   const char       *stream_name,
                   sg_mode_t         mode,
                   const SG_LINEAGE *lin)
{
    printf("# milk-stream-graph v%s\n", SG_VERSION);
    printf("# stream: %s\n", stream_name);
    printf("# mode: %s\n", sg_mode_label(mode));
    printf("# has_loop: %s\n", lin->has_loop ? "yes" : "no");

    for (int i = 0; i < lin->nb_ancestors; i++)
    {
        const SG_LINEAGE_ENTRY *e  = &lin->ancestors[i];
        const char             *sn = m->streams[e->stream_idx].name;
        printf("ANCESTOR  depth=%d  stream=%s  via=%s", e->depth, sn, e->via_name);
        if (e->is_loop)
        {
            printf("  LOOP");
        }
        printf("\n");
    }

    for (int i = 0; i < lin->nb_descendants; i++)
    {
        const SG_LINEAGE_ENTRY *e  = &lin->descendants[i];
        const char             *sn = m->streams[e->stream_idx].name;
        printf("DESCENDANT  depth=%d  stream=%s  via=%s", e->depth, sn, e->via_name);
        if (e->is_loop)
        {
            printf("  LOOP");
        }
        printf("\n");
    }

    if (lin->has_loop && lin->cycle_len > 0)
    {
        printf("CYCLE ");
        for (int i = 0; i < lin->cycle_len; i++)
        {
            if (i > 0)
            {
                printf(" -> ");
            }
            printf("%s", m->streams[lin->cycle_path[i]].name);
        }
        printf("\n");
    }
}

/**
 * sg_print_json - Output JSON formatted lineage data
 * @m:           Pointer to data model
 * @stream_name: Root stream name
 * @mode:        Active traversal mode
 * @lin:         Computed lineage data
 */
void sg_print_json(const OV_MODEL   *m,
                   const char       *stream_name,
                   sg_mode_t         mode,
                   const SG_LINEAGE *lin)
{
    printf("{\n");
    printf("  \"stream\": \"%s\",\n", stream_name);
    printf("  \"mode\": \"%s\",\n", sg_mode_label(mode));
    printf("  \"has_loop\": %s,\n", lin->has_loop ? "true" : "false");

    /* ancestors */
    printf("  \"ancestors\": [");
    for (int i = 0; i < lin->nb_ancestors; i++)
    {
        const SG_LINEAGE_ENTRY *e = &lin->ancestors[i];
        if (i > 0)
        {
            printf(",");
        }
        printf("\n    {\"stream\": \"%s\","
               " \"depth\": %d,"
               " \"via\": \"%s\","
               " \"is_loop\": %s}",
               m->streams[e->stream_idx].name, e->depth, e->via_name,
               e->is_loop ? "true" : "false");
    }
    printf("\n  ],\n");

    /* descendants */
    printf("  \"descendants\": [");
    for (int i = 0; i < lin->nb_descendants; i++)
    {
        const SG_LINEAGE_ENTRY *e = &lin->descendants[i];
        if (i > 0)
        {
            printf(",");
        }
        printf("\n    {\"stream\": \"%s\","
               " \"depth\": %d,"
               " \"via\": \"%s\","
               " \"is_loop\": %s}",
               m->streams[e->stream_idx].name, e->depth, e->via_name,
               e->is_loop ? "true" : "false");
    }
    printf("\n  ],\n");

    /* cycle path */
    printf("  \"cycle\": [");
    if (lin->has_loop)
    {
        for (int i = 0; i < lin->cycle_len; i++)
        {
            if (i > 0)
            {
                printf(", ");
            }
            printf("\"%s\"", m->streams[lin->cycle_path[i]].name);
        }
    }
    printf("]\n");
    printf("}\n");
}

/**
 * sg_print_pretty - Output TrueColor ANSI formatted interactive lineage tree
 * @m:           Pointer to data model
 * @stream_name: Root stream name
 * @mode:        Active traversal mode
 * @lin:         Computed lineage data
 */
void sg_print_pretty(const OV_MODEL   *m,
                     const char       *stream_name,
                     sg_mode_t         mode,
                     const SG_LINEAGE *lin)
{
    printf(SGC_BOLD SGC_HEADER "Stream Graph" SGC_RESET SGC_TEXT "  stream: " SGC_BOLD SGC_STREAM
                               "%s" SGC_RESET SGC_TEXT "  mode: " SGC_FPS "%s" SGC_RESET "\n\n",
           stream_name, sg_mode_label(mode));

    if (lin->nb_ancestors == 0 && lin->nb_descendants == 0)
    {
        printf(SGC_DIM "  No lineage found\n\n" SGC_RESET);
    }
    else
    {
        /* Ancestors */
        for (int i = lin->nb_ancestors - 1; i >= 0; i--)
        {
            const SG_LINEAGE_ENTRY *e  = &lin->ancestors[i];
            const char             *sn = m->streams[e->stream_idx].name;

            printf(" " SGC_DEPTH "-%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET, e->depth, sn);
            if (e->is_loop)
            {
                printf(" " SGC_BLINK SGC_LOOP "[LOOP]" SGC_RESET);
            }
            printf("\n");

            if (e->via_name[0] != '\0')
            {
                printf("      " SGC_ARROW "│" SGC_RESET "\n");
                printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
            }
        }

        /* Root */
        printf(" " SGC_DEPTH " 0 " SGC_RESET " " SGC_BOLD SGC_STREAM "%s" SGC_RESET "\n",
               stream_name);

        /* Descendants */
        for (int i = 0; i < lin->nb_descendants; i++)
        {
            const SG_LINEAGE_ENTRY *e  = &lin->descendants[i];
            const char             *sn = m->streams[e->stream_idx].name;

            if (e->via_name[0] != '\0')
            {
                printf("      " SGC_ARROW "│" SGC_RESET "\n");
                printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
            }

            printf(" " SGC_DEPTH "+%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET, e->depth, sn);
            if (e->is_loop)
            {
                printf(" " SGC_BLINK SGC_LOOP "[LOOP]" SGC_RESET);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Cycle info */
    if (lin->has_loop && lin->cycle_len > 0)
    {
        printf(SGC_BOLD SGC_LOOP " Cycle detected:" SGC_RESET " ");
        for (int i = 0; i < lin->cycle_len; i++)
        {
            if (i > 0)
            {
                printf(SGC_ARROW " -> " SGC_RESET);
            }
            printf(SGC_STREAM "%s" SGC_RESET, m->streams[lin->cycle_path[i]].name);
        }
        printf("\n\n");
    }
}
