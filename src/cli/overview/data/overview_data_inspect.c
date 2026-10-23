// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_inspect.c
 * @brief   The one stream under inspection: the UI names it, each scan copies its full metadata, semaphore
 *          PIDs and keywords into OV_MODEL.insp (the other streams only carry what the lists show).
 */

#include "overview_data_internal.h"

#include <inttypes.h>

static pthread_mutex_t s_insp_mutex = PTHREAD_MUTEX_INITIALIZER;
static char            s_insp_name[STRINGMAXLEN_IMAGE_NAME];

/** ov_scan_set_inspect - name the stream the next scans describe in full ("" or NULL: none). */
void ov_scan_set_inspect(const char *name)
{
    pthread_mutex_lock(&s_insp_mutex);
    snprintf(s_insp_name, sizeof(s_insp_name), "%s", (name != NULL) ? name : "");
    pthread_mutex_unlock(&s_insp_mutex);
}

/* Keyword value as text */
static void kw_value(const IMAGE_KEYWORD *k, char *buf, size_t n)
{
    if (k->type == 'L')
    {
        snprintf(buf, n, "%" PRId64, k->value.numl);
    }
    else if (k->type == 'D')
    {
        snprintf(buf, n, "%.10g", k->value.numf);
    }
    else
    {
        snprintf(buf, n, "%.*s", (int) sizeof(k->value.valstr), k->value.valstr);
    }
}

/** ov_scan_inspect - fill @model->insp from the cached mapping of the stream under inspection. */
void ov_scan_inspect(OV_MODEL *model)
{
    OV_STREAM_DETAIL *d = &model->insp;
    char              name[STRINGMAXLEN_IMAGE_NAME];
    pthread_mutex_lock(&s_insp_mutex);
    memcpy(name, s_insp_name, sizeof(name));
    pthread_mutex_unlock(&s_insp_mutex);
    d->valid   = 0;
    d->name[0] = '\0';
    if (name[0] == '\0')
    {
        return;
    }
    snprintf(d->name, sizeof(d->name), "%s", name);

    pthread_mutex_lock(&s_scache_mutex);
    int ci = scache_find(name);
    if (ci >= 0 && s_scache[ci].mapped && s_scache[ci].img.md != NULL)
    {
        IMAGE                *img = &s_scache[ci].img;
        const IMAGE_METADATA *md  = img->md;
        snprintf(d->version, sizeof(d->version), "%.*s", (int) sizeof(md->version), md->version);
        d->imagetype      = md->imagetype;
        d->flag           = md->flag;
        d->memsize        = img->memsize;
        d->location       = md->location;
        d->shared         = md->shared;
        d->status         = md->status;
        d->logflag        = md->logflag;
        d->write          = md->write;
        d->creationtime   = md->creationtime;
        d->lastaccesstime = md->lastaccesstime;
        d->atime          = md->atime;
        d->writetime      = md->writetime;
        d->cnt0           = md->cnt0;
        d->cnt1           = md->cnt1;
        d->cnt2           = md->cnt2;
        d->cb_size        = md->CBsize;
        d->cb_index       = md->CBindex;
        d->cb_cycle       = md->CBcycle;
        d->nb_sem         = (md->sem < IMAGE_NB_SEMAPHORE) ? md->sem : IMAGE_NB_SEMAPHORE;
        for (int i = 0; i < d->nb_sem; i++)
        {
            int has_sem          = (img->semptr != NULL && img->semptr[i] != NULL);
            d->semval[i]         = has_sem ? ImageStreamIO_semvalue(img, i) : 0;
            d->sem_read[i]       = (img->semReadPID != NULL) ? img->semReadPID[i] : 0;
            d->sem_write[i]      = (img->semWritePID != NULL) ? img->semWritePID[i] : 0;
            d->sem_read_alive[i] = (uint8_t) (d->sem_read[i] > 0 && pid_is_alive(d->sem_read[i]));
            d->sem_write_alive[i] =
                (uint8_t) (d->sem_write[i] > 0 && pid_is_alive(d->sem_write[i]));
        }
        d->nb_kw        = md->NBkw;
        d->nb_kw_listed = 0;
        for (int i = 0; img->kw != NULL && i < md->NBkw && d->nb_kw_listed < OV_INSP_MAX_KW; i++)
        {
            const IMAGE_KEYWORD *k = &img->kw[i];
            if (k->type == 'N' || k->name[0] == '\0')
            {
                continue; /* unused slot */
            }
            int j = d->nb_kw_listed++;
            snprintf(d->kw[j].name, sizeof(d->kw[j].name), "%.*s", KEYWORD_MAX_STRING, k->name);
            d->kw[j].type = k->type;
            kw_value(k, d->kw[j].value, sizeof(d->kw[j].value));
            snprintf(d->kw[j].comment, sizeof(d->kw[j].comment), "%.*s", KEYWORD_MAX_COMMENT,
                     k->comment);
        }
        d->nb_holders = s_scache[ci].nb_holders;
        for (int k = 0; k < d->nb_holders && k < OV_MAX_HOLDERS; k++)
        {
            char  path[64];
            FILE *f              = NULL;
            d->holder[k].pid     = s_scache[ci].holder[k];
            d->holder[k].comm[0] = '\0';
            snprintf(path, sizeof(path), "/proc/%d/comm", (int) d->holder[k].pid);
            if ((f = fopen(path, "re")) != NULL)
            {
                if (fgets(d->holder[k].comm, sizeof(d->holder[k].comm), f) != NULL)
                {
                    d->holder[k].comm[strcspn(d->holder[k].comm, "\n")] = '\0';
                }
                fclose(f);
            }
        }
        d->valid = 1;
    }
    pthread_mutex_unlock(&s_scache_mutex);
}
