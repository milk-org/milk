/*
 * cupti_spy.c — CUPTI callback-based GPU host-transfer monitor.
 *
 * Intercepts cudaMemcpy / cudaMemcpyAsync (and their per-thread-stream
 * variants) at the CUDA runtime API level, counting DeviceToHost and
 * HostToDevice operations.  Works transparently across ALL libraries in the
 * same process (CuPy, PyTorch, pyMilk, etc.).
 *
 * Build:  see accompanying Makefile → libcupti_spy.so
 * Use:    load from Python via ctypes (see gpu_transfer_monitor.py)
 */

#define _GNU_SOURCE

#include <cupti.h>

#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* -------------------------------------------------------------------------
 * Internal state
 * ---------------------------------------------------------------------- */

static _Atomic uint64_t g_dtoh_count = 0;
static _Atomic uint64_t g_htod_count = 0;

static CUpti_SubscriberHandle g_subscriber = NULL;

/* -------------------------------------------------------------------------
 * Parameter structs for the memcpy variants we care about.
 * These have been stable since CUDA 3.2; kind is always the 4th field.
 *
 * We use plain int / void* instead of cudaMemcpyKind / cudaStream_t so that
 * this file can be compiled with plain gcc (no nvcc required).  The field
 * layout matches the CUDA ABI exactly.
 * ---------------------------------------------------------------------- */

typedef struct
{
    void       *dst;
    const void *src;
    size_t      count;
    int         kind; /* cudaMemcpyKind */
} _sync_params;       /* cudaMemcpy, cudaMemcpy_ptds */

typedef struct
{
    void       *dst;
    const void *src;
    size_t      count;
    int         kind;   /* cudaMemcpyKind */
    void       *stream; /* cudaStream_t */
} _async_params;        /* cudaMemcpyAsync, cudaMemcpyAsync_ptsz */

/* -------------------------------------------------------------------------
 * CUPTI callback
 * ---------------------------------------------------------------------- */

static void CUPTIAPI _spy_callback(void                     *userdata,
                                   CUpti_CallbackDomain      domain,
                                   CUpti_CallbackId          cbid,
                                   const CUpti_CallbackData *info)
{
    (void) userdata;
    (void) domain;
    (void) cbid;

    /* Only inspect the call on entry (not on exit) */
    if (info->callbackSite != CUPTI_API_ENTER)
    {
        return;
    }

    const char *fn = info->functionName;
    if (!fn)
    {
        return;
    }

    int kind;

    if (strcmp(fn, "cudaMemcpy") == 0 || strcmp(fn, "cudaMemcpy_ptds") == 0)
    {
        kind = ((_sync_params *) info->functionParams)->kind;
    }
    else if (strcmp(fn, "cudaMemcpyAsync") == 0 || strcmp(fn, "cudaMemcpyAsync_ptsz") == 0)
    {
        kind = ((_async_params *) info->functionParams)->kind;
    }
    else
    {
        /*
         * cudaMemcpy2D, cudaMemcpyToSymbol, cudaMemcpyFromSymbol, etc. are
         * not counted here.  Add cases above if your workload uses them.
         *
         * For driver-API copies (cuMemcpyDtoH_v2 / cuMemcpyHtoD_v2) enable
         * CUPTI_CB_DOMAIN_DRIVER_API in cupti_spy_start() and handle them
         * here analogously.
         */
        return;
    }

    /* cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2 — stable since CUDA 1.0 */
    if (kind == 2)
    {
        atomic_fetch_add(&g_dtoh_count, 1);
    }
    else if (kind == 1)
    {
        atomic_fetch_add(&g_htod_count, 1);
    }
    /* DeviceToDevice / Unified stays on GPU — not counted */
}

/* -------------------------------------------------------------------------
 * Public API (called from Python via ctypes)
 * ---------------------------------------------------------------------- */

/**
 * cupti_spy_start — reset counters and begin intercepting memcpy calls.
 * Safe to call multiple times; each call resets the counters.
 */
void cupti_spy_start(void)
{
    atomic_store(&g_dtoh_count, 0);
    atomic_store(&g_htod_count, 0);

    if (g_subscriber != NULL)
    {
        cuptiUnsubscribe(g_subscriber);
    }

    CUptiResult r = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc) _spy_callback, NULL);
    if (r != CUPTI_SUCCESS)
    {
        fprintf(stderr, "cupti_spy: cuptiSubscribe failed (%d)\n", (int) r);
        g_subscriber = NULL;
        return;
    }

    r = cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (r != CUPTI_SUCCESS)
    {
        fprintf(stderr, "cupti_spy: cuptiEnableDomain failed (%d)\n", (int) r);
        g_subscriber = NULL;
        return;
    }
}

/**
 * cupti_spy_stop — disable interception.  Counters are still readable
 * after this call.
 */
void cupti_spy_stop(void)
{
    if (g_subscriber != NULL)
    {
        cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
        cuptiUnsubscribe(g_subscriber);
        g_subscriber = NULL;
    }
}

/** Return number of DeviceToHost transfers recorded since last start. */
uint64_t cupti_spy_get_dtoh(void)
{
    return (uint64_t) atomic_load(&g_dtoh_count);
}

/** Return number of HostToDevice transfers recorded since last start. */
uint64_t cupti_spy_get_htod(void)
{
    return (uint64_t) atomic_load(&g_htod_count);
}

/**
 * cupti_spy_write_counts — write [dtoh, htod] as two little-endian uint64_t
 * to the given file path.  Called explicitly from Python wrappers in the
 * subprocess after cupti_spy_stop().
 */
void cupti_spy_write_counts(const char *path)
{
    if (!path || *path == '\0')
    {
        return;
    }

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0)
    {
        perror("cupti_spy: could not open path");
        return;
    }

    uint64_t counts[2] = {
        atomic_load(&g_dtoh_count),
        atomic_load(&g_htod_count),
    };

    ssize_t written = write(fd, counts, sizeof(counts));
    if (written != (ssize_t) sizeof(counts))
    {
        fprintf(stderr, "cupti_spy: short write to %s\n", path);
    }

    close(fd);
}

/* -------------------------------------------------------------------------
 * Injection entry points (CUDA_INJECTION64_PATH support)
 *
 * When CUDA_INJECTION64_PATH=/path/to/libcupti_spy.so is set in the
 * environment, the CUDA runtime calls InitializeInjection() during cuInit()
 * and FinalizeInjection() at shutdown — before any application code runs or
 * after all CUDA work is done, respectively.
 *
 * Counts are persisted to the file named by CUPTI_SPY_FILE (if set) as
 * two little-endian uint64_t values: [dtoh, htod].  The parent process reads
 * them after the child exits.
 * ---------------------------------------------------------------------- */

static void _write_counts_to_file(void)
{
    const char *path = getenv("CUPTI_SPY_FILE");
    if (!path)
    {
        return;
    }

    cupti_spy_write_counts(path);
}

/*
 * Single dispatcher used by the injection subscriber.
 * During cuInit it receives CUPTI_CB_DOMAIN_RESOURCE events; once
 * CU_INIT_FINISHED fires the runtime domain is armed and subsequent calls
 * come in as CUPTI_CB_DOMAIN_RUNTIME_API events which are forwarded to the
 * counting logic in _spy_callback.
 */
static void CUPTIAPI _injection_dispatch_callback(void                     *userdata,
                                                  CUpti_CallbackDomain      domain,
                                                  CUpti_CallbackId          cbid,
                                                  const CUpti_CallbackData *info)
{
    if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        if (cbid == CUPTI_CBID_RESOURCE_CU_INIT_FINISHED)
        {
            /* CUPTI is now fully ready — arm the runtime memcpy callbacks. */
            cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
            /* Resource domain no longer needed. */
            cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE);
        }
        return;
    }

    /* Otherwise delegate to the normal counting callback. */
    _spy_callback(userdata, domain, cbid, info);
}

static int finalized = 0;

void FinalizeInjection(void)
{
    if (!finalized)
    {
        finalized = 1;
        cupti_spy_stop();
        _write_counts_to_file();
    }
}

int InitializeInjection(void)
{
    atomic_store(&g_dtoh_count, 0);
    atomic_store(&g_htod_count, 0);

    if (g_subscriber != NULL)
    {
        cuptiUnsubscribe(g_subscriber);
    }

    CUptiResult r =
        cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc) _injection_dispatch_callback, NULL);
    if (r != CUPTI_SUCCESS)
    {
        fprintf(stderr, "cupti_spy: injection cuptiSubscribe failed (%d)\n", (int) r);
        g_subscriber = NULL;
        return 0;
    }

    /* Enable the resource domain to receive CU_INIT_FINISHED.
     * The runtime domain will be enabled from that callback. */
    cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE);

    atexit(FinalizeInjection);
    return 1;
}
