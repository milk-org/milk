// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkdata.c
 * @brief   Core milk data global and initialization
 *
 * Provides the MILK_DATA milk_data global and the
 * milk_data_init() function that allocates image,
 * variable, and FPS arrays.
 *
 * This file is compiled into libmilkdata.so, which
 * is the minimal dependency for standalone fpsexec
 * programs (no readline, no CLI).
 */

#include <math.h>
#include <sys/time.h>

#include "milkdata.h"


/* Global core data instance */
MILK_DATA milk_data;


/* Buffer for variable reallocation */
#define NB_VARIABLES_BUFFER_REALLOC 200


/**
 * @brief Initialize core data arrays
 *
 * Allocates image, variable, and FPS arrays in
 * milk_data. Safe to call multiple times (checks
 * if already initialized).
 */
errno_t milk_data_init(void)
{
    struct timeval t1;

    milk_data.NB_MAX_IMAGE    = STATIC_NB_MAX_IMAGE;
    milk_data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
    milk_data.NB_MAX_FPS      = 100;
    milk_data.INVRANDMAX      = 1.0 / RAND_MAX;
    milk_data.rmSHMfile       = 0;

    /* Allocate image array */
#ifdef DATA_STATIC_ALLOC
    milk_data.NB_MAX_IMAGE = STATIC_NB_MAX_IMAGE;
#else
    {
        milk_data.image = (IMAGE *) calloc(milk_data.NB_MAX_IMAGE, sizeof(IMAGE));
        if (milk_data.image == NULL)
        {
            PRINT_ERROR("image array alloc failed");
            exit(1);
        }
    }
#endif

    for (long i = 0; i < milk_data.NB_MAX_IMAGE; i++)
    {
        milk_data.image[i].used      = 0;
        milk_data.image[i].createcnt = 0;
    }

    /* Allocate variable array */
#ifdef DATA_STATIC_ALLOC
    milk_data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
#else
    {
        milk_data.variable = (VARIABLE *) calloc(milk_data.NB_MAX_VARIABLE, sizeof(VARIABLE));
        if (milk_data.variable == NULL)
        {
            PRINT_ERROR("variable array alloc failed");
            exit(1);
        }

        milk_data.image[0].used  = 0;
        milk_data.image[0].shmfd = -1;

        long tmplong = milk_data.NB_MAX_VARIABLE;
        milk_data.NB_MAX_VARIABLE += NB_VARIABLES_BUFFER_REALLOC;

        VARIABLE *tmpvar =
            (VARIABLE *) realloc(milk_data.variable, milk_data.NB_MAX_VARIABLE * sizeof(VARIABLE));
        if (tmpvar == NULL)
        {
            PRINT_ERROR("variable realloc failed");
            exit(1);
        }
        milk_data.variable = tmpvar;

        for (long i = tmplong; i < milk_data.NB_MAX_VARIABLE; i++)
        {
            milk_data.variable[i].used = 0;
            milk_data.variable[i].type = 0;
        }
    }
#endif

    /* Allocate FPS array */
    {
        milk_data.fpsarray = (FPS *) calloc(milk_data.NB_MAX_FPS, sizeof(FPS));
        if (milk_data.fpsarray == NULL)
        {
            PRINT_ERROR("FPS array alloc failed");
            return RETURN_FAILURE;
        }

        for (int i = 0; i < milk_data.NB_MAX_FPS; i++)
        {
            milk_data.fpsarray[i].SMfd   = -1;
            milk_data.fpsarray[i].md     = NULL;
            milk_data.fpsarray[i].parray = NULL;
        }
    }

    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    return RETURN_SUCCESS;
}


/* ========================================
 * Pure-C RNG implementation
 *
 * xorshift64* for uniform generation
 * Box-Muller for Gaussian
 * Knuth for small-mu Poisson,
 * Gaussian approximation for large mu
 * ======================================== */

/**
 * Internal RNG state.
 * xorshift64* has period 2^64-1 and passes
 * BigCrush. Performance is comparable to
 * gsl_rng_rand.
 */
typedef struct
{
    uint64_t state;
    int      has_spare;
    double   spare;
} MILK_RNG;


/**
 * @brief Initializes the milk pseudo-random number generator with a seed.
 */
void milk_rng_init(uint64_t seed)
{
    MILK_RNG *rng = (MILK_RNG *) calloc(1, sizeof(MILK_RNG));
    if (rng == NULL)
    {
        PRINT_ERROR("MILK_RNG alloc failed");
        exit(1);
    }
    /* Avoid zero state (xorshift fixpoint) */
    rng->state       = (seed == 0) ? 1 : seed;
    rng->has_spare   = 0;
    rng->spare       = 0.0;
    milk_data.rndgen = rng;
}


/**
 * @brief Frees the resources associated with the milk random number generator.
 */
void milk_rng_free(void)
{
    if (milk_data.rndgen != NULL)
    {
        free(milk_data.rndgen);
        milk_data.rndgen = NULL;
    }
}


/**
 * @brief xorshift64* core step
 *
 * Returns a uniform uint64 value.
 */
static inline uint64_t xorshift64star(MILK_RNG *rng)
{
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * UINT64_C(0x2545F4914F6CDD1D);
}


/**
 * @brief Generates a uniformly distributed double-precision random number.
 */
double milk_rng_uniform(void)
{
    MILK_RNG *rng = (MILK_RNG *) milk_data.rndgen;
    /* 53-bit mantissa -> [0, 1) */
    return (xorshift64star(rng) >> 11) * (1.0 / 9007199254740992.0);
}


/**
 * @brief Gaussian N(0, sigma) via Box-Muller
 *
 * Generates pairs; caches the spare.
 */
double milk_rng_gaussian(double sigma)
{
    MILK_RNG *rng = (MILK_RNG *) milk_data.rndgen;

    if (rng->has_spare)
    {
        rng->has_spare = 0;
        return rng->spare * sigma;
    }

    double u, v, s;
    do
    {
        u = 2.0 * milk_rng_uniform() - 1.0;
        v = 2.0 * milk_rng_uniform() - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    double f       = sqrt(-2.0 * log(s) / s);
    rng->spare     = v * f;
    rng->has_spare = 1;
    return u * f * sigma;
}


/**
 * @brief Poisson random with mean mu
 *
 * Knuth algorithm for mu < 30,
 * Gaussian approximation for larger mu.
 */
long milk_rng_poisson(double mu)
{
    if (mu < 30.0)
    {
        /* Knuth's algorithm */
        double L = exp(-mu);
        long   k = 0;
        double p = 1.0;

        do
        {
            k++;
            p *= milk_rng_uniform();
        } while (p > L);
        return k - 1;
    }
    else
    {
        /* Gaussian approximation for large mu */
        double val = mu + milk_rng_gaussian(1.0) * sqrt(mu);
        if (val < 0.0)
        {
            val = 0.0;
        }
        return (long) val;
    }
}
