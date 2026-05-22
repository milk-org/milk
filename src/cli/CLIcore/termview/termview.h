/**
 * @file termview.h
 * @brief Terminal image viewer — public API.
 *
 * Renders a milk shared-memory stream to the terminal using ANSI
 * TrueColor (24-bit RGB) escape sequences.  No ncurses dependency.
 */

#ifndef _TERMVIEW_H
#define _TERMVIEW_H

#include <stdbool.h>
#include <stdint.h>

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

/** @brief Available colormaps */
typedef enum
{
    COLORMAP_GREYSCALE = 0, /**< Black → White (monochrome)             */
    COLORMAP_HEAT,          /**< Black→Blue→Cyan→Yellow→Red             */
    COLORMAP_COLD,          /**< Black→Blue→Cyan→White                  */
    COLORMAP_JET,           /**< Blue→Cyan→Green→Yellow→Red             */
    COLORMAP_INFERNO,       /**< Black→Deep Purple→Orange→Yellow        */
    COLORMAP_VIRIDIS,       /**< Deep Purple→Teal→Yellow-Green          */
    COLORMAP_MAGMA,         /**< Black→Purple→Pink→Orange→White         */
    COLORMAP_PLASMA,        /**< Dark Blue→Purple→Pink→Yellow           */
    COLORMAP_BONE,          /**< Black→Blue-Grey→Light Blue-Grey→White  */
    COLORMAP_RAINBOW,
    COLORMAP_TURBO,
    COLORMAP_OCEAN,
    COLORMAP_COPPER,
    COLORMAP_SPRING,
    COLORMAP_SUMMER,
    COLORMAP_AUTUMN,
    COLORMAP_WINTER,
    COLORMAP_NB
} termview_colormap_t;

/** @brief Pixel intensity scaling modes */
typedef enum
{
    SCALE_LINEAR = 0,  /**< Linear mapping                         */
    SCALE_SQRT,        /**< Square-root stretch                    */
    SCALE_LOG,         /**< Logarithmic stretch                    */
    SCALE_LOG_STRONG,  /**< Strong log stretch                     */
    SCALE_LOG_EXTREME, /**< Extreme log stretch                    */
    SCALE_ASINH,       /**< Asinh stretch                          */
    SCALE_SQUARED,     /**< Squared stretch                        */
    SCALE_CUBED,       /**< Cubed stretch                          */
    SCALE_NB
} termview_scale_t;

/** @brief Display range modes */
typedef enum
{
    RANGE_MINMAX = 0, /**< Full min–max of visible pixels         */
    RANGE_001_999,    /**< 0.1th–99.9th percentile                */
    RANGE_005_995,    /**< 0.5th–99.5th percentile                */
    RANGE_01_99,      /**< 1st–99th percentile                    */
    RANGE_05_95,      /**< 5th–95th percentile                    */
    RANGE_10_90,      /**< 10th–90th percentile                   */
    RANGE_15_85,      /**< 15th–85th percentile                   */
    RANGE_20_80,      /**< 20th–80th percentile                   */
    RANGE_NB
} termview_range_t;

/** @brief All runtime options for termview_screen() */
typedef struct
{
    termview_colormap_t colormap;
    termview_scale_t    scale;
    termview_range_t    range;
    bool                range_locked; /**< Freeze min/max when true      */
    double              manual_min;
    double              manual_max;
    bool                flip_h; /**< Flip image horizontally       */
    bool                flip_v; /**< Flip image vertically         */
} termview_options_t;

/**
 * Open and display a milk shared-memory stream in the terminal.
 *
 * @param imagename  Stream name (resolved via MILK_SHM_DIR or /dev/shm)
 * @param options    Initial display options
 * @return           0 on success
 */
errno_t termview_screen(const char *imagename, termview_options_t options);

#endif /* _TERMVIEW_H */
