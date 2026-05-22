/**
 * @file writer.h
 * @brief Header for the dynamic pattern writer.
 *
 * Defines shared parameters and functions for both standalone and module builds.
 */

#ifndef WRITER_H
#define WRITER_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* =============================================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                                                      */
/* =============================================================================================== */

/** @brief Global pointer to output stream name. */
extern char *out_name_ptr;

/** @brief Global pointer to image width. */
extern uint32_t *width_ptr;

/** @brief Global pointer to image height. */
extern uint32_t *height_ptr;

/** @brief Global pointer to X spatial frequency. */
extern float *freq_x_ptr;

/** @brief Global pointer to Y spatial frequency. */
extern float *freq_y_ptr;


/* =============================================================================================== */
/* SHARED FUNCTIONS                                                                                */
/* =============================================================================================== */

/**
 * @brief Core computation logic for pattern generation.
 *
 * @param fps          Pointer to the FPS structure.
 * @param processinfo  Pointer to the ProcessInfo structure.
 * @param output_image Target image stream.
 */
void writer03_compute(FUNCTION_PARAMETER_STRUCT *fps,
                      PROCESSINFO               *processinfo,
                      IMAGE                     *output_image);

/**
 * @brief Validation logic for the writer parameters.
 */
void writer03_validate();


/* =============================================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                                                  */
/* =============================================================================================== */

/**
 * @brief Shared Parameter Definition Macro for the Writer.
 *
 * Defines CLI types, FPS types, and default values for all writer parameters.
 */
#define WRITER_PARAMS(X)                                                                \
    X(FPTYPE_STRING, char *, ".out_name", "Output Stream Name", "stream03", "stream03", \
      &out_name_ptr, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))                         \
    X(FPTYPE_UINT32, uint32_t, ".width", "Stream Width", "200", 200, &width_ptr,        \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))                                        \
    X(FPTYPE_UINT32, uint32_t, ".height", "Stream Height", "200", 200, &height_ptr,     \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))                                        \
    X(FPTYPE_FLOAT32, float, ".freq_x", "Spatial Freq X", "0.1", 0.1f, &freq_x_ptr,     \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))                                        \
    X(FPTYPE_FLOAT32, float, ".freq_y", "Spatial Freq Y", "0.05", 0.05f, &freq_y_ptr,   \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

/**
 * @brief Detailed help text for the writer.
 */
#define WRITER_HELPTEXT                                                                   \
    "Example 03 Writer: Pattern Generation\n"                                             \
    "======================================\n"                                            \
    "This writer generates a dynamic 2D sine/cosine pattern and writes it to\n"           \
    "an output stream.\n\n"                                                               \
    "Computation Logic:\n"                                                                \
    "  For each output pixel (x, y):\n"                                                   \
    "    val = 0.5 * sin((x + counter) * freq_x) + 0.5 * cos((y + counter) * freq_y)\n\n" \
    "Parameters:\n"                                                                       \
    "  .out_name : Name of the output stream to be created.\n"                            \
    "  .width    : Width of the generated image.\n"                                       \
    "  .height   : Height of the generated image.\n"                                      \
    "  .freq_x   : Spatial frequency/speed in X direction.\n"                             \
    "  .freq_y   : Spatial frequency/speed in Y direction.\n\n"                           \
    "Validation:\n"                                                                       \
    "  The configuration loop ensures width and height are within reasonable\n"           \
    "  limits (minimum 1 pixel)."

#endif
