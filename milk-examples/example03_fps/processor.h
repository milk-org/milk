/**
 * @file processor.h
 * @brief Header for the ROI extraction processor.
 * 
 * Defines shared parameters and functions for both standalone and module builds.
 * Uses X-Macros to maintain a single source of truth for all function parameters.
 */

#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* =============================================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                                                      */
/* =============================================================================================== */

/** @brief Global pointer to input stream name string. */
extern char *in_name_ptr;

/** @brief Global pointer to output stream name string. */
extern char *proc_out_name_ptr;

/** @brief Global pointer to ROI size (width and height of square output). */
extern uint32_t *roi_size_ptr;

/** @brief Global pointer to X-offset in the input stream. */
extern uint32_t *off_x_ptr;


/* =============================================================================================== */
/* SHARED FUNCTIONS                                                                                */
/* =============================================================================================== */

/**
 * @brief Core computation logic for ROI extraction.
 * 
 * Extracts a square ROI from input_image and copies it to output_image.
 * 
 * @param fps          Pointer to the FPS structure (used for parameter change detection).
 * @param processinfo  Pointer to the ProcessInfo structure (used for loop synchronization).
 * @param input_image  Source image stream.
 * @param output_image Target image stream.
 */
void processor03_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *input_image, IMAGE *output_image);

/**
 * @brief Validation logic for the configuration loop.
 * 
 * Ensures that the requested off_x and roi_size are valid given the 
 * actual width of the current input stream.
 */
void processor03_validate();


/* =============================================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                                                  */
/* =============================================================================================== */

/**
 * @brief Shared Parameter Definition Macro.
 * 
 * This macro defines every parameter used by the processor. It is expanded multiple times
 * throughout the codebase to generate declarations and initializations.
 * 
 * Columns:
 * 1.  CLI_TYPE:   Type for Milk CLI argument parsing (CLIARG_*).
 * 2.  FPS_TYPE:   Type for FPS shared memory entry (FPTYPE_*).
 * 3.  C_TYPE:     Native C type of the variable.
 * 4.  KEY:        The hierarchical tag used to identify the parameter (e.g. ".off_x").
 * 5.  DESCR:      A short human-readable description for UI display.
 * 6.  DEF_STR:    Default value as a string (for CLI example usage).
 * 7.  DEF_VAL:    Default value as a literal (for C initialization).
 * 8.  PTR_ADDR:   The address of the global extern pointer defined above.
 * 9.  VAL_EXPR:   Expression used to pass the value pointer to FPS initialization.
 * 10. CLI_FLAGS:  Visibility and behavior flags for the Milk CLI ((FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), etc.).
 */
#define PROCESSOR_PARAMS(X) \
    X(   FPTYPE_STRING, char*,    ".in_name",  "Input Stream Name",  "stream03",      "stream03",      &in_name_ptr,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(   FPTYPE_STRING, char*,    ".out_name", "Output Stream Name", "stream03_proc", "stream03_proc", &proc_out_name_ptr,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_UINT32, uint32_t, ".roi_size", "ROI Size",           "50",            50,              &roi_size_ptr, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_UINT32, uint32_t, ".off_x",    "Offset X",           "0",             0,               &off_x_ptr, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

/**
 * @brief Detailed help text for the processor.
 * 
 * This multiline string is stored in the FPS metadata and can be displayed
 * by the 'milk-fpsCTRL' TUI or via the standalone help flag.
 */
#define PROCESSOR_HELPTEXT \
    "Example 03 Processor: ROI Extraction\n" \
    "=====================================\n" \
    "This processor performs a 2D Region of Interest (ROI) extraction from an\n" \
    "input stream and writes the result to an output stream.\n\n" \
    "Computation Logic:\n" \
    "  For each output pixel (x, y) in the [roi_size x roi_size] output image:\n" \
    "    out_data[y, x] = in_data[y, x + off_x]\n\n" \
    "Parameters:\n" \
    "  .in_name  : Name of the input ImageStreamIO stream.\n" \
    "  .out_name : Name of the output stream to be created/updated.\n" \
    "  .roi_size : Width and height of the square output image.\n" \
    "  .off_x    : Horizontal offset in the input image where extraction starts.\n\n" \
    "Validation:\n" \
    "  The configuration loop ensures that [off_x + roi_size] does not exceed\n" \
    "  the input image width. It automatically adjusts off_x or roi_size to\n" \
    "  maintain a valid extraction region."

#endif