/**
 * @file    pixel_dispatch.h
 * @brief   X-macro tables for datatype dispatch
 *
 * Provides FOREACH_REAL_DATATYPE to eliminate
 * copy-paste switch/if blocks when iterating
 * over image pixel types.
 *
 * Usage example -- set pixel range:
 * @code
 * #define SET_PIX_CASE(DT, ACC, CTYPE)          \
 *     case DT:                                  \
 *         for (uint32_t i = mi; i < ma; i++)    \
 *             im->array.ACC[i] = (CTYPE) val;   \
 *         break;
 *
 * switch (im->md[0].datatype) {
 *     FOREACH_REAL_DATATYPE(SET_PIX_CASE)
 *     default:
 *         PRINT_ERROR("unsupported datatype");
 *         break;
 * }
 * #undef SET_PIX_CASE
 * @endcode
 */

#ifndef PIXEL_DISPATCH_H
#define PIXEL_DISPATCH_H

/**
 * @brief X-macro for all real (non-complex) types.
 *
 * @param X  Macro taking (DTYPE, ACCESSOR, CTYPE)
 *   - DTYPE    _DATATYPE_* constant
 *   - ACCESSOR array union member (UI8, F, ...)
 *   - CTYPE    C language type
 */
#define FOREACH_REAL_DATATYPE(X)                \
    X(_DATATYPE_UINT8,  UI8,  uint8_t)          \
    X(_DATATYPE_INT8,   SI8,  int8_t)           \
    X(_DATATYPE_UINT16, UI16, uint16_t)         \
    X(_DATATYPE_INT16,  SI16, int16_t)          \
    X(_DATATYPE_UINT32, UI32, uint32_t)         \
    X(_DATATYPE_INT32,  SI32, int32_t)          \
    X(_DATATYPE_UINT64, UI64, uint64_t)         \
    X(_DATATYPE_INT64,  SI64, int64_t)          \
    X(_DATATYPE_FLOAT,  F,    float)            \
    X(_DATATYPE_DOUBLE, D,    double)

#endif /* PIXEL_DISPATCH_H */
