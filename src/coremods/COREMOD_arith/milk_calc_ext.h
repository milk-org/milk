// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_calc_ext.h
 * @brief Public C SDK for user-defined `milk.calc` / `@calc` kernels and operators (Tiers 1 & 2).
 *
 * Allows users to bring compiled C functions into `milk.calc` and `@calc` via `-l <file.c|file.so>`:
 *   - Tier 1a (`MILK_CALC_REG_BLK` / `MILK_CALC_REG_BLK2`): 1D per-pixel SIMD block kernels
 *     fused directly into the OpenMP L1 cache tile loop (`FUSE_BLK = 1024`).
 *   - Tier 1b (`MILK_CALC_REG_STENCIL`): 2D neighborhood row-strip kernels receiving `2*R + 1`
 *     pre-clamped row pointers of length `N + 2*R` (`R` in 1..3).
 *   - Tier 1c (`MILK_CALC_REG_IMG`): Single-function whole-image kernels (`IMAGE *out`,
 *     `const IMAGE *in`, `const struct par *P`) composable inside expressions.
 *   - Tier 2  (`MILK_CALC_OP_DECLARE` / `MILK_CALC_REG_OP`): Multi-input / multi-output stateful
 *     operators (`init` / `exec` / `free`) with named image and scalar ports.
 */

#ifndef MILK_CALC_EXT_H
#define MILK_CALC_EXT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ImageStreamIO/ImageStruct.h"
#include "fps_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MILK_CALC_EXT_ABI_VERSION 1
#define MILK_CALC_MAX_PORTS 8
#define MILK_CALC_MAX_PARAMS 16
#define MILK_CALC_PARAM_BYTES 256 /* largest parameter struct the engine packs for a kernel */

/**
 * enum milk_calc_ext_kind_t - Extension entry kind
 */
typedef enum
{
    MILK_EXT_BLK1    = 1, /* 1-input 1D SIMD block kernel */
    MILK_EXT_BLK2    = 2, /* 2-input 1D SIMD block kernel */
    MILK_EXT_STENCIL = 3, /* 2D KxK row-strip stencil kernel */
    MILK_EXT_IMG     = 4, /* 1-input 1-output whole-image kernel */
    MILK_EXT_OP      = 5  /* Multi-input / multi-output stateful operator */
} milk_calc_ext_kind_t;

/* C struct field types corresponding to FPS parameter type tokens (both unexpanded and hex-expanded) */
typedef int32_t  milk_ctype_FPTYPE_INT32;
typedef uint32_t milk_ctype_FPTYPE_UINT32;
typedef int64_t  milk_ctype_FPTYPE_INT64;
typedef uint64_t milk_ctype_FPTYPE_UINT64;
typedef float    milk_ctype_FPTYPE_FLOAT32;
typedef double   milk_ctype_FPTYPE_FLOAT64;
typedef int      milk_ctype_FPTYPE_ONOFF;
typedef char     milk_ctype_FPTYPE_STRING[FUNCTION_PARAMETER_STRMAXLEN];
typedef char     milk_ctype_FPTYPE_FILENAME[FUNCTION_PARAMETER_STRMAXLEN];
typedef char     milk_ctype_FPTYPE_FITSFILENAME[FUNCTION_PARAMETER_STRMAXLEN];
typedef char     milk_ctype_FPTYPE_STREAMNAME[FUNCTION_PARAMETER_STRMAXLEN];

typedef int32_t  milk_ctype_0x00000002;
typedef uint32_t milk_ctype_0x00000004;
typedef int64_t  milk_ctype_0x00000008;
typedef uint64_t milk_ctype_0x00000010;
typedef float    milk_ctype_0x00000020;
typedef double   milk_ctype_0x00000040;
typedef int      milk_ctype_0x00008000;
typedef char     milk_ctype_0x00004000[FUNCTION_PARAMETER_STRMAXLEN];
typedef char     milk_ctype_0x00000200[FUNCTION_PARAMETER_STRMAXLEN];
typedef char     milk_ctype_0x00000400[FUNCTION_PARAMETER_STRMAXLEN];
typedef char     milk_ctype_0x00002000[FUNCTION_PARAMETER_STRMAXLEN];

/**
 * struct milk_calc_param_spec_t - Metadata for one FPS-exposed parameter field
 */
typedef struct
{
    const char *name;
    uint32_t    fptype;
    uint64_t    fpflags;
    double      def_num;
    const char *descr;
    size_t      offset;
    size_t      size;
} milk_calc_param_spec_t;

/**
 * struct milk_calc_param_set_t - Parameter table for a user kernel or operator
 */
typedef struct
{
    int                           nparams;
    size_t                        struct_size;
    const milk_calc_param_spec_t *specs;
} milk_calc_param_set_t;

/* Support both 6-arg X(".kw", field, type, defval, flags, desc) and
 * 5-arg X(field, type, flags, defval, desc) parameter table macros */
#define MILK_CALC_FDEF5_(field, type, flags, defval, desc) milk_ctype_##type field;
#define MILK_CALC_FDEF6_(kw, field, type, defval, flags, desc) milk_ctype_##type field;
#define MILK_CALC_FDEF_PICK_(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define MILK_CALC_FIELD_DEF_(...) \
    MILK_CALC_FDEF_PICK_(__VA_ARGS__, MILK_CALC_FDEF6_, MILK_CALC_FDEF5_)(__VA_ARGS__)

#define MILK_CALC_PSPEC5_(field, type, flags, defval, desc) \
    { #field,                                               \
      (uint32_t) (type),                                    \
      (uint64_t) (flags),                                   \
      (double) (defval),                                    \
      (desc),                                               \
      offsetof(milk_par_s_, field),                         \
      sizeof(((milk_par_s_ *) 0)->field) },
#define MILK_CALC_PSPEC6_(kw, field, type, defval, flags, desc) \
    { #field,                                                   \
      (uint32_t) (type),                                        \
      (uint64_t) (flags),                                       \
      (double) (defval),                                        \
      (desc),                                                   \
      offsetof(milk_par_s_, field),                             \
      sizeof(((milk_par_s_ *) 0)->field) },
#define MILK_CALC_PARAM_SPEC_(...) \
    MILK_CALC_FDEF_PICK_(__VA_ARGS__, MILK_CALC_PSPEC6_, MILK_CALC_PSPEC5_)(__VA_ARGS__)

/**
 * MILK_CALC_PARAMS - Define typed parameter `struct <name>_par` (and `struct <name>`) and its FPS table
 */
#define MILK_CALC_PARAMS(name, PARAM_MACRO)                                                  \
    struct name                                                                              \
    {                                                                                        \
        PARAM_MACRO(MILK_CALC_FIELD_DEF_)                                                    \
    };                                                                                       \
    struct name##_par                                                                        \
    {                                                                                        \
        PARAM_MACRO(MILK_CALC_FIELD_DEF_)                                                    \
    };                                                                                       \
    static inline const milk_calc_param_set_t *__milk_calc_par_fn_##name(void)               \
    {                                                                                        \
        typedef struct name##_par           milk_par_s_;                                     \
        static const milk_calc_param_spec_t specs_[] = { PARAM_MACRO(MILK_CALC_PARAM_SPEC_){ \
            NULL, 0, 0, 0.0, NULL, 0, 0 } };                                                 \
        static const milk_calc_param_set_t  set_     = {                                     \
            (int) (sizeof(specs_) / sizeof(specs_[0])) - 1, sizeof(milk_par_s_), specs_      \
        };                                                                                   \
        return &set_;                                                                        \
    }

static inline const milk_calc_param_set_t *__milk_calc_par_fn_NULL(void)
{
    return NULL;
}

static inline const milk_calc_param_set_t *__milk_calc_par_fn_0(void)
{
    return NULL;
}

static inline const milk_calc_param_set_t *__milk_calc_par_fn_milk_calc_nopar(void)
{
    return NULL;
}

#define MILK_CALC_NOPARAMS milk_calc_nopar

/**
 * enum milk_calc_port_kind_t - Operator input/output port kind
 */
typedef enum
{
    MILK_PORT_IMAGE  = 1,
    MILK_PORT_SCALAR = 2
} milk_calc_port_kind_t;

/* Port synchronization and optionality flags */
#define MILK_PORT_TRIGGER 0x01u
#define MILK_PORT_LATEST 0x02u
#define MILK_PORT_OPT 0x04u
#define MILK_POST_EVERY 0x10u
#define MILK_POST_ON_CHANGE 0x20u

/**
 * struct milk_calc_port_spec_t - Metadata for one named operator port
 */
typedef struct
{
    const char           *name;
    milk_calc_port_kind_t kind;
    uint32_t              flags;
    const char           *descr;
} milk_calc_port_spec_t;

/**
 * struct milk_calc_op_ports_t - Input and output port tables of a Tier 2 operator
 */
typedef struct
{
    int                          nin;
    int                          nout;
    const milk_calc_port_spec_t *in_ports;
    const milk_calc_port_spec_t *out_ports;
} milk_calc_op_ports_t;

#define MILK_PORT_IN_IN(pname, pkind, pflags, pdesc) \
    { #pname, (milk_calc_port_kind_t) (pkind), (uint32_t) (pflags), (pdesc) },
#define MILK_PORT_IN_OUT(pname, pkind, pflags, pdesc)

#define MILK_PORT_OUT_IN(pname, pkind, pflags, pdesc)
#define MILK_PORT_OUT_OUT(pname, pkind, pflags, pdesc) \
    { #pname, (milk_calc_port_kind_t) (pkind), (uint32_t) (pflags), (pdesc) },

#define MILK_CALC_PORT_FILTER_IN_(dir, pname, pkind, pflags, pdesc) \
    MILK_PORT_IN_##dir(pname, pkind, pflags, pdesc)
#define MILK_CALC_PORT_FILTER_OUT_(dir, pname, pkind, pflags, pdesc) \
    MILK_PORT_OUT_##dir(pname, pkind, pflags, pdesc)

/**
 * MILK_CALC_OP_DECLARE - Declare a Tier 2 operator's parameter struct and port tables
 */
#define MILK_CALC_OP_DECLARE(name, PARAM_MACRO, PORT_MACRO)                                        \
    MILK_CALC_PARAMS(name, PARAM_MACRO)                                                            \
    static inline const milk_calc_op_ports_t *__milk_calc_ports_fn_##name(void)                    \
    {                                                                                              \
        static const milk_calc_port_spec_t in_specs_[]  = { PORT_MACRO(MILK_CALC_PORT_FILTER_IN_){ \
            NULL, MILK_PORT_IMAGE, 0, NULL } };                                                    \
        static const milk_calc_port_spec_t out_specs_[] = { PORT_MACRO(                            \
            MILK_CALC_PORT_FILTER_OUT_){ NULL, MILK_PORT_IMAGE, 0, NULL } };                       \
        static const milk_calc_op_ports_t  ports_       = {                                        \
            (int) (sizeof(in_specs_) / sizeof(in_specs_[0])) - 1,                                  \
            (int) (sizeof(out_specs_) / sizeof(out_specs_[0])) - 1, in_specs_, out_specs_          \
        };                                                                                         \
        return &ports_;                                                                            \
    }

/**
 * struct milk_calc_op_ctx_t - Per-call execution context passed to Tier 2 `init` and `exec`
 */
typedef struct milk_calc_op_ctx_s
{
    const char                 *op_name;
    const milk_calc_op_ports_t *ports;
    const IMAGE                *in_img[MILK_CALC_MAX_PORTS];
    double                      in_scalar[MILK_CALC_MAX_PORTS];
    int                         in_bound[MILK_CALC_MAX_PORTS];
    IMAGE                      *out_img[MILK_CALC_MAX_PORTS];
    double                      out_scalar[MILK_CALC_MAX_PORTS];
    uint8_t                     out_naxis[MILK_CALC_MAX_PORTS];
    uint32_t                    out_size[MILK_CALC_MAX_PORTS][3];
    uint8_t                     out_dtype[MILK_CALC_MAX_PORTS];
} milk_calc_op_ctx_t;

typedef milk_calc_op_ctx_t milk_op_ctx;

static inline int milk_calc_port_match_(const char *op_name, const char *port_name, const char *tok)
{
    if (port_name == NULL || tok == NULL)
    {
        return 0;
    }
    if (strcmp(tok, port_name) == 0)
    {
        return 1;
    }
    if (op_name != NULL)
    {
        size_t olen = strlen(op_name);
        if (strncmp(tok, op_name, olen) == 0 && tok[olen] == '_' &&
            strcmp(tok + olen + 1, port_name) == 0)
        {
            return 1;
        }
    }
    size_t tlen = strlen(tok);
    size_t plen = strlen(port_name);
    if (tlen > plen && tok[tlen - plen - 1] == '_' && strcmp(tok + tlen - plen, port_name) == 0)
    {
        return 1;
    }
    return 0;
}

static inline int milk_calc_in_idx_(const milk_calc_op_ctx_t *ctx, const char *tok)
{
    if (ctx == NULL || ctx->ports == NULL)
    {
        return -1;
    }
    for (int i = 0; i < ctx->ports->nin; i++)
    {
        if (milk_calc_port_match_(ctx->op_name, ctx->ports->in_ports[i].name, tok))
        {
            return i;
        }
    }
    return -1;
}

static inline int milk_calc_out_idx_(const milk_calc_op_ctx_t *ctx, const char *tok)
{
    if (ctx == NULL || ctx->ports == NULL)
    {
        return -1;
    }
    for (int i = 0; i < ctx->ports->nout; i++)
    {
        if (milk_calc_port_match_(ctx->op_name, ctx->ports->out_ports[i].name, tok))
        {
            return i;
        }
    }
    return -1;
}

static inline const IMAGE *milk_calc_op_in_(const milk_calc_op_ctx_t *ctx, const char *tok)
{
    int idx = milk_calc_in_idx_(ctx, tok);
    return (idx >= 0 && idx < MILK_CALC_MAX_PORTS) ? ctx->in_img[idx] : NULL;
}

static inline double milk_calc_op_in_scalar_(const milk_calc_op_ctx_t *ctx, const char *tok)
{
    int idx = milk_calc_in_idx_(ctx, tok);
    return (idx >= 0 && idx < MILK_CALC_MAX_PORTS) ? ctx->in_scalar[idx] : 0.0;
}

static inline IMAGE *milk_calc_op_out_(milk_calc_op_ctx_t *ctx, const char *tok)
{
    int idx = milk_calc_out_idx_(ctx, tok);
    return (idx >= 0 && idx < MILK_CALC_MAX_PORTS) ? ctx->out_img[idx] : NULL;
}

static inline double *milk_calc_op_out_scalar_(milk_calc_op_ctx_t *ctx, const char *tok)
{
    static double dummy_ = 0.0;
    int           idx    = milk_calc_out_idx_(ctx, tok);
    return (idx >= 0 && idx < MILK_CALC_MAX_PORTS) ? &ctx->out_scalar[idx] : &dummy_;
}

static inline void milk_op_set_out_shape_(milk_calc_op_ctx_t *ctx,
                                          const char         *tok,
                                          uint32_t            nx,
                                          uint32_t            ny)
{
    int idx = milk_calc_out_idx_(ctx, tok);
    if (idx >= 0 && idx < MILK_CALC_MAX_PORTS)
    {
        ctx->out_naxis[idx]   = (ny > 0) ? 2 : 1;
        ctx->out_size[idx][0] = nx;
        ctx->out_size[idx][1] = (ny > 0) ? ny : 1;
        ctx->out_size[idx][2] = 1;
    }
}

static inline void milk_op_set_out_shape3d_(milk_calc_op_ctx_t *ctx,
                                            const char         *tok,
                                            uint32_t            nx,
                                            uint32_t            ny,
                                            uint32_t            nz)
{
    int idx = milk_calc_out_idx_(ctx, tok);
    if (idx >= 0 && idx < MILK_CALC_MAX_PORTS)
    {
        ctx->out_naxis[idx]   = (nz > 1) ? 3 : ((ny > 0) ? 2 : 1);
        ctx->out_size[idx][0] = nx;
        ctx->out_size[idx][1] = (ny > 0) ? ny : 1;
        ctx->out_size[idx][2] = (nz > 0) ? nz : 1;
    }
}

static inline void milk_op_set_out_datatype_(milk_calc_op_ctx_t *ctx,
                                             const char         *tok,
                                             uint8_t             dtype)
{
    int idx = milk_calc_out_idx_(ctx, tok);
    if (idx >= 0 && idx < MILK_CALC_MAX_PORTS)
    {
        ctx->out_dtype[idx] = dtype;
    }
}

#define MILK_IN(ctx, port) milk_calc_op_in_((ctx), #port)
#define MILK_IN_SCALAR(ctx, port) milk_calc_op_in_scalar_((ctx), #port)
#define MILK_OUT(ctx, port) milk_calc_op_out_((ctx), #port)
#define MILK_OUT_SCALAR(ctx, port) (*milk_calc_op_out_scalar_((ctx), #port))
#define MILK_OUT_SCALAR_PTR(ctx, port) milk_calc_op_out_scalar_((ctx), #port)
#define milk_op_set_out_shape1d(ctx, port, nx) milk_op_set_out_shape_((ctx), #port, (nx), 0)
#define milk_op_set_out_shape(ctx, port, nx, ny) milk_op_set_out_shape_((ctx), #port, (nx), (ny))
#define milk_op_set_out_shape3d(ctx, port, nx, ny, nz) \
    milk_op_set_out_shape3d_((ctx), #port, (nx), (ny), (nz))
#define milk_op_set_out_datatype(ctx, port, dt) milk_op_set_out_datatype_((ctx), #port, (dt))

/* Kernel and operator callback function pointer types */
typedef void (*milk_calc_blk1_fn_t)(float *restrict out,
                                    const float *restrict in,
                                    const void *P,
                                    int         len);

typedef void (*milk_calc_blk2_fn_t)(float *restrict out,
                                    const float *restrict in0,
                                    const float *restrict in1,
                                    const void *P,
                                    int         len);

typedef void (*milk_calc_stencil_fn_t)(float *restrict out,
                                       const float *const *rows,
                                       const void         *P,
                                       int                 len);

typedef int (*milk_calc_img_fn_t)(IMAGE *out, const IMAGE *in, const void *P);

typedef int (*milk_calc_op_init_fn_t)(milk_calc_op_ctx_t *ctx, const void *P, void **state);

typedef int (*milk_calc_op_exec_fn_t)(milk_calc_op_ctx_t *ctx, const void *P, void *state);

typedef void (*milk_calc_op_free_fn_t)(void *state);

/**
 * struct milk_calc_ext_entry_t - Descriptor for a registered user kernel or operator
 */
typedef struct
{
    const char          *name;
    milk_calc_ext_kind_t kind;
    int                  radius;
    int                  cost;
    const milk_calc_param_set_t *(*get_params)(void);
    const milk_calc_op_ports_t *(*get_ports)(void);
    milk_calc_blk1_fn_t    blk1_fn;
    milk_calc_blk2_fn_t    blk2_fn;
    milk_calc_stencil_fn_t stencil_fn;
    milk_calc_img_fn_t     img_fn;
    milk_calc_op_init_fn_t op_init;
    milk_calc_op_exec_fn_t op_exec;
    milk_calc_op_free_fn_t op_free;
} milk_calc_ext_entry_t;

/**
 * struct milk_calc_ext_table_t - Top-level symbol (`__milk_calc_ext_table`) exported by a `.so`
 */
typedef struct
{
    uint32_t                            abi_version;
    int                                 nentries;
    const milk_calc_ext_entry_t *const *entries;
} milk_calc_ext_table_t;

#define MILK_CALC_REG_BLK(kname, fn_f, params_id, kcost)            \
    &((const milk_calc_ext_entry_t) {                               \
        .name       = #kname,                                       \
        .kind       = MILK_EXT_BLK1,                                \
        .radius     = 0,                                            \
        .cost       = (kcost),                                      \
        .get_params = &__milk_calc_par_fn_##params_id,              \
        .get_ports  = NULL,                                         \
        .blk1_fn    = (milk_calc_blk1_fn_t) (void (*)(void))(fn_f), \
    })

#define MILK_CALC_REG_BLK2(kname, fn_f, params_id, kcost)           \
    &((const milk_calc_ext_entry_t) {                               \
        .name       = #kname,                                       \
        .kind       = MILK_EXT_BLK2,                                \
        .radius     = 0,                                            \
        .cost       = (kcost),                                      \
        .get_params = &__milk_calc_par_fn_##params_id,              \
        .get_ports  = NULL,                                         \
        .blk2_fn    = (milk_calc_blk2_fn_t) (void (*)(void))(fn_f), \
    })

#define MILK_CALC_REG_STENCIL(kname, fn_f, params_id, kradius, kcost)  \
    &((const milk_calc_ext_entry_t) {                                  \
        .name       = #kname,                                          \
        .kind       = MILK_EXT_STENCIL,                                \
        .radius     = (kradius),                                       \
        .cost       = (kcost),                                         \
        .get_params = &__milk_calc_par_fn_##params_id,                 \
        .get_ports  = NULL,                                            \
        .stencil_fn = (milk_calc_stencil_fn_t) (void (*)(void))(fn_f), \
    })

#define MILK_CALC_REG_IMG(kname, fn_img, params_id)                  \
    &((const milk_calc_ext_entry_t) {                                \
        .name       = #kname,                                        \
        .kind       = MILK_EXT_IMG,                                  \
        .radius     = 0,                                             \
        .cost       = 4,                                             \
        .get_params = &__milk_calc_par_fn_##params_id,               \
        .get_ports  = NULL,                                          \
        .img_fn     = (milk_calc_img_fn_t) (void (*)(void))(fn_img), \
    })

#define MILK_CALC_REG_OP(kname, op_id, init_f, exec_f, free_f)           \
    &((const milk_calc_ext_entry_t) {                                    \
        .name       = #kname,                                            \
        .kind       = MILK_EXT_OP,                                       \
        .radius     = 0,                                                 \
        .cost       = 4,                                                 \
        .get_params = &__milk_calc_par_fn_##op_id,                       \
        .get_ports  = &__milk_calc_ports_fn_##op_id,                     \
        .op_init    = (milk_calc_op_init_fn_t) (void (*)(void))(init_f), \
        .op_exec    = (milk_calc_op_exec_fn_t) (void (*)(void))(exec_f), \
        .op_free    = (milk_calc_op_free_fn_t) (void (*)(void))(free_f), \
    })

/**
 * MILK_CALC_EXT_EXPORT - Export one or more registered kernels/operators from a translation unit
 */
#define MILK_CALC_EXT_EXPORT(...)                                                                \
    static const milk_calc_ext_entry_t *const __milk_calc_ext_list[] = { __VA_ARGS__ };          \
    __attribute__((visibility("default"))) const milk_calc_ext_table_t __milk_calc_ext_table = { \
        MILK_CALC_EXT_ABI_VERSION,                                                               \
        (int) (sizeof(__milk_calc_ext_list) / sizeof(__milk_calc_ext_list[0])),                  \
        __milk_calc_ext_list                                                                     \
    }

/* Core-tier in-memory kernel and operator registry (implemented in COREMOD_arith) */
int milk_calc_ext_register(const milk_calc_ext_entry_t *e, char *err, size_t errsz);

int milk_calc_ext_register_table(const milk_calc_ext_table_t *tbl, char *err, size_t errsz);

int milk_calc_ext_find(const char *name);

const milk_calc_ext_entry_t *milk_calc_ext_get(int idx);

int milk_calc_ext_count(void);

/**
 * MILK_CALC_MODULE_EXPORT - Export and auto-register module kernels/operators at library load time
 */
#define MILK_CALC_MODULE_EXPORT(modid, ...)                                                      \
    static const milk_calc_ext_entry_t *const __milk_calc_ext_list_##modid[] = { __VA_ARGS__ };  \
    __attribute__((visibility("default"))) const milk_calc_ext_table_t __milk_calc_ext_table = { \
        MILK_CALC_EXT_ABI_VERSION,                                                               \
        (int) (sizeof(__milk_calc_ext_list_##modid) / sizeof(__milk_calc_ext_list_##modid[0])),  \
        __milk_calc_ext_list_##modid                                                             \
    };                                                                                           \
    __attribute__((constructor)) static void __milk_calc_ext_ctor_##modid(void)                  \
    {                                                                                            \
        (void) milk_calc_ext_register_table(&__milk_calc_ext_table, NULL, 0);                    \
    }

#ifdef __cplusplus
}
#endif

#endif /* MILK_CALC_EXT_H */
