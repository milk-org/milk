# libmilkdata

Core data structure library providing the `MILK_DATA` type and image management.

## Purpose

Defines `MILK_DATA` — the global data structure holding image arrays, keywords,
and metadata. This is the foundation that all compute units interact with.

## Dependencies

- `milkfps` — FPS types
- `milkprocessinfo` — Process info tracking
- `ImageStreamIO` — Shared memory image streams

## Key Files

| File | Purpose |
|------|---------|
| `milkdata.h` | `MILK_DATA` struct, `IMGID` type, image macros |
| `milkdata_macros.h` | Helper macros for image access |

## Notes

This library has **no CLI dependency** — it can be used in standalone
executables directly. It is part of the core stack:

```
ImageStreamIO → milkprocessinfo → milkfps → milkdata
```
