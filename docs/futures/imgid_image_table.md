# IMGID and the image table

How an `IMGID` gets an image, and when that image can be found by name.

**Audience:** Internals · See also: [Programmer's Guide](../arch/programmers_guide.md) ·
[Architecture](../arch/architecture.md) · [Dependency Graph](../arch/dependency_graph.md)

!!! warning "Memo for future refactoring"

    Not a description of a settled design. Back to [Notes for the future](../notes_for_future.md).

## 1. Layering

| Layer          | Links                                  | Owns                                                                                              |
| -------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| ImageStreamIO  | –                                      | `IMAGE`, `ImageStreamIO_createIm`                                                                 |
| libfps         | ImageStreamIO, processinfo, milkcommon | `IMGID.h`: header-only handle API (`imgid_make*`, `imgid_mkimage`, `imgid_connect`)               |
| libmilkdata    | ImageStreamIO, milkcommon              | the `milk_data` global; `dcimg` is a macro for `milk_data.image`                                  |
| COREMOD_memory | both                                   | the table operations: `image_ID`, `next_avail_image_ID`, `create_image_ID_IMGID`, `RegisterIMGID` |

libfps and libmilkdata are siblings: neither links the other. `IMGID.h` therefore never names
`dcimg`. Writing `dcimg` in one of its `static inline` functions would force every includer,
compute-only files included, to include `milkdata.h` and link libmilkdata. None of libfps's own
`.c` files include `IMGID.h`; it sits in libfps because it needs `fps_streamname_parse`.

`imgid_mkimage` is the table-unaware primitive: it runs `ImageStreamIO_createIm` into whatever
`img->im` points to. Whether the image ends up in `dcimg` is the caller's decision.

## 2. Three ways to create an image from a template IMGID

| Pattern | Code                                                                   | Where                                                                                                                                                                        | Findable by name                     |
| ------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| A       | `create_image_ID_IMGID(img)`: creates directly into a `dcimg` slot     | `create_image.c`, every legacy `create_*image_ID` wrapper, `loadfits.c`                                                                                                      | yes                                  |
| B       | `calloc` + `imgid_mkimage`, then `RegisterIMGID` copies it into a slot | COREMOD_arith (`imgid_arith_helpers.h`, `imfunctions.c`, `im_complex.c`, …)                                                                                                  | yes                                  |
| C       | `calloc` + `imgid_mkimage`, no registration                            | `image_copy.c`, `stream_updateloop.c`, `stream_pixmapdecode.c`, `stream_{UDP,TCP}_receive.c`, `images2cube.c`, `breakcube.c`, `statusstat.c`, some cacao and milk-extra code | no: only through the returned handle |

C is fine for a caller that keeps the returned IMGID. It breaks any caller that later looks the
image up by name. Commit `7b621525e2` turned `load_fits_IMGID` from A into C, which broke
`load_fits(file, name, …, &ID)` for every caller; `c707aa9085` restored A. Commit `d17ef84892`
did the same to `FFT_do2dfft` and friends in milk-extra `fft/`, which broke `do2dfft` followed by
a by-name lookup of its output (DMturbulence seed generation); they now use
`createimagefromIMGID`.

On an existing name, A and B differ:

- A compares type, naxis and size, and deletes and re-creates the image on a mismatch.
- B discards the new image and points at the existing one without comparing. A caller that then
  writes into it (as `load_fits_IMGID` does with `fits_read_img`) overflows a smaller buffer.

A does not set `img->im`, `img->md` or `img->createcnt`. `loadfits_mkimage` in `loadfits.c` is A
plus those three assignments.

## 3. Known problems

- **Duplicate:** A and B do the same job, put a template IMGID into the table, with two
  implementations and different handling of an existing name. `createimagefromIMGID`
  (`COREMOD_memory/imageID.h`) and `loadfits_mkimage` are each A plus the handle update.
- **Pattern C callers looked up by name** have the latent bug `load_fits` had. Not audited.
- **`ID = 0` as a marker:** `imgid_connect` sets `img->ID = 0` for "connected, not in the
  table". 0 is also a valid `dcimg` slot, so `resolveIMGID` compares against `dcimg[0]`.
- **Docs:** [Architecture](../arch/architecture.md) says libmilkdata holds the IMGID struct; the code and
  the [Programmer's Guide](../arch/programmers_guide.md) put it in `libfps/IMGID.h`.

Planned fix: two explicit APIs, and every creation site picks one.

- **Anonymous API:** no table and no name lookup. The caller keeps the `IMAGE` handle and the
  IMGID and uses only those (what C does today, made explicit). Nothing downstream may look the
  image up by name.
- **Global API:** one registering constructor in COREMOD_memory, A plus the handle update
  (merging `createimagefromIMGID` and `loadfits_mkimage`). The image is findable by name. B
  becomes a call to it.
