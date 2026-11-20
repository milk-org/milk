# sample.sh: mlatsim + aoloop, up to the linear response

It works in `~/AOloop2` and follows
[pipelines_tour.md](../../docs/guide/pipelines_tour.md):

1. **Clone:** `mokainit clone` copies `mlatsim-conf` and `aoloop-conf`. This includes
   `scripts/`, which the `mlatsim` setup step needs.
2. **Provision:** `resolve`, then `init`, for each pipeline. For `mlatsim`, `init` writes
   `data/dm.fits`, the 16 × 16 zeros that deploy loads into `dm07disp`, its channels 02, 03 and
   10, and the delayed DM and WFS streams.
3. **Deploy:** `moka deploy -p mlatsim`, then `-p aoloop`.
4. **Link:** `moka link delaysim -p aoloop` points `aol0_wfsim` at `mlatsim_wfs` and
   `aol0_dmC` at `dm07disp03` and `aol0_dmRM` at `dm07disp02`, two channels of `mlatsim`'s DMcomb.
5. **Run:** `moka runstart -p mlatsim` starts `dmcomb`, `dmdelay` and `wfscam` (`dmturb` is on demand), then `moka mlat -p aoloop` starts
   `aoloop`'s `acquWFS` and `mlat` and waits for `mlat` to finish.
6. **Show:** `mokashow mlat -p aoloop`.
7. **Linear response:** `moka mkpokes -p aoloop` writes the poke cube
   `calib/RMmodesDM.fits`; `moka measlinresp -p aoloop` measures the response to it with mlat's
   latency and writes `calib/RMmodesWFS.fits`.

## Before it can work

- **Includes:** [aoloop-conf/conf.toml](aoloop-conf/conf.toml) still has
  `#include fpsdetails.toml`, which is only a comment. It needs `includes = ["fpsdetails.toml"]`
  above `[compute_units]`, otherwise none of the FPS defaults load.
- **Missing streams at deploy:** deploy starts the setup processes of every compute unit.
  `wfs2modes` and `modes2dm` expect `aol0_CMmodesWFS` and `aol0_CMmodesDM`, but nothing creates
  them, so deploy will probably stop with an error there.
  The options are to add `[[dataloads]]` for those streams, or to leave those compute units out
  of this test.
- **Link order:** if `modes2dm` creates `aol0_dmC` as a real file during deploy, the `delaysim`
  link can't replace it with a symlink. Linking before deploying, or creating the stream only on
  first run, would avoid that. Not yet checked which one applies.
- **`mlat` needs `acquWFS`:** `mlat` watches `aol0_imWFS2`, which only `acquWFS` writes, so
  `acquWFS` has to be running first. With all its `comp.*` steps OFF, it is not yet checked
  whether it still needs dark or mask streams to exist.
- **Hardcoded `aol0_`:** the link names are written out with loop number 0, because names on the
  left of a link aren't substituted. If `loop_number` changes, update them too.
