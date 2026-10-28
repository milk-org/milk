---
tags:
  - cli
  - scripting
  - calculator
  - units
---

# Calculator & Scripted Compute Units

The milk calculator evaluates scalar and n-dimensional image expressions in a single fused
SIMD/OpenMP pass or via in-place BLAS/LAPACK kernels. Expressions run interactively or in scripts
via `@calc`, `@test`, `@assert`, `@pipe`, and inline `@{…}`, or continuously on every frame of a
stream in a `milk-fpsexec-milk-calc` compute unit.

See also: [Scripting](scripting.md) · [Stream Slicing](slicing.md) ·
[C Extensions (`milk_calc_ext.h` & `milk-cc`)](calc_c_extensions.md) ·
[Sessions](sessions.md) · [FPS](fps.md)

## Calculator (`@calc`, `@test`, `@pipe`)

```bash
@calc "cam = cam * 0 + 5"          # image arithmetic, in place
@calc -v m "imean(cam)"            # bind the result to a bash variable
@calc -v rms,ptv 'istd(cam); imax(cam) - imin(cam)'   # the values of the last statements
@calc -a row "cam[*, 10]"          # the pixels of an image into a bash array
@calc 'rms = istd(cam); ptv = imax(cam) - imin(cam)'   # several statements
@calc "cam[0:31, 0:31] = 0"        # slices: see slicing.md
echo "mean=@{imean(cam)}"          # inline
if @test "imean(cam) > 1"; then echo bright; fi        # status 0 if true, 1 if false
```

- Names in an expression are images, calculator variables and bash variables; an unknown name is an
  error. Quote string literals: `@calc -v n 'strlen("abc")'`.
- A bash variable is read only by calculator text (`@calc`, `@test`, `@assert`, `@pipe`, `@{…}`),
  and only when the expression names it. Its value is data: `s="sqrt(16)"` is the string, not 4. A
  bash variable wins over a calculator variable of the same name, and replaces it: the calculator
  variable keeps that value after `unset`, and no image of that name can be assigned.
- A condition (`@test`, `@assert`, `@waitfor`, `if`) is a number. An image is an error (status 2):
  test `imin(im > 1)` for all pixels, `imax(im > 1)` for any.
- `-v a,b,...` binds the values of the last statements, in order; an assignment's value is the
  assigned value (`@calc -v x "y = 3"` gives 3).
- Calculator variables come back to bash only through `-v` (and `@{…}`): after
  `@calc "x = 5"`, bash has no `$x`. Write `@calc -v x "5"`, or read it later with
  `@calc -v x "x"`. Commands that set calculator variables for their caller declare them:
  `info.imstats im` sets `imsize0`... `vmin` `vmax` `vmean` `vrms` and the percentiles in bash.
- An expression made only of numbers, operators, math functions and `s.`/`f.`/`p.` properties
  is evaluated in the shell itself, in a few microseconds, without a request to the server.
- `@config digits N` sets the number of significant digits of printed results.
- A pixel of an integer image reads and writes as an integer (`im[3,4] % 10`, `im[0,0] & 0xff`).
- An image an expression makes is FLOAT, or DOUBLE when an operand is DOUBLE (`f64(...)`): integer
  images give FLOAT. `x = <expr>` replaces a local image `x` with the result, type included (an
  INT32 `x` is FLOAT after `x = x + 1`). `x += ...`, a number assigned to an image, and any
  assignment into a shared stream keep the image's type: values are truncated toward zero and
  integers wrap (300 in a UINT8 image reads 44). `@config precision` sets the precision of the
  images that module commands create, not the calculator's.
- `if <cond>: <statement>`, `if <cond> { ... } elif <cond> { ... } else { ... }` and
  `for k = start:stop[:step] { ... }` work in `@calc` as in a [unit](#compute-units-without-c).
- `@calc` alone on a line switches `milk-cli` to calc mode: each line is then a calculator
  statement, an assignment prints `name value`, a line that starts with `@` runs as a keyword, a
  bash command is an error, and `exit`, `quit` or `@calc` leave the mode (also in piped input).
  `@config calc on|off` switches the prompt from the next line typed.
- `@pipe "c1; c2"` runs several commands and expressions in one request, and stops at the first
  that fails, with its status; `@pipe "c1 |> c2"` also chains the image each stage makes as `_`.
  With no argument, `@pipe` reads the stages from stdin, one per line (`@pipe <<'EOF' ... EOF`):

```bash
@pipe "mem.mk2Dim a 64 64; a = 5; a *= 2"
@pipe "mem.mk2Dim p 32 32 |> _ = 4 |> _ *= 2.5"
```

### Operators

Operators, loosest first:

| Operators                       | Notes                                                            |
| ------------------------------- | ---------------------------------------------------------------- |
| `=` `+=` `-=` `*=` `/=`, `?:`   | `c ? a : b`: a scalar `c` evaluates only its branch (in a fused statement both run, a zero divisor in the other one is not an error); an image `c` selects per pixel |
| `\|\|`, `&&`                    | skip the right side once the result is known: `0 && (x = 1)`     |
| `\|`, `^^`, `&`                 | bitwise or, xor, and (on scalars and per pixel on images)        |
| `==` `!=`, `<` `<=` `>` `>=`    | 1 or 0; on images, a 0/1 mask per pixel                          |
| `<<` `>>`, `+` `-`, `*` `/` `%` | bit shifts (a count outside $0..63$ gives 0); `"a" + "b"` concatenates strings |
| `^`                             | power, right-associative                                         |
| unary `-` `~` `!`               | bind tighter than `^`: `-2^2` is 4; `~` is bitwise NOT           |

Integer operands give an integer result, except `/` and a negative power (`2^62 + 1` is exact,
`2^-1` is 0.5); an integer result that overflows becomes a double. A number divided by 0, or `% 0`,
is an error, and so is an image divided by the number 0; an image divided by an image gives `inf`
where the divisor is 0. Numbers may be written `0x1f`, `0o17`, `0b101`. Bitwise operators (`&`,
`|`, `^^`, `~`, `<<`, `>>`) also fuse into image expressions via `int64_t` casts (promoting the
block evaluator to `double` when constants exceed $2^{24}$). A kept scalar statement in a unit
carries integers on an exact 64-bit lane with the same rules, so `k * 3 % 1000` agrees with the
text path for any integer variable or parameter `k` (a literal above $2^{53}$ is still read as a double).

### Functions

Functions (the math ones also apply per pixel to an image):

| Kind             | Functions                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Math             | `sin cos tan asin acos atan sinh cosh tanh exp exp2 ln log log2 sqrt cbrt ceil floor round abs fabs sign posi erf erfc isnan isfinite`; `ln` is the natural logarithm, `log` base 10 |
| Math, 2-3 args   | `atan2(y,x)`, `hypot(x,y)`, `fmod(a,b)`, `min(a,b)`, `max(a,b)`, `clamp(x,lo,hi)` (also `trunc(x,lo,hi)`) |
| Datatype casts   | `f32 f64 u8 i8 u16 i16 u32 i32 u64 i64`: cast a scalar or produce an image of that type  |
| Image to number  | `itot imin imax imean istd ivar imedian norm nrm2 asum`, `perc(im,f)`, `dot(a,b)`, `trace(A)`; `irms` = `istd` |
| Image to tuple   | `m, s[, lo, hi] = istats(im)`: mean, std, min, and max; `cx, cy[, flux] = icentroid(im[, th])`: thresholded flux-weighted CoG |
| Image to integer | `icount(im)` (non-zero pixels), `iargmax(im)`, `iargmin(im)`, `iamax(im)` (index of $\max \|x_i\|$) |
| Along one axis   | `itot imin imax imean istd ivar norm nrm2 asum`, `cumsum(im[,axis])` plus an axis: `imean(cube, 2)` is the mean frame |
| Stencil / LUT    | `shift(im,dx[,dy])` (edge-clamped 2-D neighbor offset), `lut(x,tbl[,x0,dx])` (1-D linear LUT) |
| Spatial / Remap  | `medfilt3(im)`, `medfilt5(im)`, `medfilt7(im)` ($3\times3$, $5\times5$, $7\times7$ median), `gauss(im,sx[,sy])`, `interp2(im,xmap,ymap)`, `pixremap(im,map[,fill])`, `pixunmap(im,map[,n])`, `rprof(im[,cx,cy])` |
| Image to image   | `where(c,a,b)`, `transpose(im)`, `reshape(im,nx[,ny,nz])`, `vec(x0,...)`, `zeros(nx[,ny[,nz]])`, `eye(N[,M])`, `diag(X[,k])`, `matmul(A,B)`, `outer(x,y)` |
| BLAS (in-place)  | `scal(a,x)`, `axpy(a,x,y)`, `gemv(a,A,x,b,y)`, `symv(a,A,x,b,y)`, `ger(a,x,y,A)`, `syrk(a,A,b,C)`, `gemm(a,A,B,b,C)` |
| LAPACK / Solvers | `trsv(A,b[,uplo[,diag]])`, `trsm(A,B[,uplo[,diag[,a]]])`, `solve(A,B[,reg])`, `gesv(A,B)`, `posv(A,B)`, `gels(A,B)`, `inv(A)`, `pinv(A[,rtol])` |
| Coordinates/PRNG | `ix iy ir itheta (im[, cx, cy])`: per pixel x, y, radius, angle from `(cx, cy)`, by default `(xsize/2, ysize/2)`; `rand([im[, seed]])` ($[0,1)$, also `urand`), `randn([im[, seed]])` ($\mathcal{N}(0,1)$, also `grand`) |
| Strings          | `strlen toupper tolower`, `substr(s,start,len)`, `replace(s,old,new)`, `hex oct bin`     |
| Complex images   | `cplx(re,im)`, `expi(phi)`, `real imag abs arg conj`; `+ - * /` and `-z` work on them    |
| User functions   | `def f(x, a, ...) = <expr>`: inline function usable on scalars and fused into image passes |

`imin` and `imax` (and their axis forms) skip NaN pixels and give NaN only when every pixel is NaN.
`imedian` and `perc` do not handle NaN, which sorts anywhere: leave NaN pixels out with a mask,
`imedian(im, isfinite(im))`. The other reductions propagate NaN. `imedian` of an even number of
pixels is the upper of the two middle ones; `istd` (and `irms`) is the population standard
deviation. `shift(im, dx, dy)` reads `im[x+dx, y+dy]`, the nearest edge pixel past the border.
`icentroid` gives 0-based pixel coordinates.

A second image argument is a mask: `imean(im, mask)` (also `itot imin imax istd ivar imedian norm
nrm2 asum`) and `perc(im, f, mask)` reduce over the pixels where the mask is not zero; no pixel
gives NaN, or 0 for `itot`/`asum`/`norm`. A number instead is an axis. `matmul`, `gemm`, `gemv`,
`syrk`, `trsv`, `trsm`, `gesv`, `gels`, `solve`, `inv`, and `pinv` fold a top-level `transpose(A)`
directly into the BLAS/LAPACK kernel without allocating an intermediate transposed image. The BLAS
routines (`scal`, `axpy`, `gemv`, `symv`, `ger`, `syrk`, `gemm`) use strict BLAS parameter order and
mutate their destination image in-place (`gemm(alpha, A, B, beta, C)` updates `C` with
$\alpha\,\text{op}(A)\,\text{op}(B) + \beta\,C$), while algebraic assignments using `matmul(A, B)`
and `outer(x, y)` (such as `C = alpha * matmul(A, B) + beta * C`,
`dm = leak * dm - gain * matmul(Cmat, wfs)`, `A += alpha * outer(x, y)`, and
`G = alpha * matmul(transpose(A), A) + beta * G`) automatically lower to the same single in-place
BLAS pass. `reshape(im, nx[, ny, nz])` reinterprets the contiguous elements of `im` with new
dimensions (up to one dimension may be `-1` to infer it from the total element count). `medfilt3`,
`medfilt5`, and `medfilt7` apply exact $3\times3$, $5\times5$, and $7\times7$ 2-D median filters
with edge-clamped borders; `gauss(im, sigma[, sigmay])` runs a separable 2-pass 2-D Gaussian blur
(radius $\min(\lceil 3\sigma\rceil, 32)$); `interp2(im, xmap, ymap)` gathers 2-D sub-pixel
bilinearly interpolated values at `(xmap, ymap)`; `pixremap(im, map[, fill])` gathers
`im[round(map)]` (filling out-of-range indices with `fill`, default `0`), while
`pixunmap(im, map[, n])` computes the per-bin mean into a 2-D `(n, 1)` image (`n` defaults to
$\max(\text{map})+1$, negative indices ignored) and `rprof(im[, cx, cy])` computes the azimuthal
radial profile around `(cx, cy)` (default `(xsize/2, ysize/2)`).

```bash
@calc -v core 'imean(cam, ir(cam) < 10)'          # mean within 10 pixels of the centre
@calc "cam = where(cam > 4000, 0, cam)"            # one pass; a NaN in the other branch stays out
@calc --list                                       # list all pre-compiled bytecode engines & tokens
@calc --explain "out = cam * g; z = cplx(out, out)" # how each statement would run in a unit
@calc --list stencil                               # filter by keyword, opcode, or stage (--json for JSON)
@calc "ramp = ix(cam, 0, 0) / s.cam.xsize"         # 0 ... 1 along x
@calc "z = (cam - imean(cam)) / istd(cam)"         # multi-reduction in one fused expression
@calc -v r "irms(cam - imean(cam))"                # nested reduction
@calc "m, s, lo, hi = istats(cam)"                 # single-pass mean, std, min, max unpacking
@calc "cx, cy, flux = icentroid(cam, 10.0)"        # thresholded sub-pixel center of gravity
@calc "u, v = cam + 1, cam * 2"                    # single-pass multi-image assignment
@calc "lap = 4*cam - shift(cam,1,0) - shift(cam,-1,0) - shift(cam,0,1) - shift(cam,0,-1)"
@calc "cal = lut(cam, resp_curve, 0.0, 0.5)"       # 1-D linear LUT interpolation
@calc "clean = medfilt3(cam - dark)"               # exact 3x3 SIMD median filter
@calc "hp = cam - gauss(cam, 2.0)"                 # 2-D Gaussian unsharp mask
@calc "sub = psf - lut(ir(psf), rprof(psf))"       # 2-D azimuthal radial profile subtraction
@calc "vec = pixunmap(wfs, pupmap); full = pixremap(vec, pupmap, 0.0)"
@calc "warped = interp2(cam, ix(cam,0,0) + dx, iy(cam,0,0) + dy)"
@calc "def soft_thresh(x, t) = where(abs(x) > t, x, 0); out = soft_thresh(cam - dark, 5.0)"
@calc "f.loop.gain = 0.25; s.cam.kw.STREHL = 0.82" # live SHM property assignment
```

Pixels of complex images go through `real`, `imag`, `abs`, `arg`, and `conj` (`abs(z[3,4])` is a
number); on a real image, `real(im)` is `im`, `imag(im)` is a zero image of the same shape, and
`arg(im)` is `0` where pixels are $\ge 0$ and $\pi$ where pixels are $< 0$. Images combined pixel by
pixel must have the same size on each axis, or size 1 (a 2-D image applies to every frame of a
cube); `/=` takes only an image of the destination's size.

## Compute Units Without C

A compute unit is a process that runs a function on each frame of a trigger stream, with its
parameters in an FPS. Two units need no C code: `milk-fpsexec-milk-calc` runs calculator statements,
and `milk-fpsexec-milk-script` runs a bash snippet. They take the commands and options of any
fpsexec unit (`fpsinit`, `exec`, `set`, `runstart`, `runstop`, `run`, `-loopd SEC`, `-n NAME`; see
[FPS](fps.md)). Their program is given with unit options (`-e`, `-t`, `-p`, ...); the values after
the command word then set the parameters that `-p` declares, in declaration order or as
`.name=value`, as the values of any fpsexec set its parameters.

```bash
# out = (cam - dark) * gain on each frame of cam; gain is a parameter of FPS dsub, set to 2.5
milk-fpsexec-milk-calc -n dsub -t cam -p "gain:float=1" -e "out = (cam - dark) * gain" fpsinit 2.5
milk-fpsexec-milk-calc -n dsub runstart         # in the background, returns once the loop runs
milk-fpsexec-milk-calc -n dsub set 3.0          # read by the next frame (same as @set f.dsub.gain=3)
@set f.dsub.unit.expr="out = cam * gain"        # compiled at the next frame (or: -e "..." set)
@get f.dsub.unit.error                          # why the loop stopped (empty while it runs)
milk-fpsexec-milk-calc -n dsub runstop

# one-time initialization (-i) before frame 0, then per-frame leaky integrator, in this terminal
milk-fpsexec-milk-calc -n ema -t cam -p "a:float=0.1" -i "acc = cam * 0" \
    -e "acc = (1 - a) * acc + a * cam" exec .a=0.05

# statements from a file, 100 ticks per second without a trigger stream
milk-fpsexec-milk-calc -n acc -p "k:int=0 total:float=0" -f acc.calc fpsinit
milk-fpsexec-milk-calc -n acc -loopd 0.01 runstart

# a bash snippet on each frame of cam, in a shell that loaded milk.bash once
milk-fpsexec-milk-script -n mon -t cam -c '@get s.cam.mean >> cam.log' fpsinit
milk-fpsexec-milk-script -n mon runstart
```

The unit options of `milk-fpsexec-milk-calc` and the `.unit.*` FPS entries they set
(`milk-fpsexec-milk-calc -h` lists the unit options under `Unit Options` and the declared
parameters under `Parameters`):

| Option | FPS Entry | Meaning |
| --- | --- | --- |
| `-e`, `--expr` | `.unit.expr` | the statements, separated by `;` or newlines |
| `-t`, `--trigger` | `.unit.trigger` | a stream whose frames run them; none: once, or each tick of `-loopd` |
| `-p`, `--params` | `.unit.params` | parameters, `name:type[=default][:help]` as for `@args` (`int` `uint` `float` `bool` `str` `stream` `fits`): the values set them |
| `-f`, `--file` | `.unit.file` | read the statements from this file instead (`#` starts a comment) |
| `-i`, `--init` | `.unit.init` | statements run once before frame 0 (and again if `.unit.init` is set live) |
| `-I`, `--initfile` | `.unit.initfile` | read the init statements from this file instead |
| `-l`, `--lib` | `.unit.libs` | a C extension to load, `.c` (compiled) or `.so` ([C extensions](calc_c_extensions.md)) |
| `--list [PATTERN]` | | print the compiled engines, tokens and C extensions, as `@calc --list` |
| | `.unit.error` | set by the unit: why the loop stopped |

- `fpsinit`, and `exec` when the FPS does not exist yet, create the FPS from the unit options. On an
  existing FPS, `-e` `-t` `-f` `-i` `-I` (with `set`, `exec` or `runstart`) write their `.unit.*`
  entry, as `@set` does; `-p` cannot change: run `fpsinit` again to declare other parameters.
- The statements are compiled when the loop starts, and again at the frame after `.unit.expr` or
  `.unit.file` is set (set `.unit.file` again to read the file again). `.unit.init` (or
  `.unit.initfile`) runs once before frame 0 to create accumulator streams or set initial values;
  changing `.unit.expr` or `.unit.file` live keeps that state intact, while setting `.unit.init` or
  `.unit.initfile` runs the initialization again and resets the frame counters.
- Seven read-only variables are updated before each frame: `_n` (0-based frame index, `int`),
  `_t` (elapsed seconds since loop/init start, `float`), `_dt` (seconds since the previous
  frame, `float`, `0` on frame 0), from the trigger frame itself `_cnt0` (its counter),
  `_ts` (its write time, seconds since the epoch) and `_missed` (trigger frames skipped before
  it, 0 when the unit keeps up), and `_torn` (see the next point). Without a trigger stream,
  `_cnt0` is `_n`, and `_ts` and `_missed` are 0.
- Every input stream a kept statement reads, not only the trigger, is read under a torn-frame
  guard: the statement's pass is run again (up to three times) when a stream's counter moved
  under it, or a writer started a frame during it, and a writer caught mid-frame at the start is
  given up to 100 µs to finish. `_torn` counts those re-runs and the frames read while still
  being written (a writer that holds its write flag longer than that is read as it is, and not
  waited for again until it completes a frame). A steadily growing `_torn` means a producer
  faster than the unit, or a writer that never clears its flag.
- Statements of the program language, on top of the calculator's:
  - `if <scalar>: <statement>` runs a single statement when the condition is not 0
    (`if imax(cam) > 60000: skip`, `if _n % 100 == 0: post(out)`).
  - `if <cond> { ... } elif <cond> { ... } else { ... }` groups multiple statements into
    conditional blocks; conditions are kept as compiled scalar programs (`< 1` µs per check).
  - `for k = start:stop[:step] { ... }` runs a bounded loop (inclusive of `stop`, default `step`
    `1`); more than 100,000 iterations in a frame is an error.
  - `local name = <expr>` (or `L:name = <expr>` / `@L:name = <expr>`) creates a process-private
    scratch image that is not exported to `/dev/shm`. When a `local` image is consumed only by
    the immediately following compatible statement in the same basic block, the compiler merges
    both statements into one cache-resident block pass and elides the intermediate memory write.
  - `def f(x, a, ...) = <expr>` defines an inline function. When called with an image expression
    argument, the argument is evaluated once per 1024-pixel L1d block into a register slot and
    reused across all references in `<expr>`.
  - `a, b = expr1, expr2` assigns multiple images (fused into a single block pass when compatible)
    or multiple scalars; `m, s[, lo, hi] = istats(im)` unpacks mean, std, min, and max from one
    reduction pass.
  - `s.<im>.<prop>`, `f.<fps>.<param>` and `p.<proc>.<field>` can be read directly inside kept
    expressions (`FI_PROP` binary fast-path) or assigned on the left-hand side
    (`f.loop.gain = ...`, `s.out.kw.STREHL = ...`).
  - `skip` ends the tick: the statements after it do not run (the history rings are still
    updated); outputs already written are posted.
  - `post(name)` posts output `name` this tick even when no statement wrote it (readers waiting on
    it wake up); writes post by themselves.
  - `hist(s, N)` declares the ring stream `s_hist`, the last `N` frames of `s` (one more axis, slot
    `n % N` holds frame `n`), filled after the statements of each tick, so inside a tick it holds
    frames `n-1` to `n-N`; `prev(s)` and `prev(s, k)` read frame `n-1` and `n-k` of it, for `k`
    from 1 to `N-1` (`diff = cam - prev(cam)`, `bg = imean(cam_hist, 2)`), as images of `s`'s shape;
    a `prev` of an undeclared stream declares a ring of depth 2. The ring is a stream, so other
    processes read it too. A program has at most 8 rings and 128 statements.
  - `zeros(nx[, ny[, nz]])` makes a zero image of that shape (float; cast for another type,
    `out = f64(zeros(64, 64))`), so an output can be created with an explicit shape and type.
- An element-wise image assignment (`out = a*b + c`, also `+=` `-=` `*=` `/=`), with slices,
  broadcast operands (not for `/=`), bitwise operators, `shift()`, `lut()`, inline `def` functions,
  and axis reductions on either side (`out[0:63,:] = cam[0:63,:] - dark`, `hist[:,:,_n % 8] = cam`,
  `out = cam - imean(cam, 1)`), multi-image assignments (`u, v = expr1, expr2`),
  `out = matmul(...)` / `gemm` / `gemv` / `ger` / `syrk` (including scaled
  `out = alpha * matmul(...) + beta * out` and `+=` / `-=` forms), `out = transpose(...)`,
  `out = reshape(...)`, `out = reshape(matmul(...), ...)`, up to four scalar or nested reductions
  (`v = imean(cam)`, `v = irms(cam - imean(cam))`, `m, s, lo, hi = istats(cam)`, `v = dot(a, b)`,
  `v = imean(cam, mask)`), or a scalar variable assignment (`=`, `+=`, `-=`, `*=`, `/=`) combining
  numbers, parameters, variables, live SHM properties, and scalar reductions
  (`ema = 0.9 * ema + 0.1 * irms(cam - dark)`) is compiled once and runs directly on each frame
  without re-parsing; consecutive compatible element-wise assignments in the same basic block are
  automatically fused into a single cache-resident pass. Other statements are evaluated as text on
  each frame. `out /= image` is one of them: a zero pixel in the divisor is an error that stops the
  unit (write `out = out * invflat` for a compiled division).
- An output written by several statements (`out = cam * g; out = out - imean(out)`) is posted once
  per frame, after the last statement that wrote it, so readers never see the intermediate values.
- A trigger stream re-created by its owner is remapped at the next trigger timeout (2 s), and an
  owner that has exited is reported in the unit's status message; `p.<unit>.timeouts` counts the
  timeouts.
- Threads: `.procinfo.NBthread` defaults to 0, which lets the team policy decide (one thread
  below 512 K pixels, then one per 256 K pixels up to the physical cores allowed by the process's
  affinity mask, so a unit pinned with `taskset` to one core stays single-threaded). A positive
  value caps the team and OpenBLAS at that many threads. The calculator's threads are its own
  persistent team, dispatched once per block loop (no fork/join per statement): after a job the
  workers keep spinning for `MILK_CALC_SPIN_US` (default 2000 µs) before they sleep, so a unit
  ticking faster than that never wakes a sleeping worker (a wake-up costs ~100 µs, a dispatch to
  spinning workers ~1 µs). For a real-time unit at a lower rate, set `MILK_CALC_SPIN_US=-1` to keep
  the workers spinning for the life of the process (they then burn their cores), and for a unit on
  an isolated core set `MILK_CALC_SPIN_US=0` or `.procinfo.NBthread=1` so no idle worker spins at
  all. `OMP_WAIT_POLICY` still governs the OpenBLAS and other OpenMP regions.
- `-loopd SEC` ticks on an absolute deadline (`clock_nanosleep`, timer slack 1 µs below 1 ms), so
  periods of a few tens of µs hold without drift; a tick that ran long resumes from now rather
  than bursting to catch up.
- Names are shared streams: the unit is its own process, so a session's local images and calculator
  variables are out of its reach. A statement that makes a new image creates it as a stream. Shared
  streams recreated in `$MILK_SHM_DIR` with the same shape and compatible type are remapped
  automatically.
- The `-p` parameters are calculator variables that `@set` changes while the loop runs. A
  statement that assigns one (`total = itot(out)`) writes it back to the FPS on each frame, for
  `@get`, `@waitfor` and other units. A `stream` parameter (`in:stream=cam0`, `out:stream=out0`)
  binds an input or output stream name dynamically so `@set f.dsub.in=cam1` re-targets the unit on
  the next frame.
- An error stops the loop: a bad statement, or an image whose size or type changed.
- FPS strings hold 63 characters, so longer programs go in a file (`-f` or `-I`). `-p` may be longer
  on the `fpsinit` or `exec` command line that creates the FPS.

`milk-fpsexec-milk-script` takes `-c`/`--cmd` (the snippet, `.unit.cmd`), `-t` (`.unit.trigger`),
`-p` (`.unit.params`: `name:type[=default][:help]`, synced into shell variables before each tick;
the values set them), and `-f` (`.unit.file`: read the script from a file instead of `.unit.cmd`).
The 0-based frame counter `_n` is also set on each tick. A new `.unit.cmd` or `.unit.file` applies
from the next tick. A tick that fails stops the loop, with its status in `.unit.error`; end the
snippet with `|| :` to go on. Ctrl-C or `runstop` interrupts a running tick. A tick costs its
commands plus a pipe round trip (about 0.1 ms), which suits slow loops.

## Fusion & Cost Model

An expression made of same-size images, numbers, arithmetic, comparisons, bitwise operators, the
per-pixel math functions, `where`, `?:` with an image condition, `shift`, `lut`, inline `def`
functions, the coordinate/PRNG functions, live SHM numeric properties (`s.`, `f.`, `p.`), and
up to four full-image, nested, or masked scalar reductions (`(a - imean(a)) / istd(a)`,
`irms(a - imean(a))`, `imean(a, mask)`, `dot(a, b)`) runs in one fused SIMD/OpenMP pass without
temporary images, however many operators it has (and an image load is folded into the operator
that follows it, so `a*2.5 + b*0.5` is two sweeps of each block, not five) (1024×1024 floats: `x = (b*2 + c*3 - a) / 4`
~0.2 ms, `a += 1` ~0.2 ms), and assigning it to an existing image of the same size and type writes
in place. Multiple reductions over the same sub-expression (`(a - imin(a)) / (imax(a) - imin(a))`,
`m, s, lo, hi = istats(a)`) are deduplicated into a single reduction pass. Repeated `@calc`
statements reuse a 32-entry LRU compiled program cache, and OpenMP parallel regions (axis
reductions, where every thread owns its outputs, segmented reductions, the Gaussian passes and the
pixel loop) automatically size their thread team to the compute-weighted image size and physical
core count (`MILK_CALC_THREADS=N` overrides it; `OMP_NUM_THREADS` can only lower it; OpenBLAS calls
size their pool the same way).

A pass runs in float32 when every operand and destination is `float` or an 8- or 16-bit integer
(`u8 i8 u16 i16`, exact in float32: a camera frame needs no cast), and in double when any of them is
`double` or a 32- or 64-bit integer; a stencil, `lut`, `pixremap` or `interp2` on an integer image
keeps its pass in double. `@calc --explain` shows which.

`@calc --explain "<statements>"` compiles the statements as a unit would and prints, per statement,
whether it is kept (a fused pass with its operand count, precision and thread count, a pass merged
into an earlier statement, a scalar tape, a BLAS kernel) or evaluated as text each tick and why (the
limit or construct that stopped fusion); new targets are created as at a unit's start. A unit prints
the same reason once, at compile, for every statement it will evaluate as text.

Slices, broadcasting and axis reductions are part of the one pass too: a slice operand
(`a[0:63,:]`, `cube[:,:,k]`, `a[::2,-*]`) is read through a view of its image (no copy), an
operand whose extent is 1 on an axis is broadcast along it (`a - row - col`, `cube - dark`), a
sliced left-hand side (`out[0:63,:] = …`, `hist[:,:,_n % 8] = cam`) is written through a view, and
`imean(cube, 2)`, `istd(a, 0)` and the other axis forms are computed before the pixel loop and
read as operands on their own axis (`a - imean(a, 1)` subtracts each column's mean, `a - imean(a, 0)`
each row's, also when `a` is not square; assigned on their own they keep the squeezed shape, `ny`
values for `imean(a, 0)`). `where(c, a, b)` and `c ? a : b` with a scalar `c` and image branches are
fused too. Bounds that hold variables (`out[k:k+7,:]`, `ring[:,:,_n % 8]`, `prev(cam)`) are compiled
into the statement's scalar tape and re-placed each run for about a microsecond, not re-parsed; a
bound that would change the slice's shape is an error. Bounds that hold variables are resolved each time the
statement runs, also in a unit. What still takes one pass per operator: binned slices (`[::2b]`),
order statistics (`imedian`, `perc`), complex images, and a slice that selects one pixel (a
scalar). Those passes write temporary images: the calculator keeps up to four of them (256 MiB) when
a statement ends, so the statement run again, in a loop or a unit's frames, reuses them instead of
allocating and zeroing new ones (`mem.listim` lists them as `_tmpcalcpool<k>`).

## Bringing Your Own Compiled C Code (`milk_calc_ext.h` & `milk-cc`)

Users can extend `@calc` and `milk-fpsexec-milk-calc` with compiled C code (`-l <file.c|file.so>`)
or compile standalone `MILK_RUNNABLE` compute units with a single command (`milk-cc <unit.c>`),
without editing CMake files.

| Tier | Registration Macro | Best For |
| ---- | ------------------ | -------- |
| **1a: 1D SIMD Block Kernel** | `MILK_CALC_REG_BLK`, `MILK_CALC_REG_BLK2` | Per-pixel non-linearities, LUTs, 2-image blends fused into the L1 tile loop (`FUSE_BLK = 1024`) |
| **1b: 2D Neighborhood Stencil** | `MILK_CALC_REG_STENCIL` | $(2R+1)\times(2R+1)$ local neighborhood filters ($R \in \{1,2,3\}$) with pre-clamped row pointers |
| **1c: Whole-Image Kernel** | `MILK_CALC_REG_IMG` | Single-function 1-in / 1-out global image algorithms (column/row de-meaning, custom transforms) |
| **2: Stateful Multi-Port Operator** | `MILK_CALC_OP_DECLARE`, `MILK_CALC_REG_OP` | Multi-input / multi-output algorithms with `init` / `exec` / `free` state and custom output shapes |
| **3: Standalone `MILK_RUNNABLE`** | `milk-cc <unit.c>` | Full standalone `milk-fpsexec-*` binary with custom semaphore triggering |

### Auto-Compilation and Caching (`-l` / `--lib`)

Passing a `.c` file to `@calc -l my_kernels.c "..."` or
`milk-fpsexec-milk-calc -l my_kernels.c ...` automatically compiles it into
`${TMPDIR:-/tmp}/milk-calc-cache-<uid>/ext_<fnv1a64>.so` with
`-O3 -march=native -ffast-math -fopenmp-simd -fPIC -shared` and loads it via `dlopen()`. A `.c` file
is compiled as C; there is no C++ path. `MILK_CALC_LIBS` (files separated by spaces, `,` or `:`)
loads extensions in every calculator call, and a `loadlib <file>` statement loads one from
calculator text.

### FPS Parameter Exposure Rules (`MILK_CALC_PARAMS`)

Parameters declared with `MILK_CALC_PARAMS(name, PARAM_MACRO)` follow three rules in `milk.calc`
([details](calc_c_extensions.md#fps-parameter-binding-rules)):

1. **Unbound at the `.calc` call site** (`lin = nlcorr(raw - dark)`): automatically exposed in the
   unit's FPS as `.<func>.<param>`.
2. **Instance labels** (`despike3x3:stage1(lin)`) expose `.<label>.<param>`, one group per call.
3. **Bound at the `.calc` call site**, by name or by position (`clean = despike3x3(lin, th = 4.0 * rms)`,
   `despike3x3(lin, 40)`): driven by the `.calc` expression and hidden from the FPS to avoid
   parameter clutter.

For Tier 2 operators (`MILK_CALC_OP_DECLARE`), `MILK_PORT_SCALAR` outputs assigned on the LHS
(`sx, sy, flux = shslopes(cal)`) are automatically exposed as `FPFLAG_DEFAULT_OUTPUT` telemetry
parameters in the unit's FPS.

For full code examples of each tier and the complete `milk_calc_ext.h` API reference, see
**[C Extensions (`milk_calc_ext.h` & `milk-cc`)](calc_c_extensions.md)**.

---

← [Scripting](scripting.md) · [Stream Slicing](slicing.md) · [C Extensions (`milk_calc_ext.h`)](calc_c_extensions.md) · [Documentation Index](../index.md)
