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

- A name in an expression is an image if one has that name, else a bash variable, else a calculator
  variable; an unknown name is an error. A name has at most 79 characters. Quote string literals:
  `@calc -v n 'strlen("abc")'`. A calculator variable holds what was assigned last, a number or a
  string: after `s = "5"`, `s + 1` is `"51"`.
- A bash variable is read only by calculator text (`@calc`, `@test`, `@assert`, `@pipe`, `@{…}`),
  and only when the expression names it. Its value is data: `s="sqrt(16)"` is the string, not 4. A
  bash variable wins over a calculator variable of the same name, and replaces it: the calculator
  variable keeps that value after `unset`, and no image of that name can be assigned.
- A condition (`@test`, `@assert`, `@waitfor`, `if`) is a number: not 0 is true, 0 and NaN are
  false. A string is an error, and so is an image (status 2): test `imin(im > 1)` for all pixels,
  `imax(im > 1)` for any.
- `-v a,b,...` binds the values of the last statements, in order; an assignment's value is the
  assigned value (`@calc -v x "y = 3"` gives 3).
- Calculator variables come back to bash only through `-v` (and `@{…}`): after
  `@calc "x = 5"`, bash has no `$x`. Write `@calc -v x "5"`, or read it later with
  `@calc -v x "x"`. Commands that set calculator variables for their caller declare them:
  `info.imstats im` sets `imsize0`... `vmin` `vmax` `vmean` `vrms` and the percentiles in bash.
- An expression made only of numbers, operators, math functions and `s.`/`f.`/`p.` properties
  is evaluated in the shell itself, in a few microseconds, without a request to the server.
- `@config digits N` sets the number of significant digits of printed results, for this shell
  (each request carries the shell's setting).
- Statements are separated by `;` or newlines. The whole text is checked first, so a syntax error
  runs nothing; then each statement runs in turn and sees what the earlier ones did, and the first
  that fails (or, in `@test`, is false) ends the text. Properties are read when a statement runs
  and can be assigned like variables (`s.cam.kw.N += 1`). A stream name may hold `-` in an entity
  path (`s.my-cam.cnt0`); `s.cam.cnt0-1` is still a subtraction.
- A pixel of an integer image reads and writes as an integer (`im[3,4] % 10`, `im[0,0] & 0xff`).
- An image an expression makes is FLOAT, or DOUBLE when an operand is DOUBLE (`f64(...)`): integer
  images give FLOAT. `x = <expr>` replaces a local image `x` with the result, type included (an
  INT32 `x` is FLOAT after `x = x + 1`). `x += ...`, a number assigned to an image, and any
  assignment into a shared stream keep the image's type: values are truncated toward zero and
  integers wrap (300 in a UINT8 image reads 44). `@config precision` sets the precision of the
  images that module commands create, not the calculator's.
- An image `@calc` makes is local to the session; `S:name = <expr>` (or `@S:name = <expr>`) makes or
  writes the shared stream `name` instead, which other processes can read. In a
  [unit](#compute-units-without-c) it is the reverse: names are shared streams and `L:name` is
  private. Module-command image arguments take the same prefixes as `@S:name` and `@L:name`
  ([scripting](scripting.md)).
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
| `=` `+=` `-=` `*=` `/=`, `?:`   | `c ? a : b`: a scalar `c` evaluates only its branch (in a fused statement both run, a zero divisor in the other one is not an error); an image `c` selects per pixel; `im[mask] = expr` (also `+=` `-=` `*=` `/=`) writes only where `mask != 0` |
| `\|\|`, `&&`                    | skip the right side once the result is known: `0 && (x = 1)`     |
| `\|`, `^^`, `&`                 | bitwise or, xor, and (on scalars and per pixel on images)        |
| `==` `!=`, `<` `<=` `>` `>=`    | 1 or 0; on images, a 0/1 mask per pixel; comparisons may be chained (`0 <= x < n` evaluates `x` once as `(0 <= x) && (x < n)`) |
| `<<` `>>`, `+` `-`, `*` `/` `%` `@` | bit shifts (a count outside $0..63$ gives 0); `"a" + "b"` concatenates strings; `A @ B` is matrix multiplication (`matmul(A, B)`) |
| unary `-` `~` `!`, postfix `.T` | unary negation, bitwise NOT (`~`), logical NOT (`!`); `A.T` is `transpose(A)` |

`^` is not an operator (write `pow(a, b)`; `^^` is bitwise XOR). The built-in constants `pi` ($\pi$),
`nan` (`NAN`), and `inf` (`INFINITY`; `-inf` via unary `-`) are reserved: they cannot be assigned,
and no variable, image or bash variable of the same name shadows them. Built-in function names are
reserved too: a `def` or a C extension cannot take one. `S` and `L`, the letters of the
[stream prefixes](#calculator-calc-test-pipe) `S:name` and `L:name`, name images only
(`U, S, Vt = svd(A)`): they cannot be a variable, a loop variable, a `def` parameter or a unit
parameter, since `im[S:n]` would read `S:n` as a stream. Integer operands give an integer result, except `/` and a
negative power (`pow(2, 62) + 1` is exact, `pow(2, -1)` is 0.5); an integer result that overflows
becomes a double. `%` and `fmod` give C's `fmod` while $|x/y| < 2^{52}$, and NaN beyond, on numbers
and images alike (`wrap_pi` likewise while $|x| < 2^{52} \cdot 2\pi$). A number divided by 0, or
`% 0`, is an error, and so is an image divided by the number 0; an image divided by an image gives
`inf` where the divisor is 0 (except `im /= divisor`, where a zero divisor pixel is an error). Numbers may be written `0x1f`, `0o17`, `0b101`; an integer
too big for 64 bits is a double (`99999999999999999999` is `1e+20`). Bitwise
operators (`&`, `|`, `^^`, `~`, `<<`, `>>`) also fuse into image expressions via `int64_t` casts
(promoting the pass to `double` when constants exceed $2^{24}$). A kept scalar statement in a unit
carries integers on an exact 64-bit lane with the same rules, so `k * 3 % 1000` agrees with the
text path for any integer variable or parameter `k` (a literal above $2^{53}$ is still read as a
double).

A call or a slice has no space before its `(` or `[` (`imean (cam)` and `im [0]` are syntax
errors), and a slice applies to a name: give `f(x)` or `(a + b)` a name first. There is no unary
`+`. A call takes at most 16 arguments, `[e1, e2, ...]` 16 elements, a multiple assignment 8
targets, a comparison chain 8 comparisons, and a string literal 1023 bytes.

### Functions

Functions (the math ones also apply per pixel to an image):

| Kind             | Functions                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Math             | `sin cos tan asin acos atan sinh cosh tanh exp exp2 log log10 log2 sqrt cbrt ceil floor round trunc abs sign posi erf erfc isnan isfinite wrap_pi sinc`; `log` is the natural logarithm, `log10` is base 10, `trunc(x)` rounds toward zero, `wrap_pi(x)` wraps into $(-\pi, \pi]$, `sinc(x)` is $\sin(\pi x)/(\pi x)$ ($\text{sinc}(0)=1$) |
| Math, 2-3 args   | `pow(a,b)`, `atan2(y,x)`, `hypot(x,y)`, `fmod(a,b)`, `min(a,b)`, `max(a,b)`, `clamp(x,lo,hi)` |
| Datatype casts   | `f32 f64 u8 i8 u16 i16 u32 i32 u64 i64`: cast a scalar or produce an image of that type  |
| Image to number  | `itot imin imax imean istd irms ivar imedian norm asum`, `perc(im,f)`, `dot(a,b)`, `trace(A)` |
| Image to tuple   | `m, s[, lo, hi] = istats(im)`: mean, std, min, and max; `cx, cy[, flux] = icentroid(im[, th])`: thresholded flux-weighted CoG; `px, py = iargmax(im)` / `iargmin(im)` (2-D) and `px, py, pz = iargmax(cube)` / `iargmin(cube)` (3-D; `-1` when all NaN) |
| Image to integer | `icount(im)` (non-zero pixels), `iargmax(im)`, `iargmin(im)`, `iamax(im)` (index of $\max \|x_i\|$) |
| Along one axis   | `itot imin imax imean istd irms ivar norm asum`, `cumsum(im[,axis])` plus an axis: `imean(cube, 2)` is the mean frame |
| Stencil / LUT    | `shift(im,dx[,dy])` (edge-clamped 2-D neighbor offset), `roll(im,dx[,dy[,dz]])` (periodic wrap-around shift), `lut(x,tbl[,x0,dx])` (1-D linear LUT) |
| Spatial / Remap  | `medfilt3(im)`, `medfilt5(im)`, `medfilt7(im)` ($3\times3$, $5\times5$, $7\times7$ median), `imin3(im)`, `imax3(im)` ($3\times3$ min/max erosion/dilation), `boxmean(im,r)` (separable $(2r+1)\times(2r+1)$ box mean, $r \in \{1..64\}$), `conv2(im,K)` (2-D convolution with odd kernel `K` up to $31\times31$), `gauss(im,sx[,sy])`, `bblur1pix(im[,amp,niter])`, `rotate2d(im[,angle])`, `circsym(im[,cx,cy])`, `circasym(im[,cx,cy])`, `interp2(im,xmap,ymap)`, `pixremap(im,map[,fill])`, `pixunmap(im,map[,n])`, `rprof(im[,cx,cy])` |
| Temporal / Delay | `streamdelay(im[,weights], delay=..., unit=..., interp=..., avemode=..., window=..., buffsize=...)`: per-call-site ring-buffer delay, fractional interpolation, causal window average (`avemode` 1..4), or 1-D FIR filter in seconds (`unit=0`) or frames (`unit=1`) |
| Image to image   | `where(c,a,b)`, `transpose(im)` (or `im.T`), `reshape(im,nx[,ny,nz])`, `rot90(im[,k])`, `hstack(A,B,...)`, `vstack(A,B,...)`, `dstack(A,B,...)`, `cat(axis,A,B,...)`, `vec(x0,...)` (or `[x0, ...]`; `vec(im)` flattens an image), `linspace(a,b,n)`, `[expr for x = a:b[:s], ...]`, `zeros(nx[,ny[,nz]])`, `eye(N[,M])`, `diag(X[,k])`, `matmul(A,B)` (or `A @ B`), `outer(x,y)`, `einsum("subscripts",A[,B])` |
| BLAS (in-place)  | `scal(a,x)`, `axpy(a,x,y)`, `gemv(a,A,x,b,y)`, `symv(a,A,x,b,y)`, `ger(a,x,y,A)`, `syrk(a,A,b,C)`, `gemm(a,A,B,b,C)`; a transposed operand is written `A.T` |
| LAPACK / Solvers | `trsv(A,b[,uplo[,diag]])`, `trsm(A,B[,uplo[,diag[,a]]])`, `solve(A,B[,reg])`, `gesv(A,B)`, `posv(A,B[,uplo])` (`"L"` by default), `gels(A,B)`, `inv(A)`, `pinv(A[,rtol])`, `s = svd(A)` / `U, s, Vt = svd(A)` (thin SVD), `w = eigh(C)` / `w, V = eigh(C)` (symmetric eigendecomposition, ascending) |
| Spectral (FFTW)  | `fft(im)` (forward 1-D/2-D/3-D DFT to complex float, or complex double for float64 input), `ifft(im)` ($1/N$-normalized inverse DFT), `fftshift(im)` (swap half-spaces on all axes) |
| Coordinates/PRNG | `ix iy ir itheta (im[, cx, cy])`: per pixel x, y, radius, angle from `(cx, cy)`, by default `(xsize/2, ysize/2)`; `rand([im[, seed]])` ($[0,1)$), `randn([im[, seed]])` ($\mathcal{N}(0,1)$) |
| Strings          | `strlen toupper tolower`, `substr(s,start,len)`, `replace(s,old,new)`, `hex oct bin`     |
| Complex images   | `cplx(re,im)`, `expi(phi)`, `real imag abs arg conj`; `+ - * /` and `-z` work on them    |
| User functions   | `def f(x, a, ...) = <expr>`: inline function usable on scalars and fused into image passes |

`imin` and `imax` (and their axis forms) skip NaN pixels and give NaN only when every pixel is NaN.
`imedian` and `perc` do not handle NaN, which sorts anywhere: leave NaN pixels out with a mask,
`imedian(im, isfinite(im))`. The other reductions propagate NaN. `imedian` of an even number of
pixels is the upper of the two middle ones; `istd` is the population standard deviation and `irms`
the root mean square, $\sqrt{\text{mean}(x^2)}$. `shift(im, dx, dy)` reads `im[x+dx, y+dy]`, the
nearest edge pixel past the border, while `roll(im, dx, dy)` wraps around periodically
(`((x-dx) mod nx, (y-dy) mod ny)`). In slice bounds, `end` resolves to `size[axis] - 1`
(`im[1:end-1, 0:end]`), whatever variable is named `end`. `icentroid` gives 0-based pixel
coordinates, and `iargmax`/`iargmin` unpack 2-D (`px, py = iargmax(im)`) or 3-D
(`px, py, pz = iargmax(cube)`) coordinates directly (`-1` when all pixels are NaN).

`rand` and `randn` give each pixel a value that depends on the seed and on the pixel's index
only, so an image is the same whatever the thread count, and `rand(seed)` is its pixel 0. Every
distinct seed, fractional or large, is another image, from a 64-bit key; without a seed each call
draws a new one, and `rand` and `randn` of the same seed are unrelated. Both hash the pixel
index with the key, in 32-bit steps that vectorize well. For `rand` a float pixel has 24 random
bits, a double one 32 and 20 more derived from them. `randn` is a Box-Muller transform of two
such values: normal out to 6.6 $\sigma$ in a float32 pass and 8.5 $\sigma$ in a double one. On
one thread `rand` costs about 0.5 ns per pixel (1 ns in double) and `randn` about 2 ns (4.5 ns).

A second image argument is a mask: `imean(im, mask)` (also `itot imin imax istd irms ivar imedian
norm asum`) and `perc(im, f, mask)` reduce over the pixels where the mask is not zero; no pixel
gives NaN, or 0 for `itot`/`asum`/`norm`. A number instead is an axis. On the left-hand side of an
assignment, `im[mask] = expr` (and `+=` `-=` `*=` `/=`) writes only the pixels where `mask` (an
image or boolean expression of `im`'s shape) is non-zero, leaving other pixels untouched.
Matrices are column-major: a 2-D image is a matrix with `size[0]` (x) rows and `size[1]` (y)
columns, so `A[i, j]` is row `i`, column `j`; a 1-D image is a column vector; and a cube
(nx, ny, nz) is an (nx·ny) × nz matrix whose columns are its planes (`reshape(cube, nx*ny, nz)` is
the same matrix as a 2-D image).
`matmul` (or `@`), `gemm`, `gemv`, `syrk`, `trsv`, `trsm`, `gesv`, `gels`, `solve`, `inv`, and
`pinv` fold a top-level `transpose(A)` or `A.T` directly into the BLAS/LAPACK kernel without
allocating an intermediate transposed image. The BLAS routines (`scal`, `axpy`, `gemv`, `symv`,
`ger`, `syrk`, `gemm`) use strict BLAS parameter order and mutate their destination image in-place
(`gemm(alpha, A, B, beta, C)` updates `C` with $\alpha\,\text{op}(A)\,\text{op}(B) + \beta\,C$),
while algebraic assignments using `matmul(A, B)`, `A @ B`, and `outer(x, y)` (such as
`C = alpha * (A @ B) + beta * C`, `dm = leak * dm - gain * (Cmat @ wfs)`,
`A += alpha * outer(x, y)`, and `G = alpha * (A.T @ A) + beta * G`) automatically lower to the same
single in-place BLAS pass. `einsum("subscripts", A[, B])` evaluates Einstein summation contractions,
reductions, traces, diagonals, and transposes over 1-D to 3-D tensors (routing standard 2-D matrix
and vector contractions to BLAS). `reshape(im, nx[, ny, nz])` reinterprets the contiguous elements
of `im` with new dimensions (up to one dimension may be `-1`), `rot90(im[, k])` rotates a 2-D image
counter-clockwise by $90^\circ \times k$, and `hstack`, `vstack`, `dstack`, and `cat(axis, A, B, ...)`
concatenate up to 8 images. `[e1, e2, ...]` constructs a 1-D vector (`vec(e1, e2, ...)`),
`linspace(a, b, n)` builds an $n$-point linearly spaced vector, and array comprehensions
`[expr for x = a:b[:s], ...]` build 1-D, 2-D, or 3-D float images from loop bounds. `medfilt3`,
`medfilt5`, and `medfilt7` apply exact $3\times3$, $5\times5$, and $7\times7$ 2-D median filters
with edge-clamped borders; `imin3` and `imax3` apply $3\times3$ min/max filters that skip NaN pixels
(an all-NaN window gives NaN); `boxmean(im, r)` runs a separable $(2r+1)\times(2r+1)$ box mean
($r \in \{1..64\}$); `conv2(im, K)` convolves `im` with an odd-sized 2-D kernel `K` (up to
$31\times31$; the kernel is flipped, as in a convolution); `gauss(im, sigma[, sigmay])` runs a separable
2-pass 2-D Gaussian blur (radius $\min(\lceil 3\sigma\rceil, 32)$); `interp2(im, xmap, ymap)`
gathers 2-D sub-pixel bilinearly interpolated values at `(xmap, ymap)`; `pixremap(im, map[, fill])`
gathers `im[round(map)]` (filling out-of-range indices with `fill`, default `0`), while
`pixunmap(im, map[, n])` computes the per-bin mean into a 2-D `(n, 1)` image (`n` defaults to
$\max(\text{map})+1$, negative indices ignored) and `rprof(im[, cx, cy])` computes the azimuthal
radial profile around `(cx, cy)` (default `(xsize/2, ysize/2)`).

```bash
@calc -v core 'imean(cam, ir(cam) < 10)'          # mean within 10 pixels of the centre
@calc "cam[cam > 4000] = 0"                        # masked LHS assignment in one pass
@calc --list                                       # list built-in functions, operators & extensions
@calc --explain "out = cam * g; z = cplx(out, out)" # show compiled C kernel / plan per statement
@calc --list stencil                               # filter by keyword or category (--json for JSON)
@calc "ramp = ix(cam, 0, 0) / s.cam.xsize"         # 0 ... 1 along x
@calc "z = (cam - imean(cam)) / istd(cam)"         # multi-reduction in one fused expression
@calc -v r "irms(cam - imean(cam))"                # nested reduction
@calc "m, s, lo, hi = istats(cam)"                 # single-pass mean, std, min, max unpacking
@calc "cx, cy, flux = icentroid(cam, 10.0)"        # thresholded sub-pixel center of gravity
@calc "px, py = iargmax(cam)"                      # 2-D peak coordinates (px, py, pz on a 3-D cube)
@calc "u, v = cam + 1, cam * 2"                    # single-pass multi-image assignment
@calc "dm = leak * dm - gain * (Cmat @ wfs)"       # in-place BLAS gemv via @
@calc "G = A.T @ A; U, s, Vt = svd(A)"             # syrk via .T and @; thin SVD unpacking
@calc "win = [exp(-(pow(x-32, 2) + pow(y-32, 2))/50) for x = 0:63, y = 0:63]"
@calc "grid = [sin(2*pi*x/64) * cos(2*pi*y/64) for x = 0:63, y = 0:63]"
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

# per-frame leaky integrator (acc is auto-created at zero on frame 0 if it does not yet exist,
# or write out = ema(cam - dark, a) to keep that state hidden, updated just before the pass)
milk-fpsexec-milk-calc -n ema -t cam -p "a:float=0.1" \
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
| `--list [PATTERN]` | | print the built-in functions, operators, constants, and C extensions, as `@calc --list` |
| | `.unit.error` | set by the unit: why the loop stopped |

- `fpsinit`, and `exec` when the FPS does not exist yet, create the FPS from the unit options. On an
  existing FPS, `-e` `-t` `-f` `-i` `-I` (with `set`, `exec` or `runstart`) write their `.unit.*`
  entry, as `@set` does; `-p` cannot change: run `fpsinit` again to declare other parameters.
- The statements are compiled when the loop starts, and again at the frame after `.unit.expr` or
  `.unit.file` is set (set `.unit.file` again to read the file again). A change that does not
  compile is not applied: the loop keeps running the previous statements and says why on stderr
  and in its process message. `.unit.init` (or
  `.unit.initfile`) runs once before frame 0 to create accumulator streams or set initial values;
  changing `.unit.expr` or `.unit.file` live keeps that state intact, while setting `.unit.init` or
  `.unit.initfile` runs the initialization again and resets the frame counters. When a unit
  statement reads its own output stream (`acc = (1 - a) * acc + a * cam`) and that stream does not
  yet exist on frame 0, it is automatically created and zero-initialized with the shape and datatype
  of the expression's other image operands, so `-i "acc = cam * 0"` is optional.
- Seven read-only variables are updated before each frame: `_n` (0-based frame index, `int`),
  `_t` (elapsed seconds since loop/init start, `float`), `_dt` (seconds since the previous
  frame, `float`, `0` on frame 0), from the trigger frame itself `_cnt0` (its counter),
  `_ts` (its write time, seconds since the epoch) and `_missed` (trigger frames skipped before
  it, 0 when the unit keeps up), and `_torn` (see the next point). Without a trigger stream,
  `_cnt0` is `_n`, and `_ts` and `_missed` are 0.
- Every shared input stream a statement reads, not only the trigger, is read under the torn-frame
  rule that stream statistics (`s.cam.mean`) and `@grab` also follow: a read is torn when a write
  was under way as it began or ended, or the stream's counter moved during it. What read it runs
  again 100 µs later, up to 4 reads in all, then keeps the last one: an element-wise pass, a
  reduction (`imean(cam)`), or a whole-image step (a filter, a matrix or FFT function, a C
  extension), which is posted once, after its last run. A statement that would apply twice is not
  run again: one that reads what it writes (`acc += cam`, `acc = 0.9 * acc + cam`,
  `gemm(1, A, B, 1, C)`), an `ema()`, a Tier 2 extension operator. `_torn` counts the torn reads;
  a steadily growing `_torn` means a producer faster than the unit, or a writer that never clears
  its flag.
- Statements of the program language, on top of the calculator's:
  - `if <scalar>: <statement>` runs a single statement when the condition is not 0
    (`if imax(cam) > 60000: skip`, `if _n % 100 == 0: post(out)`); this form takes no `elif`
    or `else`.
  - `if <cond> { ... } elif <cond> { ... } else { ... }` groups multiple statements into
    conditional blocks (`else if` is written `elif`); conditions are kept as compiled scalar programs (`< 1` µs per check).
  - `for k = start:stop[:step] { ... }` runs a bounded loop (inclusive of `stop`, default `step`
    `1`); more than 100,000 iterations in a frame is an error.
  - `local name = <expr>` (or `L:name = <expr>` / `@L:name = <expr>`) creates a process-private
    scratch image that is not exported to `/dev/shm`. When a `local` image is consumed only by
    the immediately following compatible statement in the same basic block, the compiler merges
    both statements into one cache-resident pass and elides the intermediate memory write.
  - `def f(x, a, ...) = <expr>` defines an inline function, at most 64 of them with 1 to 8
    parameters; a call may name them (`f(im, a=2)`), and a `def` may call another one, 4 deep at
    most. A call is inlined: each parameter is replaced by its argument, so reductions,
    library functions and slices of a parameter work as if written out (`def avg(x) = imean(x)`,
    `def nrm(x) = x / imax(x)`); the compiler computes a pure argument once per pixel. An argument
    that draws random numbers or keeps state (`rand`, `ema`, `post`, a C extension) may be used
    only once in the body. A `def` is a statement of its own, not inside `if` or `for`, and
    belongs to the text that holds it: the statements do not see a `def` of `.unit.init`. In `@calc`
    its body is checked when it is defined: every name other than a parameter, and every function,
    must exist; in a unit, where the body may use a name the program creates later, at each call.
  - `ema(expr, alpha)` maintains a per-call-site persistent exponential moving average image or
    scalar, `s = (1 - alpha) * s + alpha * expr`, updated in a step of its own before the
    statement's pass (seeded with `expr` on frame 0 after `init` or recompile).
  - `streamdelay(im[, weights], delay=..., unit=..., interp=..., avemode=..., window=..., buffsize=...)`
    maintains a private per-call-site ring buffer (`COREMOD_memory` `milk_stream_delay_*` engine)
    supporting discrete or linearly interpolated (`interp=1`) delays in seconds (`unit=0`) or
    frames (`unit=1`), causal temporal window filters (`avemode`: `1` boxcar, `2` linear, `3`
    exponential, `4` Gaussian over `window`), and arbitrary 1-D FIR linear combinations via an
    optional `weights` image. Unbound parameters are automatically exposed in the unit's FPS (use
    `streamdelay:label(...)` for multiple independent instances in one unit).
  - `a, b = expr1, expr2` assigns multiple images (fused into a single pass when compatible) or
    multiple scalars; `m, s[, lo, hi] = istats(im)` unpacks mean, std, min, and max from one
    reduction pass.
  - `s.<im>.<prop>`, `f.<fps>.<param>` and `p.<proc>.<field>` can be read directly inside kept
    expressions or assigned on the left-hand side (`f.loop.gain = ...`, `s.out.kw.STREHL = ...`).
  - `skip` ends the tick: the statements after it do not run (the history rings are still
    updated); outputs already written are posted.
  - `post(name)` posts output `name` this tick even when no statement wrote it (readers waiting on
    it wake up); writes post by themselves.
  - `hist(s, N)` declares the ring stream `s_hist`, the last `N` frames of `s` (one more axis, slot
    `n % N` holds frame `n`), filled after the statements of each tick, so inside a tick it holds
    frames `n-1` to `n-N`; `prev(s)` and `prev(s, k)` read frame `n-1` and `n-k` of it, for `k`
    from 1 to `N-1` (`diff = cam - prev(cam)`, `bg = imean(cam_hist, 2)`), as images of `s`'s shape;
    a `prev` of an undeclared stream declares a ring of depth 2. The ring is a stream, so other
    processes read it too. A program has at most 16 rings and 128 statements; `hist` and `prev`
    work only in a unit (outside one, and with `--explain`, they are an error).
  - `zeros(nx[, ny[, nz]])` makes a zero image of that shape (float; cast for another type,
    `out = f64(zeros(64, 64))`), so an output can be created with an explicit shape and type.
- An element-wise image assignment (`out = a*b + c`, also `+=` `-=` `*=` `/=` and masked
  `out[mask] = ...`), with slices (including binned `[::2b]` slices and `end`), broadcast operands
  (not for `/=`), bitwise operators, complex arithmetic, `shift()`, `roll()`, `lut()`, `ema()`,
  inline `def` functions, and axis reductions on either side (`out[0:63,:] = cam[0:63,:] - dark`,
  `hist[:,:,_n % 8] = cam`, `out = cam - imean(cam, 1)`), multi-image assignments
  (`u, v = expr1, expr2`), `out = matmul(...)` / `A @ B` / `gemm` / `gemv` / `ger` / `syrk`
  (including scaled `out = alpha * (A @ B) + beta * out` and `+=` / `-=` forms),
  `out = transpose(...)` / `A.T`, `out = reshape(...)`, `out = reshape(A @ B, ...)`, up to four
  scalar or nested reductions (`v = imean(cam)`, `v = irms(cam - imean(cam))`,
  `m, s, lo, hi = istats(cam)`, `px, py = iargmax(cam)`, `v = dot(a, b)`, `v = imean(cam, mask)`),
  or a scalar variable assignment (`=`, `+=`, `-=`, `*=`, `/=`) combining numbers, parameters,
  variables, live SHM properties, and scalar reductions (`ema = 0.9 * ema + 0.1 * irms(cam - dark)`)
  is compiled once into native C code (a cached `.so`, see [Fusion & Cost Model](#fusion-cost-model))
  or a scalar/BLAS plan and runs directly on each frame without re-parsing; consecutive compatible
  element-wise assignments in the same basic block are automatically fused into a single
  cache-resident pass. In `out /= image`, a zero pixel in the divisor is caught inside the fused
  pass and stops the unit with an error.
- An output written by several statements (`out = cam * g; out = out - imean(out)`) is posted once
  per frame, after the last statement that wrote it, so readers never see the intermediate values.
- A trigger stream re-created by its owner is remapped at the next trigger timeout (2 s), and an
  owner that has exited is reported in the unit's status message; `p.<unit>.timeouts` counts the
  timeouts.
- Threads: `.procinfo.NBthread` defaults to 0, which lets the thread policy decide (one thread
  below 512 K pixels, then $\lfloor \text{pixels} / 256\text{K}\rfloor$ up to the physical cores
  allowed by the process's affinity mask, so a unit pinned with `taskset` to one core stays
  single-threaded). A positive value, or `MILK_CALC_THREADS=N`, caps the OpenMP and OpenBLAS thread
  count at that many threads (`OMP_NUM_THREADS` is respected when neither is set).
- `-loopd SEC` ticks on an absolute deadline (`clock_nanosleep`, timer slack 1 µs below 1 ms), so
  periods of a few tens of µs hold without drift; a tick that ran long resumes from now rather
  than bursting to catch up.
- Names are shared streams: the unit is its own process, so a session's local images and calculator
  variables are out of its reach. A statement that makes a new image creates it as a stream. Shared
  streams recreated in `$MILK_SHM_DIR` with the same shape and compatible type are mapped again:
  the trigger after its next timeout, the others within half a second.
- The `-p` parameters are calculator variables that `@set` changes while the loop runs. A
  statement that assigns one (`total = itot(out)`) writes it back to the FPS on each frame, for
  `@get`, `@waitfor` and other units. A `stream` parameter (`in:stream=cam0`, `out:stream=out0`)
  binds an input or output stream name dynamically so `@set f.dsub.in=cam1` re-targets the unit on
  the next frame.
- An error stops the loop: a bad statement, or an image whose size or type changed. The target of
  the statement that failed is not posted, and its write flag is cleared, so its readers do not
  see a torn frame from then on.
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

Every image expression or reduction statement is lowered to a typed AST, normalized by algebraic and
fusion passes, emitted as restricted C99 source with `#pragma omp simd` loops (rows are split across
the calculator's thread team), compiled into `<key>.so` in the kernel cache, and kept loaded in a
4096-entry in-memory table. The cache is `$MILK_CALC_CACHE` if set, else
`${XDG_CACHE_HOME:-~/.cache}/milk/calc`, which a reboot keeps; where that directory cannot be used
(no home, not writable, mounted noexec) it is `${TMPDIR:-/tmp}/milk-calc-cache-<uid>`. On a
real-time machine whose home is on a network file system, set `MILK_CALC_CACHE` to a local disk. Run-time compiles (these kernels, [`-l` C
extensions](#auto-compilation-and-caching-l-lib) and `milk-cc`) use `$MILK_CALC_CC` (default: the
compiler milk was built with) and one flag list, `MILK_CC_FLAGS` in `CMakeLists.txt`:
`-O3 -pipe -march=native -ffast-math -fno-finite-math-only -mno-recip -fopenmp-simd` (NaN tests and
divisions stay IEEE); kernels add `-fno-associative-math` so they round as scalar evaluation does.
The key covers the source, compiler, flags and CPU; the cache directory must be a real directory
owned by you (a symlink or another user's directory is refused), and beyond 4096 kernels those
loaded longest ago are deleted. Runtime scalar values, slice bounds (`im[k:k+7, :]`), `shift`/`roll`
offsets, and live SHM properties (`s.`, `f.`, `p.`) are passed in the kernel's call argument struct
rather than baked into the C source, so changing a variable or slice offset in a loop reuses the
already-loaded `.so` with zero recompilation.

An expression made of same-size images, numbers, arithmetic, comparisons, bitwise operators, the
per-pixel math functions, complex operations, `where`, `?:` with an image condition, `shift`,
`roll`, `lut`, `ema`, inline `def` functions, the coordinate/PRNG functions, live SHM numeric
properties, and up to four full-image, nested, or masked scalar reductions
(`(a - imean(a)) / istd(a)`, `irms(a - imean(a))`, `imean(a, mask)`, `dot(a, b)`) runs in one fused
SIMD/OpenMP C pass without temporary images, however many operators it has, and assigning it to an
existing image of the same size and type writes in place. Multiple reductions over the same
sub-expression (`(a - imin(a)) / (imax(a) - imin(a))`, `m, s, lo, hi = istats(a)`) are deduplicated
into a single reduction pass. Passes and reductions size their thread team to one thread below
512 K pixels, then $\lfloor \text{pixels} / 256\text{K}\rfloor$ up to the physical core count (`MILK_CALC_THREADS=N` or
`.procinfo.NBthread` overrides it; `OMP_NUM_THREADS` is used when neither is set; OpenBLAS calls
size their pool the same way).

A pass runs in float32 when every operand and destination is `float` or an 8- or 16-bit integer
(`u8 i8 u16 i16`, exact in float32: a camera frame needs no cast), and in double when any of them is
`double` or a 32- or 64-bit integer; a stencil, `lut`, `pixremap` or `interp2` on an integer image
keeps its pass in double. `@calc --explain` shows which.

`@calc --explain "<statements>"` compiles the statements as a unit would and prints, per statement,
how it executes: a fused pass with its shape, precision, operand count and thread count (the one
this session gives a pass of that size; a unit's `.procinfo.NBthread` can differ), a statement
merged into an earlier pass, a scalar statement, or a constructor or library step. The last line
gives the number of statements and of steps they run in. It does not run them, so no image or
variable is written or created. A pass whose shape is
only known at run time, such as a slice of a target the statements create, shows `a shape set at run
time`, and operand shapes that do not match are reported when the statement runs.

Slices, broadcasting and axis reductions are part of the one pass too: a slice operand
(`a[0:63,:]`, `cube[:,:,k]`, `a[::2,-*]`, `a[::2b,*]`) is read through a strided or binned view of
its image (no copy), an operand whose extent is 1 on an axis is broadcast along it
(`a - row - col`, `cube - dark`), a sliced or masked left-hand side (`out[0:63,:] = …`,
`hist[:,:,_n % 8] = cam`, `out[mask] = …`) is written in place, and `imean(cube, 2)`,
`istd(a, 0)` and the other axis forms are computed in a pre-stage before the pixel loop and read as
operands on their own axis (`a - imean(a, 1)` subtracts each column's mean, `a - imean(a, 0)` each
row's, also when `a` is not square; assigned on their own they keep the squeezed shape, `ny` values
for `imean(a, 0)`). `where(c, a, b)` and `c ? a : b` with a scalar `c` and image branches are fused
too. Bounds that hold variables (`out[k:k+7,:]`, `ring[:,:,_n % 8]`, `prev(cam)`) are evaluated each
time the statement runs and passed into the compiled C kernel; a bound that would change the slice's
shape is an error. Order statistics (`imedian`, `perc`) and whole-image library operators (the
spatial filters `medfilt*`, `imin3`, `imax3`, `boxmean`, `conv2` and `gauss`, which are COREMOD_arith
kernels with clamped edges that filter a cube plane by plane, and `fft`, `ifft`, `svd`, `eigh`,
solvers) run as pre-stages into
scratch images reused across frames from a 4-buffer (256 MiB) pool (`_tmpcalcpool<k>`). A unit plans
each new `fft` or `ifft` shape with `FFTW_MEASURE`: the first frame that uses it can take up to
seconds, and large transforms then run faster (about 5× on a 2048² complex one). `@calc` plans
instantly with `FFTW_ESTIMATE`.

## Bringing Your Own Compiled C Code (`milk_calc_ext.h` & `milk-cc`)

Users can extend `@calc` and `milk-fpsexec-milk-calc` with compiled C code (`-l <file.c|file.so>`)
or compile standalone `MILK_RUNNABLE` compute units with a single command (`milk-cc <unit.c>`),
without editing CMake files. **[C Extensions (`milk_calc_ext.h` & `milk-cc`)](calc_c_extensions.md)**
describes the tiers, their registration macros, how their parameters appear in a unit's FPS, and the
`milk_calc_ext.h` API.

### Auto-Compilation and Caching (`-l` / `--lib`)

Passing a `.c` file to `@calc --lib my_kernels.c "..."` or
`milk-fpsexec-milk-calc -l my_kernels.c ...` compiles it as a kernel is compiled (same compiler,
flags and cache directory, see [Fusion & Cost Model](#fusion-cost-model)) into `ext_<key>.so`, keyed
on the file's path and contents, reuses that `.so` until the file changes, and loads it via
`dlopen()`; a compile error reports the compiler's first error line. A `.c` file edited after it
was loaded is compiled and loaded again the next time it is named, and replaces what it
registered; a `.so` rebuilt at the same path is taken by a new session or unit only. A `.c` file
is compiled as C; there is no C++ path. `MILK_CALC_LIBS` (files separated by spaces, `,` or `:`)
loads extensions in every calculator call, and a `loadlib <file>` statement loads one from
calculator text.

---

← [Scripting](scripting.md) · [Stream Slicing](slicing.md) · [C Extensions (`milk_calc_ext.h`)](calc_c_extensions.md) · [Documentation Index](../index.md)
