---
tags: [changelog, releases]
---

# What's New

Significant features and upgrades, newest first. Minor bugfixes are omitted — see the
[commit log](https://github.com/milk-org/milk/commits/framework-dev) for the full history.

**Filter by tag** — use your browser's find (Ctrl+F) with any `#tag` below: `#cli` `#performance`
`#streams` `#fps` `#build` `#docs` `#api` `#refactor`

---

## framework-dev

### 2026-03

- **2026-03-29** — Complete CLI/Scripting layer decoupling; introduction of `libmilkscript` engine
  and standalone `milk-script` interpreter `#cli` `#refactor`
- **2026-03-26** — `@seq.NAME.prop` variable expansion for sequencer state in CLI scripts `#cli`
  `#fps` ([#183](https://github.com/milk-org/milk/issues/183))
- **2026-03-25** — `milk-seq` standalone sequencer, fpsCTRL and milk-cli integration `#fps` `#cli`
- **2026-03-25** — Module dependency system with `MODULE_DEPS` macro `#api` `#build`
  ([PR #169](https://github.com/milk-org/milk/pull/169))
- **2026-03-24** — Remove legacy `s>` stream modifier; consolidate to `@S:` syntax `#streams`
  `#refactor` ([PR #168](https://github.com/milk-org/milk/pull/168))
- **2026-03-24** — Split CLIcore monolithic sources into smaller modules `#cli` `#refactor`
  ([PR #165](https://github.com/milk-org/milk/pull/165))
- **2026-03-24** — CLI bitwise ops, string functions, processinfo integration `#cli`
  ([PR #164](https://github.com/milk-org/milk/pull/164))
- **2026-03-24** — Compound assignments, ternary operator, numeric literals, `until` loops `#cli`
  ([PR #163](https://github.com/milk-org/milk/pull/163))
- **2026-03-24** — Dual prompt/command history with type-tagged log `#cli`
  ([PR #161](https://github.com/milk-org/milk/pull/161))
- **2026-03-23** — FITSIO-like stream slicing syntax (`im[0:19,10:29]`) `#streams` `#cli`
  ([PR #157](https://github.com/milk-org/milk/pull/157))
- **2026-03-23** — CLI pipe operator `|>`, `on_fpschange`, `listim` glob, error context `#cli`
  ([PR #153](https://github.com/milk-org/milk/pull/153))
- **2026-03-23** — Notify user of adopted default arguments `#cli`
  ([PR #152](https://github.com/milk-org/milk/pull/152))
- **2026-03-23** — Advanced CLI introspection, regex `=~`, array splats, dynamic PS1 `#cli`
  ([PR #151](https://github.com/milk-org/milk/pull/151))
- **2026-03-22** — Native shell expansions and arithmetic evaluation in CLI scripts `#cli`
  ([PR #150](https://github.com/milk-org/milk/pull/150))
- **2026-03-22** — Advanced script template and argparse improvements `#cli`
  ([PR #147](https://github.com/milk-org/milk/pull/147))

### 2026-02

- **2026-02** — Remove function-pointer overhead from pixel loops `#performance`
  ([PR #146](https://github.com/milk-org/milk/pull/146))
- **2026-02** — Dynamic FIFO handling for CLI `#cli`
  ([PR #144](https://github.com/milk-org/milk/pull/144))
- **2026-02** — Typed CLI variables & subprocess shell pipe parsing `#cli`
  ([PR #141](https://github.com/milk-org/milk/pull/141))
- **2026-02** — Add `-c` (command) flag to milk-cli `#cli`
  ([PR #139](https://github.com/milk-org/milk/pull/139))
- **2026-02** — Bi-directional environment variable sync and subshell assignments `#cli`
  ([PR #138](https://github.com/milk-org/milk/pull/138))
- **2026-02** — Machine-readable JSON and TSV output for CLI `#cli` `#api`
  ([PR #136](https://github.com/milk-org/milk/pull/136))
- **2026-02** — CLI `wordexp()` simplification and smart fallback `#cli`
  ([PR #130](https://github.com/milk-org/milk/pull/130))
- **2026-02** — Pipe to shell commands (grep, sort, wc) `#cli`
  ([PR #126](https://github.com/milk-org/milk/pull/126))
- **2026-02** — `ghistory` / `lhistory` with time range, glob, session highlight `#cli`
  ([PR #124](https://github.com/milk-org/milk/pull/124))
- **2026-02** — Break help into topics; `help-<topic>` commands `#cli`
  ([PR #121](https://github.com/milk-org/milk/pull/121))
- **2026-02** — CLI scripting variables, aliases, bookmarks, and history `#cli`
  ([PR #116](https://github.com/milk-org/milk/pull/116))
- **2026-02** — Session-aware history commands `#cli`
  ([PR #111](https://github.com/milk-org/milk/pull/111))
- **2026-02** — Advanced scripting: env vars, conditional chaining, history, continuation `#cli`
  ([PR #107](https://github.com/milk-org/milk/pull/107),
  [PR #109](https://github.com/milk-org/milk/pull/109))

### 2026-01

- **2026-01** — Standalone `fpsexec` executables for image_gen (mkpolygon, mkdisk, mkspdisk) `#fps`
  `#build` ([PR #103](https://github.com/milk-org/milk/pull/103),
  [PR #104](https://github.com/milk-org/milk/pull/104))
- **2026-01** — Fully static musl fpsexec build support `#build` `#performance`
  ([PR #99](https://github.com/milk-org/milk/pull/99))
- **2026-01** — `USE_STATIC_LTO` cross-module LTO for fpsexec standalones `#build` `#performance`
  ([PR #55](https://github.com/milk-org/milk/pull/55))
- **2026-01** — Context-aware tab completion for streams and FPS `#cli`
  ([PR #92](https://github.com/milk-org/milk/pull/92))
- **2026-01** — Runtime perf sweeps 1–10: SIMD alignment, restrict, omp simd, BLAS MVM, powf
  specialization, `__builtin_memcpy`, UINT16 fast-path `#performance`
  ([PR #53](https://github.com/milk-org/milk/pull/53),
  [PR #91](https://github.com/milk-org/milk/pull/91))
- **2026-01** — PGO per-executable profile isolation `#build` `#performance`
- **2026-01** — GCC optimization infrastructure (`milk_compiler.h`) + codebase-wide annotations
  `#performance` `#api`
- **2026-01** — Reduce transitive includes in `CLIcore.h` and `fps.h` `#build` `#refactor`
  ([PR #54](https://github.com/milk-org/milk/pull/54))
- **2026-01** — MkDocs Material documentation site `#docs`
  ([PR #57](https://github.com/milk-org/milk/pull/57))
- **2026-01** — Codebase-wide IMGID API migration (streams, FFT, linalgebra, image_basic, …) `#api`
  `#refactor`
- **2026-01** — Gen 4 V2 CLI registration for all modules `#cli` `#refactor`
- **2026-01** — FPS parameter tab-completion + argument hints `#cli` `#fps`
  ([PR #71](https://github.com/milk-org/milk/pull/71))

---

## dev

_No recent significant changes. Entries will be added here when `framework-dev` features are merged
into `dev`._
